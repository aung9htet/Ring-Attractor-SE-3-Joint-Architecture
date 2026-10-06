"""What the pose network needs beyond the SO(3) calibration, measured once in NEST and stored in ``artifacts/``.

* ``gate``: readout of a link-type population that gives its value ``x`` *and* ``x²`` (the anchor
  current of a sheet neuron is a weighted sum of both).  Fitted to the measured rates of that
  population on a grid of values, like the SO(3) readouts.
* ``speed curve``: how fast a packet moves when the velocity cells of one direction are driven
  more than at rest and those of the opposite direction as much less, measured on a sheet alone,
  in all four directions.
* ``velocity``: readout of the velocity population that gives, instead of the velocity itself,
  the drive the velocity cells need for the packet to move at that velocity (the inverse of the
  speed curve).

No pose of any chain is used.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ring_so3.design import Design
from ring_so3.nest import calibrate as so3_calibrate
from ring_so3.nest.calibrate import Calibration
from ring_so3.neurons import solve_decoders

from ..sheet import DIRECTIONS, SheetDesign, SheetMap
from ..wiring import QDOT_REF, VELOCITY_SCALE
from .sheet import GATE, build_sheet

FORMAT = 1
#: a velocity cell: a slow integrate-and-fire neuron, so that its rate is proportional to its input current
VELOCITY_CELL: Dict[str, float] = {"C_m": 250.0, "tau_m": 5000.0, "V_th": 15.0, "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0,
                                   "t_ref": 1.0, "V_min": 0.0, "tau_syn_ex": 5.0, "tau_syn_in": 5.0}
#: drives (share of the full rate) at which the speed curve is measured
CURVE_POINTS = 13
#: kernel seeds each point of the curve is measured with (the cells' start phases differ)
CURVE_SEEDS = (1, 2, 3)
#: a joint counts as moving above this speed, rad/s
MOVING_RAD_S = 0.02


#: bias cells carry the fixed part of the anchor: ordinary neurons of the network driven to this rate
BIAS_CELLS = 16
BIAS_RATE_HZ = 100.0


def create_bias_cells(backend: Any, design: Design) -> Any:
    model = design.model
    cells = backend.Create("iaf_psc_alpha", BIAS_CELLS, params=model.nest_params())
    start = np.random.default_rng(77).uniform(0.0, model.V_th, BIAS_CELLS)
    current = float(model.current(np.array([BIAS_RATE_HZ]))[0])
    backend.SetStatus(cells, [{"V_m": float(v), "I_e": current} for v in start])
    return cells


def measure_bias_rate(backend: Any, design: Design, threads: int = 8, seed: int = 1, seconds: float = 10.0) -> float:
    """Rate (Hz) of one bias cell as NEST runs it (its spike times fall on the 0.1 ms grid)."""

    _kernel(backend, threads, seed)
    cells = create_bias_cells(backend, design)
    recorder = backend.Create("spike_recorder")
    backend.Connect(cells, recorder)
    backend.Simulate(1000.0)
    backend.SetStatus(recorder, {"n_events": 0})
    backend.Simulate(float(seconds) * 1000.0)
    return float(backend.GetStatus(recorder, "n_events")[0]) / (BIAS_CELLS * float(seconds))


def velocity_full_current(sheet: SheetDesign) -> float:
    """Input current (pA) at which a velocity cell fires at ``velocity_rate_hz``."""

    return float(sheet.velocity_rate_hz) * VELOCITY_CELL["C_m"] * VELOCITY_CELL["V_th"] / 1000.0


def create_velocity_cells(backend: Any, sheet: SheetDesign, seed: int) -> Any:
    cells = backend.Create("iaf_psc_alpha", int(sheet.velocity_cells), params=dict(VELOCITY_CELL))
    start = np.random.default_rng([99, int(seed)]).uniform(0.0, VELOCITY_CELL["V_th"], int(sheet.velocity_cells))
    backend.SetStatus(cells, [{"V_m": float(v), "I_e": 0.0} for v in start])
    return cells


def velocity_rest_current(sheet: SheetDesign) -> float:
    return float(sheet.velocity_rest) * velocity_full_current(sheet)


def cells_per_unit_velocity(mapping: SheetMap) -> float:
    """Cells per second the packet has to move for a velocity value of 1."""

    return QDOT_REF * VELOCITY_SCALE / mapping.cell_size


@dataclass
class PoseCalibration:
    sheet: Dict[str, Any]
    settings: Dict[str, Any]
    decoders: Dict[str, np.ndarray]
    #: packet speed (cells/s) at each of ``curve_drive``, mean of the four directions, and per direction
    curve_drive: np.ndarray
    curve_speed: np.ndarray
    curve_by_direction: np.ndarray
    fit: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arrays = {"decoders_" + key: value for key, value in self.decoders.items()}
        np.savez_compressed(path, curve_drive=self.curve_drive, curve_speed=self.curve_speed,
                            curve_by_direction=self.curve_by_direction, **arrays)
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as stream:
            json.dump({"sheet": self.sheet, "settings": self.settings, "fit": self.fit}, stream, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "PoseCalibration":
        data = np.load(path)
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as stream:
            meta = json.load(stream)
        decoders = {key[len("decoders_"):]: data[key] for key in data.files if key.startswith("decoders_")}
        return cls(meta["sheet"], meta["settings"], decoders, data["curve_drive"], data["curve_speed"],
                   data["curve_by_direction"], meta.get("fit", {}))


def _kernel(backend: Any, threads: int, seed: int) -> None:
    backend.ResetKernel()
    backend.set_verbosity("M_ERROR")
    backend.SetKernelStatus({"local_num_threads": int(threads), "rng_seed": int(seed)})


def measure_packet_velocity(backend: Any, mapping: SheetMap, drive: Sequence[float], threads: int = 8, seed: int = 1,
                            seconds: float = 3.0, tick_ms: float = 50.0) -> np.ndarray:
    """Velocity (cells/s, two components) of a free packet when the velocity cells get the drive vector
    ``drive`` (shares of their full current, one per coordinate of the sheet): the cells of a coordinate's
    positive direction get that much more than at rest, those of its negative direction as much less.
    A sheet on its own with all four groups of velocity cells, no anchor."""

    from .sheet import connect_velocity_cells

    _kernel(backend, threads, seed)
    sheet = build_sheet(backend, "sheet", mapping)
    drive = np.asarray(drive, dtype=float).reshape(2)
    heading = drive / max(float(np.max(np.abs(drive))), 1e-9)
    start_cells = np.array([mapping.n / 2.0, mapping.n / 2.0]) - 0.3 * mapping.n * heading + 0.2
    sheet.write(backend, mapping.to_scaled(start_cells))
    backend.Simulate(500.0)
    sheet.write(backend, None)
    backend.Simulate(1000.0)
    sheet.read(backend)
    full, rest = velocity_full_current(mapping.design), velocity_rest_current(mapping.design)
    for label in range(4):
        cells = create_velocity_cells(backend, mapping.design, 10 * seed + label)
        backend.SetStatus(cells, {"I_e": rest + full * float(DIRECTIONS[label] @ drive)})
        connect_velocity_cells(backend, sheet, cells, label // 2, 1 if label % 2 == 0 else -1)
    track = []
    for _ in range(int(seconds * 1000.0 / tick_ms)):
        backend.Simulate(float(tick_ms))
        place = mapping.to_cells(mapping.centre(sheet.read(backend)))
        if not np.all(np.isfinite(place)) or np.any(place < 6) or np.any(place > mapping.n - 7):
            break
        track.append(place)
    if len(track) < 8:
        return np.array([np.nan, np.nan])
    t = np.arange(len(track)) * tick_ms / 1000.0
    rel = np.array(track) - track[0]
    return np.array([np.polyfit(t, rel[:, 0], 1)[0], np.polyfit(t, rel[:, 1], 1)[0]])


def measure_packet_speed(backend: Any, mapping: SheetMap, drive: float, direction: int, threads: int = 8, seed: int = 1,
                         seconds: float = 3.0, tick_ms: float = 50.0) -> Tuple[float, float]:
    """Speed (cells/s) along label ``direction``, and across it, for a drive along that direction alone."""

    along = DIRECTIONS[int(direction)]
    velocity = measure_packet_velocity(backend, mapping, float(drive) * along, threads, seed, seconds, tick_ms)
    return float(velocity @ along), float(velocity @ np.array([-along[1], along[0]]))


def drive_for_speed(curve_drive: np.ndarray, curve_speed: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Inverse of the speed curve: the drive (0 .. 1) at which the packet moves at ``speed`` cells/s (>= 0).
    The curve is made non-decreasing first; speeds beyond its end get the full drive."""

    rising = np.maximum.accumulate(np.maximum(np.asarray(curve_speed, dtype=float), 0.0))
    rising = rising + 1e-9 * np.arange(rising.size)          # strictly rising, for the interpolation
    return np.interp(np.asarray(speed, dtype=float), rising, np.asarray(curve_drive, dtype=float))


def run(backend: Any, base: Calibration, design: Optional[Design] = None, sheet: Optional[SheetDesign] = None,
        threads: int = 8, seed: int = 1, log: Callable[[str], None] = print) -> PoseCalibration:
    design = design or Design()
    sheet = sheet or SheetDesign()
    mapping = SheetMap(sheet)
    started = time.time()
    fit: Dict[str, Any] = {}
    regularisation = float(base.settings["regularisation"])

    x = np.linspace(-1.1, 1.1, int(base.settings["scalar_points"]))
    held = np.clip(x, -1.0, 1.0)
    rates = so3_calibrate.measure_driven(backend, design, "scalar", x[:, None], base.decoders["source"], threads, seed)
    d_gate = solve_decoders(rates, np.stack([held, held ** 2], axis=1), regularisation)
    fit["gate_rms"] = float(np.sqrt(np.mean((rates @ d_gate[:, :1] - held[:, None]) ** 2)))
    fit["gate_square_rms"] = float(np.sqrt(np.mean((rates @ d_gate[:, 1:] - (held ** 2)[:, None]) ** 2)))
    log("gate: value rms %.4f, square rms %.4f (%.0f s)" % (fit["gate_rms"], fit["gate_square_rms"], time.time() - started))

    fit["bias_rate_hz"] = measure_bias_rate(backend, design, threads, seed)
    log("bias cells: %.3f Hz each" % fit["bias_rate_hz"])

    drives = np.linspace(0.0, float(sheet.velocity_span), CURVE_POINTS)
    by_direction = np.full((4, drives.size), np.nan)
    across = np.full((4, drives.size), np.nan)
    for d in range(4):
        for i, drive in enumerate(drives):
            runs = [measure_packet_speed(backend, mapping, float(drive), d, threads, s) for s in CURVE_SEEDS]
            by_direction[d, i], across[d, i] = np.nanmean([r[0] for r in runs]), np.nanmean([r[1] for r in runs])
        log("speed curve, direction %d: %s cells/s (%.0f s)" % (d, " ".join("%.2f" % v for v in by_direction[d]), time.time() - started))
    curve = np.nanmean(by_direction, axis=0)
    fit["speed_full_cells_s"] = float(curve[-1])
    fit["speed_direction_spread"] = float(np.nanmax(np.nanstd(by_direction, axis=0)))
    fit["speed_across_max"] = float(np.nanmax(np.abs(across)))

    per_unit = cells_per_unit_velocity(mapping)
    wanted = np.sign(held) * drive_for_speed(drives, curve - curve[0], np.abs(held) * per_unit)
    d_velocity = solve_decoders(rates, wanted[:, None], regularisation)
    fit["velocity_rms"] = float(np.sqrt(np.mean((rates @ d_velocity - wanted[:, None]) ** 2)))
    fit["velocity_value_at_full_speed"] = float(curve[-1] / per_unit)
    log("velocity readout: rms %.4f; the packet's top speed is a velocity value of %.2f (%.0f s)"
        % (fit["velocity_rms"], fit["velocity_value_at_full_speed"], time.time() - started))

    settings = {"format": FORMAT, "seed": int(seed), "threads": int(threads), "seconds": round(time.time() - started, 1),
                "qdot_ref": QDOT_REF, "velocity_scale": VELOCITY_SCALE, "base": so3_calibrate.artifact_name(
                    design, int(base.settings["ring_size"]), base.settings["encoder_mode"])}
    return PoseCalibration(sheet.describe(), settings, {"gate": d_gate, "velocity": d_velocity}, drives, curve, by_direction, fit)


ARTIFACTS = so3_calibrate.ARTIFACTS


def artifact_name(design: Design, ring_size: int, encoder_mode: str, sheet: SheetDesign) -> str:
    base = os.path.splitext(so3_calibrate.artifact_name(design, ring_size, encoder_mode))[0]
    return "pose_%s_s%d_u%s_k%s.npz" % (base[len("calibration_"):], sheet.size, ("%g" % sheet.cell_units).replace(".", "p"),
                                         ("%g" % sheet.shift).replace(".", "p"))


def load(design: Optional[Design] = None, ring_size: Optional[int] = None, encoder_mode: str = "fractional",
         sheet: Optional[SheetDesign] = None, directory: Optional[str] = None) -> PoseCalibration:
    design, sheet = design or Design(), sheet or SheetDesign()
    path = os.path.join(directory or ARTIFACTS, artifact_name(design, int(ring_size or design.sizes.ring), encoder_mode, sheet))
    if not os.path.exists(path):
        raise FileNotFoundError("no pose calibration %s; run scripts/calibrate_pose.py" % os.path.relpath(path))
    calibration = PoseCalibration.load(path)
    if calibration.settings.get("format") != FORMAT:
        raise FileNotFoundError("pose calibration %s was written by an older version; run scripts/calibrate_pose.py again"
                                % os.path.relpath(path))
    stored, wanted = dict(calibration.sheet), sheet.describe()
    for key in ("window", "anchor_gain", "anchor_offset"):        # these do not enter the calibration
        stored.pop(key, None)
        wanted.pop(key, None)
    if stored != wanted:
        raise FileNotFoundError("pose calibration %s is for another sheet design; run scripts/calibrate_pose.py again"
                                % os.path.relpath(path))
    return calibration


__all__ = ["ARTIFACTS", "BIAS_CELLS", "BIAS_RATE_HZ", "CURVE_POINTS", "MOVING_RAD_S", "create_bias_cells", "measure_bias_rate", "FORMAT", "PoseCalibration", "VELOCITY_CELL", "artifact_name", "velocity_rest_current",
           "cells_per_unit_velocity", "create_velocity_cells", "drive_for_speed", "load", "measure_packet_speed", "measure_packet_velocity", "run",
           "velocity_full_current"]
