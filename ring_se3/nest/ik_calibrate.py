"""The readout that turns a packet's place into a number (the *ramp readout*), measured once in NEST.

The comparison of positions needs each coordinate of the packets as a value that populations
can add and multiply.  Every coordinate is on two sheets.  The neurons of both, each at its
scaled place ``c``, send their spikes to one population that holds a pair,

    ramp  = Σ_j r_j c_j,a / (S_0 · RAMP_SPAN)            "a number that rises along the sheets"
    level = (Σ_j r_j / S_0 − 1) / LEVEL_SPAN             how strong the two packets are

with ``r_j`` the neurons' rates, ``a`` the coordinate and ``S_0`` the spikes per second of two
packets as they usually are.  The ramp alone would not do: an anchored packet is far stronger at
some places than at others, and the ramp grows with the packets' strength as well as with their
place (the amplitude sensitivity of the SO-2 repo's ramp comparator).  The population's readout
is therefore fitted to *both* values at once and gives the place, ``Σ r c / Σ r``, whatever the
strength.  Pooling the two sheets does a second thing: a packet that is weak counts little.

The sheets of the inverse-kinematics network have *level cells*
(:func:`ring_se3.nest.sheet.connect_level`), which hold a packet's strength near a set point; the
readout is fitted on sheets that have them, and its second value takes care of what they leave.

The fit is made on three real sheets, anchored through real gate populations at places on a
grid, to the centre of the spikes of the two packets in the same half second; at every place
the packets are also made weaker and stronger on purpose.  No pose of any chain is used.  The
file is stored in ``artifacts/`` as ``ramp_`` + the name of the pose calibration.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ring_so3.design import Design
from ring_so3.nest.calibrate import Calibration
from ring_so3.nest.populations import Population, connect_weights, create_population, decoded_weights, set_value
from ring_so3.nest.ring import RING_NEURON
from ring_so3.neurons import NeuronModel, Tuning, make_tuning, solve_decoders

from ..sheet import PLANES, SheetDesign, SheetMap
from . import calibrate as pose_calibrate
from .calibrate import PoseCalibration, _kernel, create_bias_cells
from .sheet import LevelDesign, Sheet, build_sheet, connect_anchor, connect_anchor_bias, connect_level

FORMAT = 1
#: neurons of one ramp readout, and the time constant of the synapses from the sheet onto it (a packet sends
#: only some 400 spikes a second: with the usual 5 ms every spike would show)
RAMP_NEURONS = 160
RAMP_TAU_MS = 30.0
#: the pair it holds: the ramp over RAMP_SPAN, and the packet's strength (1: as usual) less 1 over LEVEL_SPAN
RAMP_SPAN = 1.5
LEVEL_SPAN = 0.7
RAMP_RADIUS = 1.3
RAMP_SEED = 77
#: the noise term of the fit, as a share of the largest rate (smaller than for the other readouts: there are many more samples)
RAMP_REGULARISATION = 0.01
TICK_MS = 50.0
#: places of the calibration grid along one side of the workspace, how far the anchor goes per tick between two
#: of them (scaled units), and how long it rests at each: in all, and the part that is measured
GRID = 9
CHECK_GRID = 6
STEP_UNITS = 0.06
HOLD_TICKS = 16
MEASURE_TICKS = 10
#: places are taken inside the workspace and a little beyond its edge
PLACE_RADIUS = 1.05
#: at every place the packets are measured as they are and with these currents (pA) added to every sheet neuron,
#: which makes them weaker or stronger as an error of the anchor's height does
STRENGTH_OFFSETS_PA = (0.0, -25.0, 25.0)


def ramp_tuning(design: Design, neurons: int = RAMP_NEURONS, tau_ms: float = RAMP_TAU_MS) -> Tuning:
    """A population that holds a pair, with the slow synapse that reads a sheet."""

    model = NeuronModel(design.model.C_m, design.model.tau_m, design.model.t_ref, design.model.V_th, float(tau_ms))
    return make_tuning("phasor", int(neurons), 1000 * int(design.seed) + RAMP_SEED, model, radius=RAMP_RADIUS)


def sheets_of(coordinate: int) -> List[Tuple[int, int]]:
    """The sheets that hold a coordinate (0: x, 1: y, 2: z): ``(index of the sheet, which of its two coordinates)``."""

    return [(index, plane[1:].index(int(coordinate))) for index, plane in enumerate(PLANES) if int(coordinate) in plane[1:]]


def ramp_inputs(mapping: SheetMap, axis_in_plane: int, total_rate_hz: float) -> np.ndarray:
    """What one spike per second of every neuron of one sheet adds to the pair, (2, sheet neurons)."""

    places = mapping.places[:, int(axis_in_plane)]
    return np.stack([places / (float(total_rate_hz) * RAMP_SPAN), np.full(mapping.count, 1.0 / (float(total_rate_hz) * LEVEL_SPAN))])


def create_ramp_readout(backend: Any, tuning: Tuning, sheets: Sequence[Sheet], coordinate: int, total_rate_hz: float, name: str,
                        seed: int = 0, record: bool = False) -> Population:
    """The population that reads one coordinate from the two of ``sheets`` (xy, xz, yz) that hold it.
    ``total_rate_hz``: the spikes per second of two packets as they usually are."""

    population = create_population(backend, tuning, name, seed=seed, record=record)
    for index, axis_in_plane in sheets_of(coordinate):
        sheet = sheets[index]
        weights = tuning.input_weights(np.eye(2)) @ ramp_inputs(sheet.mapping, axis_in_plane, total_rate_hz) / population.charge
        connect_weights(backend, sheet.neurons, population.neurons, weights)
    constant = tuning.input_weights(np.array([[0.0], [-1.0 / LEVEL_SPAN]]))[:, 0]           # the "less 1" of the level
    current = np.asarray(backend.GetStatus(population.neurons, "I_e"), dtype=float) + constant
    backend.SetStatus(population.neurons, [{"I_e": float(c)} for c in current])
    return population


def pooled_centre(mapping: SheetMap, counts: np.ndarray, coordinate: int) -> float:
    """Centre, along one coordinate, of the spikes of the two sheets that hold it (``counts``: (3, neurons)); NaN without spikes."""

    total, moment = 0.0, 0.0
    for index, axis_in_plane in sheets_of(coordinate):
        c = np.asarray(counts[index], dtype=float)
        total += float(c.sum())
        moment += float(c @ mapping.places[:, axis_in_plane])
    return moment / total if total > 0 else float("nan")


@dataclass
class RampCalibration:
    settings: Dict[str, Any]
    decoders: Dict[str, np.ndarray]
    fit: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_rate_hz(self) -> float:
        return float(self.settings["total_rate_hz"])

    def tuning(self, design: Design) -> Tuning:
        return ramp_tuning(design, int(self.settings["neurons"]), float(self.settings["tau_ms"]))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **{"decoders_" + key: value for key, value in self.decoders.items()})
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as stream:
            json.dump({"settings": self.settings, "fit": self.fit}, stream, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "RampCalibration":
        data = np.load(path)
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as stream:
            meta = json.load(stream)
        return cls(meta["settings"], {key[len("decoders_"):]: data[key] for key in data.files}, meta.get("fit", {}))


def grid_places(points: int = GRID, radius: float = PLACE_RADIUS, shift: float = 0.0) -> np.ndarray:
    """Places on a cubic grid inside a ball, (k, 3), in an order in which the anchor never has far to go."""

    axis = np.linspace(-radius, radius, int(points)) + float(shift)
    rows = []
    for i, x in enumerate(axis):
        for j, y in enumerate(axis if i % 2 == 0 else axis[::-1]):
            column = axis if (i * len(axis) + j) % 2 == 0 else axis[::-1]
            rows += [(x, y, z) for z in column if x * x + y * y + z * z <= radius * radius]
    return np.array(rows)


def measure(backend: Any, design: Design, base: Calibration, pose: PoseCalibration, mapping: SheetMap, places: np.ndarray,
            tuning: Tuning, total_rate_hz: float, threads: int = 8, seed: int = 1, hold_ticks: int = HOLD_TICKS,
            measure_ticks: int = MEASURE_TICKS, step_units: float = STEP_UNITS, offsets_pa: Sequence[float] = (0.0,),
            level: Optional[LevelDesign] = None, level_decoders: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Three sheets anchored through three gate populations rest at each of ``places`` (scaled, (k, 3)), once for
    each of ``offsets_pa`` (a current added to every sheet neuron).  Per place and offset: the rates of the three
    ramp readouts (3, neurons), the centre of the spikes of the two sheets that hold each coordinate (3,), and each
    sheet's spikes per second (3,), all over the measured ticks; and how much each sheet's spikes per second differ
    from one measured tick to the next (their standard deviation, ``ripple``).  ``level``, ``level_decoders``: give
    the sheets level cells (``None``: without)."""

    _kernel(backend, threads, seed)
    sources = [create_population(backend, design.entry, "source%d" % i, seed=50 + i) for i in range(3)]
    gates = [create_population(backend, design.entry, "gate%d" % i, seed=60 + i) for i in range(3)]
    transform = np.ones((1, 1))
    for source, gate in zip(sources, gates):
        connect_weights(backend, source.neurons, gate.neurons, decoded_weights(base.decoders["source"], transform, gate))
    bias_cells = create_bias_cells(backend, design)
    sheets = []
    for name, first, second in PLANES:
        sheet = build_sheet(backend, name, mapping)
        connect_anchor(backend, sheet, [gates[first].neurons, gates[second].neurons], pose.decoders["gate"])
        connect_anchor_bias(backend, sheet, bias_cells, float(pose.fit["bias_rate_hz"]))
        if level is not None:
            connect_level(backend, sheet, design, level_decoders, level, seed=90 + len(sheets))
        sheets.append(sheet)
    readouts = [create_ramp_readout(backend, tuning, sheets, i, total_rate_hz, "ramp%d" % i, seed=70 + i, record=True) for i in range(3)]

    def tick(x: np.ndarray) -> np.ndarray:
        for source, value in zip(sources, x):
            set_value(backend, source, [float(value)])
        backend.Simulate(TICK_MS)
        return np.array([sheet.read(backend) for sheet in sheets])

    x = np.zeros(3)
    for _ in range(20):
        tick(x)
    rates, centres, totals, where, added, ripple = [], [], [], [], [], []
    seconds = int(measure_ticks) * TICK_MS / 1000.0
    resting = float(RING_NEURON["I_e"])
    for target in np.asarray(places, dtype=float):
        steps = int(np.ceil(np.linalg.norm(target - x) / step_units))
        for i in range(1, steps + 1):
            tick(x + (target - x) * i / steps)
        x = target.copy()
        for offset in offsets_pa:
            for sheet in sheets:
                backend.SetStatus(sheet.neurons, {"I_e": resting + float(offset)})
            for _ in range(int(hold_ticks) - int(measure_ticks)):
                tick(x)
            for readout in readouts:
                readout.counts(backend)
            per_tick = np.array([tick(x) for _ in range(int(measure_ticks))])
            counts = per_tick.sum(axis=0)
            rates.append([readout.counts(backend) / seconds for readout in readouts])
            centres.append([pooled_centre(mapping, counts, i) for i in range(3)])
            totals.append(counts.sum(axis=1) / seconds)
            ripple.append(np.std(per_tick.sum(axis=2), axis=0) / (TICK_MS / 1000.0))
            where.append(x.copy())
            added.append(float(offset))
        for sheet in sheets:
            backend.SetStatus(sheet.neurons, {"I_e": resting})
    return {"places": np.array(where), "rates": np.array(rates), "centres": np.array(centres), "totals": np.array(totals),
            "offsets_pa": np.array(added), "ripple": np.array(ripple)}


def _samples(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Rates and targets of all three coordinates in one list (they share one readout); places without spikes left out."""

    alive = np.isfinite(data["centres"])
    return data["rates"][alive], data["centres"][alive]


def run(backend: Any, base: Calibration, pose: PoseCalibration, level_decoders: np.ndarray, design: Optional[Design] = None,
        sheet: Optional[SheetDesign] = None, level: Optional[LevelDesign] = None, threads: int = 8, seed: int = 1,
        neurons: int = RAMP_NEURONS, tau_ms: float = RAMP_TAU_MS, log: Callable[[str], None] = print) -> RampCalibration:
    """``level_decoders``: the readout of the sum population that stays at its bounds (the ``bounded`` readout of
    :mod:`ring_so3.nest.ik_calibrate`), with which the level cells act on their sheet."""

    design, sheet, level = design or Design(), sheet or SheetDesign(), level or LevelDesign()
    mapping = SheetMap(sheet)
    started = time.time()
    tuning = ramp_tuning(design, neurons, tau_ms)
    common = dict(level=level, level_decoders=level_decoders)
    # how strong two packets usually are: a first pass with a guess, read from the sheets themselves
    first = measure(backend, design, base, pose, mapping, grid_places(4), tuning, 2.0 * level.set_rate_hz, threads, seed, **common)
    total_rate = 2.0 * float(np.median(first["totals"]))
    log("two packets send %.0f spikes a second in the median (%.0f s)" % (total_rate, time.time() - started))
    data = measure(backend, design, base, pose, mapping, grid_places(GRID), tuning, total_rate, threads, seed,
                   offsets_pa=STRENGTH_OFFSETS_PA, **common)
    rates, targets = _samples(data)
    decoders = solve_decoders(rates, targets[:, None], RAMP_REGULARISATION)
    fit: Dict[str, Any] = {"places": int(len(data["places"])), "samples": int(len(targets)), "total_rate_hz": total_rate,
                           "sheet_rate_median_hz": float(np.median(data["totals"])), "sheet_rate_sd_hz": float(np.std(data["totals"])),
                           "sheet_rate_smallest_hz": float(np.min(data["totals"])), "sheet_rate_largest_hz": float(np.max(data["totals"])),
                           "ramp_rms": float(np.sqrt(np.mean((rates @ decoders[:, 0] - targets) ** 2)))}
    log("ramp readout: %d measurements, fit rms %.4f of the reach (%.0f s)" % (fit["places"], fit["ramp_rms"], time.time() - started))
    # places between those of the grid, with another kernel seed: what the readout does where it was not fitted
    check = measure(backend, design, base, pose, mapping, grid_places(CHECK_GRID, PLACE_RADIUS - 0.07, 0.031), tuning, total_rate, threads, seed + 1,
                    offsets_pa=(0.0, -15.0), **common)
    rates, targets = _samples(check)
    error = rates @ decoders[:, 0] - targets
    fit.update(check_places=int(len(check["places"])), check_rms=float(np.sqrt(np.mean(error ** 2))),
               check_median=float(np.median(np.abs(error))), check_largest=float(np.max(np.abs(error))), check_bias=float(np.mean(error)),
               check_sheet_rate_sd_hz=float(np.std(check["totals"])), check_sheet_rate_smallest_hz=float(np.min(check["totals"])),
               anchor_rms=float(np.sqrt(np.mean((check["centres"] - check["places"]) ** 2))))
    log("on %d measurements it was not fitted on: rms %.4f, largest %.4f of the reach (%.0f s)" % (
        fit["check_places"], fit["check_rms"], fit["check_largest"], time.time() - started))
    settings = {"format": FORMAT, "seed": int(seed), "threads": int(threads), "neurons": int(neurons), "tau_ms": float(tau_ms),
                "ramp_span": RAMP_SPAN, "level_span": LEVEL_SPAN, "radius": RAMP_RADIUS, "total_rate_hz": total_rate,
                "grid": GRID, "check_grid": CHECK_GRID, "regularisation": RAMP_REGULARISATION, "hold_ticks": HOLD_TICKS, "measure_ticks": MEASURE_TICKS, "offsets_pa": list(STRENGTH_OFFSETS_PA),
                "level": level.describe(), "seconds": round(time.time() - started, 1),
                "pose": pose_calibrate.artifact_name(design, int(base.settings["ring_size"]), base.settings["encoder_mode"], sheet)}
    return RampCalibration(settings, {"ramp": decoders}, fit)


ARTIFACTS = pose_calibrate.ARTIFACTS


def artifact_name(design: Design, ring_size: int, encoder_mode: str, sheet: SheetDesign) -> str:
    return "ramp_" + pose_calibrate.artifact_name(design, ring_size, encoder_mode, sheet)[len("pose_"):]


def load(design: Optional[Design] = None, ring_size: Optional[int] = None, encoder_mode: str = "fractional",
         sheet: Optional[SheetDesign] = None, directory: Optional[str] = None) -> RampCalibration:
    design, sheet = design or Design(), sheet or SheetDesign()
    path = os.path.join(directory or ARTIFACTS, artifact_name(design, int(ring_size or design.sizes.ring), encoder_mode, sheet))
    if not os.path.exists(path):
        raise FileNotFoundError("no ramp calibration %s; run scripts/calibrate_ramp.py" % os.path.relpath(path))
    calibration = RampCalibration.load(path)
    if calibration.settings.get("format") != FORMAT:
        raise FileNotFoundError("ramp calibration %s was written by an older version; run scripts/calibrate_ramp.py again"
                                % os.path.relpath(path))
    return calibration


__all__ = ["ARTIFACTS", "CHECK_GRID", "FORMAT", "GRID", "LEVEL_SPAN", "RAMP_NEURONS", "RAMP_RADIUS", "RAMP_SPAN", "RAMP_TAU_MS", "RampCalibration",
           "artifact_name", "create_ramp_readout", "grid_places", "load", "measure", "pooled_centre", "ramp_inputs", "ramp_tuning", "run",
           "sheets_of"]
