"""Small rigs that measure one part of the pose network on its own (test 3, and the unit tests)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ring_so3.design import Design
from ring_so3.nest.calibrate import Calibration
from ring_so3.nest.populations import connect_weights, create_population, decoded_weights, set_value

from ..sheet import DIRECTIONS, SheetMap
from .calibrate import (PoseCalibration, VELOCITY_CELL, _kernel, cells_per_unit_velocity, create_bias_cells, create_velocity_cells,
                        velocity_full_current, velocity_rest_current)
from .sheet import build_sheet, connect_anchor, connect_anchor_bias, connect_velocity_cells

TICK_MS = 50.0


def measure_hold(backend: Any, mapping: SheetMap, place: Sequence[float], seconds: float = 10.0, threads: int = 8,
                 seed: int = 1) -> Dict[str, float]:
    """A packet is written at ``place`` (scaled) and left alone: how far has it moved after ``seconds``?"""

    _kernel(backend, threads, seed)
    sheet = build_sheet(backend, "sheet", mapping)
    sheet.write(backend, place)
    backend.Simulate(500.0)
    sheet.write(backend, None)
    backend.Simulate(1000.0)
    sheet.read(backend)
    track, spikes = [], []
    for _ in range(int(seconds * 1000.0 / TICK_MS)):
        backend.Simulate(TICK_MS)
        counts = sheet.read(backend)
        track.append(mapping.centre(counts))
        spikes.append(counts.sum())
    track = np.array(track)
    if not np.all(np.isfinite(track[-10:])):
        return {"alive": 0.0, "drift_cells": float("nan"), "drift_units": float("nan"), "spikes_per_tick": float(np.mean(spikes))}
    drift = float(np.linalg.norm(np.nanmean(track[-10:], axis=0) - np.nanmean(track[:10], axis=0)))
    return {"alive": 1.0, "drift_units": drift, "drift_cells": drift / mapping.cell_size, "spikes_per_tick": float(np.mean(spikes)),
            "jitter_cells": float(np.mean(np.nanstd(track - np.nanmean(track, axis=0), axis=0)) / mapping.cell_size)}


def measure_anchor(backend: Any, design: Design, base: Calibration, pose: PoseCalibration, mapping: SheetMap,
                   places: np.ndarray, threads: int = 8, seed: int = 1, hold_ticks: int = 20,
                   step_units: float = 0.04) -> List[Dict[str, float]]:
    """A sheet anchored through real gate populations, which get their two values from driven sources.
    The anchor moves to each of ``places`` (scaled, (k, 2)) and rests there; the packet's place over the
    last half of the rest is compared with it."""

    _kernel(backend, threads, seed)
    sources = [create_population(backend, design.entry, "source%d" % i, seed=50 + i) for i in range(2)]
    gates = [create_population(backend, design.entry, "gate%d" % i, seed=60 + i) for i in range(2)]
    transform = np.ones((1, 1))
    for source, gate in zip(sources, gates):
        connect_weights(backend, source.neurons, gate.neurons, decoded_weights(base.decoders["source"], transform, gate))
    sheet = build_sheet(backend, "sheet", mapping)
    connect_anchor(backend, sheet, [g.neurons for g in gates], pose.decoders["gate"])
    connect_anchor_bias(backend, sheet, create_bias_cells(backend, design), float(pose.fit["bias_rate_hz"]))

    def tick(x: np.ndarray) -> np.ndarray:
        for source, value in zip(sources, x):
            set_value(backend, source, [float(value)])
        backend.Simulate(TICK_MS)
        return sheet.read(backend)

    x = np.zeros(2)
    for _ in range(20):
        tick(x)
    rows = []
    for target in np.asarray(places, dtype=float):
        steps = int(np.ceil(np.linalg.norm(target - x) / step_units))
        for i in range(1, steps + 1):
            tick(x + (target - x) * i / steps)
        x = target.copy()
        counts = np.array([tick(x) for _ in range(int(hold_ticks))])[int(hold_ticks) // 2:]
        centre = mapping.centre(counts.sum(axis=0))
        per_tick = [np.linalg.norm(mapping.centre(c) - x) for c in counts]
        rows.append({"place": [float(v) for v in x], "error_units": float(np.linalg.norm(centre - x)),
                     "error_cells": float(np.linalg.norm(centre - x) / mapping.cell_size),
                     "tick_error_cells": float(np.nanmedian(per_tick) / mapping.cell_size), "spikes_per_tick": float(counts.sum() / len(counts))})
    return rows


def velocity_input_weights(pose: PoseCalibration, mapping: SheetMap, sign: int) -> np.ndarray:
    """Weights (velocity cells, velocity population) that turn the velocity population's value into the cells' drive."""

    charge = float(np.e * VELOCITY_CELL["tau_syn_ex"] * 1e-3)
    row = int(sign) * velocity_full_current(mapping.design) * pose.decoders["velocity"][:, 0] / charge
    return np.tile(row, (int(mapping.design.velocity_cells), 1))


def measure_velocity_chain(backend: Any, design: Design, base: Calibration, pose: PoseCalibration, mapping: SheetMap,
                           value: Sequence[float], threads: int = 8, seed: int = 1, seconds: float = 3.0) -> Dict[str, Any]:
    """A free packet moved through the real chain: two driven sources hold the velocity values ``value`` (two
    numbers in [-1, 1]), two velocity populations read them, velocity cells read those and act on the sheet.
    Returns the packet's velocity and the velocity that was asked for, both in cells/s."""

    _kernel(backend, threads, seed)
    value = np.asarray(value, dtype=float).reshape(2)
    sheet = build_sheet(backend, "sheet", mapping)
    heading = value / max(float(np.max(np.abs(value))), 1e-9)
    sheet.write(backend, mapping.to_scaled(np.array([mapping.n / 2.0, mapping.n / 2.0]) - 0.3 * mapping.n * heading + 0.2))
    backend.Simulate(500.0)
    sheet.write(backend, None)
    backend.Simulate(1000.0)
    sheet.read(backend)
    transform = np.ones((1, 1))
    for axis in range(2):
        source = create_population(backend, design.entry, "source%d" % axis, seed=70 + axis)
        relay = create_population(backend, design.entry, "velocity%d" % axis, seed=80 + axis)
        set_value(backend, source, [float(value[axis])])
        connect_weights(backend, source.neurons, relay.neurons, decoded_weights(base.decoders["source"], transform, relay))
        for sign in (1, -1):
            cells = create_velocity_cells(backend, mapping.design, 10 * seed + 2 * axis + (sign > 0))
            backend.SetStatus(cells, {"I_e": velocity_rest_current(mapping.design)})
            connect_weights(backend, relay.neurons, cells, velocity_input_weights(pose, mapping, sign))
            connect_velocity_cells(backend, sheet, cells, axis, sign)
    track = []
    for _ in range(int(seconds * 1000.0 / TICK_MS)):
        backend.Simulate(TICK_MS)
        place = mapping.to_cells(mapping.centre(sheet.read(backend)))
        if not np.all(np.isfinite(place)) or np.any(place < 6) or np.any(place > mapping.n - 7):
            break
        track.append(place)
    wanted = value * cells_per_unit_velocity(mapping)
    if len(track) < 12:
        return {"value": value.tolist(), "wanted": wanted.tolist(), "velocity": [float("nan"), float("nan")], "ticks": len(track)}
    track = np.array(track)[4:]                # the first ticks: the populations fill up
    t = np.arange(len(track)) * TICK_MS / 1000.0
    velocity = [float(np.polyfit(t, track[:, 0], 1)[0]), float(np.polyfit(t, track[:, 1], 1)[0])]
    return {"value": value.tolist(), "wanted": wanted.tolist(), "velocity": velocity, "ticks": len(track)}


__all__ = ["measure_anchor", "measure_hold", "measure_velocity_chain", "velocity_input_weights"]
