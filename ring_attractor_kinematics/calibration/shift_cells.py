"""Shift cells: how fast they move a ring attractor's bump (``artifacts/shift_cells``).

The observer's prediction moves each observer ring's bump at the chart rate, through the shift cells of
:mod:`ring_attractor_kinematics.spiking.shift_circuit`.  This module measures the bump's speed against the drive on a ring alone
(the *speed curve*), fits its inverse (the drive that gives a wanted speed), and checks the sub-gate: after the
inverse, the bump must move at the wanted speed to within ±20% over the speeds prediction needs.  How far the
bump drifts with no drive is measured beside it (it is not part of the gate).

The first motif (format 1: feedback onto the next neuron only, inhibitory) failed this gate; its measurement is
kept in the artifact under ``first_motif``.  A few other settings of the present motif are measured for the record
(``candidates``).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..neurons.ring_layout import RingMap
from ..neurons.tuning import Tunings
from ..spiking.populations import create_kernel, create_population, set_value
from ..spiking.ring_attractor import build_ring_attractor
from ..spiking.shift_circuit import ShiftSettings, build_shift_cells
from .artifacts import ARTIFACTS, Artifact

FORMAT = 2
TICK_MS = 50.0
#: drives of the curve (a source's value: its positive half to the right side, its negative half to the left)
DRIVES = tuple(np.round(np.concatenate([-np.geomspace(1.0, 0.01, 15), [0.0], np.geomspace(0.01, 1.0, 15)]), 5))
SEEDS = (1, 2, 3, 4, 5)
#: each speed of the gate is timed over this long, on three kernel seeds
CHECK_SECONDS = 5.0
#: the speeds prediction needs: chart rates up to 2 rad/s (1 rad/s joints, 1/cos pitch held at 2), from 0.1 rad/s
NEEDED_RAD_S = (0.1, 2.0)
SUB_GATE = 0.20
#: other settings of the motif, measured for the record (rest, drive, feedback, shift, spread)
CANDIDATES = ({"rest_pa": 200000.0, "drive_pa": 150000.0, "feedback": 15.0, "shift": 12},
              {"rest_pa": 250000.0, "drive_pa": 200000.0, "feedback": 10.0, "shift": 14},
              {"rest_pa": 100000.0, "drive_pa": 120000.0, "feedback": 25.0, "shift": 10})


def measure_speed(backend: Any, tunings: Tunings, half_decoders: np.ndarray, drive: float, settings: ShiftSettings,
                  threads: int, seed: int, seconds: float = 3.0, size: int = 200) -> Tuple[float, float]:
    """Speed (rad/s) of a free bump on a circular ring with the drive held at ``drive``, and the median number of
    ring neurons that fire in a tick (the bump's width)."""

    create_kernel(backend, threads, seed)
    ring = build_ring_attractor(backend, "observer", RingMap.circle(size))
    plus = create_population(backend, tunings.entry, "drive_plus", seed=1)
    minus = create_population(backend, tunings.entry, "drive_minus", seed=2)
    build_shift_cells(backend, ring, (plus, half_decoders), (minus, half_decoders), settings)
    for _ in range(10):                                   # write a bump, then let it go
        ring.drive(backend, 0.0)
        backend.Simulate(TICK_MS)
        ring.read(backend)
    backend.SetStatus(ring.generators, {"rate": 0.0})
    set_value(backend, plus, [max(drive, 0.0)])
    set_value(backend, minus, [max(-drive, 0.0)])
    track: List[float] = []
    width: List[int] = []
    for _ in range(int(seconds * 1000.0 / TICK_MS)):
        backend.Simulate(TICK_MS)
        counts = ring.read(backend)
        track.append(ring.held_angle())
        width.append(int(np.count_nonzero(counts)))
    angles = np.unwrap(np.array(track)[np.isfinite(track)])
    if angles.size < 10:
        return float("nan"), 0.0
    t = np.arange(angles.size) * TICK_MS / 1000.0
    return float(np.polyfit(t[5:], angles[5:], 1)[0]), float(np.median(width))


def inverse(drives: np.ndarray, speeds: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """The drive that gives each wanted speed (the curve made monotonic, interpolated; beyond its ends: the ends)."""

    order = np.argsort(drives)
    d, s = np.asarray(drives, dtype=float)[order], np.maximum.accumulate(np.asarray(speeds, dtype=float)[order])
    s = s + 1e-9 * np.arange(s.size)
    return np.interp(wanted, s, d)


def curve(backend: Any, tunings: Tunings, half: np.ndarray, settings: ShiftSettings, threads: int,
          seeds: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    rows = [[measure_speed(backend, tunings, half, d, settings, threads, s) for d in DRIVES] for s in seeds]
    speeds = np.array([[r[0] for r in row] for row in rows])
    widths = np.array([[r[1] for r in row] for row in rows])
    return speeds, widths


def first_motif(directory: str) -> Dict[str, Any]:
    """The format-1 measurement (the first motif), carried over when this calibration replaces it."""

    path = os.path.join(directory, "shift_cells.json")
    if not os.path.exists(path):
        return {}
    meta = json.load(open(path, encoding="utf-8"))
    if meta.get("settings", {}).get("format") == 1:
        fit = meta["fit"]
        return {"settings": meta["settings"]["shift"], "top_rad_s": float(np.nanmax(np.abs(fit["achieved_rad_s"]))),
                "worst_ratio_error": fit["worst_ratio_error"], "passes": fit["passes"]}
    return meta.get("first_motif", {})


def run(backend: Any, tunings: Tunings, bounded: Artifact, settings: ShiftSettings = ShiftSettings(), threads: int = 8,
        directory: str = ARTIFACTS, log: Callable[[str], None] = print) -> Artifact:
    started = time.time()
    first = first_motif(directory)
    half = bounded.arrays["decoders_half"]
    drives = np.array(DRIVES)
    by_seed, widths = curve(backend, tunings, half, settings, threads, SEEDS)
    speed_curve = np.nanmean(by_seed, axis=0)
    log("speed curve (rad/s): %s (%.0f s)" % (" ".join("%.2f" % v for v in speed_curve), time.time() - started))
    wanted = np.concatenate([-np.geomspace(NEEDED_RAD_S[1], NEEDED_RAD_S[0], 6), np.geomspace(NEEDED_RAD_S[0], NEEDED_RAD_S[1], 6)])
    reachable = bool(np.nanmax(speed_curve) >= NEEDED_RAD_S[1] and np.nanmin(speed_curve) <= -NEEDED_RAD_S[1])
    check_drives = inverse(drives, speed_curve, wanted)
    achieved = np.array([np.nanmean([measure_speed(backend, tunings, half, float(d), settings, threads, s + 10, CHECK_SECONDS)[0]
                                     for s in SEEDS[:3]]) for d in check_drives])
    still = [measure_speed(backend, tunings, half, 0.0, settings, threads, s + 20, CHECK_SECONDS)[0] for s in SEEDS[:3]]
    ratio = achieved / wanted
    worst = float(np.nanmax(np.abs(ratio - 1.0)))
    drift = float(np.nanmax(np.abs(still)))
    passes = bool(reachable and worst <= SUB_GATE)
    fit: Dict[str, Any] = {"reachable": reachable, "wanted_rad_s": wanted.tolist(), "check_drives": check_drives.tolist(),
                           "achieved_rad_s": achieved.tolist(), "ratio": ratio.tolist(), "worst_ratio_error": worst,
                           "drift_rad_s": drift, "sub_gate": SUB_GATE, "passes": passes,
                           "top_rad_s": float(min(np.nanmax(speed_curve), -np.nanmin(speed_curve))),
                           "width_median": float(np.median(widths)), "width_max": float(np.max(widths))}
    log("sub-gate: worst |achieved/wanted − 1| = %.2f over %.1f–%.1f rad/s, drift %.3f rad/s → %s (%.0f s)" % (
        worst, NEEDED_RAD_S[0], NEEDED_RAD_S[1], drift, "passes" if passes else "fails", time.time() - started))
    candidates = []
    for changes in CANDIDATES:
        other = ShiftSettings(**{**settings.describe(), **changes})
        speeds, other_widths = curve(backend, tunings, half, other, threads, SEEDS[:2])
        candidates.append({"settings": other.describe(), "speed_curve": np.nanmean(speeds, axis=0).tolist(),
                           "width_median": float(np.median(other_widths))})
    meta = {"settings": {"format": FORMAT, "shift": settings.describe(), "drives": list(DRIVES), "seeds": list(SEEDS),
                         "seconds": round(time.time() - started, 1)},
            "fit": fit, "candidates": candidates, "first_motif": first}
    return Artifact("shift_cells", {"drives": drives, "speed_curve": speed_curve, "speed_by_seed": by_seed,
                                    "width_by_seed": widths}, meta)
