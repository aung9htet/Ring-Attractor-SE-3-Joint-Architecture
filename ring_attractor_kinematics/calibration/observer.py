"""The observer method's own read-outs and the cue's delay (``artifacts/observer``).

* ``secant``: a second read-out of an observer ring's cos/sin population, ``1 / (2 max(cos θ, 0.5))`` (the chart
  rate of lift needs ``1 / cos pitch``, held at 2);
* ``drive_plus``, ``drive_minus``: read-outs of a rate population (a sum holding speed / ``rate_full``) that give
  the shift cells' drive for that speed, through the inverse of their speed curve (``artifacts/shift_cells``): the
  positive half to the right side, the negative half to the left;
* ``feature_scale``: the factor that makes the cos/sin sums of a free observer bump a pair of length 1 (the shift
  cells' resting feedback makes an observer bump larger than a joint ring's);
* ``delay_ms``: how much later the delayed cos/sin populations get the observer rings' spikes: the cue's lag behind
  the arm less the cos/sin's lag behind their bump (:func:`measure_delay`), so the innovation compares the cue with
  what the chart held when the cue was made.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Tuple

import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.geometry.rotations import lift_pitch_yaw, nearest_rotation

from ..config import IkDesign, ObserverDesign
from ..network.pose_ik_network import PoseIkNetwork, ik_options
from ..neurons.ring_layout import RingMap
from ..neurons.tuning import Tunings, solve_decoders
from ..rules.observer_rule import CHARTS, smooth_path
from ..spiking.populations import create_kernel, create_population, set_value
from ..spiking.ring_attractor import build_ring_attractor
from ..spiking.shift_circuit import ShiftSettings, build_shift_cells
from .artifacts import ARTIFACTS, Artifact, Calibrations
from .shift_cells import inverse
from .value_populations import measure_cos_sin, measure_driven

FORMAT = 1
#: the delay first assumed, and the delays the measurement compares (ms)
FIRST_DELAY_MS = 150.0
DELAYS_MS = tuple(range(0, 401, 5))
TRAINING_SEED = 20261187
TRAINING_SPEED = 0.6


def fit_secant(backend: Any, tunings: Tunings, values: Artifact, threads: int, seed: int) -> Tuple[np.ndarray, float]:
    size = int(values.meta["settings"]["ring_size"])
    profile = values.arrays["ring_profile"]
    offsets = (np.arange(size) + size // 2) % size - size // 2
    scale = 1.0 / float(abs(np.sum(profile * np.exp(1j * offsets * 2.0 * np.pi / size))))
    tuning = tunings.cos_sin(float(values.meta["settings"]["feature_tau"]))
    angles, rates, _ = measure_cos_sin(backend, tuning, scale, list(np.arange(0.0, size, 1.37)), size, threads, seed)
    target = (1.0 / (2.0 * np.maximum(np.cos(angles), 0.5)))[:, None]
    decoders = solve_decoders(rates, target, float(values.meta["settings"]["regularisation"]))
    return decoders, float(np.sqrt(np.mean((rates @ decoders - target) ** 2)))


def fit_drive(backend: Any, tunings: Tunings, values: Artifact, shift: Artifact, rate_full: float, threads: int,
              seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    x = np.concatenate([np.linspace(-1.6, 1.6, 97), np.linspace(-0.25, 0.25, 61)])
    inputs = np.stack([np.clip(x, -1.1, 1.1), x - np.clip(x, -1.1, 1.1)], axis=1)
    rates = measure_driven(backend, tunings, "scalar", inputs, values.arrays["decoders_source"], threads, seed)
    drive = inverse(shift.arrays["drives"], shift.arrays["speed_curve"], np.clip(x, -1.0, 1.0) * float(rate_full))
    target = np.stack([np.maximum(drive, 0.0), np.maximum(-drive, 0.0)], axis=1)
    regularisation = float(values.meta["settings"]["regularisation"])
    plus, minus = solve_decoders(rates, target[:, :1], regularisation), solve_decoders(rates, target[:, 1:], regularisation)
    error = np.concatenate([rates @ plus - target[:, :1], rates @ minus - target[:, 1:]])
    return plus, minus, float(np.sqrt(np.mean(error ** 2)))


def _lag_ms(late: np.ndarray, early: np.ndarray, pick: Tuple[Tuple[int, int], ...]) -> Tuple[float, list]:
    """The delay (ms, of :data:`DELAYS_MS`) by which series ``late`` (ticks, chart, angle) best follows ``early``."""

    ticks = np.arange(late.shape[0]) * 50.0
    fine = np.arange(0.0, ticks[-1] + 50.0, 5.0)

    def resample(series: np.ndarray) -> np.ndarray:
        good = np.isfinite(series)
        return np.interp(fine, ticks[good], np.unwrap(series[good]))

    scores = []
    for d in DELAYS_MS:
        shift = int(d / 5)
        errors = [np.sin(resample(late[:, c, j])[shift:] - resample(early[:, c, j])[: fine.size - shift])[200:] ** 2 for c, j in pick]
        scores.append(float(np.mean(np.concatenate(errors))))
    return float(DELAYS_MS[int(np.argmin(scores))]), scores


def observer_profile(backend: Any, tunings: Tunings, shift: Artifact, bounded: Artifact, size: int, threads: int, seed: int,
                     ticks: int = 40) -> np.ndarray:
    """Mean spike count per tick of a free observer bump (shift cells attached, drive 0), centred on its peak."""

    create_kernel(backend, threads, seed)
    ring = build_ring_attractor(backend, "observer", RingMap.circle(size))
    plus = create_population(backend, tunings.entry, "drive_plus", seed=1)
    minus = create_population(backend, tunings.entry, "drive_minus", seed=2)
    half = bounded.arrays["decoders_half"]
    build_shift_cells(backend, ring, (plus, half), (minus, half), ShiftSettings(**shift.meta["settings"]["shift"]))
    set_value(backend, plus, [0.0])
    set_value(backend, minus, [0.0])
    for _ in range(10):
        ring.drive(backend, 0.0)
        backend.Simulate(50.0)
        ring.read(backend)
    backend.SetStatus(ring.generators, {"rate": 0.0})
    collected = []
    for t in range(int(ticks)):
        backend.Simulate(50.0)
        counts = ring.read(backend)
        if t >= 10 and np.isfinite(ring.index):
            collected.append(np.roll(counts, -int(round(ring.index))))
    return np.mean(collected, axis=0) / 0.05


def measure_delay(backend: Any, directory: str, threads: int, seed: int, seconds: float = 30.0,
                  log: Callable[[str], None] = print) -> Dict[str, Any]:
    """The delay the comparison needs: ``(cue behind the arm) − (observer cos/sin behind the observer's bump)``.

    The observer's bump stands for the arm now, so its cos/sin, delayed by the difference, show what it held when the
    cue was made.  Both lags are read through the same read-out cells, whose smoothing cancels in the difference.
    Measured along a training path (its own seed) with the observer rings written at the arm's true angles."""

    chain = get_chain("tiago7")
    rng = np.random.default_rng(TRAINING_SEED)
    start = chain.random_pose(rng)
    path = smooth_path(chain, start, [chain.random_pose(rng) for _ in range(12)], TRAINING_SPEED)[: int(seconds / 0.05)]
    network = PoseIkNetwork(backend, chain, Calibrations.load(directory), Tunings(), ik_options(threads=threads, seed=seed),
                            IkDesign(task="pose"), method="observer")
    truth, cue, pair, bump = [], [], [], []
    for t, q in enumerate(path):
        rotation = chain.rotation(q)
        for c, u in enumerate(CHARTS):
            for ring, angle in zip(network.observer_rings[c], lift_pitch_yaw(u.T @ rotation)):
                ring.drive(backend, float(angle))
        tick = network.step_ik(q, lift_pitch_yaw(rotation), kinematics.position(chain, q))
        seen = nearest_rotation(np.asarray(tick.forward.tool, dtype=float)) if np.all(np.isfinite(tick.forward.tool)) else None
        truth.append([lift_pitch_yaw(u.T @ rotation) for u in CHARTS])
        cue.append([lift_pitch_yaw(u.T @ seen) if seen is not None else (np.nan,) * 3 for u in CHARTS])
        pair.append(tick.observer.pair_angles)
        bump.append(tick.observer.angles)
        if t % 100 == 0:
            log("delay: tick %d of %d" % (t, len(path)))
    pick = ((0, 0), (0, 2), (1, 1))          # lift and yaw of chart 0, pitch of chart 1: away from the poles on this path
    cue_lag, cue_scores = _lag_ms(np.array(cue, dtype=float), np.array(truth, dtype=float), pick)
    pair_lag, pair_scores = _lag_ms(np.array(pair, dtype=float), np.array(bump, dtype=float), pick)
    return {"delay_ms": float(max(cue_lag - pair_lag, 1.0)), "cue_lag_ms": cue_lag, "pair_lag_ms": pair_lag,
            "cue_scores": cue_scores, "pair_scores": pair_scores, "delays_ms": list(DELAYS_MS), "training_seed": TRAINING_SEED,
            "training_speed": TRAINING_SPEED, "seconds": float(seconds)}


def run(backend: Any, tunings: Tunings, values: Artifact, bounded: Artifact, shift: Artifact, design: ObserverDesign = ObserverDesign(),
        threads: int = 8, seed: int = 1, directory: str = ARTIFACTS, log: Callable[[str], None] = print) -> Artifact:
    started = time.time()
    size = int(values.meta["settings"]["ring_size"])
    profile = observer_profile(backend, tunings, shift, bounded, size, threads, seed)
    offsets = (np.arange(size) + size // 2) % size - size // 2
    feature_scale = 1.0 / float(abs(np.sum(profile * np.exp(1j * offsets * 2.0 * np.pi / size))))
    secant, secant_rms = fit_secant(backend, tunings, values, threads, seed)
    plus, minus, drive_rms = fit_drive(backend, tunings, values, shift, design.rate_full, threads, seed)
    log("secant fit rms %.4f, drive fit rms %.4f (%.0f s)" % (secant_rms, drive_rms, time.time() - started))
    arrays = {"decoders_secant": secant, "decoders_drive_plus": plus, "decoders_drive_minus": minus, "profile": profile}
    settings = {"format": FORMAT, "seed": int(seed), "rate_full": float(design.rate_full), "shift_cells": shift.meta["settings"]["shift"]}
    fit: Dict[str, Any] = {"secant_rms": secant_rms, "drive_rms": drive_rms, "delay_ms": FIRST_DELAY_MS, "feature_scale": feature_scale,
                           "joint_feature_scale": 1.0 / float(abs(np.sum(values.arrays["ring_profile"] * np.exp(1j * offsets * 2.0 * np.pi / size)))),
                           "bump_rate_hz": float(profile.sum())}
    Artifact("observer", arrays, {"settings": settings, "fit": fit}).save(directory)
    delay = measure_delay(backend, directory, threads, seed, log=log)
    fit.update(delay_ms=delay["delay_ms"], delay=delay)
    log("cue delay %.0f ms (%.0f s)" % (delay["delay_ms"], time.time() - started))
    settings["seconds"] = round(time.time() - started, 1)
    return Artifact("observer", arrays, {"settings": settings, "fit": fit})
