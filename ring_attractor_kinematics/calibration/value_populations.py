"""Decoders of every kind of value population, fitted to rates measured in NEST (``artifacts/value_populations``).

Each kind is measured once, in small rigs that run side by side in one kernel:

* ``source``: a sum population driven by a constant current (the sender in the other rigs, and the goal input);
* ``scalar``: a sum population driven by a source's spikes; fitted to give back the value;
* ``product``: driven by two sources on a grid of pairs; fitted to give ``a·b``;
* ``norm``: driven by two sources inside the unit disc; fitted to give the pair's length (cos pitch);
* ``feature``: the cos/sin population of a real joint ring whose bump is written at many positions.

The fit never sees a pose of any arm: sources take values on a grid, the ring is swept index by index.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from ..neurons.ring_layout import RingMap
from ..neurons.tuning import Tuning, Tunings, solve_decoders
from ..spiking.populations import Population, connect_weights, create_kernel, create_population, decoded_weights, set_value
from ..spiking.ring_attractor import build_ring_attractor
from .artifacts import Artifact

FORMAT = 2
SETTLE_MS = 300.0
MEASURE_MS = 1000.0
TICK_MS = 50.0


def measured_rates(backend: Any, populations: Sequence[Population], settle_ms: float = SETTLE_MS, measure_ms: float = MEASURE_MS) -> np.ndarray:
    backend.Simulate(float(settle_ms))
    for population in populations:
        population.counts(backend)
    backend.Simulate(float(measure_ms))
    return np.array([population.counts(backend) for population in populations]) / (measure_ms / 1000.0)


def measure_source(backend: Any, tunings: Tunings, values: np.ndarray, threads: int, seed: int) -> np.ndarray:
    """Rates (values × neurons) of the sum population driven by constant currents."""

    create_kernel(backend, threads, seed)
    populations = []
    for i, x in enumerate(values):
        population = create_population(backend, tunings.entry, "source%d" % i, seed=i, record=True)
        set_value(backend, population, [x])
        populations.append(population)
    return measured_rates(backend, populations)


def measure_driven(backend: Any, tunings: Tunings, kind: str, inputs: np.ndarray, source_decoders: np.ndarray,
                   threads: int, seed: int, target_tuning: Tuning = None) -> np.ndarray:
    """Rates (inputs × neurons) of a population of ``kind`` driven by sources' spikes; each input row is one rig."""

    tuning = target_tuning or tunings.of_kind(kind)
    create_kernel(backend, threads, seed)
    targets = []
    for i, point in enumerate(np.atleast_2d(inputs)):
        target = create_population(backend, tuning, "%s%d" % (kind, i), seed=1000 + i, record=True)
        for d, value in enumerate(point):
            source = create_population(backend, tunings.entry, "source%d_%d" % (i, d), seed=2000 + 7 * i + d)
            set_value(backend, source, [value])
            transform = np.zeros((tuning.dims, 1))
            transform[min(d, tuning.dims - 1), 0] = 1.0
            connect_weights(backend, source.neurons, target.neurons, decoded_weights(source_decoders, transform, target))
        targets.append(target)
    return measured_rates(backend, targets)


def measure_ring_profile(backend: Any, size: int, threads: int, seed: int, positions: Sequence[int] = (20, 60, 100, 140, 180),
                         ticks: int = 40) -> np.ndarray:
    """Mean rate of every ring neuron around a held bump, rolled so the bump is centred on index 0."""

    profiles = []
    for position in positions:
        create_kernel(backend, threads, seed + int(position))
        mapping = RingMap.circle(size)
        ring = build_ring_attractor(backend, "ring", mapping)
        collected = []
        for t in range(int(ticks)):
            ring.drive(backend, mapping.angle(position))
            backend.Simulate(TICK_MS)
            counts = ring.read(backend)
            if t >= ticks // 2:
                collected.append(counts / (TICK_MS / 1000.0))
        profiles.append(np.roll(np.mean(collected, axis=0), -int(position)))
    return np.mean(profiles, axis=0)


def measure_cos_sin(backend: Any, tuning: Tuning, scale: float, positions: Sequence[float], size: int, threads: int, seed: int,
                    settle_ticks: int = 6, hold_ticks: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep a bump over a circular ring; returns (written angles, cos/sin rates, ring index error), one row per held tick."""

    create_kernel(backend, threads, seed)
    mapping = RingMap.circle(size)
    ring = build_ring_attractor(backend, "ring", mapping)
    population = create_population(backend, tuning, "cos_sin", record=True)
    angles_of = mapping.neuron_angles()
    basis = np.stack([np.cos(angles_of), np.sin(angles_of)], axis=1)
    connect_weights(backend, ring.neurons, population.neurons, tuning.input_weights(np.eye(2)) @ (scale * basis).T / population.charge)
    angles, rates, index_error = [], [], []
    for position in positions:
        for t in range(int(settle_ticks + hold_ticks)):
            ring.drive(backend, mapping.angle(float(position)))
            backend.Simulate(TICK_MS)
            ring.read(backend)
            counts = population.counts(backend)
            if t >= settle_ticks:
                angles.append(mapping.angle(float(position)))
                rates.append(counts / (TICK_MS / 1000.0))
                index_error.append(mapping.index_error(ring.index, float(position)))
    return np.array(angles), np.array(rates), np.array(index_error)


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run(backend: Any, tunings: Tunings, ring_size: int = 200, cos_sin_tau_ms: float = 15.0, threads: int = 8, seed: int = 1,
        scalar_points: int = 61, product_points: int = 25, log: Callable[[str], None] = print) -> Artifact:
    started = time.time()
    regularisation = tunings.regularisation
    fit: Dict[str, Any] = {}
    x = np.linspace(-1.1, 1.1, int(scalar_points))
    held = np.clip(x, -1.0, 1.0)[:, None]
    source_rates = measure_source(backend, tunings, x, threads, seed)
    d_source = solve_decoders(source_rates, held, regularisation)
    fit["source_rms"] = rms(source_rates @ d_source, held)
    scalar_rates = measure_driven(backend, tunings, "scalar", x[:, None], d_source, threads, seed)
    d_scalar = solve_decoders(scalar_rates, held, regularisation)
    fit["scalar_rms"] = rms(scalar_rates @ d_scalar, held)
    log("source and sum: fit rms %.4f, %.4f (%.0f s)" % (fit["source_rms"], fit["scalar_rms"], time.time() - started))

    axis = np.linspace(-1.1, 1.1, int(product_points))
    a, b = [g.ravel() for g in np.meshgrid(axis, axis, indexing="ij")]
    pairs = np.stack([a, b], axis=1)
    target = (np.clip(a, -1.0, 1.0) * np.clip(b, -1.0, 1.0))[:, None]
    product_rates = measure_driven(backend, tunings, "product", pairs, d_source, threads, seed)
    d_product = solve_decoders(product_rates, target, regularisation)
    fit["product_rms"] = rms(product_rates @ d_product, target)
    disc = pairs[np.hypot(pairs[:, 0], pairs[:, 1]) <= 1.15]
    norm_target = np.minimum(np.hypot(disc[:, 0], disc[:, 1]), 1.0)[:, None]
    norm_rates = measure_driven(backend, tunings, "norm", disc, d_source, threads, seed)
    d_norm = solve_decoders(norm_rates, norm_target, regularisation)
    fit["norm_rms"] = rms(norm_rates @ d_norm, norm_target)
    log("product and norm: fit rms %.4f, %.4f (%.0f s)" % (fit["product_rms"], fit["norm_rms"], time.time() - started))

    profile = measure_ring_profile(backend, ring_size, threads, seed)
    offsets = (np.arange(ring_size) + ring_size // 2) % ring_size - ring_size // 2
    scale = 1.0 / float(abs(np.sum(profile * np.exp(1j * offsets * 2.0 * np.pi / ring_size))))
    angles, rates, index_error = measure_cos_sin(backend, tunings.cos_sin(cos_sin_tau_ms), scale, list(np.arange(0.0, ring_size, 1.37)),
                                                 ring_size, threads, seed)
    targets = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    d_feature = solve_decoders(rates, targets, regularisation)
    fit.update(feature_rms=rms(rates @ d_feature, targets), ring_total_rate_hz=float(profile.sum()),
               ring_active_neurons=int(np.sum(profile > 0.05 * profile.max())),
               ring_index_error_mean=float(np.mean(index_error)), ring_index_error_sd=float(np.std(index_error)))
    log("cos/sin: fit rms %.4f; the ring holds the written position to %+.2f ± %.2f indices (%.0f s)" % (
        fit["feature_rms"], fit["ring_index_error_mean"], fit["ring_index_error_sd"], time.time() - started))
    settings = {"format": FORMAT, "ring_size": int(ring_size), "encoder_mode": "fractional", "feature_tau": float(cos_sin_tau_ms),
                "seed": int(seed), "regularisation": float(regularisation), "scalar_points": int(scalar_points),
                "product_points": int(product_points), "settle_ms": SETTLE_MS, "measure_ms": MEASURE_MS,
                "seconds": round(time.time() - started, 1)}
    arrays = {"ring_profile": profile, "decoders_source": d_source, "decoders_scalar": d_scalar, "decoders_norm": d_norm,
              "decoders_product": d_product, "decoders_feature": d_feature}
    return Artifact("value_populations", arrays, {"design": tunings.describe(), "settings": settings, "fit": fit})
