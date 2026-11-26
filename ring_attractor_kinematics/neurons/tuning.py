"""Value populations: how ``n`` neurons hold a number (or a pair) in their rates, and how it is read back.

Neuron ``i`` gets the current ``J_i = gain_i · (encoder_i · x) / radius + bias_i``: it responds to one
direction of ``x`` above its own threshold and reaches its own top rate at the edge of the range.
A value, or a function of it, is read back as a weighted sum of the rates; the weights (*decoders*)
are a least-squares fit with a noise term (:func:`solve_decoders`).  This is the population code of
Eliasmith & Anderson (2003).

Three kinds (:func:`make_tuning`):

* ``scalar``: one number in [−1, 1]; encoders +1 and −1 (push-pull halves);
* ``phasor``: a pair (cos, sin), encoders all around the circle;
* ``product``: a pair (a, b), encoders on the four diagonals; read out as ``ab = ((a+b)² − (a−b)²)/4``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..config import Sizes
from .lif import NeuronModel

KINDS = ("scalar", "phasor", "product")
#: ramp read-out: neurons, synapse (ms), radius and the seed offset of its tuning
RAMP_NEURONS = 160
RAMP_TAU_MS = 30.0
RAMP_RADIUS = 1.3
RAMP_SEED = 77


@dataclass
class Tuning:
    kind: str
    n: int
    dims: int
    radius: float
    encoders: np.ndarray        # (n, dims)
    gain: np.ndarray            # (n,) pA per unit of encoder · x / radius
    bias: np.ndarray            # (n,) pA
    model: NeuronModel = field(default_factory=NeuronModel)
    seed: int = 0

    def currents(self, x: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(x, dtype=float))
        if values.shape[1] != self.dims:
            values = values.reshape(-1, self.dims)
        return self.gain[None, :] * (values @ self.encoders.T) / self.radius + self.bias[None, :]

    def rates(self, x: np.ndarray) -> np.ndarray:
        """Rates (samples × neurons) for constant inputs."""

        return self.model.rate(self.currents(x))

    def input_weights(self, transform: np.ndarray) -> np.ndarray:
        """``(n, k)``: currents per unit of each of ``k`` values that enter through ``transform`` (dims × k)."""

        return self.gain[:, None] * (self.encoders @ np.asarray(transform, dtype=float)) / self.radius

    def with_bias(self, bias: np.ndarray) -> "Tuning":
        return Tuning(self.kind, self.n, self.dims, self.radius, self.encoders, self.gain, np.asarray(bias, dtype=float),
                      self.model, self.seed)


def make_tuning(kind: str, n: int, seed: int, model: Optional[NeuronModel] = None,
                intercepts: Tuple[float, float] = (-0.9, 0.9), max_rates: Tuple[float, float] = (100.0, 200.0),
                radius: Optional[float] = None) -> Tuning:
    """``n`` neurons with thresholds and top rates drawn from ``seed``."""

    if kind not in KINDS:
        raise ValueError("kind must be one of %r" % (KINDS,))
    model = model or NeuronModel()
    rng = np.random.default_rng(int(seed))
    n = int(n)
    if kind == "scalar":
        encoders = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)[:, None]
        default_radius = 1.0
    elif kind == "phasor":
        angles = 2.0 * np.pi * (np.arange(n) + rng.uniform(0.0, 1.0)) / n
        encoders = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        default_radius = 1.3            # room for a pair somewhat longer than 1
    else:
        signs = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
        encoders = signs[np.arange(n) % 4] / 2.0
        default_radius = 1.0
    cut = rng.uniform(intercepts[0], intercepts[1], n)
    top = rng.uniform(max_rates[0], max_rates[1], n)
    gain = (model.current(top) - model.rheobase) / (1.0 - cut)
    bias = model.rheobase - gain * cut
    return Tuning(kind, n, encoders.shape[1], float(radius or default_radius), encoders, gain, bias, model, int(seed))


def solve_decoders(rates: np.ndarray, targets: np.ndarray, regularisation: float = 0.1) -> np.ndarray:
    """Decoders ``D`` (neurons × outputs) with ``rates @ D ≈ targets``.

    Least squares with a ridge term that stands for noise on the rates of ``regularisation`` times the
    largest rate, so no weight relies on a difference of rates smaller than that noise.
    """

    a = np.asarray(rates, dtype=float)
    y = np.asarray(targets, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    sigma = float(regularisation) * float(np.max(a)) if a.size else 0.0
    gram = a.T @ a + (sigma ** 2) * a.shape[0] * np.eye(a.shape[1])
    return np.linalg.solve(gram, a.T @ y)


@dataclass
class Tunings:
    """The tuning of every kind of value population, drawn from one seed (the *network seed* of the calibrations)."""

    sizes: Sizes = field(default_factory=Sizes)
    model: NeuronModel = field(default_factory=NeuronModel)
    seed: int = 1
    regularisation: float = 0.02

    def __post_init__(self) -> None:
        base = 1000 * int(self.seed)
        self.feature = make_tuning("phasor", self.sizes.feature, base + 1, self.model)
        self.product = make_tuning("product", self.sizes.product, base + 2, self.model)
        self.entry = make_tuning("scalar", self.sizes.entry, base + 3, self.model)
        self.norm = make_tuning("phasor", self.sizes.norm, base + 4, self.model, radius=1.2)

    def of_kind(self, kind: str) -> Tuning:
        return {"phasor": self.feature, "product": self.product, "scalar": self.entry, "norm": self.norm}[kind]

    def cos_sin(self, tau_syn_ms: float) -> Tuning:
        """The cos/sin population: the phasor tuning with the slower synapse that reads a ring."""

        return make_tuning("phasor", self.sizes.feature, self.feature.seed, self.model.with_synapse(tau_syn_ms),
                           radius=self.feature.radius)

    def level(self, tau_syn_ms: float) -> Tuning:
        """The level cells: the sum population with the slow synapse that reads a sheet."""

        return make_tuning("scalar", self.sizes.entry, self.entry.seed, self.model.with_synapse(tau_syn_ms))

    def ramp(self, neurons: int = RAMP_NEURONS, tau_syn_ms: float = RAMP_TAU_MS) -> Tuning:
        """A ramp read-out: a pair population with the slow synapse that reads a sheet."""

        return make_tuning("phasor", int(neurons), 1000 * int(self.seed) + RAMP_SEED, self.model.with_synapse(tau_syn_ms),
                           radius=RAMP_RADIUS)

    def describe(self) -> Dict[str, object]:
        return {"sizes": self.sizes.__dict__, "model": self.model.__dict__, "seed": int(self.seed),
                "regularisation": float(self.regularisation)}
