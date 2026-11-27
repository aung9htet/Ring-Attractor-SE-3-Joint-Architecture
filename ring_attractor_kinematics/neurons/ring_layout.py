"""Rings of neurons: where an angle sits on a ring, and how a ring attractor is connected (NumPy only).

**Angle ↔ ring index** (:class:`RingMap`), as in the SO-2 repo:

* circular: ``index = angle / 2π · N``;
* bounded: the joint's limits lie on an arc covering ``arc_fraction`` of the ring, centred on ``N/2``;
  around the seam the ring is out of range, and its neurons get less bias current (*wall*), so no bump rests there.

**Ring attractor**: every neuron excites its neighbours and inhibits the rest with a difference of two
Gaussians over the ring distance (0 .. 50 for the far side).  One bump of activity forms and stays.

**Encoder**: one Poisson generator per neuron, 200 Hz on an 11-neuron window centred on the
(fractional) position of the angle: the two neurons at the window's ends get the share of the rate
the fraction leaves them, so the stimulus is centred on the angle itself, not on the nearest index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from reaching_benchmark.geometry.chains import Joint

ARC_FRACTION = 0.70
WALL_MARGIN = 15
MAX_DISTANCE = 50.0
EXCITATION_SD = 10.0
INHIBITION_SD = 5.0
OUTSIDE_CURRENT_SCALE = 0.9
STIMULUS_RATE_HZ = 200.0
STIMULUS_WEIGHT = 4.5e3
STIMULUS_HALF_WIDTH = 5


@dataclass(frozen=True)
class RingMap:
    size: int
    lower: float
    upper: float
    circular: bool
    arc_fraction: float = ARC_FRACTION

    @classmethod
    def for_joint(cls, joint: Joint, size: int) -> "RingMap":
        return cls(int(size), float(joint.lower), float(joint.upper), bool(joint.circular))

    @classmethod
    def circle(cls, size: int) -> "RingMap":
        return cls(int(size), -float(np.pi), float(np.pi), True)

    @property
    def span(self) -> float:
        """Ring indices the joint range covers."""

        return float(self.size) if self.circular else float(round(self.arc_fraction * self.size))

    @property
    def first(self) -> float:
        """Ring index of the lower limit (0 on a circular ring, where index 0 is angle 0)."""

        return 0.0 if self.circular else (self.size - self.span) / 2.0

    @property
    def arc(self) -> Tuple[float, float]:
        return self.first, self.first + self.span

    @property
    def radians_per_index(self) -> float:
        return 2.0 * np.pi / self.size if self.circular else (self.upper - self.lower) / self.span

    def position(self, angle: float) -> float:
        """Fractional ring position of an angle (beyond the limits: the arc's ends)."""

        if self.circular:
            return float(np.mod(float(angle), 2.0 * np.pi) / (2.0 * np.pi) * self.size)
        phase = min(max((float(angle) - self.lower) / (self.upper - self.lower), 0.0), 1.0)
        return float(self.first + phase * self.span)

    def angle(self, index: float) -> float:
        """Angle of a (fractional) ring index."""

        if self.circular:
            return float(np.angle(np.exp(2j * np.pi * float(index) / self.size)))
        return self.lower + (float(index) - self.first) * self.radians_per_index

    def neuron_angles(self) -> np.ndarray:
        """Angle each neuron stands for; on a bounded ring the line through the arc goes on past its ends."""

        indices = np.arange(self.size, dtype=float)
        if self.circular:
            return 2.0 * np.pi * indices / self.size
        return self.lower + (indices - self.first) * self.radians_per_index

    def wall(self, margin: int = WALL_MARGIN) -> Tuple[int, int]:
        """Ring indices between which a bump may rest on a bounded ring."""

        first, last = self.arc
        return max(int(round(first)) - int(margin), 0), min(int(round(last)) + int(margin), self.size - 1)

    def population_vector_index(self, counts: np.ndarray) -> float:
        """Fractional ring index of the centre of the activity (NaN for a silent ring)."""

        values = np.asarray(counts, dtype=float)
        total = float(values.sum())
        if total <= 0.0:
            return float("nan")
        vector = np.sum(values * np.exp(2j * np.pi * np.arange(self.size) / self.size)) / total
        index = float((np.angle(vector) % (2.0 * np.pi)) * self.size / (2.0 * np.pi))
        if self.circular:
            return index
        middle = self.size / 2.0                    # on the side of the seam nearer the arc
        return middle + ((index - middle + self.size / 2.0) % self.size - self.size / 2.0)

    def index_error(self, a: float, b: float) -> float:
        """``a − b`` in ring indices (the short way round on a circular ring)."""

        difference = float(a) - float(b)
        if self.circular:
            difference = (difference + self.size / 2.0) % self.size - self.size / 2.0
        return difference


def ring_distances(size: int, max_distance: float = MAX_DISTANCE) -> np.ndarray:
    """Distance of every neuron from neuron 0 along the ring (0 .. ``max_distance`` at the far side)."""

    half = int(size / 2)
    if size % 2 == 0:
        up = np.linspace(0, max_distance, half + 1)[1:half]
        down = np.linspace(max_distance, 0, half + 1)[1:half]
        return np.concatenate([[0.0], up, [max_distance], down])
    up = np.linspace(0, max_distance, half + 1)[1:]
    down = np.linspace(max_distance, 0, half + 1)[:half]
    return np.concatenate([[0.0], up, down])


def ring_weight(distance: np.ndarray, sd_1: float = EXCITATION_SD, sd_2: float = INHIBITION_SD) -> np.ndarray:
    """Recurrent weight (pA) for a distance: a difference of two Gaussians, rounded as in the SO-2 repo."""

    d = np.asarray(distance, dtype=float)
    weight = 3.0 * ((sd_2 * np.exp(-(d ** 2) / (2 * sd_1 ** 2)) - sd_1 * np.exp(-(d ** 2) / (2 * sd_2 ** 2))) / (sd_2 - sd_1))
    return np.around(weight, 3)


def ring_weight_matrix(size: int) -> np.ndarray:
    """``(pre, post)`` recurrent weights, self-connection included."""

    profile = ring_weight(ring_distances(int(size)))
    pre = np.arange(int(size))[:, None]
    shift = np.arange(int(size))[None, :]
    matrix = np.zeros((int(size), int(size)))
    matrix[pre, (pre + shift) % int(size)] = profile[None, :]
    return matrix


def wall_profile(size: int, low: float, high: float, outside_scale: float = OUTSIDE_CURRENT_SCALE) -> np.ndarray:
    """Bias-current factor per neuron: 1 between ``low`` and ``high``, ``outside_scale`` elsewhere."""

    indices = np.arange(int(size), dtype=float)
    outside = (np.maximum(low - indices, 0.0) + np.maximum(indices - high, 0.0)) > 0
    return 1.0 - outside * (1.0 - float(outside_scale))


def bump_rates(size: int, center: int, half_width: int = STIMULUS_HALF_WIDTH, rate_hz: float = STIMULUS_RATE_HZ) -> np.ndarray:
    rates = np.zeros(int(size))
    for offset in range(-int(half_width), int(half_width) + 1):
        rates[(int(center) + offset) % int(size)] = float(rate_hz)
    return rates


def fractional_bump_rates(size: int, position: float, half_width: int = STIMULUS_HALF_WIDTH,
                          rate_hz: float = STIMULUS_RATE_HZ) -> np.ndarray:
    """The stimulus window centred on a fractional position: a blend of the two nearest whole windows."""

    low = int(np.floor(position))
    fraction = float(position) - low
    return (1.0 - fraction) * bump_rates(size, low, half_width, rate_hz) + fraction * bump_rates(size, low + 1, half_width, rate_hz)


def bump_spread(counts: np.ndarray, index: float) -> float:
    """Share of a ring's spikes further than 20 indices from the centre of its activity (two bumps: large)."""

    total = float(np.sum(counts))
    if total <= 0 or not np.isfinite(index):
        return float("nan")
    size = counts.size
    distance = np.abs((np.arange(size) - index + size / 2.0) % size - size / 2.0)
    return float(np.sum(counts[distance > 20]) / total)
