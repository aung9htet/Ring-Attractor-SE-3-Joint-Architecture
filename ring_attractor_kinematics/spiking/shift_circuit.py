"""Shift cells: two gain populations under a ring attractor that move its bump at a commanded speed.

Both sides are gated neuron by neuron by the bump: the gate alone is below a gain neuron's threshold, and with
the rest current both sides fire at a resting rate under the bump.  A rate drives one side up and the other down
(its positive half to the right side, its negative half to the left).  Each side feeds back onto the ring with
the ring's own excitatory reach, ``shift`` indices ahead (right) or behind (left): at rest the two pushes cancel;
a rate tips them, and the bump drifts toward the stronger side, faster the larger the rate.

The observer's prediction uses one pair per observer ring (``calibration/shift_cells.py`` measures the speed
against the drive and fits its inverse).  The first motif tried (feedback onto the next neuron only, and
inhibitory) moved the bump at 0.9 rad/s at most; this one reaches more than the 2 rad/s the observer needs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from ..neurons.lif import GAIN_NEURON
from .feedforward_ring import connect_shifted, decoded_current
from .populations import Population
from .ring_attractor import Ring


@dataclass(frozen=True)
class ShiftSettings:
    gate_weight: float = 450000.0        # ring neuron i → gain neuron i of both sides
    rest_pa: float = 150000.0            # constant current into both sides: with the gate, a resting rate under the bump
    drive_pa: float = 120000.0           # at a full drive: added to one side, taken from the other
    feedback: float = 20.0               # pA: the sum of one gain neuron's weights onto the ring
    shift: int = 10                      # ring indices between a gain neuron and the centre of what it excites
    spread: float = 3.0                  # s.d. (indices) of that excitation

    def describe(self) -> Dict[str, float]:
        return asdict(self)

    def profile(self) -> List[Tuple[int, float]]:
        """``(offset from shift, weight)`` of one gain neuron's feedback (Gaussian, out to three s.d.)."""

        span = int(np.ceil(3.0 * self.spread))
        offsets = np.arange(-span, span + 1)
        weights = np.exp(-offsets ** 2 / (2.0 * self.spread ** 2))
        weights = float(self.feedback) * weights / weights.sum()
        return [(int(d), float(w)) for d, w in zip(offsets, weights)]


@dataclass
class ShiftCells:
    left: Any
    right: Any
    settings: ShiftSettings

    @property
    def neurons(self) -> int:
        return 2 * len(self.left)


def build_shift_cells(backend: Any, ring: Ring, plus: Tuple[Population, np.ndarray], minus: Tuple[Population, np.ndarray],
                      settings: ShiftSettings) -> ShiftCells:
    """Two gain populations under ``ring``.  ``plus`` / ``minus``: (population, decoders) of the drive's positive and
    negative halves (each in [0, 1])."""

    wrap = bool(ring.mapping.circular)
    sides = []
    for _ in range(2):
        gain = backend.Create("iaf_psc_alpha", ring.size, params=dict(GAIN_NEURON, I_e=float(settings.rest_pa)))
        backend.Connect(ring.neurons, gain, "one_to_one", {"weight": float(settings.gate_weight)})
        sides.append(gain)
    left, right = sides
    decoded_current(backend, plus[0], plus[1], right, settings.drive_pa)
    decoded_current(backend, minus[0], minus[1], right, -settings.drive_pa)
    decoded_current(backend, minus[0], minus[1], left, settings.drive_pa)
    decoded_current(backend, plus[0], plus[1], left, -settings.drive_pa)
    for offset, weight in settings.profile():
        connect_shifted(backend, right, ring.neurons, int(settings.shift) + offset, weight, wrap)
        connect_shifted(backend, left, ring.neurons, -int(settings.shift) + offset, weight, wrap)
    return ShiftCells(left, right, settings)
