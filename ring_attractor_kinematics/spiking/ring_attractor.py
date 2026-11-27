"""The ring attractor: joint rings and goal rings, each with the encoder that writes an angle on it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..neurons.lif import RING_NEURON
from ..neurons.ring_layout import (STIMULUS_HALF_WIDTH, STIMULUS_RATE_HZ, STIMULUS_WEIGHT, WALL_MARGIN, RingMap,
                                   fractional_bump_rates, ring_weight_matrix, wall_profile)
from .populations import attach_recorder, first_id


@dataclass
class Ring:
    """A ring of ring neurons in the kernel, with its recorder (and, for an attractor, its encoder's generators)."""

    name: str
    mapping: RingMap
    neurons: Any
    recorder: Any
    first_id: int
    generators: Any = None
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    index: float = float("nan")
    target_index: float = float("nan")

    @property
    def size(self) -> int:
        return self.mapping.size

    def drive(self, backend: Any, angle: float) -> None:
        """Before a tick: write ``angle`` on the ring (the encoder stimulates every tick)."""

        self.target_index = self.mapping.position(angle)
        rates = fractional_bump_rates(self.size, self.target_index, STIMULUS_HALF_WIDTH, STIMULUS_RATE_HZ)
        backend.SetStatus(self.generators, [{"rate": float(r)} for r in rates])

    def read(self, backend: Any) -> np.ndarray:
        """After a tick: the tick's spike counts, and the centre of the activity."""

        events = backend.GetStatus(self.recorder, "events")[0]
        senders = np.asarray(events["senders"], dtype=int)
        self.counts = np.bincount(senders - self.first_id, minlength=self.size).astype(float)[: self.size]
        backend.SetStatus(self.recorder, {"n_events": 0})
        self.index = self.mapping.population_vector_index(self.counts)
        return self.counts

    def held_angle(self) -> float:
        """The angle the ring holds now (NaN while silent)."""

        return self.mapping.angle(self.index) if np.isfinite(self.index) else float("nan")

    def ring_error(self) -> float:
        """Distance of the bump from the written position, ring indices."""

        if not (np.isfinite(self.index) and np.isfinite(self.target_index)):
            return float("nan")
        return self.mapping.index_error(self.index, self.target_index)


def build_ring_attractor(backend: Any, name: str, mapping: RingMap) -> Ring:
    """A ring attractor (recurrent difference-of-Gaussians weights) with a Poisson encoder per neuron.
    On a bounded ring the neurons outside the arc get 90% of the bias, so no bump rests there."""

    size = mapping.size
    neurons = backend.Create("iaf_psc_alpha", size, params=dict(RING_NEURON))
    if not mapping.circular:
        low, high = mapping.wall(WALL_MARGIN)
        backend.SetStatus(neurons, [{"I_e": RING_NEURON["I_e"] * float(f)} for f in wall_profile(size, low, high)])
    backend.Connect(neurons, neurons, {"rule": "all_to_all"}, {"weight": np.ascontiguousarray(ring_weight_matrix(size).T)})
    generators = backend.Create("poisson_generator", size, params={"rate": 0.0})
    backend.Connect(generators, neurons, "one_to_one", {"weight": STIMULUS_WEIGHT})
    recorder = attach_recorder(backend, neurons)
    return Ring(name, mapping, neurons, recorder, first_id(backend, neurons), generators)


def build_plain_ring(backend: Any, name: str, mapping: RingMap, bias_pa: Optional[float] = None) -> Ring:
    """Ring neurons with no recurrent weights and no encoder (a written goal ring)."""

    params = dict(RING_NEURON, I_e=float(bias_pa)) if bias_pa is not None else dict(RING_NEURON)
    neurons = backend.Create("iaf_psc_alpha", mapping.size, params=params)
    recorder = attach_recorder(backend, neurons)
    return Ring(name, mapping, neurons, recorder, first_id(backend, neurons))
