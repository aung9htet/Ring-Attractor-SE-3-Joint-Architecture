"""Feed-forward rings: rings with no attractor of their own, written from outside.

* **Written goal ring + gain pair** (:func:`build_goal_writing`): one joint's goal ring, written beside the
  joint's ring by two gain populations.  Both sides are gated by the joint's bump and fire alike at rest;
  the share halves push their rates apart, and the right side writes ``shift`` indices up the ring, the
  left side as far down.  The centre of the goal ring's activity leads the joint's bump by up to ``shift``
  indices: the joint's next step.
* **Readout rings** (:func:`build_readout_ring`): lift, pitch and yaw of the hand as bumps.  Each neuron is
  driven by the pair (cos, sin) of its angle, projected on its preferred direction, above a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np

from ..config import GainPairSettings
from ..neurons.lif import GAIN_NEURON, NeuronModel, alpha_charge
from ..neurons.ring_layout import RingMap
from .populations import Population, attach_recorder, connect_weights, first_id
from .ring_attractor import Ring, build_plain_ring

#: a readout-ring neuron starts to fire when its drive exceeds this, and fires at OUTPUT_PEAK_HZ at a full pair
OUTPUT_THRESHOLD = 0.25
OUTPUT_PEAK_HZ = 150.0


@dataclass
class GoalWriting:
    """One joint's written goal ring and its gain pair (left: down the ring, right: up)."""

    ring: Ring
    left: Any
    right: Any
    recorders: Tuple[Any, Any]
    settings: GainPairSettings

    def gain_counts(self, backend: Any) -> np.ndarray:
        """Spikes of the left and right gain populations since the last call."""

        out = []
        for recorder in self.recorders:
            out.append(float(backend.GetStatus(recorder, "n_events")[0]))
            backend.SetStatus(recorder, {"n_events": 0})
        return np.array(out)

    @property
    def neurons(self) -> int:
        return self.ring.size * 3


def connect_shifted(backend: Any, pre: Any, post: Any, shift: int, weight: float, wrap: bool) -> None:
    """``pre[i] → post[i + shift]``; around the seam only when ``wrap``."""

    size, shift = len(pre), int(shift)
    if shift == 0:
        backend.Connect(pre, post, "one_to_one", {"weight": float(weight)})
        return
    s = abs(shift)
    if shift > 0:
        backend.Connect(pre[0:size - s], post[s:size], "one_to_one", {"weight": float(weight)})
        if wrap:
            backend.Connect(pre[size - s:size], post[0:s], "one_to_one", {"weight": float(weight)})
    else:
        backend.Connect(pre[s:size], post[0:size - s], "one_to_one", {"weight": float(weight)})
        if wrap:
            backend.Connect(pre[0:s], post[size - s:size], "one_to_one", {"weight": float(weight)})


def decoded_current(backend: Any, source: Population, decoders: np.ndarray, target: Any, pa_per_unit: float) -> None:
    """Hand ``pa_per_unit`` × (decoded value of ``source``) to every neuron of ``target`` (gain neurons) as a current."""

    per_hz = float(pa_per_unit) * np.asarray(decoders, dtype=float)[:, 0]
    excitatory, inhibitory = alpha_charge(GAIN_NEURON["tau_syn_ex"]), alpha_charge(GAIN_NEURON["tau_syn_in"])
    weights = np.where(per_hz >= 0.0, per_hz / excitatory, per_hz / inhibitory)
    backend.Connect(source.neurons, target, {"rule": "all_to_all"},
                    {"weight": np.ascontiguousarray(np.tile(weights[None, :], (len(target), 1))), "delay": 1.0})


def build_goal_writing(backend: Any, name: str, mapping: RingMap, joint_ring: Ring, plus: Tuple[Population, np.ndarray],
                       minus: Tuple[Population, np.ndarray], settings: GainPairSettings) -> GoalWriting:
    """``plus`` / ``minus``: the two share halves (population, decoders); the plus half moves the goal up the ring."""

    s = settings
    wrap = bool(mapping.circular)
    ring = build_plain_ring(backend, name, mapping, s.ring_bias_pa)
    sides: List[Any] = []
    recorders: List[Any] = []
    for _ in range(2):
        gain = backend.Create("iaf_psc_alpha", mapping.size, params=dict(GAIN_NEURON, I_e=float(s.rest_pa)))
        backend.Connect(joint_ring.neurons, gain, "one_to_one", {"weight": float(s.gate_weight)})
        recorders.append(attach_recorder(backend, gain))
        sides.append(gain)
    left, right = sides
    decoded_current(backend, plus[0], plus[1], right, s.share_drive_pa)
    decoded_current(backend, minus[0], minus[1], right, -s.share_drive_pa)
    decoded_current(backend, minus[0], minus[1], left, s.share_drive_pa)
    decoded_current(backend, plus[0], plus[1], left, -s.share_drive_pa)
    connect_shifted(backend, right, ring.neurons, +int(s.shift), float(s.write_weight), wrap)
    connect_shifted(backend, left, ring.neurons, -int(s.shift), float(s.write_weight), wrap)
    return GoalWriting(ring, left, right, (recorders[0], recorders[1]), s)


def build_readout_ring(backend: Any, name: str, size: int, model: NeuronModel,
                       drives: Sequence[Sequence[Tuple[Any, float]]], index: int) -> Population:
    """One readout ring.  ``drives``: for the cos and the sin direction, a list of ``(source, coefficient)`` where a
    source is ``(population, decoders)`` or ``None`` for the constant 1."""

    theta = 2.0 * np.pi * np.arange(size) / size
    gain = float(model.current(np.array([OUTPUT_PEAK_HZ]))[0] - model.rheobase) / (1.0 - OUTPUT_THRESHOLD)
    neurons = backend.Create("iaf_psc_alpha", size, params=model.nest_params())
    bias = np.full(size, model.rheobase - gain * OUTPUT_THRESHOLD)
    for terms, direction in zip(drives, (np.cos(theta), np.sin(theta))):
        for source, coefficient in terms:
            if source is None:
                bias = bias + gain * coefficient * direction
            else:
                population, decoders = source
                connect_weights(backend, population.neurons, neurons, gain * coefficient * np.outer(direction, decoders[:, 0]) / model.charge)
    start = np.random.default_rng([7, int(index)]).uniform(0.0, model.V_th, size)
    backend.SetStatus(neurons, [{"I_e": float(c), "V_m": float(v)} for c, v in zip(bias, start)])
    recorder = attach_recorder(backend, neurons)
    return Population(name, neurons, size, first_id(backend, neurons), None, recorder, model.charge)
