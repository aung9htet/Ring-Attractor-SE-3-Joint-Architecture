"""The neuron: NEST's ``iaf_psc_alpha`` leaky integrate-and-fire model, and the parameter sets this work uses.

Three parameter sets:

* :class:`NeuronModel` — the value populations (cos/sin, products, sums, shares, gates, level cells,
  ramp read-outs, readout-ring neurons, bias cells);
* :data:`RING_NEURON` — the joint, goal and sheet neurons (the ring neuron of the SO-2 ring attractor);
* :data:`GAIN_NEURON` — the gain pairs that write each joint's goal ring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

#: ``iaf_psc_alpha`` parameters of a ring neuron (SO-2 repo, Table 1 of its paper)
RING_NEURON: Dict[str, float] = {
    "V_m": 0.0, "E_L": 0.0, "V_th": 0.8, "V_reset": 0.0, "tau_syn_ex": 27.0, "tau_syn_in": 50.0,
    "I_e": 450000.0, "tau_m": 7.04, "C_m": 3960000.0, "t_ref": 0.0,
}

#: ``iaf_psc_alpha`` parameters of a gain neuron (SO-2 repo)
GAIN_NEURON: Dict[str, float] = {
    "V_m": 0.0, "E_L": 0.0, "V_th": 1.35, "V_reset": 0.0, "tau_syn_ex": 27.0, "tau_syn_in": 50.0,
    "I_e": 0.0, "tau_m": 7.04, "C_m": 3960000.0, "t_ref": 0.0,
}


def alpha_charge(tau_syn_ms: float) -> float:
    """Mean current (pA) one input firing at 1 Hz delivers per pA of weight.

    NEST's alpha current is ``w·(e/τ)·t·exp(−t/τ)`` (peak ``w``), whose integral is ``w·e·τ``.
    """

    return float(np.e * float(tau_syn_ms) * 1e-3)


@dataclass(frozen=True)
class NeuronModel:
    """``iaf_psc_alpha`` parameters of a value-population neuron (pF, ms, mV, pA)."""

    C_m: float = 250.0
    tau_m: float = 20.0
    t_ref: float = 2.0
    V_th: float = 15.0          # above the resting and reset potential, which are 0
    tau_syn: float = 5.0

    @property
    def rheobase(self) -> float:
        """Smallest constant current that makes the neuron fire, pA."""

        return self.C_m * self.V_th / self.tau_m

    def rate(self, current: np.ndarray) -> np.ndarray:
        """Firing rate (Hz) for a constant input current (pA)."""

        j = np.asarray(current, dtype=float)
        out = np.zeros_like(j)
        above = j > self.rheobase * (1.0 + 1e-12)
        out[above] = 1000.0 / (self.t_ref + self.tau_m * np.log(j[above] / (j[above] - self.rheobase)))
        return out

    def current(self, rate_hz: np.ndarray) -> np.ndarray:
        """Constant current (pA) that gives a firing rate (Hz); the inverse of :meth:`rate` above 0."""

        period = 1000.0 / np.asarray(rate_hz, dtype=float) - self.t_ref
        return self.rheobase / (1.0 - np.exp(-period / self.tau_m))

    @property
    def charge(self) -> float:
        return alpha_charge(self.tau_syn)

    def with_synapse(self, tau_syn_ms: float) -> "NeuronModel":
        """The same neuron with another synaptic time constant (slow synapses read rings and sheets)."""

        return NeuronModel(self.C_m, self.tau_m, self.t_ref, self.V_th, float(tau_syn_ms))

    def nest_params(self) -> Dict[str, float]:
        return {"C_m": self.C_m, "tau_m": self.tau_m, "t_ref": self.t_ref, "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0,
                "V_th": self.V_th, "tau_syn_ex": self.tau_syn, "tau_syn_in": self.tau_syn, "I_e": 0.0}
