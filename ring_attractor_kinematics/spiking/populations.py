"""Value populations and readout cells in a NEST kernel.

The kernel is passed in as ``backend`` (the ``nest`` module): nothing in this package imports NEST, so
everything up to the first ``Create`` runs without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..neurons.lif import NeuronModel, alpha_charge
from ..neurons.tuning import Tuning

#: the current that holds a neuron far below threshold: a silenced population sends nothing
SILENT_PA = -1.0e6
#: a lesioned neuron's threshold, mV: no input makes it fire (the readout cells' threshold)
NEVER_FIRES_MV = 1.0e12
#: readout cell: a leaky integrator that never fires; its membrane potential is the smoothed weighted sum of its input
READOUT = {"tau_m": 50.0, "C_m": 250.0, "tau_syn": 5.0, "mv_per_unit": 10.0}


@dataclass
class Population:
    name: str
    neurons: Any
    n: int
    first_id: int
    tuning: Optional[Tuning] = None
    recorder: Any = None
    #: mean current (pA) per Hz of a sender per pA of weight, for weights into this population
    charge: float = 0.0

    def counts(self, backend: Any) -> np.ndarray:
        """Spikes of every neuron since the last call (needs a recorder)."""

        events = backend.GetStatus(self.recorder, "events")[0]
        senders = np.asarray(events["senders"], dtype=int)
        out = np.bincount(senders - self.first_id, minlength=self.n).astype(float)[: self.n]
        backend.SetStatus(self.recorder, {"n_events": 0})
        return out


def first_id(backend: Any, neurons: Any) -> int:
    return int(np.min(backend.GetStatus(neurons, "global_id")))


def attach_recorder(backend: Any, neurons: Any) -> Any:
    recorder = backend.Create("spike_recorder")
    backend.Connect(neurons, recorder)
    return recorder


def create_population(backend: Any, tuning: Tuning, name: str, seed: int = 0, record: bool = False) -> Population:
    """``tuning.n`` neurons with the population's bias currents; start potentials spread out (seeded)."""

    model: NeuronModel = tuning.model
    neurons = backend.Create("iaf_psc_alpha", tuning.n, params=model.nest_params())
    start = np.random.default_rng([int(seed), tuning.seed]).uniform(0.0, model.V_th, tuning.n)
    backend.SetStatus(neurons, [{"I_e": float(b), "V_m": float(v)} for b, v in zip(tuning.bias, start)])
    return Population(name, neurons, tuning.n, first_id(backend, neurons), tuning,
                      attach_recorder(backend, neurons) if record else None, model.charge)


def set_value(backend: Any, population: Population, x: Sequence[float]) -> None:
    """Drive a population with a constant value directly (a *source*: goal positions, calibration inputs)."""

    current = population.tuning.currents(np.asarray(x, dtype=float).reshape(1, -1))[0]
    backend.SetStatus(population.neurons, [{"I_e": float(c)} for c in current])


def add_bias(backend: Any, population: Population, extra_pa: np.ndarray) -> np.ndarray:
    """Add a constant current to every neuron (the constant part of a value); returns the new bias."""

    current = np.asarray(backend.GetStatus(population.neurons, "I_e"), dtype=float) + np.asarray(extra_pa, dtype=float)
    backend.SetStatus(population.neurons, [{"I_e": float(c)} for c in current])
    return current


def connect_weights(backend: Any, pre: Any, post: Any, weights_post_by_pre: np.ndarray, delay: float = 1.0) -> None:
    """All-to-all with one weight (pA) per pair; ``weights_post_by_pre`` has shape (post, pre)."""

    matrix = np.ascontiguousarray(np.asarray(weights_post_by_pre, dtype=float))
    backend.Connect(pre, post, {"rule": "all_to_all"}, {"weight": matrix, "delay": float(delay)})


def decoded_weights(decoders: np.ndarray, transform: np.ndarray, post: Population) -> np.ndarray:
    """Weights (post, pre) that hand ``transform @ decoded values`` of the sender to ``post``."""

    return post.tuning.input_weights(transform) @ np.asarray(decoders, dtype=float).T / post.charge


def switch_populations(backend: Any, populations: Sequence[Population], on: bool, threshold: float, rng: np.random.Generator) -> None:
    """Silence populations, or give them their own currents back with every neuron at a fresh potential."""

    for population in populations:
        if on:
            backend.SetStatus(population.neurons, [{"I_e": float(i), "V_m": float(v)}
                                                   for i, v in zip(population.tuning.bias, rng.uniform(0.0, threshold, population.n))])
        else:
            backend.SetStatus(population.neurons, {"I_e": SILENT_PA})


@dataclass
class Silenced:
    """Lesioned neurons, with what it takes to give them back (see :func:`silence_nodes`)."""

    #: the NodeCollections as given, and per collection the (V_th, V_m) of every neuron when it was silenced
    nodes: List[Any] = field(default_factory=list)
    saved: List[List[Tuple[float, float]]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return int(sum(len(saved) for saved in self.saved))


def silence_nodes(backend: Any, collections: Sequence[Any]) -> Silenced:
    """Lesion every neuron of ``collections``: raise its threshold to :data:`NEVER_FIRES_MV`, so it sends nothing.

    A threshold holds whatever the input, unlike a large negative current (:data:`SILENT_PA`). A written goal ring's
    neuron, for one, can take nearly 1 µA from a single gain neuron. The neuron goes on integrating its input;
    :func:`restore_nodes` gives back its threshold and the potential it had when it was silenced."""

    out = Silenced()
    for nodes in collections:
        status = backend.GetStatus(nodes, ["V_th", "V_m"])
        out.nodes.append(nodes)
        out.saved.append([(float(threshold), float(potential)) for threshold, potential in status])
        backend.SetStatus(nodes, {"V_th": NEVER_FIRES_MV})
    return out


def restore_nodes(backend: Any, silenced: Silenced) -> None:
    """Undo :func:`silence_nodes`."""

    for nodes, saved in zip(silenced.nodes, silenced.saved):
        backend.SetStatus(nodes, [{"V_th": threshold, "V_m": potential} for threshold, potential in saved])


def random_subset(nodes: Any, size: int, fraction: float, rng: np.random.Generator) -> Optional[Any]:
    """Exactly ``round(fraction · size)`` neurons of ``nodes``, drawn without replacement; ``None`` when that is none."""

    chosen = np.sort(rng.choice(int(size), int(round(float(fraction) * int(size))), replace=False))
    return nodes[[int(i) for i in chosen]] if chosen.size else None


@dataclass
class ReadoutCells:
    """Cells whose membrane potential is a weighted sum of spikes, smoothed with ``tau_m`` (they never fire)."""

    neurons: Any
    n: int
    settings: Dict[str, float] = field(default_factory=lambda: dict(READOUT))

    @property
    def pa_per_unit(self) -> float:
        """Input current that holds the potential at one unit of value."""

        return self.settings["mv_per_unit"] * self.settings["C_m"] / self.settings["tau_m"]

    @property
    def charge(self) -> float:
        return alpha_charge(self.settings["tau_syn"])

    def connect(self, backend: Any, pre: Population, decoders: np.ndarray, cell: int) -> None:
        """``decoders``: (pre neurons, 1), value per Hz, into readout cell ``cell``."""

        d = np.asarray(decoders, dtype=float)
        connect_weights(backend, pre.neurons, self.neurons[int(cell)], (d * self.pa_per_unit / self.charge).T)

    def set_constants(self, backend: Any, constants: Sequence[float]) -> None:
        values = np.asarray(constants, dtype=float)
        backend.SetStatus(self.neurons, [{"I_e": float(v * self.pa_per_unit)} for v in values])

    def values(self, backend: Any) -> np.ndarray:
        return np.asarray(backend.GetStatus(self.neurons, "V_m"), dtype=float) / self.settings["mv_per_unit"]


def create_readout_cells(backend: Any, n: int, tau_ms: float) -> ReadoutCells:
    s = dict(READOUT, tau_m=float(tau_ms))
    params = {"C_m": s["C_m"], "tau_m": s["tau_m"], "t_ref": 0.0, "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0,
              "V_th": 1.0e12, "tau_syn_ex": s["tau_syn"], "tau_syn_in": s["tau_syn"], "I_e": 0.0}
    return ReadoutCells(backend.Create("iaf_psc_alpha", int(n), params=params), int(n), s)


def create_kernel(backend: Any, threads: int, seed: int) -> None:
    """A fresh NEST kernel."""

    backend.ResetKernel()
    backend.set_verbosity("M_ERROR")
    backend.SetKernelStatus({"local_num_threads": int(threads), "rng_seed": int(seed)})
