"""2-D attractor sheets in the kernel, with their anchor (gates + bias cells) and their level cells.

* **Sheet**: ring neurons on a square, with the recurrent weights of :class:`~ring_attractor_kinematics.neurons.sheet_layout.SheetMap`.
* **Anchor**: the gates (one value population per coordinate) give neuron ``c`` the current ``gain·(c·x − x²/2)``;
  the bias cells (16 neurons at a steady 100 Hz) add the fixed part ``offset − gain/2·|c|²``.  Both enter
  through one synapse type, so they rise and fall together when the anchor is switched.
* **Level cells** — a set-point loop (with the warm/cold ramp read-outs, the Hammel-style homeostasis of this
  work): a population that holds how far the packet's spike rate is from its set point and sends every sheet
  neuron the same current against it.  The anchor's height is a small difference of large currents and is off
  by some tens of pA from place to place; an error of the height is the same current at every neuron, so the
  same current at every neuron takes it back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import LevelDesign
from ..neurons.lif import RING_NEURON, NeuronModel, alpha_charge
from ..neurons.ring_layout import STIMULUS_WEIGHT
from ..neurons.sheet_layout import SheetMap
from ..neurons.tuning import Tunings
from .populations import Population, attach_recorder, connect_weights, create_population, first_id

#: receptors of a sheet neuron: the ring neuron's excitatory and inhibitory synapse, and the anchor's, which
#: carries currents of both signs with one time constant
EXCITATORY, INHIBITORY, ANCHOR = 1, 2, 3
ANCHOR_TAU_MS = 20.0
#: bias cells carry the fixed part of the anchor
BIAS_CELLS = 16
BIAS_RATE_HZ = 100.0


def sheet_neuron() -> Dict[str, Any]:
    """The ring neuron as ``iaf_psc_alpha_multisynapse`` (the same dynamics) with the anchor's synapse added."""

    params = {key: value for key, value in RING_NEURON.items() if key not in ("tau_syn_ex", "tau_syn_in")}
    params["tau_syn"] = [RING_NEURON["tau_syn_ex"], RING_NEURON["tau_syn_in"], ANCHOR_TAU_MS]
    return params


def connect_signed(backend: Any, pre: Any, post: Any, weights_post_by_pre: np.ndarray) -> int:
    """Positive weights on the excitatory synapse, negative ones on the inhibitory synapse; returns the synapse count."""

    w = np.asarray(weights_post_by_pre, dtype=float)
    pre_ids, post_ids = np.array(pre.tolist()), np.array(post.tolist())
    made = 0
    for mask, receptor in ((w > 0.0, EXCITATORY), (w < 0.0, INHIBITORY)):
        rows, cols = np.nonzero(mask)
        if rows.size:
            backend.Connect(pre_ids[cols], post_ids[rows], "one_to_one",
                            {"weight": w[rows, cols], "delay": np.ones(rows.size), "receptor_type": np.full(rows.size, receptor)})
            made += int(rows.size)
    return made


@dataclass
class Sheet:
    name: str
    mapping: SheetMap
    neurons: Any
    recorder: Any
    first_id: int
    generators: Any = None
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    history: List[np.ndarray] = field(default_factory=list)
    level: Optional[Population] = None

    def write(self, backend: Any, scaled: Optional[Sequence[float]], radius_cells: float = 2.0, rate_hz: float = 200.0) -> None:
        """Stimulate the neurons around a place directly (tests: a packet without an anchor); ``None`` stops it."""

        if self.generators is None:
            if scaled is None:
                return
            self.generators = backend.Create("poisson_generator", self.mapping.count, params={"rate": 0.0})
            backend.Connect(self.generators, self.neurons, "one_to_one", {"weight": STIMULUS_WEIGHT, "receptor_type": EXCITATORY})
        if scaled is None:
            backend.SetStatus(self.generators, {"rate": 0.0})
            return
        distance = np.linalg.norm(self.mapping.cells - self.mapping.to_cells(scaled), axis=1)
        backend.SetStatus(self.generators, [{"rate": float(rate_hz) if d <= radius_cells else 0.0} for d in distance])

    def read(self, backend: Any) -> np.ndarray:
        events = backend.GetStatus(self.recorder, "events")[0]
        senders = np.asarray(events["senders"], dtype=int)
        self.counts = np.bincount(senders - self.first_id, minlength=self.mapping.count).astype(float)[: self.mapping.count]
        backend.SetStatus(self.recorder, {"n_events": 0})
        self.history = (self.history + [self.counts])[-int(self.mapping.design.window):]
        return self.counts

    def window_counts(self) -> np.ndarray:
        return np.sum(self.history, axis=0) if self.history else np.zeros(self.mapping.count)


def build_sheet(backend: Any, name: str, mapping: SheetMap) -> Sheet:
    neurons = backend.Create("iaf_psc_alpha_multisynapse", mapping.count, params=sheet_neuron())
    connect_signed(backend, neurons, neurons, mapping.weight_matrix().T)
    recorder = attach_recorder(backend, neurons)
    return Sheet(name, mapping, neurons, recorder, first_id(backend, neurons))


def connect_anchor(backend: Any, sheet: Sheet, gates: Sequence[Any], decoders: np.ndarray) -> None:
    """The moving part of the anchor: gate ``a`` holds coordinate ``x_a``; read with ``decoders`` (neurons × 2: x and x²)."""

    gain = sheet.mapping.anchor_gain_per_unit()
    places = sheet.mapping.places
    d = np.asarray(decoders, dtype=float)
    for axis, gate in enumerate(gates):
        current = gain * (places[:, axis:axis + 1] * d[:, 0][None, :] - 0.5 * d[:, 1][None, :])        # (sheet neuron, gate neuron)
        backend.Connect(gate, sheet.neurons, {"rule": "all_to_all"},
                        {"weight": np.ascontiguousarray(current / alpha_charge(ANCHOR_TAU_MS)), "delay": 1.0, "receptor_type": ANCHOR})


def create_bias_cells(backend: Any, model: NeuronModel) -> Any:
    cells = backend.Create("iaf_psc_alpha", BIAS_CELLS, params=model.nest_params())
    start = np.random.default_rng(77).uniform(0.0, model.V_th, BIAS_CELLS)
    current = float(model.current(np.array([BIAS_RATE_HZ]))[0])
    backend.SetStatus(cells, [{"V_m": float(v), "I_e": current} for v in start])
    return cells


def connect_anchor_bias(backend: Any, sheet: Sheet, cells: Any, rate_hz: float) -> None:
    """The fixed part of the anchor, from bias cells firing at ``rate_hz`` each (measured: the calibration's bias rate)."""

    count = len(cells)
    per_train = sheet.mapping.anchor_bias() / (count * float(rate_hz) * alpha_charge(ANCHOR_TAU_MS))
    backend.Connect(cells, sheet.neurons, {"rule": "all_to_all"},
                    {"weight": np.ascontiguousarray(np.tile(per_train[:, None], (1, count))), "delay": 1.0, "receptor_type": ANCHOR})


def connect_level_cells(backend: Any, sheet: Sheet, tunings: Tunings, decoders: np.ndarray, level: LevelDesign,
                        seed: int = 0, record: bool = False) -> Population:
    """Give ``sheet`` its level cells.  ``decoders``: the bounded readout (stays at ±1 outside its range)."""

    tuning = tunings.level(level.tau_ms)
    cells = create_population(backend, tuning, "level_" + sheet.name, seed=seed, record=record)
    per_hz = 1.0 / (float(level.set_rate_hz) * float(level.span))
    connect_weights(backend, sheet.neurons, cells.neurons,
                    tuning.input_weights(np.ones((1, 1))) @ np.full((1, sheet.mapping.count), per_hz) / cells.charge)
    constant = tuning.input_weights(np.array([[-1.0 / float(level.span)]]))[:, 0]
    current = np.asarray(backend.GetStatus(cells.neurons, "I_e"), dtype=float) + constant
    backend.SetStatus(cells.neurons, [{"I_e": float(c)} for c in current])
    cells.tuning = tuning.with_bias(current)
    weights = -float(level.gain_pa) * np.asarray(decoders, dtype=float)[:, 0] / alpha_charge(ANCHOR_TAU_MS)
    backend.Connect(cells.neurons, sheet.neurons, {"rule": "all_to_all"},
                    {"weight": np.ascontiguousarray(np.tile(weights[None, :], (sheet.mapping.count, 1))), "delay": 1.0,
                     "receptor_type": ANCHOR})
    sheet.level = cells
    return cells
