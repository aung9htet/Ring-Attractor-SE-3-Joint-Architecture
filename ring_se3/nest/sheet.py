"""One attractor sheet in the NEST kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ring_so3.nest.ring import RING_NEURON, STIMULUS_WEIGHT

from ..sheet import SheetMap


#: receptors of a sheet neuron: the ring neuron's excitatory and inhibitory synapse; a fast inhibitory one that
#: only the velocity cells use; and one for the anchor, which carries currents of both signs with one time
#: constant (through the ring neuron's two synapses the negative part would outlast the positive part
#: whenever the anchor changes, and that silences the packet)
EXCITATORY, INHIBITORY, GATE, ANCHOR = 1, 2, 3, 4
ANCHOR_TAU_MS = 20.0


def sheet_neuron(gate_tau_ms: float) -> Dict[str, Any]:
    """The SO-2 ring neuron as ``iaf_psc_alpha_multisynapse`` (the same dynamics) with one more synapse."""

    params = {key: value for key, value in RING_NEURON.items() if key not in ("tau_syn_ex", "tau_syn_in")}
    params["tau_syn"] = [RING_NEURON["tau_syn_ex"], RING_NEURON["tau_syn_in"], float(gate_tau_ms), ANCHOR_TAU_MS]
    return params


def charge(weight_sign: float) -> float:
    """Mean current (pA) a 1 Hz input delivers per pA of weight into a sheet neuron (the ring neuron's two synapses)."""

    tau = RING_NEURON["tau_syn_ex"] if weight_sign >= 0 else RING_NEURON["tau_syn_in"]
    return float(np.e * tau * 1e-3)


def connect_signed(backend: Any, pre: Any, post: Any, weights_post_by_pre: np.ndarray, smallest: float = 0.0) -> int:
    """One synapse per pair with a weight above ``smallest`` in size: positive weights on the excitatory
    synapse, negative ones on the inhibitory synapse.  Returns the number of synapses."""

    w = np.asarray(weights_post_by_pre, dtype=float)
    pre_ids, post_ids = np.array(pre.tolist()), np.array(post.tolist())
    made = 0
    for mask, receptor in ((w > smallest, EXCITATORY), (w < -smallest, INHIBITORY)):
        rows, cols = np.nonzero(mask)
        if rows.size:
            backend.Connect(pre_ids[cols], post_ids[rows], "one_to_one",
                            {"weight": w[rows, cols], "delay": np.ones(rows.size), "receptor_type": np.full(rows.size, receptor)})
            made += int(rows.size)
    return made


def current_weights(current_per_hz: np.ndarray) -> np.ndarray:
    """Synaptic weights (pA) that deliver ``current_per_hz`` (pA of mean current per Hz of the sender)."""

    c = np.asarray(current_per_hz, dtype=float)
    return np.where(c >= 0, c / charge(1.0), c / charge(-1.0))


@dataclass
class Sheet:
    name: str
    mapping: SheetMap
    neurons: Any
    generators: Any
    recorder: Any
    first_id: int
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    history: List[np.ndarray] = field(default_factory=list)

    def write(self, backend: Any, scaled: Optional[Sequence[float]], radius_cells: float = 2.0, rate_hz: float = 200.0) -> None:
        """Stimulate the neurons around a place directly (tests and calibration: a packet without an anchor);
        ``None`` stops it.  The stimulus generators are made on first use: the pose network never needs them."""

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

    def place(self, window: Optional[int] = None) -> np.ndarray:
        """Scaled place of the packet from the last ``window`` ticks (default: the design's)."""

        recent = self.history[-int(window):] if window else self.history
        return self.mapping.centre(np.sum(recent, axis=0)) if recent else np.array([np.nan, np.nan])

    def spikes(self) -> float:
        return float(np.sum(self.history)) if self.history else 0.0


def build_sheet(backend: Any, name: str, mapping: SheetMap) -> Sheet:
    neurons = backend.Create("iaf_psc_alpha_multisynapse", mapping.count, params=sheet_neuron(mapping.design.velocity_tau_ms))
    connect_signed(backend, neurons, neurons, mapping.weight_matrix().T)
    recorder = backend.Create("spike_recorder")
    backend.Connect(neurons, recorder)
    return Sheet(name, mapping, neurons, None, recorder, int(np.min(backend.GetStatus(neurons, "global_id"))))


def connect_anchor(backend: Any, sheet: Sheet, gates: Sequence[Any], decoders: np.ndarray) -> None:
    """The moving part of the anchor: two populations, each holding one coordinate ``x`` of the anchored place,
    read with ``decoders`` (neurons x 2: ``x`` and ``x²``), give neuron ``c`` the current ``gain · (c·x - x²/2)``."""

    gain = sheet.mapping.anchor_gain_per_unit()
    places = sheet.mapping.places
    d = np.asarray(decoders, dtype=float)
    for axis, gate in enumerate(gates):
        current = gain * (places[:, axis:axis + 1] * d[:, 0][None, :] - 0.5 * d[:, 1][None, :])        # (sheet neuron, gate neuron)
        weights = current / float(np.e * ANCHOR_TAU_MS * 1e-3)
        backend.Connect(gate, sheet.neurons, {"rule": "all_to_all"},
                        {"weight": np.ascontiguousarray(weights), "delay": 1.0, "receptor_type": ANCHOR})


def connect_anchor_bias(backend: Any, sheet: Sheet, cells: Any, rate_hz: float) -> None:
    """The fixed part of the anchor (``offset - gain/2 · |c|²`` for the neuron at ``c``), carried by cells that
    fire at a steady ``rate_hz`` each and through the same synapse as the moving part, so that both parts
    rise and fall together when the anchor is switched."""

    count = len(cells)
    per_spike_train = sheet.mapping.anchor_bias() / (count * float(rate_hz) * float(np.e * ANCHOR_TAU_MS * 1e-3))
    backend.Connect(cells, sheet.neurons, {"rule": "all_to_all"},
                    {"weight": np.ascontiguousarray(np.tile(per_spike_train[:, None], (1, count))), "delay": 1.0,
                     "receptor_type": ANCHOR})


def connect_velocity_cells(backend: Any, sheet: Sheet, cells: Any, axis_in_plane: int, sign: int) -> None:
    """Velocity cells for a motion along one coordinate of the sheet silence the group that points against it."""

    from ..sheet import against

    group = sheet.mapping.group(against(axis_in_plane, sign))
    targets = sheet.neurons[[int(i) for i in group]]
    backend.Connect(cells, targets, "all_to_all", {"weight": -abs(float(sheet.mapping.design.velocity_weight)), "receptor_type": GATE})


__all__ = ["ANCHOR", "ANCHOR_TAU_MS", "EXCITATORY", "GATE", "INHIBITORY", "Sheet", "build_sheet", "charge", "connect_anchor", "connect_anchor_bias", "connect_signed",
           "connect_velocity_cells", "current_weights", "sheet_neuron"]
