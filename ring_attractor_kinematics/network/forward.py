"""The forward network (Figure 1, stages ① and ②): joint rings in, the hand's orientation and position out.

:class:`ForwardNetwork` builds, in one NEST kernel and in this order:

1. **joint rings** — a ring attractor per joint, with its encoder, and a **cos/sin population** reading it;
2. **link orientation** — the product and link-orientation populations of the forward wiring;
3. **readout cells** — nine per link and nine for the hand (the decoded matrix entries, for measurement);
4. **readout rings** — lift, pitch and yaw of the hand as bumps (with the norm population for cos pitch);
5. **position cells** — the end point as three sums of link-orientation entries (and three readout cells);
6. **forward sheets** — three 2-D attractor sheets (xy, xz, yz) anchored by three **gates** and the
   **bias cells**, each with **level cells** when ``options.level``.

The order is kept as the measurements were made: NEST hands out node IDs in creation order, and the
encoders' Poisson spikes depend on them.  :meth:`ForwardNetwork.step` writes joint angles, runs one tick
and returns a :class:`ForwardTick`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reaching_benchmark.geometry.chains import Chain
from reaching_benchmark.geometry.rotations import nearest_rotation

from ..calibration.artifacts import Calibrations
from ..config import NetworkOptions
from ..neurons.ring_layout import RingMap
from ..neurons.sheet_layout import PLANES, SheetMap, sheet_position
from ..neurons.tuning import Tunings
from ..spiking.feedforward_ring import build_readout_ring
from ..spiking.populations import (NEVER_FIRES_MV, SILENT_PA, Population, Silenced, add_bias, connect_weights,
                                   create_kernel, create_population, create_readout_cells, decoded_weights, random_subset,
                                   silence_nodes, switch_populations)
from ..spiking.ring_attractor import Ring, build_ring_attractor
from ..spiking.sheet import (BIAS_CELLS, Sheet, build_sheet, connect_anchor, connect_anchor_bias, connect_level_cells,
                             create_bias_cells)
from .forward_wiring import ONE, Combo, ForwardWiring, Node, Product, forward_wiring

READOUT_RING_NAMES = ("lift", "pitch", "yaw")
#: ticks of readout-ring spikes summed for one reading of its angle
READOUT_RING_WINDOW = 4
Source = Tuple[Population, np.ndarray]
#: the classes of neurons a diffuse lesion can hit (:meth:`ForwardNetwork.lesion_classes`); ``all`` is every class
LESION_CLASSES = ("rings", "value", "sheets")
#: one population of a lesion class: its name, its NodeCollection, its size
Group = Tuple[str, Any, int]


@dataclass
class ForwardTick:
    t_ms: float
    #: angle each joint ring holds (NaN while silent) and its distance from the written position (ring indices)
    held: np.ndarray
    ring_error: np.ndarray
    #: decoded orientation of every link (n, 3, 3) and of the hand (3, 3), raw
    links: np.ndarray
    tool: np.ndarray
    #: lift, pitch, yaw on the readout rings (rad; NaN while silent) and their spike counts
    angles: np.ndarray
    ring_spikes: np.ndarray
    #: scaled position held by the position cells
    cells_scaled: np.ndarray
    #: scaled place of each sheet's packet from this tick (3, 2), and the sheets' spikes this tick (3, neurons)
    places: np.ndarray
    sheet_counts: np.ndarray
    #: scaled position from the sheets over their reading window, and the difference of each coordinate's two copies
    sheets_scaled: np.ndarray
    disagreement: np.ndarray
    #: the same in metres, base frame
    position: np.ndarray
    cells_position: np.ndarray

    def tool_rotation(self) -> np.ndarray:
        return nearest_rotation(self.tool)


class ForwardNetwork:
    def __init__(self, backend: Any, chain: Chain, calibrations: Calibrations, tunings: Optional[Tunings] = None,
                 options: Optional[NetworkOptions] = None) -> None:
        self.backend = backend
        self.chain = chain
        self.calibrations = calibrations
        self.tunings = tunings or Tunings()
        self.options = options or NetworkOptions()
        if self.options.level and not calibrations.has("bounded_readouts"):
            raise ValueError("level cells need the bounded read-out (python -m ring_attractor_kinematics calibrate bounded_readouts)")
        self.wiring: ForwardWiring = forward_wiring(chain)
        self.space = self.wiring.space
        self.mapping = SheetMap(self.options.sheet)
        self.t_ms = 0.0
        self.rings: List[Ring] = []
        self.cos_sin: List[Population] = []
        self.sources: Dict[Node, Source] = {}
        self.products: Dict[Node, Population] = {}
        self.entries: Dict[Node, Population] = {}
        self.readout_rings: Dict[str, Population] = {}
        self.position_cells: List[Population] = []
        self.gates: List[Population] = []
        self.sheets: List[Sheet] = []
        self.level_cells: List[Population] = []
        self.anchor_on = False
        self.build_log: List[str] = []
        self._readout_ring_history: List[np.ndarray] = []
        create_kernel(backend, self.options.threads, self.options.seed)
        self._build_joint_rings()
        self._build_link_orientation()
        self._build_readout_cells()
        self._build_readout_rings()
        self._build_position_cells()
        self._build_forward_sheets()
        self.build_log.append("%s forward network: %d joints, %d products, %d link populations, %d neurons, %d synapses" % (
            chain.name, chain.n, len(self.products), len(self.entries), self.neuron_count(),
            int(backend.GetKernelStatus("num_connections"))))

    # -- construction --------------------------------------------------------
    def feed(self, population: Population, combo: Combo, dim: int) -> None:
        """Hand the value of ``combo`` (a weighted sum of nodes) to dimension ``dim`` of ``population``."""

        transform = np.zeros((population.tuning.dims, 1))
        transform[dim, 0] = 1.0
        for node, coefficient in combo.items():
            if node == ONE:
                add_bias(self.backend, population, population.tuning.input_weights(transform)[:, 0] * float(coefficient))
            else:
                source, decoders = self.sources[node]
                connect_weights(self.backend, source.neurons, population.neurons,
                                decoded_weights(decoders * float(coefficient), transform, population))

    def connect_ring_to_cos_sin(self, ring: Ring, population: Population, delay_ms: float = 1.0, scale: Optional[float] = None) -> None:
        """``scale``: the factor that makes a held bump's sums a pair of length 1 (default: a joint ring's)."""

        angles = ring.mapping.neuron_angles()
        factor = self.calibrations.feature_scale(ring.mapping) if scale is None else float(scale)
        basis = factor * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        connect_weights(self.backend, ring.neurons, population.neurons,
                        population.tuning.input_weights(np.eye(2)) @ basis.T / population.charge, delay=delay_ms)

    def _noise(self, populations: Sequence[Population]) -> None:
        if self.options.noise_pa > 0.0:
            noise = self.backend.Create("noise_generator", params={"mean": 0.0, "std": float(self.options.noise_pa), "dt": 1.0})
            for population in populations:
                self.backend.Connect(noise, population.neurons)

    def _build_joint_rings(self) -> None:
        b, record = self.backend, self.options.record
        tuning = self.tunings.cos_sin(self.calibrations.cos_sin_tau_ms)
        decoders = self.calibrations.decoders["feature"]
        for k, joint in enumerate(self.chain.joints):
            ring = build_ring_attractor(b, "joint_ring%d" % (k + 1), RingMap.for_joint(joint, self.calibrations.ring_size))
            cos_sin = create_population(b, tuning, "cos_sin%d" % (k + 1), seed=10 + k, record=record)
            self.connect_ring_to_cos_sin(ring, cos_sin)
            self.rings.append(ring)
            self.cos_sin.append(cos_sin)
            for part in (0, 1):
                self.sources[("feat", k, part)] = (cos_sin, decoders[:, part:part + 1])

    def _build_link_orientation(self) -> None:
        b, record, decoders = self.backend, self.options.record, self.calibrations.decoders
        for step in self.wiring.steps:
            if isinstance(step, Product):
                population = create_population(b, self.tunings.product, "product%d" % step.index, seed=100 + step.index, record=record)
                self.feed(population, dict(step.operand), 0)
                self.feed(population, {step.feature: 1.0}, 1)
                self.products[step.node] = population
                self.sources[step.node] = (population, decoders["product"])
            else:
                population = create_population(b, self.tunings.entry, "link%d_%d%d" % (step.link, step.row, step.col),
                                               seed=5000 + 100 * step.link + 10 * step.row + step.col, record=record)
                self.feed(population, dict(step.terms), 0)
                self.entries[step.node] = population
                self.sources[step.node] = (population, decoders["scalar"])

    def _build_readout_cells(self) -> None:
        matrices = list(self.wiring.links) + [self.wiring.tool]
        self.matrix_cells = create_readout_cells(self.backend, 9 * len(matrices), self.options.readout_tau_ms)
        constants = np.zeros(self.matrix_cells.n)
        for m, matrix in enumerate(matrices):
            for i in range(3):
                for j in range(3):
                    cell = 9 * m + 3 * i + j
                    for node, coefficient in matrix[i][j].items():
                        if node == ONE:
                            constants[cell] += coefficient
                        else:
                            population, decoders = self.sources[node]
                            self.matrix_cells.connect(self.backend, population, decoders * coefficient, cell)
        self.matrix_cells.set_constants(self.backend, constants)

    def _build_readout_rings(self) -> None:
        """Lift = atan2(r32, r33), pitch = atan2(−r31, |(r11, r21)|), yaw = atan2(r21, r11), as bumps."""

        b, tool = self.backend, self.wiring.tool
        self.norm = create_population(b, self.tunings.norm, "pitch_norm", seed=9000, record=self.options.record)
        self.feed(self.norm, tool[0][0], 0)
        self.feed(self.norm, tool[1][0], 1)
        norm_source = (self.norm, self.calibrations.decoders["norm"])

        def terms(combo: Combo, sign: float = 1.0) -> List[Tuple[Optional[Source], float]]:
            return [(None if node == ONE else self.sources[node], sign * coefficient) for node, coefficient in combo.items()]

        drives = {"lift": (terms(tool[2][2]), terms(tool[2][1])),
                  "pitch": ([(norm_source, 1.0)], terms(tool[2][0], -1.0)),
                  "yaw": (terms(tool[0][0]), terms(tool[1][0]))}
        for index, name in enumerate(READOUT_RING_NAMES):
            self.readout_rings[name] = build_readout_ring(b, name, int(self.tunings.sizes.output_ring), self.tunings.model,
                                                          drives[name], index)
        self._noise(self.cos_sin + list(self.products.values()) + list(self.entries.values()) + [self.norm])

    def _build_position_cells(self) -> None:
        b, record = self.backend, self.options.record
        for count, cell in enumerate(self.wiring.position_cells):
            population = create_population(b, self.tunings.entry, "pos_%d" % cell.node[1], seed=30000 + 100 * (1 + count), record=record)
            self.feed(population, dict(cell.terms), 0)
            self.position_cells.append(population)
            self.sources[cell.node] = (population, self.calibrations.decoders["scalar"])
        self.position_readout = create_readout_cells(b, 3, self.options.readout_tau_ms)
        for i, node in enumerate(self.wiring.position):
            population, decoders = self.sources[node]
            self.position_readout.connect(b, population, decoders, i)

    def _build_forward_sheets(self) -> None:
        b, o = self.backend, self.options
        for i in range(3):
            gate = create_population(b, self.tunings.entry, "gate%d" % i, seed=40000 + i, record=o.record)
            self.feed(gate, {self.wiring.position[i]: 1.0}, 0)
            self.gates.append(gate)
        self.bias_cells = create_bias_cells(b, self.tunings.model)
        self.bias_current = float(b.GetStatus(self.bias_cells, "I_e")[0])
        for name, first, second in PLANES:
            sheet = build_sheet(b, name, self.mapping)
            connect_anchor(b, sheet, [self.gates[first].neurons, self.gates[second].neurons], self.calibrations.decoders["gate"])
            connect_anchor_bias(b, sheet, self.bias_cells, self.calibrations.bias_rate_hz)
            if o.level:
                self.level_cells.append(connect_level_cells(b, sheet, self.tunings, self.calibrations.decoders["bounded"],
                                                            o.level_design, seed=46000 + len(self.sheets), record=o.record))
            self.sheets.append(sheet)
        self._noise(self.position_cells + self.gates)
        self.anchor_on = True
        self.set_anchor(True)

    # -- switches ------------------------------------------------------------
    def set_anchor(self, on: bool) -> None:
        """Anchor off: gates, bias cells and level cells are silenced, and the packets stay where they are.
        Switching on puts every one of those neurons at a fresh potential below threshold."""

        b, model = self.backend, self.tunings.model
        rng = np.random.default_rng([4242, int(self.t_ms)])
        switch_populations(b, self.gates, on, model.V_th, rng)
        if on:
            b.SetStatus(self.bias_cells, [{"I_e": self.bias_current, "V_m": float(v)} for v in rng.uniform(0.0, model.V_th, BIAS_CELLS)])
        else:
            b.SetStatus(self.bias_cells, {"I_e": SILENT_PA})
        switch_populations(b, self.level_cells, on, model.V_th, rng)
        self.anchor_on = bool(on)

    # -- running -------------------------------------------------------------
    def step(self, q: Sequence[float]) -> ForwardTick:
        """Write the joint angles on the joint rings, simulate one tick, return what the network holds."""

        b = self.backend
        for ring, angle in zip(self.rings, np.asarray(q, dtype=float).reshape(-1)):
            ring.drive(b, float(angle))
        b.Simulate(float(self.options.tick_ms))
        self.t_ms += float(self.options.tick_ms)
        for ring in self.rings:
            ring.read(b)
        held = np.array([ring.held_angle() for ring in self.rings])
        ring_error = np.array([ring.ring_error() for ring in self.rings])
        matrices = self.matrix_cells.values(b).reshape(-1, 3, 3)
        counts = np.array([self.readout_rings[name].counts(b) for name in READOUT_RING_NAMES])
        self._readout_ring_history = (self._readout_ring_history + [counts])[-READOUT_RING_WINDOW:]
        window = np.sum(self._readout_ring_history, axis=0)
        vector = window @ np.exp(2j * np.pi * np.arange(window.shape[1]) / window.shape[1])
        ring_spikes = window.sum(axis=1)
        angles = np.where(ring_spikes > 0, np.angle(vector), np.nan)
        sheet_counts = np.array([sheet.read(b) for sheet in self.sheets])
        places = np.array([self.mapping.centre(c) for c in sheet_counts])
        sheets_scaled, disagreement = sheet_position(self.mapping, np.array([sheet.window_counts() for sheet in self.sheets]))
        cells_scaled = self.position_readout.values(b)
        return ForwardTick(self.t_ms, held, ring_error, matrices[:-1], matrices[-1], angles, ring_spikes, cells_scaled, places,
                           sheet_counts, sheets_scaled, disagreement, self.space.metres(sheets_scaled), self.space.metres(cells_scaled))

    def position_from_counts(self, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Position (m) and disagreement (m) from summed spike counts of the three sheets."""

        scaled, disagreement = sheet_position(self.mapping, counts)
        return self.space.metres(scaled), disagreement * self.space.radius

    def ring_counts(self) -> np.ndarray:
        return np.array([ring.counts for ring in self.rings])

    def readout_ring_counts(self) -> np.ndarray:
        size = int(self.tunings.sizes.output_ring)
        return self._readout_ring_history[-1] if self._readout_ring_history else np.zeros((3, size))

    def forward_neuron_count(self) -> int:
        n = sum(ring.size for ring in self.rings) + sum(p.n for p in self.cos_sin)
        n += sum(p.n for p in self.products.values()) + sum(p.n for p in self.entries.values())
        n += sum(p.n for p in self.readout_rings.values()) + self.norm.n
        n += sum(p.n for p in self.position_cells) + sum(p.n for p in self.gates)
        n += len(self.sheets) * self.mapping.count + BIAS_CELLS + sum(p.n for p in self.level_cells)
        return int(n)

    def neuron_count(self) -> int:
        return self.forward_neuron_count()

    # -- lesions -------------------------------------------------------------
    def lesion_classes(self) -> Dict[str, List[Group]]:
        """Every population that does the controller's work, by class, in build order:

        * ``rings``: the joint rings;
        * ``value``: the value populations (cos/sin, products, link entries, position cells, gates);
        * ``sheets``: the attractor sheets, their level cells and their bias cells.

        Left out: the readout rings and the norm population feeding the pitch ring, which only display the hand's
        lift, pitch and yaw, and the readout cells, which measure and never fire."""

        value = self.cos_sin + list(self.products.values()) + list(self.entries.values()) + self.position_cells + self.gates
        return {"rings": [(ring.name, ring.neurons, ring.size) for ring in self.rings],
                "value": [(p.name, p.neurons, p.n) for p in value],
                "sheets": ([(sheet.name, sheet.neurons, sheet.mapping.count) for sheet in self.sheets]
                           + [(p.name, p.neurons, p.n) for p in self.level_cells] + [("bias_cells", self.bias_cells, BIAS_CELLS)])}

    def lesion_fraction(self, fraction: float, seed: int, classes: Sequence[str] = ("all",)) -> Dict[str, Any]:
        """Diffuse neuron loss for the rest of the run. In every population of ``classes`` (``all``: every class),
        exactly ``round(fraction · n)`` neurons, drawn at random, are silenced (:func:`silence_nodes`).

        Each population draws from its own stream, seeded by ``seed`` and its place in :meth:`lesion_classes`, so it
        loses the same neurons whichever other classes are lesioned. Returns a description of the lesion."""

        unknown = set(classes) - set(LESION_CLASSES) - {"all"}
        if unknown:
            raise ValueError("unknown lesion classes %s (known: all, %s)" % (sorted(unknown), ", ".join(LESION_CLASSES)))
        chosen = list(LESION_CLASSES) if "all" in classes else [c for c in LESION_CLASSES if c in classes]
        groups = self.lesion_classes()
        collections, by_class = [], {}
        for c, name in enumerate(LESION_CLASSES):
            if name not in chosen:
                continue
            lost = total = 0
            for index, (_, nodes, size) in enumerate(groups[name]):
                subset = random_subset(nodes, size, fraction, np.random.default_rng([int(seed), c, index]))
                if subset is not None:
                    collections.append(subset)
                    lost += len(subset)
                total += size
            by_class[name] = {"populations": len(groups[name]), "neurons": total, "silenced": lost}
        silenced: Silenced = silence_nodes(self.backend, collections)
        return {"fraction": float(fraction), "classes": chosen, "lesion_seed": int(seed), "silenced": silenced.count,
                "of": int(sum(c["neurons"] for c in by_class.values())), "network_neurons": self.neuron_count(), "by_class": by_class,
                "how": "threshold raised to %g mV from the build on (no input makes the neuron fire); never restored" % NEVER_FIRES_MV}
