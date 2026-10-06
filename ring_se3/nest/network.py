"""The spiking pose network: the SO(3) network of :mod:`ring_so3` plus position, velocity and the three sheets.

:class:`PoseNetwork` first builds the SO(3) network unchanged (joint rings, cos / sin
populations, products, link orientations, output rings) and then adds, in the same kernel:

* the populations of a :class:`~ring_se3.wiring.PoseWiring`: three position cells, the
  levers, the joint-velocity inputs, the products of the velocity, three velocity populations
  (all of the kinds the SO(3) network already has, with its calibrated readouts);
* three *gate* populations, one per coordinate, that hand the position to the sheets, and a few
  *bias cells* that carry the fixed part of the anchor (silencing both switches the anchor off);
* *velocity cells*: for each coordinate and sign a few slow integrate-and-fire neurons.  While a
  joint moves they all fire at a resting rate, and a velocity raises the rate on one side and
  lowers it on the other (silencing them switches path integration off);
* the three attractor sheets (:mod:`ring_se3.sheet`);
* readout cells for the position cells and the velocity populations.

:meth:`PoseNetwork.step` writes joint angles and joint velocities, runs one tick and
returns a :class:`PoseTick`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ring_so3.design import Design
from ring_so3.kinematics import Chain
from ring_so3.nest.calibrate import Calibration
from ring_so3.nest.network import FkNetwork, NetworkOptions, Tick
from ring_so3.nest.populations import (Population, connect_weights, create_population, create_readout_cells, decoded_weights,
                                       set_value)

from ..sheet import PLANES, SheetDesign, SheetMap, combine
from ..wiring import Mul, PoseWiring, pose_wiring
from .calibrate import (BIAS_CELLS, MOVING_RAD_S, PoseCalibration, create_bias_cells, create_velocity_cells,
                        velocity_rest_current)
from .rigs import velocity_input_weights
from .sheet import Sheet, build_sheet, connect_anchor, connect_anchor_bias, connect_velocity_cells

SILENT_PA = -1.0e6


@dataclass
class PoseOptions(NetworkOptions):
    #: build the velocity path (joint-velocity inputs, products, velocity populations and cells); without it the sheets are anchored only
    velocity: bool = True
    #: the anchor is given the position this many seconds ahead (position + velocity x lead); needs ``velocity``
    anchor_lead_s: float = 0.0
    #: start with the anchor / path integration switched on
    anchor: bool = True
    path_integration: bool = True
    #: three more velocity populations whose values are set from outside (the shift-register experiment:
    #: a velocity taken from where the packets have just been, instead of from the joint velocities)
    trail: bool = False
    sheet: SheetDesign = field(default_factory=SheetDesign)


@dataclass
class PoseTick:
    so3: Tick
    #: scaled position held by the position cells and scaled velocity held by the velocity populations (NaN without the velocity path)
    cells_scaled: np.ndarray
    velocity_scaled: np.ndarray
    #: scaled place of each sheet's packet from this tick's spikes, (3, 2); NaN while a sheet is silent
    places: np.ndarray
    #: spike counts of the three sheets in this tick, (3, neurons)
    sheet_counts: np.ndarray
    #: scaled position from the sheets over their reading window, and the difference of the two copies of each coordinate
    sheets_scaled: np.ndarray
    disagreement: np.ndarray
    #: the same in metres, base frame
    position: np.ndarray
    cells_position: np.ndarray

    @property
    def t_ms(self) -> float:
        return self.so3.t_ms

    def __getattr__(self, name: str) -> Any:        # held, ring_error, links, tool, angles, ring_spikes, tool_rotation
        if name == "so3":
            raise AttributeError(name)
        return getattr(self.so3, name)


def sheet_position(mapping: SheetMap, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled position and disagreement from spike counts of the three sheets, (3, neurons)."""

    return combine({name: mapping.centre(counts[i]) for i, (name, _, _) in enumerate(PLANES)})


class PoseNetwork(FkNetwork):
    def __init__(self, backend: Any, chain: Chain, calibration: Calibration, pose_calibration: PoseCalibration,
                 design: Optional[Design] = None, options: Optional[PoseOptions] = None) -> None:
        self.pose_calibration = pose_calibration
        self.pose_options = options or PoseOptions()
        self.pose: PoseWiring = pose_wiring(chain, self.pose_options.relay, self.pose_options.velocity)
        self.space = self.pose.space
        self.mapping = SheetMap(self.pose_options.sheet)
        self.muls: Dict[Any, Population] = {}
        self.relays: Dict[Any, Population] = {}
        self.qdot: List[Population] = []
        self.gates: List[Population] = []
        self.velocity_cells: Dict[Tuple[int, int], Any] = {}
        self.sheets: List[Sheet] = []
        self.trail_sources: List[Population] = []
        self.trail_relays: List[Population] = []
        self.anchor_on = False
        self.path_integration_on = False
        self._moving = False
        self._trail_moving = False
        self._qdot_written = np.zeros(chain.n)
        super().__init__(backend, chain, calibration, design, self.pose_options)
        self._build_pose()

    # -- construction --------------------------------------------------------
    def _build_pose(self) -> None:
        b, o, cal, pose_cal = self.backend, self.pose_options, self.calibration, self.pose_calibration
        entry, product = self.design.entry, self.design.product
        for k in range(self.chain.n if o.velocity else 0):
            population = create_population(b, entry, "qdot%d" % (k + 1), seed=45000 + k, record=o.record)
            set_value(b, population, [0.0])
            self.qdot.append(population)
            self.sources[("qdot", k)] = (population, cal.decoders["source"])
        for step in self.pose.steps:
            if isinstance(step, Mul):
                population = create_population(b, product, "mul%d" % step.index, seed=20000 + step.index, record=o.record)
                self._feed(population, dict(step.a), 0)
                self._feed(population, dict(step.b), 1)
                self.muls[step.node] = population
                self.sources[step.node] = (population, cal.decoders["product"])
            else:
                seed = 30000 + 100 * (1 + len(self.relays))
                population = create_population(b, entry, "_".join(str(v) for v in step.node), seed=seed, record=o.record)
                self._feed(population, dict(step.terms), 0)
                self.relays[step.node] = population
                self.sources[step.node] = (population, cal.decoders["scalar"])

        # readout cells: position (3) and velocity (3)
        nodes = list(self.pose.position) + list(self.pose.velocity_nodes)
        self.pose_cells = create_readout_cells(b, len(nodes), {"tau_m": float(o.readout_tau_ms)})
        for cell, node in enumerate(nodes):
            population, decoders = self.sources[node]
            self.pose_cells.connect(b, population, decoders, [cell])

        # gates: the position (plus, with a lead, velocity x lead) as handed to the sheets
        lead = float(o.anchor_lead_s) * self.pose.qdot_ref * self.pose.velocity_scale if o.velocity else 0.0
        for i in range(3):
            population = create_population(b, entry, "gate%d" % i, seed=40000 + i, record=o.record)
            combo = {self.pose.position[i]: 1.0}
            if lead:
                combo[self.pose.velocity_nodes[i]] = lead
            self._feed(population, combo, 0)
            self.gates.append(population)

        # velocity cells: (coordinate, sign) -> a few slow integrate-and-fire neurons
        if o.velocity:
            for i in range(3):
                source, _ = self.sources[self.pose.velocity_nodes[i]]
                for sign in (1, -1):
                    cells = create_velocity_cells(b, o.sheet, seed=10 * i + (sign > 0))
                    connect_weights(b, source.neurons, cells, velocity_input_weights(pose_cal, self.mapping, sign))
                    self.velocity_cells[(i, sign)] = cells

        # the shift-register experiment: a second way into the velocity cells, set from outside
        if o.velocity and o.trail:
            transform = np.ones((1, 1))
            for i in range(3):
                source = create_population(b, entry, "trail_source%d" % i, seed=47000 + i, record=o.record)
                relay = create_population(b, entry, "trail%d" % i, seed=47100 + i, record=o.record)
                set_value(b, source, [0.0])
                connect_weights(b, source.neurons, relay.neurons, decoded_weights(cal.decoders["source"], transform, relay))
                for sign in (1, -1):
                    connect_weights(b, relay.neurons, self.velocity_cells[(i, sign)], velocity_input_weights(pose_cal, self.mapping, sign))
                self.trail_sources.append(source)
                self.trail_relays.append(relay)

        self.bias_cells = create_bias_cells(b, self.design)
        self.bias_current = float(b.GetStatus(self.bias_cells, "I_e")[0])
        for name, first, second in PLANES:
            sheet = build_sheet(b, name, self.mapping)
            connect_anchor(b, sheet, [self.gates[first].neurons, self.gates[second].neurons], pose_cal.decoders["gate"])
            connect_anchor_bias(b, sheet, self.bias_cells, float(pose_cal.fit["bias_rate_hz"]))
            for axis_in_plane, coordinate in enumerate((first, second)):
                for sign in (1, -1):
                    if (coordinate, sign) in self.velocity_cells:
                        connect_velocity_cells(b, sheet, self.velocity_cells[(coordinate, sign)], axis_in_plane, sign)
            self.sheets.append(sheet)

        if o.noise_pa > 0.0:
            noise = b.Create("noise_generator", params={"mean": 0.0, "std": float(o.noise_pa), "dt": 1.0})
            for population in list(self.muls.values()) + list(self.relays.values()) + self.gates:
                b.Connect(noise, population.neurons)
        self.anchor_on = True
        self.path_integration_on = bool(o.velocity)
        self.set_anchor(bool(o.anchor))
        self.set_path_integration(bool(o.path_integration))
        self.build_log.append("pose: %d products, %d sum populations, 3 sheets of %d x %d, %d neurons, %d synapses" % (
            len(self.muls), len(self.relays), self.mapping.n, self.mapping.n, self.neuron_count(),
            int(b.GetKernelStatus("num_connections"))))

    # -- switches ------------------------------------------------------------
    def set_anchor(self, on: bool) -> None:
        """Anchor off: the gate populations and the bias cells are silenced; the sheets then get nothing from the position.

        Switching on puts every one of those neurons at a fresh potential between rest and threshold: a neuron
        that was held far below rest would otherwise come back late, each at its own time, and the sheets would
        get a wrong anchor for a few ticks."""

        b, model = self.backend, self.design.model
        rng = np.random.default_rng([4242, int(self.t_ms)])
        for gate in self.gates:
            if on:
                b.SetStatus(gate.neurons, [{"I_e": float(i), "V_m": float(v)}
                                           for i, v in zip(gate.tuning.bias, rng.uniform(0.0, model.V_th, gate.n))])
            else:
                b.SetStatus(gate.neurons, {"I_e": SILENT_PA})
        if on:
            b.SetStatus(self.bias_cells, [{"I_e": self.bias_current, "V_m": float(v)} for v in rng.uniform(0.0, model.V_th, BIAS_CELLS)])
        else:
            b.SetStatus(self.bias_cells, {"I_e": SILENT_PA})
        self.anchor_on = bool(on)

    def set_path_integration(self, on: bool) -> None:
        self.path_integration_on = bool(on) and bool(self.velocity_cells)
        self._drive_velocity_cells(self._moving)

    def _drive_velocity_cells(self, moving: bool) -> None:
        """Path integration on: all velocity cells get their resting drive while a joint moves, nothing while the arm is still."""

        rest = velocity_rest_current(self.pose_options.sheet) if moving else 0.0
        for cells in self.velocity_cells.values():
            self.backend.SetStatus(cells, {"I_e": rest if self.path_integration_on else SILENT_PA})
        self._moving = bool(moving)

    # -- running -------------------------------------------------------------
    def set_trail_velocity(self, velocity_m_s: Optional[Sequence[float]]) -> None:
        """Hand a velocity of the end point (m/s) to the velocity cells through the trail populations; ``None``: zero."""

        if not self.trail_sources:
            raise ValueError("the network was built without the trail populations (PoseOptions(trail=True))")
        scale = self.space.radius * self.pose.qdot_ref * self.pose.velocity_scale
        values = np.zeros(3) if velocity_m_s is None else np.clip(np.asarray(velocity_m_s, dtype=float) / scale, -1.0, 1.0)
        for source, value in zip(self.trail_sources, values):
            set_value(self.backend, source, [float(value)])
        self._trail_moving = bool(np.any(np.abs(values) > 0.01))

    def step(self, q: Sequence[float], qdot: Optional[Sequence[float]] = None) -> PoseTick:      # type: ignore[override]
        b = self.backend
        if self.qdot:
            rates = np.zeros(self.chain.n) if qdot is None else np.asarray(qdot, dtype=float).reshape(-1)
            values = self.pose.qdot_values(rates)
            for k, (population, value) in enumerate(zip(self.qdot, values)):
                if value != self._qdot_written[k]:
                    set_value(b, population, [float(value)])
            self._qdot_written = values
            moving = bool(np.any(np.abs(rates) > MOVING_RAD_S)) or self._trail_moving
            if moving != self._moving:
                self._drive_velocity_cells(moving)
        tick = super().step(q)
        counts = np.array([sheet.read(b) for sheet in self.sheets])
        places = np.array([self.mapping.centre(c) for c in counts])
        window = np.array([np.sum(sheet.history, axis=0) for sheet in self.sheets])
        sheets_scaled, disagreement = sheet_position(self.mapping, window)
        held = self.pose_cells.values(b)
        cells_scaled = held[:3]
        velocity_scaled = held[3:6] if held.size >= 6 else np.full(3, np.nan)
        return PoseTick(tick, cells_scaled, velocity_scaled, places, counts, sheets_scaled, disagreement,
                        self.space.metres(sheets_scaled), self.space.metres(cells_scaled))

    def position_from_counts(self, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Position (m) and disagreement (m) from summed spike counts of the three sheets."""

        scaled, disagreement = sheet_position(self.mapping, counts)
        return self.space.metres(scaled), disagreement * self.space.radius

    def velocity_m_s(self, tick: PoseTick) -> np.ndarray:
        return tick.velocity_scaled * self.space.radius * self.pose.qdot_ref * self.pose.velocity_scale

    def so3_neuron_count(self) -> int:
        return super().neuron_count()

    def neuron_count(self) -> int:
        n = super().neuron_count()
        n += sum(p.n for p in self.muls.values()) + sum(p.n for p in self.relays.values())
        n += sum(p.n for p in self.qdot) + sum(p.n for p in self.gates) + 2 * sum(p.n for p in self.trail_sources)
        n += len(self.velocity_cells) * int(self.pose_options.sheet.velocity_cells) + len(self.sheets) * self.mapping.count + BIAS_CELLS
        return int(n)

    def silence_sheets(self, fraction: float, seed: int = 0) -> int:
        """Switch off a random ``fraction`` of every sheet's neurons for good (robustness test); returns how many."""

        rng = np.random.default_rng(int(seed))
        dead = 0
        for sheet in self.sheets:
            chosen = np.flatnonzero(rng.uniform(size=self.mapping.count) < float(fraction))
            if chosen.size:
                self.backend.SetStatus(sheet.neurons[[int(i) for i in chosen]], {"I_e": SILENT_PA})
            dead += int(chosen.size)
        return dead


__all__ = ["PoseNetwork", "PoseOptions", "PoseTick", "sheet_position"]
