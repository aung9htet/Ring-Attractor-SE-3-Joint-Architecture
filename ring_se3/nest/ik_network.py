"""The pose network with the pose inverse-kinematics layer on top (open loop: the arm is moved from outside).

:class:`PoseIkNetwork` is a :class:`~ring_se3.nest.network.PoseNetwork` (built without its
velocity path, and with level cells on its sheets) that also holds

* for the orientation goal, three goal rings (yaw, pitch, lift) with their encoders and a
  cos / sin population each, as in :mod:`ring_so3.nest.ik_network`;
* for the position goal, three *goal gates* (the goal's x, y, z as numbers, set from outside) and,
  with the position goal ``sheets``, three *goal sheets* that those gates anchor: the goal is a
  packet on each of them, as the hand's position is a packet on each forward sheet;
* the *ramp readouts*: per coordinate one population that reads the two goal sheets that hold it
  and one that reads the two forward sheets (:mod:`ring_se3.nest.ik_calibrate`);
* the populations of a :class:`~ring_se3.ik.wiring.PoseIkWiring`: the goal matrix, the two
  comparisons (turn and offset), per joint the direction in which it moves the end point, its
  share (two halves) and what is left after it;
* near a joint limit, inhibition from that joint's ring onto the half that points beyond it;
* per joint a gain pair and a goal ring (:mod:`ring_so3.nest.goal_ring`);
* readout cells for the comparisons, the shares and what is left.

:meth:`PoseIkNetwork.step_ik` writes joint angles and a pose goal, simulates one tick and
returns both the forward readout and the layer's values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ring_so3.design import Design
from ring_so3.ik import rules as so3_rules
from ring_so3.ik.wiring import Half, Mul, TURN_SCALE
from ring_so3.kinematics import Chain
from ring_so3.mapping import RingMap
from ring_so3.nest.calibrate import Calibration, feature_tuning
from ring_so3.nest.goal_ring import GoalRing, WrittenRingSettings, build_written_ring
from ring_so3.nest.ik_calibrate import IkCalibration
from ring_so3.nest.populations import (Population, ReadoutCells, connect_weights, create_population, create_readout_cells,
                                       decoded_weights, set_value)
from ring_so3.nest.ring import JointRing, build_joint_ring
from ring_so3.wiring import Node

from ..ik.design import PoseIkDesign
from ..ik.wiring import PoseIkWiring, pose_ik_wiring, ramp_node, ramp_nodes
from ..sheet import PLANES
from .calibrate import BIAS_CELLS, PoseCalibration, create_bias_cells
from .ik_calibrate import RampCalibration, create_ramp_readout
from .network import SILENT_PA, PoseNetwork, PoseOptions, PoseTick, sheet_position, switch_populations
from .sheet import Sheet, build_sheet, connect_anchor, connect_anchor_bias, connect_level

GOAL_NAMES = ("yaw", "pitch", "lift")          # the order of the goal chain
#: the inhibition near a limit keeps rising beyond the limit up to this many times the full hold
LIMIT_HOLD_MAX = 2.0


def ik_options(**changes: Any) -> PoseOptions:
    """The pose network as the inverse kinematics needs it: no velocity path (the sheets are anchored), level cells on."""

    settings: Dict[str, Any] = dict(velocity=False, level=True)
    settings.update(changes)
    return PoseOptions(**settings)


@dataclass
class PoseIkTick:
    tick: PoseTick
    #: the comparison of orientations (rad) and of positions (reaches); zeros for the half the task leaves out
    turn: np.ndarray
    offset: np.ndarray
    #: each joint's share (rad)
    shares: np.ndarray
    #: what is left after every joint: of the turn (n, 3) (rad) and of the offset (n, 3) (reaches)
    rests: np.ndarray
    rests_position: np.ndarray
    #: lift, pitch, yaw the goal rings hold (NaN while silent, or without an orientation goal)
    goal_held: np.ndarray
    #: scaled place of each goal sheet's packet from this tick's spikes, (3, 2), and the spikes, (3, neurons)
    goal_places: np.ndarray
    goal_sheet_counts: np.ndarray
    #: the goal position the goal sheets hold over their reading window, m (NaN without goal sheets)
    goal_position: np.ndarray
    #: what the ramp readouts hold, m: goal and hand (NaN without them)
    ramp_goal: np.ndarray
    ramp_hand: np.ndarray
    #: the angle each joint's goal ring holds (NaN while silent, or without goal rings) and its lead
    #: over the joint's state ring in ring indices
    joint_goal: np.ndarray
    lead: np.ndarray

    @property
    def left(self) -> np.ndarray:
        """What the whole arm could not take: turn (rad), then offset (reaches)."""

        return np.concatenate([self.rests[-1], self.rests_position[-1]])


class PoseIkNetwork(PoseNetwork):
    def __init__(self, backend: Any, chain: Chain, calibration: Calibration, pose_calibration: PoseCalibration,
                 ik_calibration: IkCalibration, ramp_calibration: Optional[RampCalibration] = None, design: Optional[Design] = None,
                 options: Optional[PoseOptions] = None, ik_design: Optional[PoseIkDesign] = None, goal_ring_settings: Any = None) -> None:
        """``ik_calibration``: the readouts of :mod:`ring_so3.nest.ik_calibrate` (a share's half, a sum that stays at its
        bounds).  ``ramp_calibration``: the readout of a packet's place; needed for the position goal ``sheets``."""

        self.ik_design = ik_design or PoseIkDesign()
        self.ik_calibration = ik_calibration
        self.ramp_calibration = ramp_calibration
        self.goal_ring_settings = goal_ring_settings
        options = options or ik_options()
        if options.mode != "chain" or not options.relay or not options.output_rings:
            raise ValueError("the IK layer needs the joint-by-joint forward network with its link populations and output rings")
        self.on_sheets = self.ik_design.position and self.ik_design.position_goal == "sheets"
        if self.on_sheets and (ramp_calibration is None or not options.level):
            raise ValueError("the position goal on sheets needs the ramp calibration and sheets with level cells (PoseOptions(level=True))")
        self.goal_rings: List[JointRing] = []
        self.goal_features: List[Population] = []
        self.goal_sources: List[Population] = []
        self.goal_gates: List[Population] = []
        self.goal_sheets: List[Sheet] = []
        self.goal_level_cells: List[Population] = []
        self.goal_bias_cells: Any = None
        self.ramp_readouts: Dict[Node, Population] = {}
        self.ik_populations: Dict[Node, Population] = {}
        self.ik_cells: Optional[ReadoutCells] = None
        self.joint_goal_rings: List[GoalRing] = []
        self._joint_goal_history: List[np.ndarray] = []
        self._goal_written = np.full(3, np.nan)
        self.goal_anchor_on = False
        super().__init__(backend, chain, calibration, pose_calibration, design, options, level_decoders=ik_calibration.decoders["bounded"])
        self._build_ik()

    # -- construction --------------------------------------------------------
    def _build_ik(self) -> None:
        b, o, cal, d = self.backend, self.pose_options, self.calibration, self.ik_design
        size = int(cal.settings["ring_size"])
        self.ik_wiring: PoseIkWiring = pose_ik_wiring(self.chain, self.pose, d, size)
        entry = self.design.entry
        if d.orientation:
            tuning = feature_tuning(self.design, cal.settings["feature_tau"])
            mapping = RingMap(size, -np.pi, np.pi, True)
            angles = mapping.neuron_angles()
            basis = cal.feature_scale(mapping) * np.stack([np.cos(angles), np.sin(angles)], axis=1)
            for c, name in enumerate(GOAL_NAMES):
                ring = build_joint_ring(b, "goal_" + name, mapping, mode=o.encoder)
                feature = create_population(b, tuning, "goal_feature_" + name, seed=60 + c, record=o.record)
                connect_weights(b, ring.neurons, feature.neurons, tuning.input_weights(np.eye(2)) @ basis.T / feature.charge)
                self.goal_rings.append(ring)
                self.goal_features.append(feature)
                for part in (0, 1):
                    self.sources[("gfeat", c, part)] = (feature, cal.decoders["feature"][:, part:part + 1])
        if d.position:
            self._build_position_goal()

        half_of = {node: (k, sign) for k, pair in enumerate(self.ik_wiring.shares) for node, sign in zip(pair, (1, -1))}
        for index, step in enumerate(self.ik_wiring.steps):
            seed = 50000 + index
            if isinstance(step, Mul):
                population = create_population(b, self.design.product, "ik_product%d" % index, seed=seed, record=o.record)
                self._feed(population, dict(step.a), 0)
                self._feed(population, dict(step.b), 1)
                decoders = cal.decoders["product"]
            else:
                population = create_population(b, entry, "ik_%s" % "_".join(str(p) for p in step.node), seed=seed, record=o.record)
                self._feed(population, dict(step.terms), 0)
                if isinstance(step, Half):
                    decoders = self.ik_calibration.decoders["half"]
                    if step.node in half_of:
                        self._hold_at_limit(population, *half_of[step.node])
                else:           # goal matrix, levers and directions stay within what a population holds; comparisons and rests can ask for more
                    decoders = cal.decoders["scalar"] if step.node[0] in ("g", "lever", "jac") else self.ik_calibration.decoders["bounded"]
            self.ik_populations[step.node] = population
            self.sources[step.node] = (population, decoders)

        w = self.ik_wiring
        nodes = list(w.turn) + list(w.offset) + [node for pair in w.shares for node in pair]
        nodes += [node for rest in w.rests + w.rests_position for node in rest]
        nodes += list(w.ramps) + ([("gpos", i) for i in range(3)] if d.position else [])
        self._ik_nodes: List[Node] = list(dict.fromkeys(nodes))                 # a rest that a joint does not change is listed once
        self.ik_cells = create_readout_cells(b, len(self._ik_nodes), {"tau_m": float(o.readout_tau_ms)})
        for cell, node in enumerate(self._ik_nodes):
            population, decoders = self.sources[node]
            self.ik_cells.connect(b, population, decoders, [cell])
        if d.goal_ring != "none":
            settings = self.goal_ring_settings or WrittenRingSettings()
            for k, (plus, minus) in enumerate(w.shares):
                self.joint_goal_rings.append(build_written_ring(b, "joint_goal%d" % (k + 1), self.rings[k].mapping, self.rings[k],
                                                                self.sources[plus], self.sources[minus], settings))
        if o.noise_pa > 0.0:
            noise = b.Create("noise_generator", params={"mean": 0.0, "std": float(o.noise_pa), "dt": 1.0})
            for population in self.goal_features + self.goal_gates + list(self.ramp_readouts.values()) + list(self.ik_populations.values()):
                b.Connect(noise, population.neurons)
        summary = w.summary()
        self.build_log.append("IK layer (%s, position goal on %s, %s): %d products, %d sum populations, %d share halves, %d neurons in all" % (
            d.task, d.position_goal if d.position else "nothing", "cascade" if d.cascade else "all at once",
            summary["products"], summary["sum_populations"], summary["share_populations"], self.neuron_count()))

    def _build_position_goal(self) -> None:
        b, o, cal, pose_cal, entry = self.backend, self.pose_options, self.calibration, self.pose_calibration, self.design.entry
        transform = np.ones((1, 1))
        for i in range(3):
            source = create_population(b, entry, "goal_source%d" % i, seed=48000 + i, record=o.record)
            gate = create_population(b, entry, "goal_gate%d" % i, seed=48100 + i, record=o.record)
            set_value(b, source, [0.0])
            connect_weights(b, source.neurons, gate.neurons, decoded_weights(cal.decoders["source"], transform, gate))
            self.goal_sources.append(source)
            self.goal_gates.append(gate)
            self.sources[("gpos", i)] = (gate, cal.decoders["scalar"])
        self._goal_written = np.zeros(3)
        if not self.on_sheets:
            return
        self.goal_bias_cells = create_bias_cells(b, self.design)
        self.goal_bias_current = float(b.GetStatus(self.goal_bias_cells, "I_e")[0])
        for index, (name, first, second) in enumerate(PLANES):
            sheet = build_sheet(b, "goal_" + name, self.mapping)
            connect_anchor(b, sheet, [self.goal_gates[first].neurons, self.goal_gates[second].neurons], pose_cal.decoders["gate"])
            connect_anchor_bias(b, sheet, self.goal_bias_cells, float(pose_cal.fit["bias_rate_hz"]))
            self.goal_level_cells.append(connect_level(b, sheet, self.design, self.level_decoders, o.level_design,
                                                       seed=48200 + index, record=o.record))
            self.goal_sheets.append(sheet)
        ramp = self.ramp_calibration
        tuning = ramp.tuning(self.design)
        for s, (side, sheets) in enumerate((("goal", self.goal_sheets), ("hand", self.sheets))):
            for i in range(3):
                population = create_ramp_readout(b, tuning, sheets, i, ramp.total_rate_hz, "ramp_%s%d" % (side, i),
                                                 seed=48300 + 10 * s + i, record=o.record)
                self.ramp_readouts[ramp_node(side, i)] = population
                self.sources[ramp_node(side, i)] = (population, ramp.decoders["ramp"])
        self.goal_anchor_on = True

    def _hold_at_limit(self, population: Population, joint: int, sign: int) -> None:
        """The joint's own ring holds down the half of its share that points beyond a limit."""

        ring = self.rings[joint]
        mapping = ring.mapping
        if mapping.circular:
            return
        taper = float(self.ik_design.limit_taper_indices)
        first, last = mapping.arc
        indices = np.arange(mapping.size, dtype=float)
        ramp = (indices - (last - taper)) / taper if sign > 0 else ((first + taper) - indices) / taper
        ramp = np.clip(ramp, 0.0, LIMIT_HOLD_MAX)
        per_hz = ramp / float(self.calibration.ring_profile.sum())            # a bump at index i gives about ramp(i)
        transform = np.array([[-1.0]])
        connect_weights(self.backend, ring.neurons, population.neurons,
                        population.tuning.input_weights(transform) @ per_hz[None, :] / population.charge)

    # -- switches ------------------------------------------------------------
    def set_goal_anchor(self, on: bool) -> None:
        """Goal anchor off: the goal gates, their bias cells and the goal sheets' level cells are silenced.  The goal
        packets then stay where they are by themselves: the goal sheets remember the goal."""

        if not self.goal_sheets:
            return
        b, model = self.backend, self.design.model
        rng = np.random.default_rng([2424, int(self.t_ms)])
        switch_populations(b, self.goal_gates, on, model.V_th, rng)
        if on:
            b.SetStatus(self.goal_bias_cells, [{"I_e": self.goal_bias_current, "V_m": float(v)} for v in rng.uniform(0.0, model.V_th, BIAS_CELLS)])
        else:
            b.SetStatus(self.goal_bias_cells, {"I_e": SILENT_PA})
        switch_populations(b, self.goal_level_cells, on, model.V_th, rng)
        self.goal_anchor_on = bool(on)

    # -- running -------------------------------------------------------------
    def step_ik(self, q: Sequence[float], goal_lpy: Sequence[float], goal_position: Sequence[float],
                qdot: Optional[Sequence[float]] = None) -> PoseIkTick:
        """Write joint angles and a pose goal (lift, pitch, yaw; position in metres, base frame), simulate one tick,
        return what the network holds."""

        b, d = self.backend, self.ik_design
        if d.orientation:
            lift, pitch, yaw = [float(v) for v in goal_lpy]
            for ring, angle in zip(self.goal_rings, (yaw, pitch, lift)):
                ring.drive(b, angle)
        if d.position:
            scaled = np.clip(self.space.scaled(goal_position), -1.0, 1.0)
            for i, (source, value) in enumerate(zip(self.goal_sources, scaled)):
                if value != self._goal_written[i]:
                    set_value(b, source, [float(value)])
            self._goal_written = scaled
        tick = self.step(q, qdot)
        held = np.full(3, np.nan)
        if self.goal_rings:
            for c, ring in enumerate(self.goal_rings):
                ring.read(b)
                held[2 - c] = ring.held_angle()                         # the rings are yaw, pitch, lift
        mapping = self.mapping
        goal_counts = np.zeros((3, mapping.count))
        goal_places, goal_position_held = np.full((3, 2), np.nan), np.full(3, np.nan)
        if self.goal_sheets:
            goal_counts = np.array([sheet.read(b) for sheet in self.goal_sheets])
            goal_places = np.array([mapping.centre(c) for c in goal_counts])
            window = np.array([np.sum(sheet.history, axis=0) for sheet in self.goal_sheets])
            goal_position_held = self.space.metres(sheet_position(mapping, window)[0])
        n = self.chain.n
        joint_goal, lead = np.full(n, np.nan), np.full(n, np.nan)
        if self.joint_goal_rings:
            counts = np.array([goal.ring.read(b) for goal in self.joint_goal_rings])
            self._joint_goal_history = (self._joint_goal_history + [counts])[-int(d.goal_window):]
            window = np.sum(self._joint_goal_history, axis=0)
            for k, (goal, state) in enumerate(zip(self.joint_goal_rings, self.rings)):
                index = goal.ring.mapping.population_vector_index(window[k])
                if np.isfinite(index):
                    joint_goal[k] = goal.ring.mapping.angle(index)
                    if np.isfinite(state.index):
                        lead[k] = goal.ring.mapping.index_error(index, state.index)
        values = dict(zip(self._ik_nodes, self.ik_cells.values(b)))
        w = self.ik_wiring
        rests = np.array([[values[node] for node in rest] for rest in w.rests]) / TURN_SCALE if w.turn else np.zeros((len(w.rests_position), 3))
        rests_position = (np.array([[values[node] for node in rest] for rest in w.rests_position]) / d.position_scale
                          if w.offset else np.zeros((len(w.rests), 3)))
        ramp_goal, ramp_hand = np.full(3, np.nan), np.full(3, np.nan)
        if w.ramps:
            ramp_goal = self.space.metres([values[node] for node in ramp_nodes("goal")])
            ramp_hand = self.space.metres([values[node] for node in ramp_nodes("hand")])
        return PoseIkTick(tick, w.turn_vector(values), w.offset_vector(values), w.share_values(values), rests, rests_position, held,
                          goal_places, goal_counts, goal_position_held, ramp_goal, ramp_hand, joint_goal, lead)

    def joint_goal_counts(self) -> np.ndarray:
        """Spike counts of every joint's goal ring in the last tick, (joints, ring size)."""

        if not self._joint_goal_history:
            return np.zeros((self.chain.n, int(self.calibration.settings["ring_size"])))
        return self._joint_goal_history[-1]

    def goal_counts(self) -> np.ndarray:
        """Spike counts of the goal rings in the last tick, rows lift, pitch, yaw (zeros without an orientation goal)."""

        if not self.goal_rings:
            return np.zeros((3, int(self.calibration.settings["ring_size"])))
        return np.array([self.goal_rings[c].counts for c in (2, 1, 0)])

    def ik_neuron_count(self) -> int:
        """Neurons of everything this class adds to the pose network."""

        n = sum(ring.size for ring in self.goal_rings) + sum(p.n for p in self.goal_features)
        n += sum(p.n for p in self.goal_sources) + sum(p.n for p in self.goal_gates) + len(self.goal_sheets) * self.mapping.count
        n += sum(p.n for p in self.goal_level_cells) + (BIAS_CELLS if self.goal_sheets else 0)
        n += sum(p.n for p in self.ramp_readouts.values()) + sum(p.n for p in self.ik_populations.values())
        n += sum(goal.neurons for goal in self.joint_goal_rings)
        return int(n)

    def neuron_count(self) -> int:
        return int(super().neuron_count() + self.ik_neuron_count())

    def silence_ik(self, fraction: float, seed: int = 0) -> int:
        """Switch off a random ``fraction`` of the neurons of comparison and cascade for good; returns how many."""

        rng = np.random.default_rng(int(seed))
        dead = 0
        for node in sorted(self.ik_populations, key=repr):
            population = self.ik_populations[node]
            for index in np.flatnonzero(rng.uniform(size=population.n) < float(fraction)):
                self.backend.SetStatus(population.neurons[int(index)], {"I_e": SILENT_PA})
                dead += 1
        return dead

    def exact(self, q: Sequence[float], goal_lpy: Sequence[float], goal_position: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """What the layer's own wiring gives with exact products and sums: turn, offset, shares, what is left."""

        w = self.ik_wiring
        room = so3_rules.limit_room(self.chain, q, self.ik_design, int(self.calibration.settings["ring_size"]))
        values = w.evaluate(q, goal_lpy, goal_position, room=room)
        return w.turn_vector(values), w.offset_vector(values), w.share_values(values), w.left_vector(values)


__all__ = ["GOAL_NAMES", "PoseIkNetwork", "PoseIkTick", "ik_options"]
