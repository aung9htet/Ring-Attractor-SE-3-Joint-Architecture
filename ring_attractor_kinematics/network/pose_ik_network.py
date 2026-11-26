"""The pose IK network (Figure 1, stages ③–⑤): the forward network plus goal, comparison, cascade and goal writing.

:class:`PoseIkNetwork` is a :class:`~ring_attractor_kinematics.network.forward.ForwardNetwork` (with level cells on its sheets)
that also builds, in this order:

3. **goal** — three goal rings (yaw, pitch, lift) with their cos/sin populations; three **goal gates** set from
   outside and, for positions compared on sheets, three **goal sheets** they anchor (with their own bias and
   level cells) and the six **ramp read-outs** (goal: cold, hand: warm);
4. **comparison and cascade** — the populations of the IK wiring, in step order: the goal matrix, the **turn**,
   the **offset**, the levers, and per joint its direction, its two **share halves** (held down near a joint
   limit by the joint's own ring) and what it leaves;
5. **goal writing** — per joint a **gain pair** and a **written goal ring**: the joint's next step.

**Methods.**  ``matrix`` (the old method): the joint rings hold the joint angles, and the forward network shows the
arm about 0.2 s late; the turn reads the hand's late matrix.  ``lead`` (the new method): the joint rings are written
``lead_k · q̇_k`` ahead, so the forward network — orientation, axes, levers and position — shows the arm about on
time; separate **motor rings** hold the true joint angles for what needs them (the gain pairs' gate, the limit hold,
the lead read-out).  ``observer``: the joint rings hold the joint angles as in the old method, and an **orientation
observer** (network/observer_wiring.py, Figure 2c) gives the turn its matrix: observer rings in two charts, moved
by shift cells at the chart rates of the commanded angular velocity and corrected toward the late matrix; the
selector picks a chart and the matrix gates pass its matrix, or the late one when confidence is low.  The controller
knows ``q̇``: it is what it commanded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reaching_benchmark.geometry.chains import Chain
from reaching_benchmark.geometry.rotations import lift_pitch_yaw, nearest_rotation

from ..calibration.artifacts import Calibrations
from ..config import METHODS, GainPairSettings, IkDesign, NetworkOptions, ObserverDesign
from ..neurons.lif import RING_NEURON
from ..neurons.ring_layout import RingMap
from ..neurons.sheet_layout import PLANES, sheet_position
from ..neurons.tuning import Tunings
from ..rules.cascade_rule import limit_room
from ..rules.observer_rule import CHARTS
from ..spiking.feedforward_ring import GoalWriting, build_goal_writing
from ..spiking.populations import (SILENT_PA, Population, ReadoutCells, Silenced, connect_weights, create_population,
                                   create_readout_cells, decoded_weights, restore_nodes, set_value, silence_nodes,
                                   switch_populations)
from ..spiking.ramp_readout import create_ramp_readout
from ..spiking.ring_attractor import Ring, build_ring_attractor
from ..spiking.shift_circuit import ShiftCells, ShiftSettings, build_shift_cells
from ..spiking.sheet import BIAS_CELLS, Sheet, build_sheet, connect_anchor, connect_anchor_bias, connect_level_cells, create_bias_cells
from .forward import ForwardNetwork, ForwardTick, Group
from .ik_wiring import TURN_SCALE, Half, IkWiring, Mul, ik_wiring, ramp_node, ramp_nodes
from .observer_wiring import GATE_ON, QDOT_FULL, Norm, ObserverWiring, observer_wiring

GOAL_RING_NAMES = ("yaw", "pitch", "lift")          # the order of the goal chain
#: ticks the observer rings stay silent when seeding starts: a ring neuron sits at its threshold, so even the 1% of an
#: old bump's synaptic current left after 0.2 s lights it again; after 0.4 s it is gone
OBSERVER_CLEAR_TICKS = 8
#: the current that keeps the observer's shift cells silent while its rings are seeded (their gate brings µA)
SHIFT_SILENT_PA = 20.0 * SILENT_PA
#: the inhibition near a limit keeps rising beyond the limit up to this many times the full hold
LIMIT_HOLD_MAX = 2.0
#: what a joint lesion silences (:meth:`PoseIkNetwork.lesion_joint`): ``motor``, the joint's gain pair and written goal
#: ring; ``allocation``, its two share halves
JOINT_LESION_SITES = ("motor", "allocation")


def ik_options(**changes: Any) -> NetworkOptions:
    """Network options as the IK needs them: level cells on the forward sheets."""

    settings: Dict[str, Any] = dict(level=True)
    settings.update(changes)
    return NetworkOptions(**settings)


@dataclass
class IkTick:
    forward: ForwardTick
    #: the turn (rad) and the offset (reaches); zeros for the half the task leaves out
    turn: np.ndarray
    offset: np.ndarray
    #: each joint's share (rad), and what is left after every joint of turn (n, 3) and offset (n, 3)
    shares: np.ndarray
    rests: np.ndarray
    rests_position: np.ndarray
    #: lift, pitch, yaw the goal rings hold
    goal_held: np.ndarray
    #: goal sheets: places (3, 2), spikes (3, neurons), and the position they hold (m)
    goal_places: np.ndarray
    goal_sheet_counts: np.ndarray
    goal_position: np.ndarray
    #: what the ramp read-outs hold, m: goal (cold) and hand (warm)
    ramp_goal: np.ndarray
    ramp_hand: np.ndarray
    #: the angle each joint's written goal ring holds, and its lead over the joint ring (ring indices)
    joint_goal: np.ndarray
    lead: np.ndarray
    #: the observer method: what its observer holds and chose
    observer: Optional["ObserverTick"] = None

    @property
    def left(self) -> np.ndarray:
        """What the whole arm could not take: turn (rad), then offset (reaches)."""

        return np.concatenate([self.rests[-1], self.rests_position[-1]])


@dataclass
class JointLesion:
    """One joint's motor or allocation neurons, silenced until :meth:`PoseIkNetwork.restore`."""

    site: str
    joint: int
    #: (name, neurons) of every population silenced
    populations: List[Tuple[str, int]]
    silenced: Silenced

    @property
    def neurons(self) -> int:
        return self.silenced.count

    def describe(self) -> Dict[str, Any]:
        return {"site": self.site, "joint": self.joint, "populations": [name for name, _ in self.populations], "neurons": self.neurons}


@dataclass
class ObserverTick:
    #: lift, pitch, yaw each chart's observer rings hold (2, 3), rad (NaN while silent)
    angles: np.ndarray
    #: the chart the selector holds, its confidence (fitness), and whether the late matrix goes to the turn
    selected: int
    confidence: float
    fallback: bool
    #: the matrix the turn reads (the sum of the matrix gates), the rings' speeds (2, 3) rad/s, the innovations (2, 3)
    #: and each chart's fitness
    hand: np.ndarray
    rates: np.ndarray
    innovations: np.ndarray
    fitness: np.ndarray
    selector: np.ndarray
    #: the angles the observer's cos/sin populations hold (2, 3): what the delayed comparison is calibrated on
    pair_angles: np.ndarray


class PoseIkNetwork(ForwardNetwork):
    def __init__(self, backend: Any, chain: Chain, calibrations: Calibrations, tunings: Optional[Tunings] = None,
                 options: Optional[NetworkOptions] = None, ik_design: Optional[IkDesign] = None,
                 gain_pairs: Optional[GainPairSettings] = None, method: str = "matrix",
                 observer_design: Optional[ObserverDesign] = None) -> None:
        if method not in METHODS:
            raise ValueError("method must be one of %r" % (METHODS,))
        self.method = method
        self.motor_rings: List[Ring] = []
        self.ik_design = ik_design or IkDesign()
        self.observer_design = observer_design or ObserverDesign()
        if method == "observer" and not (calibrations.has("observer") and calibrations.has("shift_cells") and self.ik_design.orientation):
            raise ValueError("the observer method needs an orientation task and the shift_cells and observer calibrations")
        self.observer_wiring: Optional[ObserverWiring] = None
        self.observer_rings: List[List[Ring]] = []
        self.observer_pairs: List[Population] = []
        self.observer_sources: List[Population] = []
        self.observer_populations: Dict[Any, Population] = {}
        self.shift_cells: List[ShiftCells] = []
        self.observer_seeding = False
        self._observer_cleared = False
        self._observer_clear_left = 0
        self.gain_pairs = gain_pairs or GainPairSettings()
        options = options or ik_options()
        self.on_sheets = self.ik_design.position and self.ik_design.position_goal == "sheets"
        if self.on_sheets and (not calibrations.has("ramp_readout") or not options.level):
            raise ValueError("positions compared on sheets need the ramp read-out calibration and level cells")
        self.goal_rings: List[Ring] = []
        self.goal_cos_sin: List[Population] = []
        self.goal_sources: List[Population] = []
        self.goal_gates: List[Population] = []
        self.goal_sheets: List[Sheet] = []
        self.goal_level_cells: List[Population] = []
        self.goal_bias_cells: Any = None
        self.ramp_readouts: Dict[Any, Population] = {}
        self.ik_populations: Dict[Any, Population] = {}
        self.goal_writing: List[GoalWriting] = []
        self._joint_goal_history: List[np.ndarray] = []
        self._goal_written = np.full(3, np.nan)
        self.goal_anchor_on = False
        super().__init__(backend, chain, calibrations, tunings, options)
        hand, defaults = None, None
        if method == "observer":
            self.observer_wiring = observer_wiring(chain, self.observer_design, self.wiring)
            hand, defaults = self.observer_wiring.hand, self.observer_wiring.hand_defaults()
        self.ik_wiring: IkWiring = ik_wiring(chain, self.ik_design, calibrations.ring_size, self.wiring, hand=hand, hand_defaults=defaults)
        self._build_goal()
        self._build_observer()
        self._build_comparison_and_cascade()
        self._build_goal_writing()
        self._noise(self.goal_cos_sin + self.goal_gates + list(self.ramp_readouts.values()) + list(self.ik_populations.values())
                    + self.observer_pairs + list(self.observer_populations.values()))
        summary = self.ik_wiring.summary()
        self.build_log.append("IK (%s, positions on %s): %d products, %d sums, %d share halves, %d neurons in all" % (
            self.ik_design.task, self.ik_design.position_goal if self.ik_design.position else "-", summary["products"],
            summary["sum_populations"], summary["share_populations"], self.neuron_count()))

    # -- construction --------------------------------------------------------
    @property
    def state_rings(self) -> List[Ring]:
        """The rings that hold the true joint angles: the joint rings, or with ``lead`` the motor rings."""

        return self.motor_rings if self.method == "lead" else self.rings

    def _build_goal(self) -> None:
        b, o, d = self.backend, self.options, self.ik_design
        if self.method == "lead":
            for k, joint in enumerate(self.chain.joints):
                self.motor_rings.append(build_ring_attractor(b, "motor_ring%d" % (k + 1), RingMap.for_joint(joint, self.calibrations.ring_size)))
        if d.orientation:
            tuning = self.tunings.cos_sin(self.calibrations.cos_sin_tau_ms)
            for c, name in enumerate(GOAL_RING_NAMES):
                ring = build_ring_attractor(b, "goal_" + name, RingMap.circle(self.calibrations.ring_size))
                cos_sin = create_population(b, tuning, "goal_cos_sin_" + name, seed=60 + c, record=o.record)
                self.connect_ring_to_cos_sin(ring, cos_sin)
                self.goal_rings.append(ring)
                self.goal_cos_sin.append(cos_sin)
                for part in (0, 1):
                    self.sources[("gfeat", c, part)] = (cos_sin, self.calibrations.decoders["feature"][:, part:part + 1])
        if not d.position:
            return
        decoders = self.calibrations.decoders
        for i in range(3):
            source = create_population(b, self.tunings.entry, "goal_source%d" % i, seed=48000 + i, record=o.record)
            gate = create_population(b, self.tunings.entry, "goal_gate%d" % i, seed=48100 + i, record=o.record)
            set_value(b, source, [0.0])
            connect_weights(b, source.neurons, gate.neurons, decoded_weights(decoders["source"], np.ones((1, 1)), gate))
            self.goal_sources.append(source)
            self.goal_gates.append(gate)
            self.sources[("gpos", i)] = (gate, decoders["scalar"])
        self._goal_written = np.zeros(3)
        if not self.on_sheets:
            return
        self.goal_bias_cells = create_bias_cells(b, self.tunings.model)
        self.goal_bias_current = float(b.GetStatus(self.goal_bias_cells, "I_e")[0])
        for index, (name, first, second) in enumerate(PLANES):
            sheet = build_sheet(b, "goal_" + name, self.mapping)
            connect_anchor(b, sheet, [self.goal_gates[first].neurons, self.goal_gates[second].neurons], decoders["gate"])
            connect_anchor_bias(b, sheet, self.goal_bias_cells, self.calibrations.bias_rate_hz)
            self.goal_level_cells.append(connect_level_cells(b, sheet, self.tunings, decoders["bounded"], o.level_design,
                                                             seed=48200 + index, record=o.record))
            self.goal_sheets.append(sheet)
        tuning = self.tunings.ramp(self.calibrations.ramp_neurons, self.calibrations.ramp_tau_ms)
        for s, (side, sheets) in enumerate((("goal", self.goal_sheets), ("hand", self.sheets))):
            for i in range(3):
                population = create_ramp_readout(b, tuning, sheets, i, self.calibrations.ramp_total_rate_hz, "ramp_%s%d" % (side, i),
                                                 seed=48300 + 10 * s + i, record=o.record)
                self.ramp_readouts[ramp_node(side, i)] = population
                self.sources[ramp_node(side, i)] = (population, decoders["ramp"])
        self.goal_anchor_on = True

    def _build_observer(self) -> None:
        """The observer band (Figure 2c): joint-velocity sources, observer rings and their cos/sin (now and delayed),
        the step list of network/observer_wiring.py, the selector's own loops, the matrix gates and the shift cells."""

        w = self.observer_wiring
        if w is None:
            return
        b, record, decoders = self.backend, self.options.record, self.calibrations.decoders
        tuning = self.tunings.cos_sin(self.calibrations.cos_sin_tau_ms)
        delay, scale = self.calibrations.observer_delay_ms, self.calibrations.observer_feature_scale
        for k in range(self.chain.n):
            source = create_population(b, self.tunings.entry, "observer_qdot%d" % (k + 1), seed=56000 + k, record=record)
            set_value(b, source, [0.0])
            self.observer_sources.append(source)
            self.sources[("qdot", k)] = (source, decoders["source"])
        for c in range(2):
            rings = []
            for j, name in enumerate(("lift", "pitch", "yaw")):
                ring = build_ring_attractor(b, "observer%d_%s" % (c, name), RingMap.circle(self.calibrations.ring_size))
                now = create_population(b, tuning, "observer%d_cos_sin_%s" % (c, name), seed=56100 + 10 * c + j, record=record)
                past = create_population(b, tuning, "observer%d_past_%s" % (c, name), seed=56200 + 10 * c + j, record=record)
                self.connect_ring_to_cos_sin(ring, now, scale=scale)
                self.connect_ring_to_cos_sin(ring, past, delay_ms=delay, scale=scale)
                for part in (0, 1):
                    self.sources[("ofeat", c, j, part)] = (now, decoders["feature"][:, part:part + 1])
                    self.sources[("opast", c, j, part)] = (past, decoders["feature"][:, part:part + 1])
                if j == 1:
                    self.sources[("osec", c)] = (now, decoders["secant"])
                rings.append(ring)
                self.observer_pairs += [now, past]
            self.observer_rings.append(rings)
        for index, step in enumerate(w.steps):
            seed = 57000 + index
            name = "observer_" + "_".join(str(p) for p in step.node)
            if isinstance(step, Mul):
                population = create_population(b, self.tunings.product, name, seed=seed, record=record)
                self.feed(population, dict(step.a), 0)
                self.feed(population, dict(step.b), 1)
                read = decoders["product"]
            elif isinstance(step, Norm):
                population = create_population(b, self.tunings.norm, name, seed=seed, record=record)
                self.feed(population, dict(step.a), 0)
                self.feed(population, dict(step.b), 1)
                read = decoders["norm"]
            else:
                population = create_population(b, self.tunings.entry, name, seed=seed, record=record)
                self.feed(population, dict(step.terms), 0)
                if isinstance(step, Half):
                    read = decoders["half"]
                else:
                    read = decoders["scalar"] if step.node[0] in ("o", "route") else decoders["bounded"]
            self.observer_populations[step.node] = population
            self.sources[step.node] = (population, read)
        for node, combo in w.recurrent.items():
            self.feed(self.observer_populations[node], combo, 0)
        for gate in w.gates:
            for by in gate.by:
                self._gate(self.observer_populations[gate.node], self.observer_populations[by])
        settings = ShiftSettings(**self.calibrations.shift_settings)
        for c in range(2):
            for j in range(3):
                rate = self.observer_populations[w.rates[c][j]]
                self.shift_cells.append(build_shift_cells(b, self.observer_rings[c][j], (rate, decoders["drive_plus"]),
                                                          (rate, decoders["drive_minus"]), settings))
        self._observer_nodes = ([node for row in w.rates for node in row] + [("oinn", c, j) for c in range(2) for j in range(3)]
                                + [("ofit", c) for c in range(2)] + [("osel", c) for c in range(2)]
                                + [("ofall", c) for c in range(2)] + [("okeep",)]
                                + [("route", r, i, j) for r in range(3) for i in range(3) for j in range(3)]
                                + [("ofeat", c, j, part) for c in range(2) for j in range(3) for part in (0, 1)])
        self.observer_readout: ReadoutCells = create_readout_cells(b, len(self._observer_nodes), self.options.readout_tau_ms)
        for cell, node in enumerate(self._observer_nodes):
            population, read = self.sources[node]
            self.observer_readout.connect(b, population, read, cell)

    def _gate(self, target: Population, gate: Population) -> None:
        """Silence every neuron of ``target`` while ``gate`` (a half population) holds more than GATE_ON.

        Only the gate's neurons that are silent at 0 and fire at 1 take part, each with the same inhibitory weight
        onto every neuron of ``target``: an open gate sends nothing at all, and at GATE_ON the inhibition is twice the
        strongest current a neuron of ``target`` takes at 1.5 times its range."""

        tuning, model = gate.tuning, gate.tuning.model
        quiet = model.rate(tuning.currents(np.zeros((1, 1)))[0]) < 1.0
        loud = model.rate(tuning.currents(np.ones((1, 1)))[0]) > 20.0
        chosen = np.flatnonzero(quiet & loud)
        at_threshold = float(np.sum(model.rate(tuning.currents(np.full((1, 1), GATE_ON))[0])[chosen]))
        target_tuning = target.tuning
        strongest = float(max(np.max(np.abs(target_tuning.currents(np.full((1, target_tuning.dims), x)))) for x in (-1.5, 1.5)))
        weight = -2.0 * strongest / (max(at_threshold, 1.0) * target.charge)
        self.backend.Connect(gate.neurons[[int(i) for i in chosen]], target.neurons, {"rule": "all_to_all"}, {"weight": weight, "delay": 1.0})

    def _build_comparison_and_cascade(self) -> None:
        b, record, decoders, w = self.backend, self.options.record, self.calibrations.decoders, self.ik_wiring
        half_of = {node: (k, sign) for k, pair in enumerate(w.shares) for node, sign in zip(pair, (1, -1))}
        for index, step in enumerate(w.steps):
            seed = 50000 + index
            if isinstance(step, Mul):
                population = create_population(b, self.tunings.product, "ik_product%d" % index, seed=seed, record=record)
                self.feed(population, dict(step.a), 0)
                self.feed(population, dict(step.b), 1)
                read = decoders["product"]
            else:
                population = create_population(b, self.tunings.entry, "ik_%s" % "_".join(str(p) for p in step.node), seed=seed, record=record)
                self.feed(population, dict(step.terms), 0)
                if isinstance(step, Half):
                    read = decoders["half"]
                    if step.node in half_of:
                        self._hold_at_limit(population, *half_of[step.node])
                else:       # goal matrix, levers and directions stay in range; comparisons and rests can ask for more
                    read = decoders["scalar"] if step.node[0] in ("g", "lever", "jac") else decoders["bounded"]
            self.ik_populations[step.node] = population
            self.sources[step.node] = (population, read)
        nodes = list(w.turn) + list(w.offset) + [node for pair in w.shares for node in pair]
        nodes += [node for rest in w.rests + w.rests_position for node in rest]
        nodes += list(w.ramps) + ([("gpos", i) for i in range(3)] if self.ik_design.position else [])
        self._ik_nodes = list(dict.fromkeys(nodes))
        self.ik_readout: ReadoutCells = create_readout_cells(b, len(self._ik_nodes), self.options.readout_tau_ms)
        for cell, node in enumerate(self._ik_nodes):
            population, read = self.sources[node]
            self.ik_readout.connect(b, population, read, cell)

    def _hold_at_limit(self, population: Population, joint: int, sign: int) -> None:
        """The joint's own ring holds down the share half that points beyond a limit."""

        mapping = self.state_rings[joint].mapping
        if mapping.circular:
            return
        taper = float(self.ik_design.limit_taper_indices)
        first, last = mapping.arc
        indices = np.arange(mapping.size, dtype=float)
        ramp = (indices - (last - taper)) / taper if sign > 0 else ((first + taper) - indices) / taper
        per_hz = np.clip(ramp, 0.0, LIMIT_HOLD_MAX) / float(self.calibrations.ring_profile.sum())
        connect_weights(self.backend, self.state_rings[joint].neurons, population.neurons,
                        population.tuning.input_weights(np.array([[-1.0]])) @ per_hz[None, :] / population.charge)

    def _build_goal_writing(self) -> None:
        for k, (plus, minus) in enumerate(self.ik_wiring.shares):
            self.goal_writing.append(build_goal_writing(self.backend, "goal_ring%d" % (k + 1), self.state_rings[k].mapping, self.state_rings[k],
                                                        self.sources[plus], self.sources[minus], self.gain_pairs))

    # -- switches ------------------------------------------------------------
    def set_goal_anchor(self, on: bool) -> None:
        """Goal anchor off: goal gates, their bias cells and the goal sheets' level cells are silenced; the goal
        packets then stay where they are (the goal sheets remember the goal)."""

        if not self.goal_sheets:
            return
        b, model = self.backend, self.tunings.model
        rng = np.random.default_rng([2424, int(self.t_ms)])
        switch_populations(b, self.goal_gates, on, model.V_th, rng)
        if on:
            b.SetStatus(self.goal_bias_cells, [{"I_e": self.goal_bias_current, "V_m": float(v)}
                                               for v in rng.uniform(0.0, model.V_th, BIAS_CELLS)])
        else:
            b.SetStatus(self.goal_bias_cells, {"I_e": SILENT_PA})
        switch_populations(b, self.goal_level_cells, on, model.V_th, rng)
        self.goal_anchor_on = bool(on)

    # -- running -------------------------------------------------------------
    def step_ik(self, q: Sequence[float], goal_lpy: Sequence[float], goal_position: Sequence[float],
                written: Optional[Sequence[float]] = None, velocity: Optional[Sequence[float]] = None,
                seed: bool = False) -> IkTick:
        """Write joint angles and a pose goal (lift, pitch, yaw; position in m), simulate one tick, read everything.
        ``written``: with ``lead``, the angles written ahead on the joint rings (the motor rings get ``q``).
        ``velocity``: with ``observer``, the joint velocities the controller commanded (rad/s); ``seed``: write the
        observer rings at the angles the network's own late matrix shows (while the arm is placed) instead."""

        b, d, w = self.backend, self.ik_design, self.ik_wiring
        if self.observer_wiring is not None:
            self._drive_observer(velocity, seed)
        for ring, angle in zip(self.motor_rings, np.asarray(q, dtype=float)):
            ring.drive(b, float(angle))
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
        tick = self.step(written if (self.method == "lead" and written is not None) else q)
        for ring in self.motor_rings:
            ring.read(b)
        held = np.full(3, np.nan)
        for c, ring in enumerate(self.goal_rings):
            ring.read(b)
            held[2 - c] = ring.held_angle()
        goal_counts = np.zeros((3, self.mapping.count))
        goal_places, goal_position_held = np.full((3, 2), np.nan), np.full(3, np.nan)
        if self.goal_sheets:
            goal_counts = np.array([sheet.read(b) for sheet in self.goal_sheets])
            goal_places = np.array([self.mapping.centre(c) for c in goal_counts])
            window = np.array([sheet.window_counts() for sheet in self.goal_sheets])
            goal_position_held = self.space.metres(sheet_position(self.mapping, window)[0])
        n = self.chain.n
        joint_goal, lead = np.full(n, np.nan), np.full(n, np.nan)
        if self.goal_writing:
            counts = np.array([goal.ring.read(b) for goal in self.goal_writing])
            self._joint_goal_history = (self._joint_goal_history + [counts])[-int(d.goal_window):]
            window = np.sum(self._joint_goal_history, axis=0)
            for k, (goal, joint_ring) in enumerate(zip(self.goal_writing, self.state_rings)):
                index = goal.ring.mapping.population_vector_index(window[k])
                if np.isfinite(index):
                    joint_goal[k] = goal.ring.mapping.angle(index)
                    if np.isfinite(joint_ring.index):
                        lead[k] = goal.ring.mapping.index_error(index, joint_ring.index)
        values = dict(zip(self._ik_nodes, self.ik_readout.values(b)))
        rests = np.array([[values[node] for node in rest] for rest in w.rests]) / TURN_SCALE if w.turn else np.zeros((len(w.rests_position), 3))
        rests_position = (np.array([[values[node] for node in rest] for rest in w.rests_position]) / d.position_scale
                          if w.offset else np.zeros((len(w.rests), 3)))
        ramp_goal, ramp_hand = np.full(3, np.nan), np.full(3, np.nan)
        if w.ramps:
            ramp_goal = self.space.metres([values[node] for node in ramp_nodes("goal")])
            ramp_hand = self.space.metres([values[node] for node in ramp_nodes("hand")])
        self._last_tool = tick.tool
        return IkTick(tick, w.turn_vector(values), w.offset_vector(values), w.share_values(values), rests, rests_position, held,
                      goal_places, goal_counts, goal_position_held, ramp_goal, ramp_hand, joint_goal, lead,
                      self._read_observer() if self.observer_wiring is not None else None)

    def _drive_observer(self, velocity: Optional[Sequence[float]], seed: bool) -> None:
        """Set the joint-velocity sources; while the arm is placed (``seed``), write the observer rings instead.

        Seeding starts from empty rings: for its first OBSERVER_CLEAR_TICKS ticks the observer rings and their shift
        cells are silenced (a bump left where the arm was would stay beside the new one, as a far stimulus raises a
        second bump, and a short pause is not enough: ring neurons sit at their threshold, so what is left of the old
        bump's own synaptic currents lights it again);
        then the rings are written at the chart angles of the network's own late matrix, once that matrix is close to
        a rotation, and the shift cells stay silent until seeding ends."""

        b = self.backend
        speeds = np.zeros(self.chain.n) if (seed or velocity is None) else np.asarray(velocity, dtype=float)
        for source, v in zip(self.observer_sources, speeds):
            set_value(b, source, [float(np.clip(v / QDOT_FULL, -1.0, 1.0))])
        rings = [ring for chart in self.observer_rings for ring in chart]
        if seed and not self.observer_seeding:
            for cells in self.shift_cells:          # their gate alone brings a few µA under a written bump: hold them well below
                b.SetStatus(cells.left, {"I_e": SHIFT_SILENT_PA})
                b.SetStatus(cells.right, {"I_e": SHIFT_SILENT_PA})
            for ring in rings:
                b.SetStatus(ring.neurons, {"I_e": SILENT_PA})
                b.SetStatus(ring.generators, {"rate": 0.0})
            self.observer_seeding, self._observer_cleared, self._observer_clear_left = True, True, OBSERVER_CLEAR_TICKS - 1
            return
        if seed:
            if self._observer_clear_left > 0:
                self._observer_clear_left -= 1
                return
            if self._observer_cleared:
                rng = np.random.default_rng([4242, int(self.t_ms)])
                for ring in rings:
                    b.SetStatus(ring.neurons, [{"I_e": RING_NEURON["I_e"], "V_m": float(v)} for v in rng.uniform(0.0, RING_NEURON["V_th"], ring.size)])
                self._observer_cleared = False
            last = getattr(self, "_last_tool", None)
            if last is not None and np.all(np.isfinite(last)) and np.linalg.norm(last.T @ last - np.eye(3)) < 0.3:
                rotation = nearest_rotation(np.asarray(last, dtype=float))
                for c, u in enumerate(CHARTS):
                    for ring, angle in zip(self.observer_rings[c], lift_pitch_yaw(u.T @ rotation)):
                        ring.drive(b, float(angle))
        elif self.observer_seeding:
            for ring in rings:
                b.SetStatus(ring.generators, {"rate": 0.0})
            for cells in self.shift_cells:
                b.SetStatus(cells.left, {"I_e": float(cells.settings.rest_pa)})
                b.SetStatus(cells.right, {"I_e": float(cells.settings.rest_pa)})
            self.observer_seeding = False

    @property
    def observer_seed_ticks(self) -> int:
        """Ticks a placement must hold for the observer to be cleared and written again (0 without an observer)."""

        return OBSERVER_CLEAR_TICKS + 8 if self.observer_wiring is not None else 0

    def restart_observer(self) -> None:
        """The next seeding tick clears the observer rings again (the session calls this once the arm is placed: a
        bump dragged along a fast placement can leave a second one behind)."""

        self.observer_seeding = False

    def _read_observer(self) -> ObserverTick:
        b, w = self.backend, self.observer_wiring
        angles = np.full((2, 3), np.nan)
        for c, rings in enumerate(self.observer_rings):
            for j, ring in enumerate(rings):
                ring.read(b)
                angles[c, j] = ring.held_angle()
        values = dict(zip(self._observer_nodes, self.observer_readout.values(b)))
        selector = np.array([values[("osel", c)] for c in range(2)])
        selected = int(np.argmax(selector))
        fitness = np.array([values[("ofit", c)] for c in range(2)])
        fallback = bool(values[("ofall", 0)] + values[("ofall", 1)] > GATE_ON)
        hand = np.array([[sum(values[("route", r, i, j)] for r in range(3)) for j in range(3)] for i in range(3)])
        pairs = np.array([[[values[("ofeat", c, j, part)] for part in (0, 1)] for j in range(3)] for c in range(2)])
        return ObserverTick(angles, selected, float(fitness[selected]), fallback, hand, w.rate_values(values),
                            np.array([[values[("oinn", c, j)] for j in range(3)] for c in range(2)]), fitness, selector,
                            np.arctan2(pairs[..., 1], pairs[..., 0]))

    def joint_goal_counts(self) -> np.ndarray:
        if not self._joint_goal_history:
            return np.zeros((self.chain.n, self.calibrations.ring_size))
        return self._joint_goal_history[-1]

    def goal_counts(self) -> np.ndarray:
        """Spike counts of the goal rings in the last tick, rows lift, pitch, yaw."""

        if not self.goal_rings:
            return np.zeros((3, self.calibrations.ring_size))
        return np.array([self.goal_rings[c].counts for c in (2, 1, 0)])

    def ik_neuron_count(self) -> int:
        n = sum(ring.size for ring in self.goal_rings + self.motor_rings) + sum(p.n for p in self.goal_cos_sin)
        n += sum(p.n for p in self.goal_sources) + sum(p.n for p in self.goal_gates) + len(self.goal_sheets) * self.mapping.count
        n += sum(p.n for p in self.goal_level_cells) + (BIAS_CELLS if self.goal_sheets else 0)
        n += sum(p.n for p in self.ramp_readouts.values()) + sum(p.n for p in self.ik_populations.values())
        n += sum(goal.neurons for goal in self.goal_writing)
        n += sum(ring.size for rings in self.observer_rings for ring in rings) + sum(p.n for p in self.observer_pairs)
        n += sum(p.n for p in self.observer_sources) + sum(p.n for p in self.observer_populations.values())
        n += sum(cells.neurons for cells in self.shift_cells)
        return int(n)

    def neuron_count(self) -> int:
        return int(self.forward_neuron_count() + self.ik_neuron_count())

    # -- lesions -------------------------------------------------------------
    def lesion_classes(self) -> Dict[str, List[Group]]:
        """The forward network's classes (:meth:`ForwardNetwork.lesion_classes`) with the IK's populations added:

        * ``rings``: motor rings, pose-goal rings, written goal rings, gain pairs (and the observer's rings and shift cells);
        * ``value``: goal cos/sin, goal sources and gates, ramp read-outs, every comparison and cascade population (and
          the observer's populations);
        * ``sheets``: goal sheets, their level cells and their bias cells."""

        groups = super().lesion_classes()
        rings = groups["rings"]
        rings += [(ring.name, ring.neurons, ring.size) for ring in self.motor_rings + self.goal_rings]
        for k, goal in enumerate(self.goal_writing):
            rings += [(goal.ring.name, goal.ring.neurons, goal.ring.size), ("gain_pair%d_left" % (k + 1), goal.left, goal.ring.size),
                      ("gain_pair%d_right" % (k + 1), goal.right, goal.ring.size)]
        rings += [(ring.name, ring.neurons, ring.size) for chart in self.observer_rings for ring in chart]
        for index, cells in enumerate(self.shift_cells):
            rings += [("shift%d_left" % index, cells.left, len(cells.left)), ("shift%d_right" % index, cells.right, len(cells.right))]
        value = (self.goal_cos_sin + self.goal_sources + self.goal_gates + list(self.ramp_readouts.values()) + list(self.ik_populations.values())
                 + self.observer_sources + self.observer_pairs + list(self.observer_populations.values()))
        groups["value"] += [(p.name, p.neurons, p.n) for p in value]
        groups["sheets"] += [(sheet.name, sheet.neurons, sheet.mapping.count) for sheet in self.goal_sheets]
        groups["sheets"] += [(p.name, p.neurons, p.n) for p in self.goal_level_cells]
        if self.goal_bias_cells is not None:
            groups["sheets"].append(("goal_bias_cells", self.goal_bias_cells, BIAS_CELLS))
        return groups

    def joint_lesion_targets(self, joint: int, site: str) -> List[Group]:
        """What a lesion of one joint silences (``joint`` counts from 0; the names count from 1):

        * ``motor``: the joint's gain pair (2 × ring size) and written goal ring (ring size). Nothing writes the joint's
          next step any more, so its target goes missing and the joint holds its angle. The cascade still hands it its
          share and subtracts it from what later joints see.
        * ``allocation``: the joint's two share halves (2 × sum population). Its share reads 0, so the gain pair writes
          the goal ring on the joint's own bump and the joint holds. The relays that pass on what is left subtract
          nothing for it, so every later joint is handed the whole remainder."""

        k = int(joint)
        if not 0 <= k < self.chain.n:
            raise ValueError("joint %d: the arm has joints 0 to %d" % (k, self.chain.n - 1))
        if site == "motor":
            goal = self.goal_writing[k]
            return [("gain_pair%d_left" % (k + 1), goal.left, goal.ring.size), ("gain_pair%d_right" % (k + 1), goal.right, goal.ring.size),
                    (goal.ring.name, goal.ring.neurons, goal.ring.size)]
        if site == "allocation":
            plus, minus = self.ik_wiring.shares[k]
            return [("share_half%d_plus" % (k + 1), self.ik_populations[plus].neurons, self.ik_populations[plus].n),
                    ("share_half%d_minus" % (k + 1), self.ik_populations[minus].neurons, self.ik_populations[minus].n)]
        raise ValueError("lesion site must be one of %r" % (JOINT_LESION_SITES,))

    def lesion_joint(self, joint: int, site: str) -> JointLesion:
        """Silence one joint's ``motor`` or ``allocation`` neurons (:meth:`joint_lesion_targets`) until :meth:`restore`.

        Every neuron's threshold is raised out of reach (``spiking.populations.silence_nodes``), which holds whatever its
        input. A negative current would not: a written goal ring's neuron can take nearly 1 µA from one gain neuron."""

        targets = self.joint_lesion_targets(joint, site)
        silenced = silence_nodes(self.backend, [nodes for _, nodes, _ in targets])
        return JointLesion(site, int(joint), [(name, size) for name, _, size in targets], silenced)

    def restore(self, lesion: JointLesion) -> None:
        """Give a lesion's neurons back their thresholds and the potentials they had when silenced."""

        restore_nodes(self.backend, lesion.silenced)

    def exact(self, q: Sequence[float], goal_lpy: Sequence[float], goal_position: Sequence[float]) -> Tuple[np.ndarray, ...]:
        """What the network's own wiring gives with exact products and sums: turn, offset, shares, what is left."""

        w = self.ik_wiring
        values = w.evaluate(q, goal_lpy, goal_position, room=limit_room(self.chain, q, self.ik_design, self.calibrations.ring_size))
        return w.turn_vector(values), w.offset_vector(values), w.share_values(values), w.left_vector(values)
