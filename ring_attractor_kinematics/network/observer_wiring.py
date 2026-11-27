"""The observer method's orientation observer as small steps (no neurons): Figure 2c, as built.

Nodes added to those of the forward wiring:

* ``("qdot", k)``: joint k's commanded velocity / :data:`QDOT_FULL` (a source, set every tick);
* ``("omega", i)``: the angular velocity ``ω = Σ_k a_k q̇_k`` / :data:`OMEGA_FULL` (axes from link entries);
* ``("ofeat", c, j, 0 | 1)``: cos / sin of observer ring ``j`` (0: lift, 1: pitch, 2: yaw) of chart ``c``;
* ``("opast", c, j, 0 | 1)``: the same, ``delay`` ms late (the chart's estimate of what the cue shows now);
* ``("osec", c)``: ``1 / (2 max(cos pitch, 0.5))``, a second read-out of chart c's pitch pair;
* ``("onorm", c)``: the cue's cos pitch in chart c (a norm population, as the forward network's);
* ``("oinn", c, j)``: the *innovation* ``sin(cue − past)`` of angle j (lift and yaw scaled by the cue's cos pitch);
* ``("orate", c, j)``: the speed of observer ring j: chart rate + innovation / correction time, / ``rate_full``;
* ``("ofit", c)``: chart c's *fitness*: its cos pitch − weight · Σ |innovation|;
* ``("osel", c)``: the *selector*: 1 while chart c is selected (two cells that excite themselves and inhibit
  each other, so the choice holds until the other chart's fitness is larger by the margin);
* ``("ofall", c)``: the *fallback* while chart c is selected and its fitness is below the threshold;
* ``("okeep",)``: on while there is no fallback;
* ``("route", r, i, j)``: the *matrix gates*: entry (i, j) of the observer matrix of chart r (0, 1) or of the
  late matrix (r = 2), each silenced unless its route is the one chosen; the turn takes their sum.

A chart's observer matrix is ``U_c · R(lift, pitch, yaw)``, built by the goal-matrix wiring of the IK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reaching_benchmark.geometry.chains import GOAL_CHAIN, Chain
from reaching_benchmark.geometry.rotations import AXES

from ..config import ObserverDesign
from ..rules.observer_rule import CHARTS
from .forward_wiring import ONE, Combo, Entry, ForwardWiring, Matrix, Node, Product, forward_wiring, value_of, weighted_sum
from .ik_wiring import Half, IkStep, Mul, Relay, _MulTable, _terms

#: the joint-velocity sources hold q̇ / QDOT_FULL; the angular velocity is held as ω / OMEGA_FULL (rad/s)
QDOT_FULL = 2.0
OMEGA_FULL = 3.0
LIFT, PITCH, YAW = 0, 1, 2
#: which goal-chain feature each observer ring stands for (the goal chain turns yaw, pitch, lift in that order)
GOAL_FEATURE = {YAW: 0, PITCH: 1, LIFT: 2}
#: the selector: gain on the fitness difference and self-excitation (switches at a difference of (a − ½) / K)
SELECTOR_SELF = 0.75
#: the fallback cells and the cue-validity cells: gain on how far below their threshold a value is
FALLBACK_GAIN = 5.0
VALIDITY_GAIN = 10.0
#: a gated population is silenced while its gate holds more than this
GATE_ON = 0.5


@dataclass(frozen=True)
class Norm:
    """A norm population: the length of the pair (a, b)."""

    node: Node
    a: Tuple[Tuple[Node, float], ...]
    b: Tuple[Tuple[Node, float], ...]


@dataclass(frozen=True)
class Gate:
    """``node`` (a population) is silenced while any of ``by`` holds more than :data:`GATE_ON`."""

    node: Node
    by: Tuple[Node, ...]


@dataclass
class ObserverWiring:
    chain: Chain
    forward: ForwardWiring
    design: ObserverDesign
    steps: List[object] = field(default_factory=list)
    gates: List[Gate] = field(default_factory=list)
    #: what the turn reads in place of the hand's matrix: the sum of the three routes, per entry
    hand: Optional[Matrix] = None
    rates: List[List[Node]] = field(default_factory=list)          # [chart][ring]
    #: recurrent terms of the selector cells, connected after every population exists
    recurrent: Dict[Node, Combo] = field(default_factory=dict)

    @property
    def muls(self) -> List[Mul]:
        return [s for s in self.steps if isinstance(s, Mul)]

    @property
    def relays(self) -> List[Relay]:
        return [s for s in self.steps if isinstance(s, Relay)]

    @property
    def halves(self) -> List[Half]:
        return [s for s in self.steps if isinstance(s, Half)]

    @property
    def norms(self) -> List[Norm]:
        return [s for s in self.steps if isinstance(s, Norm)]

    def summary(self) -> Dict[str, int]:
        return {"products": len(self.muls), "sum_populations": len(self.relays), "halves": len(self.halves),
                "norms": len(self.norms), "gated": len(self.gates)}

    def evaluate(self, q: Sequence[float], qdot: Sequence[float], angles: np.ndarray, past: np.ndarray, selected: int,
                 fallback: bool) -> Dict[Node, float]:
        """Exact value of every node.  ``q``: the joint angles the joint rings hold (the cue and the axes come from
        them); ``angles``, ``past``: (2, 3) lift, pitch, yaw of each chart now and ``delay`` ago; ``selected``,
        ``fallback``: the selector's state (the cells hold it; the caller keeps it)."""

        values = self.forward.evaluate(q)
        for k, v in enumerate(qdot):
            values[("qdot", k)] = float(np.clip(v / QDOT_FULL, -1.0, 1.0))
        for c in range(2):
            for j in range(3):
                for part, f in ((0, np.cos), (1, np.sin)):
                    values[("ofeat", c, j, part)] = float(f(angles[c][j]))
                    values[("opast", c, j, part)] = float(f(past[c][j]))
            values[("osec", c)] = 1.0 / (2.0 * max(float(np.cos(angles[c][PITCH])), 0.5))
            values[("osel", c)] = 1.0 if c == selected else 0.0
        gates = {gate.node: gate for gate in self.gates}
        for step in self.steps:
            if isinstance(step, Mul):
                values[step.node] = value_of(dict(step.a), values) * value_of(dict(step.b), values)
            elif isinstance(step, Norm):
                values[step.node] = float(min(np.hypot(value_of(dict(step.a), values), value_of(dict(step.b), values)), 1.0))
            elif isinstance(step, Half):
                if step.node[0] in ("osel",):
                    continue
                values[step.node] = float(np.clip(value_of(dict(step.terms), values), 0.0, 1.0))
            else:
                values[step.node] = float(np.clip(value_of(dict(step.terms), values), -1.0, 1.0))
            if step.node[0] == "ofall":
                values[step.node] = 1.0 if (fallback and step.node[1] == selected) else 0.0
            if step.node[0] == "okeep":
                values[step.node] = 0.0 if fallback else 1.0
            gate = gates.get(step.node)
            if gate is not None and all(g in values for g in gate.by) and any(values[g] > GATE_ON for g in gate.by):
                values[step.node] = 0.0
        for gate in self.gates:                  # gates whose cells come later in the list (the matrix gates)
            if any(values.get(g, 0.0) > GATE_ON for g in gate.by):
                values[gate.node] = 0.0
        return values

    def hand_defaults(self) -> Dict[Node, Combo]:
        """Each matrix gate's value with the late route open (what the IK's exact evaluation assumes)."""

        out: Dict[Node, Combo] = {}
        for i in range(3):
            for j in range(3):
                for r in range(3):
                    out[("route", r, i, j)] = dict(self.forward.tool[i][j]) if r == 2 else {}
        return out

    def hand_matrix(self, values: Dict[Node, float]) -> np.ndarray:
        return np.array([[value_of(self.hand[i][j], values) for j in range(3)] for i in range(3)])

    def rate_values(self, values: Dict[Node, float]) -> np.ndarray:
        """(2, 3) ring speeds, rad/s (as the rate populations hold them, before the shift cells' top speed)."""

        return np.array([[values[node] * self.design.rate_full for node in row] for row in self.rates])


def _chart_entry(tool: Matrix, u: np.ndarray, a: int, b: int) -> Combo:
    """Entry (a, b) of ``U_cᵀ R`` as a combo of the hand's entries."""

    return weighted_sum([tool[i][b] for i in range(3)], [float(u[i, a]) for i in range(3)])


def _observer_matrix(wiring: ObserverWiring, c: int) -> Matrix:
    """``R(lift, pitch, yaw)`` of chart c from its observer cos/sin, with the goal chain's own wiring."""

    goal = forward_wiring(GOAL_CHAIN, with_position=False)
    inverse_feature = {v: k for k, v in GOAL_FEATURE.items()}

    def node(n: Node) -> Node:
        if n == ONE:
            return ONE
        if n[0] == "feat":
            return ("ofeat", c, inverse_feature[n[1]], n[2])
        return ("o", c, n)

    def combo(cb: Combo) -> Combo:
        return {node(n): v for n, v in cb.items()}

    for step in goal.steps:
        if isinstance(step, Product):
            wiring.steps.append(Mul(-1, _terms(combo(dict(step.operand))), ((node(step.feature), 1.0),), node(step.node)))
        elif isinstance(step, Entry):
            wiring.steps.append(Relay(node(step.node), _terms(combo(dict(step.terms)))))
    return [[combo(goal.tool[i][j]) for j in range(3)] for i in range(3)]


def observer_wiring(chain: Chain, design: Optional[ObserverDesign] = None, forward: Optional[ForwardWiring] = None) -> ObserverWiring:
    design = design or ObserverDesign()
    forward = forward or forward_wiring(chain)
    wiring = ObserverWiring(chain, forward, design)
    steps: List[IkStep] = wiring.steps         # type: ignore[assignment]
    table = _MulTable(steps, prefix="omul")
    tool = forward.tool
    u_of = CHARTS                               # U₀ = I, U₁ = R_y(π/2): the second chart's poles lie where the first has pitch 0

    # angular velocity from the joint axes (link entries) and the commanded velocities
    omega_terms: List[List[Combo]] = [[], [], []]
    for k, joint in enumerate(chain.joints):
        axis = [forward.links[k][i][AXES.index(joint.axis)] for i in range(3)]
        for i in range(3):
            omega_terms[i].append(table.times(axis[i], {("qdot", k): 1.0}))
    omega: List[Node] = []
    for i in range(3):
        node = ("omega", i)
        steps.append(Relay(node, _terms(weighted_sum(omega_terms[i], [QDOT_FULL / OMEGA_FULL] * len(omega_terms[i])))))
        omega.append(node)

    rate_full, tc = float(design.rate_full), float(design.correction_s)
    route_nodes: List[List[List[Node]]] = []
    for c, u in enumerate(u_of):
        w = [weighted_sum([{omega[i]: 1.0} for i in range(3)], [float(u[i, r]) for i in range(3)]) for r in range(3)]   # U_cᵀ ω
        f = {(j, p): {("ofeat", c, j, p): 1.0} for j in range(3) for p in (0, 1)}
        m1, m2 = table.times(f[YAW, 0], w[0]), table.times(f[YAW, 1], w[1])
        m3, m4 = table.times(f[YAW, 1], w[0]), table.times(f[YAW, 0], w[1])
        a_node = ("oa", c)
        steps.append(Relay(a_node, _terms(weighted_sum([m1, m2]))))                      # (cy ωx + sy ωy) / Ω
        lift = table.times({a_node: 1.0}, {("osec", c): 1.0})                             # lift rate / 2Ω
        yaw_extra = table.times(f[PITCH, 1], lift)                                        # sp · lift rate / 2Ω
        chart_rate = [weighted_sum([lift], [2.0 * OMEGA_FULL / rate_full]),
                      weighted_sum([m3, m4], [-OMEGA_FULL / rate_full, OMEGA_FULL / rate_full]),
                      weighted_sum([w[2], yaw_extra], [OMEGA_FULL / rate_full, 2.0 * OMEGA_FULL / rate_full])]
        # the cue in this chart, straight from the hand's entries, against the estimate of `delay` ago
        norm = ("onorm", c)
        steps.append(Norm(norm, _terms(_chart_entry(tool, u, 0, 0)), _terms(_chart_entry(tool, u, 1, 0))))
        invalid = ("oinvalid", c)
        threshold = float(np.cos(np.radians(design.cue_limit_deg)))
        steps.append(Half(invalid, _terms({ONE: VALIDITY_GAIN * threshold + GATE_ON, norm: -VALIDITY_GAIN})))
        cue = {LIFT: (_chart_entry(tool, u, 2, 2), _chart_entry(tool, u, 2, 1)),
               PITCH: ({norm: 1.0}, weighted_sum([_chart_entry(tool, u, 2, 0)], [-1.0])),
               YAW: (_chart_entry(tool, u, 0, 0), _chart_entry(tool, u, 1, 0))}
        rates, innovations = [], []
        for j in range(3):
            cos_cue, sin_cue = cue[j]
            terms = weighted_sum([table.times(sin_cue, {("opast", c, j, 0): 1.0}), table.times(cos_cue, {("opast", c, j, 1): 1.0})], [1.0, -1.0])
            node = ("oinn", c, j)
            steps.append(Relay(node, _terms(terms)))
            wiring.gates.append(Gate(node, (invalid,)))
            innovations.append(node)
            rate = ("orate", c, j)
            steps.append(Relay(rate, _terms(weighted_sum([chart_rate[j], {node: 1.0}], [1.0, 1.0 / (tc * rate_full)]))))
            rates.append(rate)
        wiring.rates.append(rates)
        absolute = []
        for j, node in enumerate(innovations):
            for sign in (1, -1):
                half = ("oabs", c, j, sign)
                steps.append(Half(half, ((node, float(sign)),)))
                absolute.append(half)
        steps.append(Relay(("ofit", c), _terms(weighted_sum([{("ofeat", c, PITCH, 0): 1.0}] + [{h: 1.0} for h in absolute],
                                                             [1.0] + [-float(design.confidence_weight)] * len(absolute)))))
        # the observer matrix, re-expressed in the base frame: U_c · R_c
        matrix = _observer_matrix(wiring, c)
        route = []
        for i in range(3):
            row = []
            for j in range(3):
                node = ("route", c, i, j)
                steps.append(Relay(node, _terms(weighted_sum([matrix[r][j] for r in range(3)], [float(u[i, r]) for r in range(3)]))))
                row.append(node)
            route.append(row)
        route_nodes.append(route)

    # selection: the selector cells (bistable), the fallback cells, and the matrix gates
    gain = (SELECTOR_SELF - 0.5) / float(design.selector_margin)
    for c in range(2):
        node = ("osel", c)
        steps.append(Half(node, _terms({ONE: 0.5, ("ofit", c): gain, ("ofit", 1 - c): -gain})))
        wiring.recurrent[node] = {node: SELECTOR_SELF, ("osel", 1 - c): -SELECTOR_SELF}
    for c in range(2):
        steps.append(Half(("ofall", c), _terms({ONE: FALLBACK_GAIN * float(design.fallback_confidence) - 3.0 + GATE_ON,
                                                ("ofit", c): -FALLBACK_GAIN, ("osel", c): 3.0})))
    steps.append(Half(("okeep",), _terms({ONE: 1.0, ("ofall", 0): -1.0, ("ofall", 1): -1.0})))
    late = []
    for i in range(3):
        row = []
        for j in range(3):
            node = ("route", 2, i, j)
            steps.append(Relay(node, _terms(tool[i][j])))
            wiring.gates.append(Gate(node, (("okeep",),)))
            row.append(node)
        late.append(row)
    for c in range(2):
        for i in range(3):
            for j in range(3):
                wiring.gates.append(Gate(route_nodes[c][i][j], (("osel", 1 - c), ("ofall", 0), ("ofall", 1))))
    wiring.hand = [[{route_nodes[0][i][j]: 1.0, route_nodes[1][i][j]: 1.0, late[i][j]: 1.0} for j in range(3)] for i in range(3)]
    return wiring
