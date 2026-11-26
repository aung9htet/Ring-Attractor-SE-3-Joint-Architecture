"""The comparison and the cascade as small steps on top of the forward wiring (no neurons).

Nodes added to those of :mod:`ring_attractor_kinematics.network.forward_wiring`:

* ``("gfeat", c, part)``: cos / sin of goal ring c (0: yaw, 1: pitch, 2: lift);
* ``("g", node)``: the *goal matrix*, built from the goal rings with the forward chain's own wiring;
* ``("turn", i)``: the *turn* ``e = ½ Σ_j r_j × g_j`` × :data:`TURN_SCALE` (18 products, 3 sums);
* ``("ramp", side, i)``: coordinate i read from the goal or the forward sheets (*ramp read-outs*, from outside);
* ``("gpos", i)``: the goal position as a number (position goal ``cells``);
* ``("offset", i)``: ``d = (goal − hand) / reach`` × ``position_scale``;
* ``("lever", k, i)``: the lever of joint k over its longest length;
* ``("jac", k, i)``: ``c_k = a_k × lever_k``, the direction in which joint k moves the end point;
* ``("ikmul", j)``: a product of two earlier values;
* ``("share", k, ±1)``: the two halves of joint k's share (*share halves*), each in [0, 1] of the joint's cap;
* ``("rest", k, i)``, ``("restp", k, i)``: what is left of turn and offset after joint k.

Kinds of population: :class:`Mul` (a product), :class:`Relay` (a sum, held in [−1, 1]), :class:`Half`
(the positive part of a sum, held in [0, 1]).  Constants are folded: joint 1 turns about a fixed axis,
so its share and what it leaves need no product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from reaching_benchmark.geometry.chains import GOAL_CHAIN, Chain, lever_lengths, link_offsets
from reaching_benchmark.geometry.rotations import AXES

from ..config import IkDesign
from ..rules.cascade_rule import Levers, levers, normalisers, share_caps
from .forward_wiring import (ONE, Combo, Entry, ForwardWiring, Node, Product, forward_wiring, value_of,
                             weighted_sum)

Terms = Tuple[Tuple[Node, float], ...]
EPS = 1e-12
#: the turn is held as TURN_SCALE · e: a turn of 0.5 rad fills a population's range
TURN_SCALE = 2.0
SIDES = ("goal", "hand")


@dataclass(frozen=True)
class Mul:
    """A product population ``a × b``; a product of the goal matrix keeps the goal chain's node name."""

    index: int
    a: Terms
    b: Terms
    name: Optional[Node] = None

    @property
    def node(self) -> Node:
        return self.name if self.name is not None else ("ikmul", self.index)


@dataclass(frozen=True)
class Relay:
    """A sum population (held in [−1, 1])."""

    node: Node
    terms: Terms


@dataclass(frozen=True)
class Half:
    """A share half: the positive part of a sum, held in [0, 1]."""

    node: Node
    terms: Terms


IkStep = Union[Mul, Relay, Half]


def _terms(combo: Combo) -> Terms:
    return tuple(sorted(combo.items(), key=lambda item: repr(item[0])))


class _MulTable:
    """Hands out one product population per distinct pair of operands."""

    def __init__(self, steps: List[IkStep], prefix: str = "ikmul") -> None:
        """``prefix``: the products' node names (another wiring on the same network uses its own)."""

        self.steps = steps
        self.prefix = prefix
        self.known: Dict[Tuple[Terms, Terms], Mul] = {}

    @staticmethod
    def _split(combo: Combo) -> Tuple[Terms, float]:
        if len(combo) == 1 and ONE not in combo:
            (node, scale), = combo.items()
            return ((node, 1.0),), float(scale)
        return _terms(combo), 1.0

    def times(self, a: Combo, b: Combo) -> Combo:
        a, b = weighted_sum([a]), weighted_sum([b])
        if not a or not b:
            return {}
        if set(a) == {ONE}:
            return weighted_sum([{node: c * a[ONE] for node, c in b.items()}])
        if set(b) == {ONE}:
            return weighted_sum([{node: c * b[ONE] for node, c in a.items()}])
        (ta, sa), (tb, sb) = self._split(a), self._split(b)
        key = (ta, tb)
        if key not in self.known:
            self.known[key] = Mul(len(self.known), ta, tb, None if self.prefix == "ikmul" else (self.prefix, len(self.known)))
            self.steps.append(self.known[key])
        return {self.known[key].node: sa * sb}


def ramp_node(side: str, coordinate: int) -> Node:
    return ("ramp", side, int(coordinate))


def ramp_nodes(side: str) -> List[Node]:
    return [ramp_node(side, i) for i in range(3)]


def _goal_node(node: Node) -> Node:
    if node == ONE:
        return ONE
    if node[0] == "feat":
        return ("gfeat", node[1], node[2])
    return ("g", node)


def _goal_combo(combo: Combo) -> Combo:
    return {_goal_node(node): c for node, c in combo.items()}


@dataclass
class IkWiring:
    chain: Chain
    forward: ForwardWiring
    design: IkDesign
    caps: np.ndarray                                   # the largest share of every joint, rad
    lever: Levers
    norms: np.ndarray                                  # N_k
    steps: List[IkStep] = field(default_factory=list)
    turn: List[Node] = field(default_factory=list)
    offset: List[Node] = field(default_factory=list)
    shares: List[Tuple[Node, Node]] = field(default_factory=list)       # (plus, minus) per joint
    rests: List[List[Node]] = field(default_factory=list)
    rests_position: List[List[Node]] = field(default_factory=list)
    goal: Optional[List[List[Combo]]] = None
    jacobian: List[List[Combo]] = field(default_factory=list)
    #: the hand's matrix the turn reads (default: link n's entries); with the observer method, the matrix gates
    hand: Optional[List[List[Combo]]] = None
    #: values the exact evaluation gives nodes of ``hand`` it does not compute (the matrix gates: the late route open)
    hand_defaults: Dict[Node, Combo] = field(default_factory=dict)
    #: where each part starts in ``steps``: (name, joint or −1, index of its first step)
    sections: List[Tuple[str, int, int]] = field(default_factory=list)

    def mark(self, name: str, joint: int = -1) -> None:
        self.sections.append((name, int(joint), len(self.steps)))

    def part_of(self) -> Dict[Node, Tuple[str, int]]:
        """Which part of Figure 1 every step belongs to: goal, turn, offset, levers, direction, share, rest."""

        out: Dict[Node, Tuple[str, int]] = {}
        for index, (name, joint, first) in enumerate(self.sections):
            last = self.sections[index + 1][2] if index + 1 < len(self.sections) else len(self.steps)
            for step in self.steps[first:last]:
                out[step.node] = (name, joint)
        return out

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
    def ramps(self) -> List[Node]:
        if not self.design.position or self.design.position_goal != "sheets":
            return []
        return ramp_nodes("goal") + ramp_nodes("hand")

    def outside_values(self, goal_lpy: Sequence[float], goal_scaled: Sequence[float], hand_scaled: Sequence[float]) -> Dict[Node, float]:
        """Exact values of what the layer is given: goal cos/sin and the two positions."""

        lift, pitch, yaw = [float(v) for v in goal_lpy]
        out: Dict[Node, float] = {}
        for c, angle in enumerate((yaw, pitch, lift)):
            out[("gfeat", c, 0)], out[("gfeat", c, 1)] = float(np.cos(angle)), float(np.sin(angle))
        goal, hand = np.asarray(goal_scaled, dtype=float), np.asarray(hand_scaled, dtype=float)
        for i in range(3):
            out[("gpos", i)] = float(goal[i])
        for side, values in (("goal", goal), ("hand", hand)):
            for i, node in enumerate(ramp_nodes(side)):
                out[node] = float(values[i])
        return out

    def evaluate(self, q: Sequence[float], goal_lpy: Sequence[float], goal_position: Sequence[float],
                 room: Optional[np.ndarray] = None) -> Dict[Node, float]:
        """Exact value of every node.  ``goal_position`` in metres; ``room``: (n, 2) joint-limit factors."""

        values = self.forward.evaluate(q)
        hand = [values[node] for node in self.forward.position]
        values.update(self.outside_values(goal_lpy, self.forward.space.scaled(goal_position), hand))
        for node, combo in self.hand_defaults.items():
            values.setdefault(node, value_of(combo, values))
        joints = {node: k for k, pair in enumerate(self.shares) for node in pair}
        for step in self.steps:
            if isinstance(step, Mul):
                values[step.node] = value_of(dict(step.a), values) * value_of(dict(step.b), values)
            elif isinstance(step, Half):
                x = value_of(dict(step.terms), values)
                if room is not None and step.node in joints:
                    x -= 1.0 - float(room[joints[step.node], 1 if step.node[2] > 0 else 0])
                values[step.node] = float(np.clip(x, 0.0, 1.0))
            else:
                values[step.node] = float(np.clip(value_of(dict(step.terms), values), -1.0, 1.0))
        return values

    def turn_vector(self, values: Dict[Node, float]) -> np.ndarray:
        return np.array([values[node] for node in self.turn]) / TURN_SCALE if self.turn else np.zeros(3)

    def offset_vector(self, values: Dict[Node, float]) -> np.ndarray:
        return np.array([values[node] for node in self.offset]) / self.design.position_scale if self.offset else np.zeros(3)

    def share_values(self, values: Dict[Node, float]) -> np.ndarray:
        return np.array([self.caps[k] * (values[plus] - values[minus]) for k, (plus, minus) in enumerate(self.shares)])

    def left_vector(self, values: Dict[Node, float]) -> np.ndarray:
        """What the whole arm could not take: turn (rad), then offset (reaches)."""

        turn = np.array([values[node] for node in self.rests[-1]]) / TURN_SCALE if self.turn else np.zeros(3)
        offset = np.array([values[node] for node in self.rests_position[-1]]) / self.design.position_scale if self.offset else np.zeros(3)
        return np.concatenate([turn, offset])

    def summary(self) -> Dict[str, int]:
        return {"products": len(self.muls), "sum_populations": len(self.relays), "share_populations": len(self.halves),
                "sheet_readouts": len(self.ramps)}


def _goal_matrix(wiring: IkWiring) -> List[List[Combo]]:
    """The goal matrix from the three goal rings: the forward chain's own wiring for yaw, pitch, lift."""

    goal = forward_wiring(GOAL_CHAIN, with_position=False)
    for step in goal.steps:
        if isinstance(step, Product):
            wiring.steps.append(Mul(-1, _terms(_goal_combo(dict(step.operand))), ((_goal_node(step.feature), 1.0),), _goal_node(step.node)))
        elif isinstance(step, Entry):
            wiring.steps.append(Relay(_goal_node(step.node), _terms(_goal_combo(dict(step.terms)))))
    return [[_goal_combo(goal.tool[i][j]) for j in range(3)] for i in range(3)]


def _turn(wiring: IkWiring, table: _MulTable) -> List[Combo]:
    """``e = ½ Σ_j r_j × g_j`` from the hand's matrix (link 7's entries, or the matrix gates) and the goal matrix."""

    tool = wiring.hand if wiring.hand is not None else wiring.forward.tool
    wiring.mark("goal")
    wiring.goal = _goal_matrix(wiring)
    wiring.mark("turn")
    components = []
    for i in range(3):
        a, b = (i + 1) % 3, (i + 2) % 3
        terms = []
        for j in range(3):                       # (r_j × g_j)_i = r_aj g_bj − r_bj g_aj
            terms.append(table.times(tool[a][j], wiring.goal[b][j]))
            terms.append(weighted_sum([table.times(tool[b][j], wiring.goal[a][j])], [-1.0]))
        components.append(weighted_sum(terms, [0.5] * len(terms)))
    return components


def _offset(wiring: IkWiring) -> List[Combo]:
    """``d = goal − hand`` per coordinate: ramp read-outs of goal and forward sheets, or goal number − position cell."""

    if wiring.design.position_goal == "cells":
        return [{("gpos", i): 1.0, wiring.forward.position[i]: -1.0} for i in range(3)]
    return [{ramp_node("goal", i): 1.0, ramp_node("hand", i): -1.0} for i in range(3)]


def _levers(wiring: IkWiring) -> List[List[Combo]]:
    """The lever of every joint over its longest length (empty where the joint moves nothing)."""

    forward, chain = wiring.forward, wiring.chain
    offsets, lengths, n = link_offsets(chain), lever_lengths(chain), chain.n
    out: List[List[Combo]] = []
    for k in range(n):
        if float(wiring.lever.mean_square[k]) <= EPS:
            out.append([{}, {}, {}])
        elif k == 0:
            out.append([{forward.position[i]: 1.0} for i in range(3)])
        else:
            lever: List[Combo] = []
            for i in range(3):
                vectors = [weighted_sum([forward.links[m - 1][i][j] for j in range(3)], offsets[m]) for m in range(k + 1, n + 1)]
                combo = weighted_sum(vectors, [1.0 / lengths[k]] * len(vectors))
                if not combo:
                    lever.append({})
                    continue
                node = ("lever", k, i)
                wiring.steps.append(Relay(node, _terms(combo)))
                lever.append({node: 1.0})
            out.append(lever)
    return out


def ik_wiring(chain: Chain, design: Optional[IkDesign] = None, ring_size: int = 200,
              forward: Optional[ForwardWiring] = None, hand: Optional[List[List[Combo]]] = None,
              hand_defaults: Optional[Dict[Node, Combo]] = None) -> IkWiring:
    """``hand``: what the turn reads as the hand's matrix (the observer method's matrix gates), with
    ``hand_defaults`` for the exact evaluation."""

    design = design or IkDesign()
    if design.normaliser == "exact":
        raise ValueError("the exact normaliser divides by a value that changes with the pose: it is for the exact rule only")
    forward = forward or forward_wiring(chain)
    lever = levers(chain)
    wiring = IkWiring(chain, forward, design, share_caps(chain, design, ring_size), lever, normalisers(design, lever),
                      hand=hand, hand_defaults=dict(hand_defaults or {}))
    table = _MulTable(wiring.steps)
    alpha, beta, scale = design.turn_weight, design.move_weight, float(design.position_scale)

    if design.orientation:
        for i, combo in enumerate(_turn(wiring, table)):
            wiring.steps.append(Relay(("turn", i), _terms(weighted_sum([combo], [TURN_SCALE]))))
            wiring.turn.append(("turn", i))
    levers_: List[List[Combo]] = []
    if design.position:
        wiring.mark("offset")
        for i, combo in enumerate(_offset(wiring)):
            wiring.steps.append(Relay(("offset", i), _terms(weighted_sum([combo], [scale]))))
            wiring.offset.append(("offset", i))
        wiring.mark("levers")
        levers_ = _levers(wiring)

    rest_turn: List[Combo] = [{node: 1.0} for node in wiring.turn]
    rest_offset: List[Combo] = [{node: 1.0} for node in wiring.offset]
    for k, joint in enumerate(chain.joints):
        axis = [forward.links[k][i][AXES.index(joint.axis)] for i in range(3)]
        ratio = float(lever.ratios[k])
        moving = design.position and float(lever.mean_square[k]) > EPS
        cross: List[Combo] = [{}, {}, {}]
        wiring.mark("direction", k)
        if moving:
            for i in range(3):
                j, m = (i + 1) % 3, (i + 2) % 3
                combo = weighted_sum([table.times(axis[j], levers_[k][m]), table.times(axis[m], levers_[k][j])], [1.0, -1.0])
                if combo:
                    wiring.steps.append(Relay(("jac", k, i), _terms(combo)))
                    cross[i] = {("jac", k, i): 1.0}
        wiring.jacobian.append(cross)

        wiring.mark("share", k)
        asked: List[Combo] = []
        weights: List[float] = []
        if design.orientation:                   # TURN_SCALE × (a_k · e_k)
            asked.append(weighted_sum([table.times(axis[i], rest_turn[i]) for i in range(3)]))
            weights.append(alpha / (TURN_SCALE * wiring.norms[k] * wiring.caps[k]))
        if moving:                               # position_scale × (c_k · d_k)
            asked.append(weighted_sum([table.times(cross[i], rest_offset[i]) for i in range(3)]))
            weights.append(beta * ratio / (scale * wiring.norms[k] * wiring.caps[k]))
        share = weighted_sum(asked, weights)
        plus, minus = ("share", k, 1), ("share", k, -1)
        wiring.steps.append(Half(plus, _terms(share)))
        wiring.steps.append(Half(minus, _terms(weighted_sum([share], [-1.0]))))
        wiring.shares.append((plus, minus))
        if not design.cascade:
            continue
        wiring.mark("rest", k)
        taken = {plus: 1.0, minus: -1.0}
        if design.orientation:
            nodes = []
            for i in range(3):
                node = ("rest", k + 1, i)
                wiring.steps.append(Relay(node, _terms(weighted_sum([rest_turn[i], table.times(axis[i], taken)],
                                                                    [1.0, -TURN_SCALE * wiring.caps[k]]))))
                nodes.append(node)
            rest_turn = [{node: 1.0} for node in nodes]
            wiring.rests.append(nodes)
        if design.position:
            nodes = []
            for i in range(3):
                moved = table.times(cross[i], taken) if moving else {}
                if not moved:                    # this joint does not move the end point along i
                    nodes.append(next(iter(rest_offset[i])))
                    continue
                node = ("restp", k + 1, i)
                wiring.steps.append(Relay(node, _terms(weighted_sum([rest_offset[i], moved], [1.0, -scale * ratio * wiring.caps[k]]))))
                nodes.append(node)
            rest_offset = [{node: 1.0} for node in nodes]
            wiring.rests_position.append(nodes)
    if not design.cascade:
        wiring.rests.append(list(wiring.turn))
        wiring.rests_position.append(list(wiring.offset))
    return wiring
