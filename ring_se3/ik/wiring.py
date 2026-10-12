"""Pose comparison and pose cascade as small steps on top of the pose wiring.

The pose wiring (:mod:`ring_se3.wiring`) gives every entry of every link's orientation and the
three position nodes.  :func:`pose_ik_wiring` adds, as a list of steps the spiking network and
the exact evaluation both follow (the step kinds are those of :mod:`ring_so3.ik.wiring`):

* ``("gfeat", c, part)``, ``("g", node)``: cos and sin of the three goal rings and the goal matrix
  built from them, as in the SO-3 repo;
* ``("turn", i)``: the comparison of orientations ``e``, times :data:`TURN_SCALE`;
* ``("ramp", side, i)``: coordinate i of the packets of one set of sheets, read as a number from
  the two sheets that hold it (``side``: ``"goal"`` or ``"hand"``); given from outside, like the
  goal rings' cos and sin;
* ``("gpos", i)``: the goal position as a number (position goal ``cells``), from outside;
* ``("offset", i)``: the comparison of positions ``d = (goal − hand) / reach``, times the design's
  ``position_scale``.  For ``sheets`` goal ramp minus hand ramp (the ramp comparator); for
  ``cells`` the goal number minus the position cell;
* ``("lever", k, i)``: the lever of joint k over its largest length (taken from the pose wiring
  when that was built with its velocity path);
* ``("jac", k, i)``: ``c_k = a_k × lever_k``: the direction in which joint k moves the end point.
  The end point goes ``m_k = ρ_k c_k`` reaches per radian;
* ``("ikmul", j)``: a product of two earlier values;
* ``("share", k, +1)`` and ``("share", k, -1)``: the two halves of joint k's share, each between 0
  and 1, in units of the joint's cap: share = cap_k · (plus − minus);
* ``("rest", k, i)`` and ``("restp", k, i)``: what is left of turn and offset after joint k (in
  the units of ``turn`` and ``offset``); ``k = n`` is what the whole arm could not take.

Constants are folded: joint 1 turns about a fixed axis, so its turn part and what it leaves of
the turn need no product, and its ``c_1`` is a plain sum of the position nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ring_so3.ik import rules as so3
from ring_so3.ik.wiring import (Half, IkStep, Mul, Relay, TURN_SCALE, _MulTable, _goal_steps, _sum, _terms, value_of)
from ring_so3.kinematics import AXES, Chain
from ring_so3.wiring import Combo, Node

from ..chains import lever_lengths, link_offsets
from ..wiring import PoseWiring, pose_wiring
from ..wiring import Relay as PoseRelay
from . import rules
from .design import PoseIkDesign

EPS = 1e-12
SIDES = ("goal", "hand")


def ramp_node(side: str, coordinate: int) -> Node:
    return ("ramp", side, int(coordinate))


def ramp_nodes(side: str) -> List[Node]:
    """The three readouts of one set of sheets: x, y, z."""

    return [ramp_node(side, i) for i in range(3)]


@dataclass
class PoseIkWiring:
    chain: Chain
    pose: PoseWiring
    design: PoseIkDesign
    caps: np.ndarray                                   # the largest share of every joint, rad
    lever: rules.Levers
    norms: np.ndarray                                  # what every joint's share is divided by
    steps: List[IkStep] = field(default_factory=list)
    turn: List[Node] = field(default_factory=list)                      # empty when the task has no orientation
    offset: List[Node] = field(default_factory=list)                    # empty when the task has no position
    shares: List[Tuple[Node, Node]] = field(default_factory=list)       # (plus, minus) per joint
    rests: List[List[Node]] = field(default_factory=list)               # after joint 1 .. n: turn nodes
    rests_position: List[List[Node]] = field(default_factory=list)      # after joint 1 .. n: offset nodes
    goal: Optional[List[List[Combo]]] = None                            # the goal matrix
    jacobian: List[List[Combo]] = field(default_factory=list)           # c_k per joint, three combos
    #: where each part of the layer starts in ``steps``: (name, joint or -1, index of its first step)
    sections: List[Tuple[str, int, int]] = field(default_factory=list)

    def mark(self, name: str, joint: int = -1) -> None:
        self.sections.append((name, int(joint), len(self.steps)))

    def parts(self) -> List[Dict[str, object]]:
        """Products, sums and share halves of every part, in the order they are built: the goal matrix (``goal``), the
        comparison of orientations (``turn``) and of positions (``offset``), the levers (``levers``), and per joint
        the direction in which it moves the end point (``direction``), its share (``share``) and what it leaves (``rest``)."""

        out = []
        for index, (name, joint, first) in enumerate(self.sections):
            last = self.sections[index + 1][2] if index + 1 < len(self.sections) else len(self.steps)
            steps = self.steps[first:last]
            out.append({"part": name, "joint": joint, "products": sum(isinstance(s, Mul) for s in steps),
                        "sums": sum(isinstance(s, Relay) for s in steps), "halves": sum(isinstance(s, Half) for s in steps)})
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
        """The sheet readouts the comparison of positions takes (none for the position goal ``cells``)."""

        if not self.design.position or self.design.position_goal != "sheets":
            return []
        return ramp_nodes("goal") + ramp_nodes("hand")

    # -- exact evaluation ----------------------------------------------------
    def outside_values(self, goal_lpy: Sequence[float], goal_scaled: Sequence[float], hand_scaled: Sequence[float]) -> Dict[Node, float]:
        """Exact values of what the layer is given from outside: goal cos / sin, and the position as it is handed over."""

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
                 pose_values: Optional[Dict[Node, float]] = None, room: Optional[np.ndarray] = None) -> Dict[Node, float]:
        """Exact value of every node.  ``goal_position``: metres, base frame.  ``room``: (n, 2) factors on the
        minus / plus half of each share (joint limits)."""

        values = dict(pose_values) if pose_values is not None else self.pose.evaluate(q)
        hand = [values[node] for node in self.pose.position]
        values.update(self.outside_values(goal_lpy, self.pose.space.scaled(goal_position), hand))
        joints = {node: k for k, pair in enumerate(self.shares) for node in pair}
        for step in self.steps:
            if isinstance(step, Mul):
                values[step.node] = value_of(dict(step.a), values) * value_of(dict(step.b), values)
            elif isinstance(step, Half):
                x = value_of(dict(step.terms), values)
                if room is not None and step.node in joints:
                    x -= 1.0 - float(room[joints[step.node], 1 if step.node[2] > 0 else 0])      # the state ring holds this half down
                values[step.node] = float(np.clip(x, 0.0, 1.0))
            else:
                values[step.node] = float(np.clip(value_of(dict(step.terms), values), -1.0, 1.0))
        return values

    def turn_vector(self, values: Dict[Node, float]) -> np.ndarray:
        """The comparison of orientations ``e`` (rad); zeros when the task has none."""

        return np.array([values[node] for node in self.turn]) / TURN_SCALE if self.turn else np.zeros(3)

    def offset_vector(self, values: Dict[Node, float]) -> np.ndarray:
        """The comparison of positions ``d`` (reaches); zeros when the task has none."""

        return np.array([values[node] for node in self.offset]) / self.design.position_scale if self.offset else np.zeros(3)

    def share_values(self, values: Dict[Node, float]) -> np.ndarray:
        """Each joint's share, rad."""

        return np.array([self.caps[k] * (values[plus] - values[minus]) for k, (plus, minus) in enumerate(self.shares)])

    def left_vector(self, values: Dict[Node, float]) -> np.ndarray:
        """What the whole arm could not take: turn (rad), then offset (reaches)."""

        turn = np.array([values[node] for node in self.rests[-1]]) / TURN_SCALE if self.turn else np.zeros(3)
        offset = np.array([values[node] for node in self.rests_position[-1]]) / self.design.position_scale if self.offset else np.zeros(3)
        return np.concatenate([turn, offset])

    def summary(self) -> Dict[str, int]:
        return {"products": len(self.muls), "sum_populations": len(self.relays), "share_populations": len(self.halves),
                "sheet_readouts": len(self.ramps)}


def _matrix_comparison(wiring: PoseIkWiring, table: _MulTable) -> List[Combo]:
    """``e = ½ Σ_j r_j × g_j`` from the decoded matrix and the goal matrix (the SO(3) layer's comparison)."""

    tool = wiring.pose.so3.tool
    wiring.mark("goal")
    wiring.goal = _goal_steps(wiring)
    wiring.mark("turn")
    components = []
    for i in range(3):
        a, b = (i + 1) % 3, (i + 2) % 3
        terms = []
        for j in range(3):                       # (r_j x g_j)_i = r_aj g_bj - r_bj g_aj
            terms.append(table.times(tool[a][j], wiring.goal[b][j]))
            terms.append(_sum([table.times(tool[b][j], wiring.goal[a][j])], [-1.0]))
        components.append(_sum(terms, [0.5] * len(terms)))
    return components


def _position_comparison(wiring: PoseIkWiring) -> List[Combo]:
    """``d = goal − hand`` in reaches, per coordinate."""

    if wiring.design.position_goal == "cells":
        return [{("gpos", i): 1.0, wiring.pose.position[i]: -1.0} for i in range(3)]
    return [{ramp_node("goal", i): 1.0, ramp_node("hand", i): -1.0} for i in range(3)]


def _levers(wiring: PoseIkWiring) -> List[List[Combo]]:
    """The lever of every joint over its largest length, three combos each (empty where the joint has none)."""

    pose, chain = wiring.pose, wiring.chain
    offsets, lengths, n = link_offsets(chain), lever_lengths(chain), chain.n
    known = {step.node for step in pose.steps if isinstance(step, PoseRelay)}
    out: List[List[Combo]] = []
    for k in range(n):
        if float(wiring.lever.mean_square[k]) <= EPS:          # no lever, or one along the joint's own axis: it moves nothing
            out.append([{}, {}, {}])
        elif k == 0:
            out.append([{pose.position[i]: 1.0} for i in range(3)])
        else:
            lever: List[Combo] = []
            for i in range(3):
                node = ("lever", k, i)
                if node not in known:
                    vectors = [_sum([pose.so3.links[m - 1][i][j] for j in range(3)], offsets[m]) for m in range(k + 1, n + 1)]
                    combo = _sum(vectors, [1.0 / lengths[k]] * len(vectors))
                    if not combo:
                        lever.append({})
                        continue
                    wiring.steps.append(Relay(node, _terms(combo)))
                lever.append({node: 1.0})
            out.append(lever)
    return out


def pose_ik_wiring(chain: Chain, pose: Optional[PoseWiring] = None, design: Optional[PoseIkDesign] = None,
                   ring_size: int = 200) -> PoseIkWiring:
    design = design or PoseIkDesign()
    if design.normaliser == "exact":
        raise ValueError("the exact normaliser divides by a value that changes with the pose: it is for the rule without neurons")
    pose = pose or pose_wiring(chain, relay=True, velocity=False)
    lever = rules.levers(chain)
    wiring = PoseIkWiring(chain, pose, design, so3.share_caps(chain, design, ring_size), lever, rules.normalisers(design, lever))
    table = _MulTable(wiring.steps)
    alpha, beta, scale = design.turn_weight, design.move_weight, float(design.position_scale)

    if design.orientation:
        for i, combo in enumerate(_matrix_comparison(wiring, table)):
            wiring.steps.append(Relay(("turn", i), _terms(_sum([combo], [TURN_SCALE]))))
            wiring.turn.append(("turn", i))
    if design.position:
        wiring.mark("offset")
        for i, combo in enumerate(_position_comparison(wiring)):
            wiring.steps.append(Relay(("offset", i), _terms(_sum([combo], [scale]))))
            wiring.offset.append(("offset", i))
        wiring.mark("levers")
        levers = _levers(wiring)

    rest_turn: List[Combo] = [{node: 1.0} for node in wiring.turn]
    rest_offset: List[Combo] = [{node: 1.0} for node in wiring.offset]
    for k, joint in enumerate(chain.joints):
        axis = [pose.so3.links[k][i][AXES.index(joint.axis)] for i in range(3)]
        ratio = float(lever.ratios[k])
        moving = design.position and float(lever.mean_square[k]) > EPS          # a lever along the joint's own axis moves nothing
        cross: List[Combo] = [{}, {}, {}]
        wiring.mark("direction", k)
        if moving:
            for i in range(3):
                j, m = (i + 1) % 3, (i + 2) % 3
                combo = _sum([table.times(axis[j], levers[k][m]), table.times(axis[m], levers[k][j])], [1.0, -1.0])
                if combo:
                    wiring.steps.append(Relay(("jac", k, i), _terms(combo)))
                    cross[i] = {("jac", k, i): 1.0}
        wiring.jacobian.append(cross)

        wiring.mark("share", k)
        asked: List[Combo] = []
        weights: List[float] = []
        if design.orientation:                   # TURN_SCALE x (a_k · e_k)
            asked.append(_sum([table.times(axis[i], rest_turn[i]) for i in range(3)]))
            weights.append(alpha / (TURN_SCALE * wiring.norms[k] * wiring.caps[k]))
        if moving:                               # position_scale x (c_k · d_k); the end point goes ratio x c_k per radian
            asked.append(_sum([table.times(cross[i], rest_offset[i]) for i in range(3)]))
            weights.append(beta * ratio / (scale * wiring.norms[k] * wiring.caps[k]))
        share = _sum(asked, weights)
        plus, minus = ("share", k, 1), ("share", k, -1)
        wiring.steps.append(Half(plus, _terms(share)))
        wiring.steps.append(Half(minus, _terms(_sum([share], [-1.0]))))
        wiring.shares.append((plus, minus))
        if not design.cascade:
            continue
        wiring.mark("rest", k)
        taken = {plus: 1.0, minus: -1.0}
        if design.orientation:
            nodes = []
            for i in range(3):
                node = ("rest", k + 1, i)
                wiring.steps.append(Relay(node, _terms(_sum([rest_turn[i], table.times(axis[i], taken)], [1.0, -TURN_SCALE * wiring.caps[k]]))))
                nodes.append(node)
            rest_turn = [{node: 1.0} for node in nodes]
            wiring.rests.append(nodes)
        if design.position:
            nodes = []
            for i in range(3):
                moved = table.times(cross[i], taken) if moving else {}
                if not moved:                    # this joint does not move the end point along i: what is left stays as it is
                    nodes.append(next(iter(rest_offset[i])))
                    continue
                node = ("restp", k + 1, i)
                wiring.steps.append(Relay(node, _terms(_sum([rest_offset[i], moved], [1.0, -scale * ratio * wiring.caps[k]]))))
                nodes.append(node)
            rest_offset = [{node: 1.0} for node in nodes]
            wiring.rests_position.append(nodes)
    if not design.cascade:
        wiring.rests.append(list(wiring.turn))
        wiring.rests_position.append(list(wiring.offset))
    return wiring


__all__ = ["PoseIkWiring", "SIDES", "TURN_SCALE", "pose_ik_wiring", "ramp_node", "ramp_nodes"]
