"""The forward kinematics as a list of small steps: what each value population computes (no neurons).

Values are named by *nodes*:

* ``("feat", k, 0 | 1)``: cos / sin of joint k (the *cos/sin populations*, read from joint ring k);
* ``("prod", i)``: a *product population*: (a weighted sum of earlier nodes) × (one cos or sin);
* ``("entry", link, row, col)``: a *link-orientation population*: one entry of a link's orientation;
* ``("pos", i)``: a *position cell*: one coordinate of the end point, ``(p − centre) / reach``;
* ``ONE``: the constant 1 (a bias current).

A matrix is a 3 × 3 table of *combos* (``{node: coefficient}``).  :func:`forward_wiring` works joint by
joint, ``R_k = R_(k−1) · A_k · Rot(axis_k, q_k)``: the fixed rotation ``A_k`` only re-labels columns; the
joint then mixes two columns, ``col_u' = c·col_u + s·col_v`` and ``col_v' = −s·col_u + c·col_v``, four
products per row; the third column is passed on untouched.  Every new entry that is a sum of several
products gets its own link-orientation population.  The position is then a weighted sum of entries,
``p = Σ_m R_m · o_(m+1)``, with no product at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from reaching_benchmark.geometry.chains import Chain, Workspace, link_offsets, workspace
from reaching_benchmark.geometry.rotations import AXES, other_axes

Node = Tuple
Combo = Dict[Node, float]
Terms = Tuple[Tuple[Node, float], ...]
ONE: Node = ("one",)
EPS = 1e-12


def feature_node(joint: int, part: int) -> Node:
    """``part`` 0: cos, 1: sin."""

    return ("feat", int(joint), int(part))


@dataclass(frozen=True)
class Product:
    """A product population: ``operand × feature``."""

    index: int
    operand: Terms
    joint: int
    part: int

    @property
    def node(self) -> Node:
        return ("prod", self.index)

    @property
    def feature(self) -> Node:
        return feature_node(self.joint, self.part)


@dataclass(frozen=True)
class Entry:
    """A link-orientation population: one entry of link ``link``'s orientation, a sum of earlier nodes."""

    link: int          # 1 .. n
    row: int
    col: int
    terms: Terms

    @property
    def node(self) -> Node:
        return ("entry", self.link, self.row, self.col)


@dataclass(frozen=True)
class PositionCell:
    """A position cell: one scaled coordinate of the end point, a weighted sum of link-orientation entries."""

    node: Node
    terms: Terms


Step = Union[Product, Entry]
Matrix = List[List[Combo]]


def add_term(target: Combo, node: Node, coefficient: float) -> None:
    value = target.get(node, 0.0) + float(coefficient)
    if abs(value) < EPS:
        target.pop(node, None)
    else:
        target[node] = value


def combo_sum(*combos: Combo) -> Combo:
    out: Combo = {}
    for combo in combos:
        for node, coefficient in combo.items():
            add_term(out, node, coefficient)
    return out


def scaled(combo: Combo, factor: float) -> Combo:
    return {node: c * factor for node, c in combo.items() if abs(c * factor) >= EPS}


def weighted_sum(combos: Sequence[Combo], weights: Optional[Sequence[float]] = None) -> Combo:
    """``Σ_i w_i · combo_i``, dropping coefficients below EPS (the summing of the pose and IK layers)."""

    out: Combo = {}
    for index, combo in enumerate(combos):
        weight = 1.0 if weights is None else float(weights[index])
        for node, c in combo.items():
            out[node] = out.get(node, 0.0) + weight * c
    return {node: float(c) for node, c in out.items() if abs(c) >= EPS}


def value_of(combo: Combo, values: Dict[Node, float]) -> float:
    return float(sum(c * values[node] for node, c in combo.items()))


def _times_matrix(matrix: Matrix, fixed: np.ndarray) -> Matrix:
    return [[combo_sum(*[scaled(matrix[i][m], float(fixed[m][j])) for m in range(3)]) for j in range(3)] for i in range(3)]


def _identity() -> Matrix:
    return [[({ONE: 1.0} if i == j else {}) for j in range(3)] for i in range(3)]


def _needs_population(combo: Combo) -> bool:
    nodes = [node for node in combo if node != ONE]
    return len(nodes) >= 2 or (len(nodes) == 1 and ONE in combo)


class _ProductTable:
    """Hands out one product population per distinct (operand, joint, cos | sin)."""

    def __init__(self, steps: List[Step]) -> None:
        self.steps = steps
        self.known: Dict[Tuple, Product] = {}

    def times_feature(self, operand: Combo, joint: int, part: int) -> Combo:
        """Combo of ``operand × feature`` (a constant part of the operand needs no product)."""

        out: Combo = {}
        if ONE in operand:
            add_term(out, feature_node(joint, part), operand[ONE])
        rest = {node: c for node, c in operand.items() if node != ONE}
        if not rest:
            return out
        if len(rest) == 1:
            (node, scale), = rest.items()
            key_operand: Terms = ((node, 1.0),)
        else:
            key_operand, scale = tuple(sorted(rest.items())), 1.0
        key = (key_operand, int(joint), int(part))
        if key not in self.known:
            self.known[key] = Product(len(self.known), key_operand, int(joint), int(part))
            self.steps.append(self.known[key])
        add_term(out, self.known[key].node, scale)
        return out


@dataclass
class ForwardWiring:
    chain: Chain
    space: Workspace
    #: products and link-orientation entries, in the order they are built
    steps: List[Step] = field(default_factory=list)
    #: orientation of link 1 .. n and of the hand, as tables of combos
    links: List[Matrix] = field(default_factory=list)
    tool: Matrix = field(default_factory=_identity)
    #: the three position cells
    position_cells: List[PositionCell] = field(default_factory=list)

    @property
    def products(self) -> List[Product]:
        return [s for s in self.steps if isinstance(s, Product)]

    @property
    def entries(self) -> List[Entry]:
        return [s for s in self.steps if isinstance(s, Entry)]

    @property
    def position(self) -> List[Node]:
        return [cell.node for cell in self.position_cells]

    def evaluate(self, q: Sequence[float], product: Optional[Callable[[float, float, Product], float]] = None) -> Dict[Node, float]:
        """Value of every node for joint angles ``q`` (exact, or with ``product`` in place of the exact product)."""

        values: Dict[Node, float] = {ONE: 1.0}
        for k, angle in enumerate(np.asarray(q, dtype=float).reshape(-1)):
            values[feature_node(k, 0)], values[feature_node(k, 1)] = float(np.cos(angle)), float(np.sin(angle))
        for step in self.steps:
            if isinstance(step, Product):
                a = sum(coefficient * values[node] for node, coefficient in step.operand)
                values[step.node] = product(a, values[step.feature], step) if product is not None else a * values[step.feature]
            else:
                values[step.node] = sum(coefficient * values[node] for node, coefficient in step.terms)
        for cell in self.position_cells:
            values[cell.node] = value_of(dict(cell.terms), values)
        return values

    @staticmethod
    def matrix(values: Dict[Node, float], combos: Matrix) -> np.ndarray:
        return np.array([[sum(c * values[node] for node, c in combos[i][j].items()) for j in range(3)] for i in range(3)])

    def tool_matrix(self, values: Dict[Node, float]) -> np.ndarray:
        return self.matrix(values, self.tool)

    def position_m(self, values: Dict[Node, float]) -> np.ndarray:
        return self.space.metres([values[node] for node in self.position])

    def depths(self) -> Dict[Node, int]:
        """Populations a signal passes from a joint ring to each node (a cos/sin population is 1)."""

        depth: Dict[Node, int] = {ONE: 0}
        for k in range(self.chain.n):
            depth[feature_node(k, 0)] = depth[feature_node(k, 1)] = 1
        for step in self.steps:
            if isinstance(step, Product):
                depth[step.node] = 1 + max([depth[node] for node, _ in step.operand] + [depth[step.feature]])
            else:
                depth[step.node] = 1 + max(depth[node] for node, _ in step.terms)
        return depth

    def summary(self) -> Dict[str, int]:
        depth = self.depths()
        tool_nodes = {node for row in self.tool for combo in row for node in combo if node != ONE}
        return {"joints": self.chain.n, "products": len(self.products), "entries": len(self.entries),
                "depth": max([depth[node] for node in tool_nodes] or [0])}


def forward_wiring(chain: Chain, with_position: bool = True) -> ForwardWiring:
    """Joint by joint; ``with_position``: add the three position cells (the goal chain has no position)."""

    wiring = ForwardWiring(chain, workspace(chain) if with_position else Workspace(np.zeros(3), 1.0))
    table = _ProductTable(wiring.steps)
    matrix = _identity()
    for k, joint in enumerate(chain.joints):
        moved = _times_matrix(matrix, joint.fixed_matrix())
        a, (u, v) = AXES.index(joint.axis), other_axes(joint.axis)
        new: Matrix = [[{} for _ in range(3)] for _ in range(3)]
        for i in range(3):
            new[i][a] = moved[i][a]
            new[i][u] = combo_sum(table.times_feature(moved[i][u], k, 0), table.times_feature(moved[i][v], k, 1))
            new[i][v] = combo_sum(scaled(table.times_feature(moved[i][u], k, 1), -1.0), table.times_feature(moved[i][v], k, 0))
            for col in (u, v):
                if _needs_population(new[i][col]):
                    entry = Entry(k + 1, i, col, tuple(sorted(new[i][col].items())))
                    wiring.steps.append(entry)
                    new[i][col] = {entry.node: 1.0}
        matrix = new
        wiring.links.append([[dict(combo) for combo in row] for row in matrix])
    wiring.tool = _times_matrix(matrix, np.array(chain.tool_fixed, dtype=float))
    if not with_position:
        return wiring

    offsets, n = link_offsets(chain), chain.n
    link_vectors = [[weighted_sum([wiring.links[m - 1][i][j] for j in range(3)], offsets[m]) for i in range(3)] for m in range(1, n + 1)]
    for i in range(3):
        combo = weighted_sum([vector[i] for vector in link_vectors], [1.0 / wiring.space.radius] * n)
        wiring.position_cells.append(PositionCell(("pos", i), tuple(sorted(combo.items()))))
    return wiring
