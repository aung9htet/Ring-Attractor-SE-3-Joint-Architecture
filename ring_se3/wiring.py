"""Position and velocity of the end point as small steps on top of the SO(3) wiring.

:func:`pose_wiring` takes the joint-by-joint wiring of :mod:`ring_so3.wiring` (which gives
every link's orientation as a table of nodes) and adds:

* ``("pos", i)``: the position, scaled to the workspace, ``(p - centre) / radius``.  A
  weighted sum of link-orientation entries with the link offsets as weights; no product.
* ``("lever", k, i)``: the lever of joint k (from its origin to the end point) over its
  largest length ``L_k``.  Also a weighted sum.  The lever of joint 1 is the position itself.
* ``("qdot", k)``: the velocity of joint k over ``qdot_ref``, given from outside.
* ``("mul", j)``: a product of two earlier values: first ``w_k = qdot_k · axis_k`` (the axis
  is a column of link k), then the cross product ``w_k × lever_k``, two products per component.
* ``("vel", i)``: the velocity of the end point over ``radius · qdot_ref · velocity_scale``,
  the weighted sum of the cross products (a *velocity population*; the *velocity cells* that
  act on the sheets read these).

Constants are folded: a product with a constant factor is no product (joint 1 of every
chain here turns about a fixed axis).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ring_so3.kinematics import AXES, Chain
from ring_so3.wiring import ONE, Combo, Node, Wiring, chain_wiring

from .chains import Workspace, lever_lengths, link_offsets, workspace

Terms = Tuple[Tuple[Node, float], ...]
EPS = 1e-12
#: joint velocity that is given to the network as 1, rad/s
QDOT_REF = 1.0
#: the end point's velocity is given to the sheets as ``v / (radius · QDOT_REF · VELOCITY_SCALE)``
VELOCITY_SCALE = 1.5


@dataclass(frozen=True)
class Mul:
    index: int
    a: Terms
    b: Terms

    @property
    def node(self) -> Node:
        return ("mul", self.index)


@dataclass(frozen=True)
class Relay:
    """A population that holds a weighted sum of earlier values."""

    node: Node
    terms: Terms


PoseStep = Union[Mul, Relay]


def _clean(combo: Combo) -> Combo:
    return {node: float(c) for node, c in combo.items() if abs(c) >= EPS}


def _sum(combos: Sequence[Combo], weights: Optional[Sequence[float]] = None) -> Combo:
    out: Combo = {}
    for index, combo in enumerate(combos):
        weight = 1.0 if weights is None else float(weights[index])
        for node, c in combo.items():
            out[node] = out.get(node, 0.0) + weight * c
    return _clean(out)


def _terms(combo: Combo) -> Terms:
    return tuple(sorted(combo.items()))


def value_of(combo: Combo, values: Dict[Node, float]) -> float:
    return float(sum(c * values[node] for node, c in combo.items()))


@dataclass
class PoseWiring:
    chain: Chain
    so3: Wiring
    space: Workspace
    velocity: bool
    steps: List[PoseStep] = field(default_factory=list)
    #: the three position nodes and, with ``velocity``, the three velocity nodes
    position: List[Node] = field(default_factory=list)
    velocity_nodes: List[Node] = field(default_factory=list)
    qdot_ref: float = QDOT_REF
    velocity_scale: float = VELOCITY_SCALE

    @property
    def muls(self) -> List[Mul]:
        return [s for s in self.steps if isinstance(s, Mul)]

    @property
    def relays(self) -> List[Relay]:
        return [s for s in self.steps if isinstance(s, Relay)]

    def qdot_values(self, qdot: Sequence[float]) -> np.ndarray:
        return np.clip(np.asarray(qdot, dtype=float).reshape(-1) / self.qdot_ref, -1.0, 1.0)

    def evaluate(self, q: Sequence[float], qdot: Optional[Sequence[float]] = None,
                 so3_values: Optional[Dict[Node, float]] = None,
                 mul: Optional[Callable[[float, float, Mul], float]] = None,
                 relay: Optional[Callable[[float, Relay], float]] = None) -> Dict[Node, float]:
        """Value of every node.  ``mul`` and ``relay`` replace the exact product and sum by what a population gives."""

        values = dict(so3_values) if so3_values is not None else self.so3.evaluate(q)
        rates = self.qdot_values(np.zeros(self.chain.n) if qdot is None else qdot)
        for k in range(self.chain.n):
            values[("qdot", k)] = float(rates[k])
        for step in self.steps:
            if isinstance(step, Mul):
                a, b = value_of(dict(step.a), values), value_of(dict(step.b), values)
                values[step.node] = mul(a, b, step) if mul is not None else a * b
            else:
                x = value_of(dict(step.terms), values)
                values[step.node] = relay(x, step) if relay is not None else x
        return values

    def position_m(self, values: Dict[Node, float]) -> np.ndarray:
        return self.space.metres([values[node] for node in self.position])

    def velocity_m_s(self, values: Dict[Node, float]) -> np.ndarray:
        scale = self.space.radius * self.qdot_ref * self.velocity_scale
        return np.array([values[node] for node in self.velocity_nodes]) * scale

    def summary(self) -> Dict[str, int]:
        out = dict(self.so3.summary())
        out.update({"pose_products": len(self.muls), "pose_relays": len(self.relays)})
        return out


class _MulTable:
    def __init__(self, steps: List[PoseStep]) -> None:
        self.steps = steps
        self.known: Dict[Tuple[Terms, Terms], Mul] = {}

    @staticmethod
    def _split(combo: Combo) -> Tuple[Terms, float]:
        """A single node with a coefficient is handed on as the node, with the coefficient outside."""

        if len(combo) == 1 and ONE not in combo:
            (node, scale), = combo.items()
            return ((node, 1.0),), float(scale)
        return _terms(combo), 1.0

    def times(self, a: Combo, b: Combo) -> Combo:
        a, b = _clean(a), _clean(b)
        if not a or not b:
            return {}
        if set(a) == {ONE}:
            return _clean({node: c * a[ONE] for node, c in b.items()})
        if set(b) == {ONE}:
            return _clean({node: c * b[ONE] for node, c in a.items()})
        (ta, sa), (tb, sb) = self._split(a), self._split(b)
        key = (ta, tb)
        if key not in self.known:
            self.known[key] = Mul(len(self.known), ta, tb)
            self.steps.append(self.known[key])
        return {self.known[key].node: sa * sb}


def pose_wiring(chain: Chain, relay: bool = True, velocity: bool = True) -> PoseWiring:
    so3 = chain_wiring(chain, relay)
    space = workspace(chain)
    wiring = PoseWiring(chain, so3, space, bool(velocity))
    offsets, lengths, n = link_offsets(chain), lever_lengths(chain), chain.n

    # d_m = R_m · o_(m+1), m = 1 .. n, as combos in metres (m = 0 is the constant centre)
    link_vectors: List[List[Combo]] = []
    for m in range(1, n + 1):
        table = so3.links[m - 1]
        link_vectors.append([_sum([table[i][j] for j in range(3)], offsets[m]) for i in range(3)])

    for i in range(3):
        combo = _sum([vector[i] for vector in link_vectors], [1.0 / space.radius] * n)
        wiring.steps.append(Relay(("pos", i), _terms(combo)))
        wiring.position.append(("pos", i))
    if not velocity:
        return wiring

    table = _MulTable(wiring.steps)
    crosses: List[List[Combo]] = []
    weights: List[float] = []
    for k in range(n):
        if lengths[k] < EPS:
            continue
        if k == 0:
            lever: List[Combo] = [{("pos", i): 1.0} for i in range(3)]
        else:
            lever = []
            for i in range(3):
                combo = _sum([vector[i] for vector in link_vectors[k:]], [1.0 / lengths[k]] * (n - k))
                if combo:
                    wiring.steps.append(Relay(("lever", k, i), _terms(combo)))
                    lever.append({("lever", k, i): 1.0})
                else:
                    lever.append({})
        axis = AXES.index(chain.joints[k].axis)
        turning = [table.times(so3.links[k][i][axis], {("qdot", k): 1.0}) for i in range(3)]
        cross = []
        for i in range(3):
            j, m = (i + 1) % 3, (i + 2) % 3
            cross.append(_sum([table.times(turning[j], lever[m]), table.times(turning[m], lever[j])], [1.0, -1.0]))
        crosses.append(cross)
        weights.append(lengths[k] / (space.radius * wiring.velocity_scale))
    for i in range(3):
        combo = _sum([cross[i] for cross in crosses], weights)
        wiring.steps.append(Relay(("vel", i), _terms(combo)))
        wiring.velocity_nodes.append(("vel", i))
    return wiring


__all__ = ["Mul", "PoseStep", "PoseWiring", "QDOT_REF", "Relay", "VELOCITY_SCALE", "pose_wiring", "value_of"]
