"""Chains of revolute joints, and the arms this work is built for (NumPy only).

Joint ``k`` first applies a fixed rotation ``A_k`` (how its frame sits in the previous link) and
then turns about one coordinate axis by its angle::

    R_k  = R_(k−1) · A_k · Rot(axis_k, q_k)        orientation of link k
    R_ee = R_n · A_tool                            orientation of the hand

The arms (:data:`MODEL_NAMES`):

* ``generic2`` .. ``generic7``: joints about x, y, z, x, y, z, x, 0.2 m apart, angles on the whole circle;
* ``tiago2`` .. ``tiago7``: the first joints of the TIAGo arm in the frame of ``torso_lift_link``, with
  the axes, link rotations, offsets and limits of the robot's URDF.  ``tiago7`` ends at
  ``arm_tool_link``; a shorter arm ends at the origin of the next joint, so every joint moves its end
  point (``tiago4`` ends at the wrist).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .rotations import IDENTITY, rot, rpy_matrix

GENERIC_AXES = ("x", "y", "z", "x", "y", "z", "x")
GENERIC_LINK_M = 0.2
H = float(np.pi / 2.0)

#: name, origin xyz (m), origin rpy (rad), lower, upper (rad), velocity limit (rad/s)
TIAGO_ARM = (
    ("arm_1_joint", (0.15505, 0.014, -0.151), (0.0, 0.0, -H), 0.0, 2.748893571891069, 1.95),
    ("arm_2_joint", (0.125, 0.0195, -0.031), (H, 0.0, 0.0), -1.5707963267948966, 1.090830782496456, 1.95),
    ("arm_3_joint", (0.0895, 0.0, -0.0015), (-H, 0.0, H), -3.5342917352885173, 1.5707963267948966, 2.35),
    ("arm_4_joint", (-0.02, -0.027, -0.222), (-H, -H, 0.0), -0.39269908169872414, 2.356194490192345, 2.35),
    ("arm_5_joint", (-0.162, 0.02, 0.027), (H, -H, -H), -2.0943951023931953, 2.0943951023931953, 1.95),
    ("arm_6_joint", (0.0, 0.0, 0.15), (0.0, -H, -H), -1.413716694115407, 1.413716694115407, 1.76),
    ("arm_7_joint", (0.0, 0.0, 0.0), (H, 0.0, H), -2.0943951023931953, 2.0943951023931953, 1.76),
)
TIAGO_TOOL = ("arm_tool_joint", (0.0, 0.0, 0.046), (H, -H, float(np.pi)))
TIAGO_BASE_LINK = "torso_lift_link"


@dataclass(frozen=True)
class Joint:
    name: str
    #: coordinate axis the joint turns about, in its own frame
    axis: str
    #: rotation of the joint's frame in the previous link's frame at angle 0
    fixed: Tuple[Tuple[float, ...], ...] = IDENTITY
    #: origin of the joint in the previous link's frame, m
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    lower: float = -float(np.pi)
    upper: float = float(np.pi)
    #: True: the angle lives on the whole circle; False: its limits are mapped onto an arc of the ring
    circular: bool = True
    max_velocity: float = 2.0

    def fixed_matrix(self) -> np.ndarray:
        return np.array(self.fixed, dtype=float)


@dataclass(frozen=True)
class Chain:
    name: str
    joints: Tuple[Joint, ...]
    tool_fixed: Tuple[Tuple[float, ...], ...] = IDENTITY
    tool_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    description: str = ""

    @property
    def n(self) -> int:
        return len(self.joints)

    def link_rotations(self, q: Sequence[float]) -> List[np.ndarray]:
        """Orientation of every link, ``R_1 .. R_n``."""

        rotation, out = np.eye(3), []
        for joint, angle in zip(self.joints, self._angles(q)):
            rotation = rotation @ joint.fixed_matrix() @ rot(joint.axis, angle)
            out.append(rotation)
        return out

    def rotation(self, q: Sequence[float]) -> np.ndarray:
        """Orientation of the hand."""

        return self.link_rotations(q)[-1] @ np.array(self.tool_fixed, dtype=float)

    def frames(self, q: Sequence[float]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """``(origin, orientation)`` of the base, of every link and of the hand, in the base frame."""

        origin, rotation = np.zeros(3), np.eye(3)
        out = [(origin.copy(), rotation.copy())]
        for joint, angle in zip(self.joints, self._angles(q)):
            origin = origin + rotation @ np.array(joint.offset, dtype=float)
            rotation = rotation @ joint.fixed_matrix() @ rot(joint.axis, angle)
            out.append((origin.copy(), rotation.copy()))
        origin = origin + rotation @ np.array(self.tool_offset, dtype=float)
        out.append((origin, rotation @ np.array(self.tool_fixed, dtype=float)))
        return out

    def limits(self) -> np.ndarray:
        return np.array([[j.lower, j.upper] for j in self.joints], dtype=float)

    def clip(self, q: Sequence[float]) -> np.ndarray:
        limits = self.limits()
        return np.clip(np.asarray(q, dtype=float), limits[:, 0], limits[:, 1])

    def random_pose(self, rng: np.random.Generator) -> np.ndarray:
        limits = self.limits()
        return rng.uniform(limits[:, 0], limits[:, 1])

    def circular_mask(self) -> np.ndarray:
        return np.array([j.circular for j in self.joints])

    def start_pose(self) -> np.ndarray:
        """Where trials start by default: all zeros on circular joints, the middle of the range otherwise."""

        return np.zeros(self.n) if self.joints[0].circular else self.limits().mean(axis=1)

    def _angles(self, q: Sequence[float]) -> np.ndarray:
        angles = np.asarray(q, dtype=float).reshape(-1)
        if angles.size != self.n:
            raise ValueError("%s has %d joints, got %d angles" % (self.name, self.n, angles.size))
        return angles


@dataclass(frozen=True)
class Workspace:
    """A ball that holds every position the end point can take; the network is given ``(p − centre) / radius``."""

    centre: np.ndarray
    radius: float

    def scaled(self, position: Sequence[float]) -> np.ndarray:
        return (np.asarray(position, dtype=float) - self.centre) / self.radius

    def metres(self, scaled: Sequence[float]) -> np.ndarray:
        return np.asarray(scaled, dtype=float) * self.radius + self.centre


def _matrix_tuple(matrix: np.ndarray) -> Tuple[Tuple[float, ...], ...]:
    """The TIAGo link rotations are multiples of 90°: store exact 0 and ±1."""

    cleaned = np.where(np.abs(matrix) < 1e-12, 0.0, matrix)
    cleaned = np.where(np.abs(np.abs(cleaned) - 1.0) < 1e-12, np.sign(cleaned), cleaned)
    return tuple(tuple(float(v) for v in row) for row in cleaned)


def generic_chain(n: int) -> Chain:
    if not 1 <= int(n) <= len(GENERIC_AXES):
        raise ValueError("generic chains have 1 to %d joints" % len(GENERIC_AXES))
    joints = tuple(Joint("q%d" % (k + 1), GENERIC_AXES[k], offset=(0.0, 0.0, 0.0 if k == 0 else GENERIC_LINK_M))
                   for k in range(int(n)))
    return Chain("generic%d" % n, joints, tool_offset=(0.0, 0.0, GENERIC_LINK_M),
                 description="joints about %s, angles on the whole circle" % ", ".join(GENERIC_AXES[: int(n)]))


def tiago_chain(n: int = 7) -> Chain:
    if not 1 <= int(n) <= len(TIAGO_ARM):
        raise ValueError("the TIAGo arm has %d joints" % len(TIAGO_ARM))
    joints = tuple(Joint(name, "z", fixed=_matrix_tuple(rpy_matrix(*rpy)), offset=tuple(xyz), lower=lower, upper=upper,
                         circular=False, max_velocity=velocity)
                   for name, xyz, rpy, lower, upper, velocity in TIAGO_ARM[: int(n)])
    if int(n) == len(TIAGO_ARM):
        _, xyz, rpy = TIAGO_TOOL
        return Chain("tiago7", joints, tool_fixed=_matrix_tuple(rpy_matrix(*rpy)), tool_offset=tuple(xyz),
                     description="TIAGo arm, arm_1 .. arm_7, ending at arm_tool_link")
    following = TIAGO_ARM[int(n)]
    return Chain("tiago%d" % n, joints, tool_offset=tuple(following[1]),
                 description="TIAGo arm, arm_1 .. arm_%d, ending at the origin of %s" % (n, following[0]))


#: the twelve arms of the tests and benchmarks
MODEL_NAMES: List[str] = ["generic%d" % n for n in range(2, 8)] + ["tiago%d" % n for n in range(2, 8)]
CHAINS: Dict[str, Chain] = {**{"generic%d" % n: generic_chain(n) for n in range(1, 8)},
                            **{"tiago%d" % n: tiago_chain(n) for n in range(1, 8)}}

#: the three goal rings as a chain, yaw about z, pitch about y, lift about x: its orientation is the goal matrix
GOAL_CHAIN = Chain("goal", (Joint("yaw", "z"), Joint("pitch", "y"), Joint("lift", "x")),
                   description="R = Rot(z, yaw) Rot(y, pitch) Rot(x, lift)")


def get_chain(name: str) -> Chain:
    try:
        return CHAINS[name]
    except KeyError:
        raise ValueError("unknown chain %r; known: %s" % (name, ", ".join(sorted(CHAINS)))) from None


def link_offsets(chain: Chain) -> List[np.ndarray]:
    """``o_1 .. o_(n+1)``: the origin of joint k in link k−1, and last the end point in link n."""

    return [np.array(j.offset, dtype=float) for j in chain.joints] + [np.array(chain.tool_offset, dtype=float)]


def lever_lengths(chain: Chain) -> np.ndarray:
    """``L_k``: the longest the lever from joint k's origin to the end point can be."""

    lengths = np.array([np.linalg.norm(o) for o in link_offsets(chain)])
    return np.array([lengths[k + 1:].sum() for k in range(chain.n)])


def workspace(chain: Chain) -> Workspace:
    """Centre: the origin of the first joint (it does not move).  Radius (the *reach*): the summed links after it."""

    return Workspace(link_offsets(chain)[0].copy(), float(lever_lengths(chain)[0]))

