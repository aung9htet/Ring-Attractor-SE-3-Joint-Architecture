"""Exact position, velocity and pose of the end point of a chain (no neurons).

With ``R_k`` the orientation of link k (``R_0`` the identity) and ``o_k`` the fixed origin
of joint k in link k-1 (``o_(n+1)``: the end point in link n)::

    origin of joint k:   c_k = Σ_(m<k)  R_m · o_(m+1)
    end point:           p   = Σ_(m<=n) R_m · o_(m+1)             a weighted sum of matrix entries
    lever of joint k:    l_k = p - c_k = Σ_(m>=k) R_m · o_(m+1)
    velocity:            v   = Σ_k  q̇_k · (a_k × l_k)             a_k: the turning axis, a column of R_k
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ring_so3.kinematics import AXES, Chain

from .chains import link_offsets


def link_rotations(chain: Chain, q: Sequence[float]) -> List[np.ndarray]:
    """``R_0 .. R_n``."""

    return [np.eye(3)] + chain.link_rotations(q)


def joint_origins(chain: Chain, q: Sequence[float]) -> np.ndarray:
    """``c_1 .. c_n`` and the end point, (n + 1, 3), in the base frame."""

    rotations, offsets = link_rotations(chain, q), link_offsets(chain)
    points, origin = [], np.zeros(3)
    for m in range(chain.n + 1):
        origin = origin + rotations[m] @ offsets[m]
        points.append(origin.copy())
    return np.array(points)


def position(chain: Chain, q: Sequence[float]) -> np.ndarray:
    return joint_origins(chain, q)[-1]


def joint_axes(chain: Chain, q: Sequence[float]) -> np.ndarray:
    """Turning axis of every joint in the base frame, (n, 3): the column of ``R_k`` the joint leaves alone."""

    rotations = chain.link_rotations(q)
    return np.array([rotations[k][:, AXES.index(joint.axis)] for k, joint in enumerate(chain.joints)])


def levers(chain: Chain, q: Sequence[float]) -> np.ndarray:
    points = joint_origins(chain, q)
    return points[-1][None, :] - points[:-1]


def jacobian(chain: Chain, q: Sequence[float]) -> np.ndarray:
    """(3, n): column k is the velocity of the end point per rad/s of joint k."""

    return np.cross(joint_axes(chain, q), levers(chain, q)).T


def velocity(chain: Chain, q: Sequence[float], qdot: Sequence[float]) -> np.ndarray:
    return jacobian(chain, q) @ np.asarray(qdot, dtype=float)


def pose(chain: Chain, q: Sequence[float]) -> np.ndarray:
    """4 x 4 homogeneous matrix of the end point in the base frame."""

    out = np.eye(4)
    out[:3, :3] = chain.rotation(q)
    out[:3, 3] = position(chain, q)
    return out


def position_error(decoded: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(decoded, dtype=float) - np.asarray(true, dtype=float)))


__all__ = ["jacobian", "joint_axes", "joint_origins", "levers", "link_rotations", "pose", "position", "position_error", "velocity"]
