"""Rotations and angles: the conventions every other module uses (NumPy only).

Orientation is read as lift, pitch and yaw, ``R = Rz(yaw) · Ry(pitch) · Rx(lift)``:

    lift  = atan2(r32, r33)
    pitch = −asin(r31)
    yaw   = atan2(r21, r11)

Pitch is the middle angle, so at pitch = ±90° lift and yaw lose their meaning (the chart's pole).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

AXES = ("x", "y", "z")
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def rot(axis: str, angle: float) -> np.ndarray:
    """Rotation about one coordinate axis."""

    c, s = float(np.cos(angle)), float(np.sin(angle))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError("unknown axis %r" % (axis,))


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Fixed-axis roll, pitch, yaw as a URDF gives them: ``Rz(yaw) Ry(pitch) Rx(roll)``."""

    return rot("z", yaw) @ rot("y", pitch) @ rot("x", roll)


def other_axes(axis: str) -> Tuple[int, int]:
    """The columns ``(u, v)`` a turn about ``axis`` mixes: ``col_u' = c·col_u + s·col_v``, ``col_v' = −s·col_u + c·col_v``."""

    a = AXES.index(axis)
    return (a + 1) % 3, (a + 2) % 3


def lift_pitch_yaw(rotation: np.ndarray) -> Tuple[float, float, float]:
    r = np.asarray(rotation, dtype=float)
    pitch = -np.arcsin(np.clip(r[2, 0], -1.0, 1.0))
    lift = np.arctan2(r[2, 1], r[2, 2])
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return float(lift), float(pitch), float(yaw)


def from_lift_pitch_yaw(lift: float, pitch: float, yaw: float) -> np.ndarray:
    return rot("z", yaw) @ rot("y", pitch) @ rot("x", lift)


def nearest_rotation(matrix: np.ndarray) -> np.ndarray:
    """The proper rotation closest to a 3 × 3 matrix (least squares, by SVD)."""

    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) of the single rotation that turns orientation ``a`` into ``b``."""

    relative = np.asarray(a, dtype=float).T @ np.asarray(b, dtype=float)
    return float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


def orientation_error_deg(estimate: np.ndarray, true: np.ndarray) -> float:
    """Orientation error in degrees; ``estimate`` may be a raw decoded matrix, it is first made a rotation."""

    matrix = np.asarray(estimate, dtype=float)
    if not np.all(np.isfinite(matrix)):
        return float("nan")
    return float(np.degrees(rotation_angle(nearest_rotation(matrix), true)))


def orthonormality_defect(matrix: np.ndarray) -> float:
    """How far a decoded 3 × 3 matrix is from a rotation: Frobenius norm of ``MᵀM − I``."""

    m = np.asarray(matrix, dtype=float)
    return float(np.linalg.norm(m.T @ m - np.eye(3)))


def wrap(angle):
    """Angle(s) on the circle, in (−π, π]."""

    return np.angle(np.exp(1j * np.asarray(angle, dtype=float)))


def skew(vector: Sequence[float]) -> np.ndarray:
    """The matrix ``[w]×`` with ``[w]× v = w × v``."""

    w = np.asarray(vector, dtype=float)
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rodrigues(vector: Sequence[float]) -> np.ndarray:
    """The rotation by ``|vector|`` rad about ``vector``: ``I + sin θ [k]× + (1 − cos θ) [k]×²``."""

    w = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.eye(3)
    k = skew(w / angle)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def turn_between(rotation: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """The turn ``e = ½ Σ_j r_j × g_j`` over the columns of ``R`` and ``G``: ``sin(Δ)·n`` for the rotation by ``Δ``
    about ``n`` that takes ``R`` into ``G`` (base frame)."""

    r, g = np.asarray(rotation, dtype=float), np.asarray(goal, dtype=float)
    return 0.5 * sum(np.cross(r[:, j], g[:, j]) for j in range(3))
