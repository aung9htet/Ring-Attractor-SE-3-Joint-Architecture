"""Cyclic coordinate descent (Wang & Chen 1991; Kenwright 2012): one joint at a time, from the wrist to the base, each
turned by the angle that brings the hand closest to the goal, the hand's pose recomputed after every joint, joint limits
kept by clamping; sweeps repeat until the hand is at the goal or a sweep limit is reached.

**One joint's angle** (Wang & Chen's combined position and orientation step). Turning joint k by φ about its axis ``a``
(through its origin ``c``) turns every vector ``u`` after it into ``R(φ)u``, and for any ``u``, ``v``::

    vᵀR(φ)u = (a·u)(a·v) + [v·u − (a·u)(a·v)] cos φ + v·(a × u) sin φ

The task's weighted error, θ² + w²·(‖p* − p‖ / reach)² for small θ, is smallest where
``α Σ_j g_j·R(φ)r_j + (2β / reach²) (p* − c)·R(φ)(p − c)`` is largest (``r_j``, ``g_j`` the columns of the hand's and
the goal's rotation; α = 1, β = w² = 9, each 0 when the task does not ask for it). That sum is ``K₀ + K₁ cos φ + K₂ sin φ``,
so ``φ = atan2(K₂, K₁)``: one closed-form angle per joint.

It reads the measured joint angles and the true geometry every tick, and solves for the written goal from there: no
internal model, no delay and nothing learned.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import Chain, workspace
from reaching_benchmark.geometry.rotations import rotation_angle, wrap
from reaching_benchmark.task import IkDesign

#: at most this many sweeps per tick
SWEEPS = 50
#: a sweep stops early when the hand is this close: orientation (deg) and position (fraction of reach)
STOP_DEG = 0.05
STOP_REACH = 0.0005
#: the step toward the solution is sent as a target this many times ahead of the joint, over κ (as every controller's)
GAIN = 1.0


def best_angle(axis: np.ndarray, origin: np.ndarray, hand_rotation: np.ndarray, hand_position: np.ndarray,
               goal_rotation: np.ndarray, goal_position: np.ndarray, design: IkDesign, radius: float) -> float:
    """The turn of one joint (rad) that makes the task's weighted error smallest, all other joints held."""

    a = np.asarray(axis, dtype=float)
    pairs = []
    if design.turn_weight:
        pairs += [(design.turn_weight, hand_rotation[:, j], goal_rotation[:, j]) for j in range(3)]
    if design.move_weight:
        pairs.append((2.0 * design.move_weight / radius ** 2, hand_position - origin, goal_position - origin))
    k1 = k2 = 0.0
    for weight, u, v in pairs:
        au, av = float(a @ u), float(a @ v)
        k1 += weight * (float(v @ u) - au * av)
        k2 += weight * float(v @ np.cross(a, u))
    return float(np.arctan2(k2, k1))


def errors(chain: Chain, q: np.ndarray, goal_rotation: np.ndarray, goal_position: np.ndarray, radius: float) -> Tuple[float, float]:
    """Orientation (deg) and position (fraction of reach) of the hand from the goal."""

    return (float(np.degrees(rotation_angle(chain.rotation(q), goal_rotation))),
            float(np.linalg.norm(goal_position - kinematics.position(chain, q)) / radius))


def solve(chain: Chain, q: np.ndarray, goal_rotation: np.ndarray, goal_position: np.ndarray, design: IkDesign,
          sweeps: int = SWEEPS, radius: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """CCD from ``q`` toward the goal: the joint angles it ends at, and the sweeps it took."""

    radius = float(radius if radius is not None else workspace(chain).radius)
    q = np.array(q, dtype=float)
    limits, circular = chain.limits(), chain.circular_mask()
    done = 0
    for done in range(1, int(sweeps) + 1):
        for k in reversed(range(chain.n)):
            axes = kinematics.joint_axes(chain, q)
            origins = kinematics.joint_origins(chain, q)
            phi = best_angle(axes[k], origins[k], chain.rotation(q), origins[-1], goal_rotation, goal_position, design, radius)
            angle = q[k] + phi
            q[k] = wrap(angle) if circular[k] else float(np.clip(angle, limits[k, 0], limits[k, 1]))
        degrees, reach = errors(chain, q, goal_rotation, goal_position, radius)
        if (not design.orientation or degrees <= STOP_DEG) and (not design.position or reach <= STOP_REACH):
            break
    return q, done
