"""What every controller is held to, and how the task's errors are measured (NumPy only).

**Step limits.** A joint may be told to move at most ``lead_indices`` *steps* per tick, and a step toward a joint limit
fades to nothing over the last ``limit_taper_indices`` steps before it. A step is 1/200 of a joint's range: on a
circular joint 2π/200, on a bounded joint (upper − lower) / 140 (the range covers 70% of a 200-index ring). These are the
units of the ring-attractor kinematics model's joint rings; every other controller is held to the same caps in radians,
so the comparison is fair.

**Goal travel.** The goal a controller is given (the *written* goal) travels toward the final goal at ``goal_speed``
(lift, pitch, yaw, rad/s) and ``position_speed`` (reaches/s) every tick, for every controller alike.

**Errors.** The orientation comparison is the turn ``e = ½ Σ_j r_j × g_j`` from the hand's rotation to the goal's; the
position comparison is the offset ``d = (p* − p) / reach``. Both are held within bounds (``TURN_BOUND``; ``1 /
position_scale`` reaches) before a controller sees them.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .geometry import kinematics
from .geometry.chains import Chain, Joint, workspace
from .geometry.rotations import from_lift_pitch_yaw, turn_between, wrap
from .task import IkDesign

#: steps per joint range on a circular joint; a bounded joint's range covers ``ARC_FRACTION`` of them
STEP_RESOLUTION = 200
ARC_FRACTION = 0.70
#: the turn is held within this (rad) before a controller sees it
TURN_BOUND = 0.5


def step_radians(joint: Joint, resolution: int = STEP_RESOLUTION) -> float:
    """One step of a joint, rad (one index of the model's ring for that joint)."""

    if joint.circular:
        return 2.0 * np.pi / int(resolution)
    return (float(joint.upper) - float(joint.lower)) / float(round(ARC_FRACTION * int(resolution)))


def step_caps(chain: Chain, design: IkDesign, resolution: int = STEP_RESOLUTION) -> np.ndarray:
    """The largest step of every joint per tick, rad: ``lead_indices`` steps."""

    return np.array([design.lead_indices * step_radians(joint, resolution) for joint in chain.joints])


def limit_room(chain: Chain, q: Sequence[float], design: IkDesign, resolution: int = STEP_RESOLUTION) -> np.ndarray:
    """``(n, 2)``: how much of a negative / positive step a joint may take: 1 away from its limits, falling to 0 over
    ``limit_taper_indices`` steps before the limit it would move toward."""

    out = np.ones((chain.n, 2))
    for k, joint in enumerate(chain.joints):
        if joint.circular:
            continue
        taper = max(design.limit_taper_indices * step_radians(joint, resolution), 1e-9)
        out[k, 0] = np.clip((float(q[k]) - joint.lower) / taper, 0.0, 1.0)
        out[k, 1] = np.clip((joint.upper - float(q[k])) / taper, 0.0, 1.0)
    return out


def faded(share: float, cap: float, room_minus: float, room_plus: float, fade: str = "subtract") -> float:
    """A step near a joint limit.  ``scale``: the step times the room.  ``subtract``: the step less (1 − room) caps,
    never below nothing (what inhibition from the joint ring does to a share population)."""

    room = float(room_plus if share > 0 else room_minus)
    if fade == "scale":
        return float(share) * room
    return float(np.sign(share) * max(abs(share) - (1.0 - room) * float(cap), 0.0))


def goal_matrix(goal_lpy: Sequence[float]) -> np.ndarray:
    return from_lift_pitch_yaw(*[float(v) for v in goal_lpy])


def turn(rotation: np.ndarray, goal_lpy: Sequence[float]) -> np.ndarray:
    """The comparison of orientations: the turn from ``rotation`` to the goal."""

    return turn_between(rotation, goal_matrix(goal_lpy))


def offset(chain: Chain, q: Sequence[float], goal_position: Sequence[float]) -> np.ndarray:
    """``(p* − p) / reach`` (``goal_position`` in metres, base frame)."""

    return (np.asarray(goal_position, dtype=float) - kinematics.position(chain, q)) / workspace(chain).radius


def bounded(vector: np.ndarray, bound: Optional[float]) -> np.ndarray:
    """Every component held within ±bound (a population that is full); ``None``: as it is."""

    return vector if bound is None else np.clip(vector, -float(bound), float(bound))


def travel(current_lpy: np.ndarray, goal_lpy: Sequence[float], step: float) -> np.ndarray:
    """Move lift, pitch, yaw toward a goal by at most ``step`` each; lift and yaw the short way round."""

    goal = np.asarray(goal_lpy, dtype=float)
    difference = np.array([wrap(goal[0] - current_lpy[0]), goal[1] - current_lpy[1], wrap(goal[2] - current_lpy[2])])
    moved = current_lpy + np.clip(difference, -step, step)
    return np.array([wrap(moved[0]), moved[1], wrap(moved[2])])


def travel_position(current: np.ndarray, goal: Sequence[float], step: float) -> np.ndarray:
    """Move a point toward a goal by at most ``step`` on the straight line between them."""

    difference = np.asarray(goal, dtype=float) - current
    distance = float(np.linalg.norm(difference))
    return current + difference * (min(1.0, float(step) / distance) if distance > 0 else 0.0)


def step_joints(chain: Chain, q: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
    limits = chain.limits()
    moved = q + velocity * dt
    return np.where(chain.circular_mask(), wrap(moved), np.clip(moved, limits[:, 0], limits[:, 1]))


def first_time_within(t: np.ndarray, ok: np.ndarray) -> float:
    """The first time from which ``ok`` stays true to the end (NaN when it is false at the end)."""

    ok = np.asarray(ok, dtype=bool)
    if ok.size == 0 or not ok[-1]:
        return float("nan")
    outside = np.flatnonzero(~ok)
    return float(np.asarray(t)[outside[-1] + 1 if outside.size else 0])


def weighted_jacobian(axes: np.ndarray, move: np.ndarray, turn_: np.ndarray, offset_: np.ndarray, design: IkDesign):
    """The task's Jacobian ``J`` (6 × n) and error ``E`` (6,), weighed as the task asks: ``E = (√α e, √β d)``,
    ``J_k = (√α a_k, √β m_k)`` with α = 1 (0 when orientation is not asked for) and β = w² (0 when position is not)."""

    ra, rb = np.sqrt(design.turn_weight), np.sqrt(design.move_weight)
    jacobian = np.concatenate([ra * np.asarray(axes, dtype=float).T, rb * np.asarray(move, dtype=float).T])
    return jacobian, np.concatenate([ra * np.asarray(turn_, dtype=float), rb * np.asarray(offset_, dtype=float)])


def fade_steps(steps: np.ndarray, caps: np.ndarray, room: Optional[np.ndarray]) -> np.ndarray:
    """Every joint's step scaled by its room toward the limit it moves to (``faded(..., "scale")``)."""

    if room is None:
        return steps
    return np.array([faded(steps[k], caps[k], room[k, 0], room[k, 1], "scale") for k in range(steps.size)])
