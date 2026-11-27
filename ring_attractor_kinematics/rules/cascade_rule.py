"""The pose inverse-kinematics rule in exact arithmetic (no neurons): what the spiking network approximates.

**Comparison.**  Goal and current pose become six numbers in the base frame:

* the *turn* ``e = ½ Σ_j r_j × g_j`` (rad): about which axis, and by about how much, the hand still has to turn;
* the *offset* ``d = (p* − p) / reach``: where, and how far, the hand still has to go.

**What a joint does.**  Joint k turns everything after it about its axis ``a_k``.  Per radian the hand turns
by ``a_k`` and its end point moves ``m_k = a_k × l_k / reach``.

**Cascade.**  With balance w (a reach of offset counts as w radians of turn)::

    e_1 = e,  d_1 = d
    t_k = clip( (a_k · e_k + w² m_k · d_k) / N_k , ±cap_k )      (nothing toward a reached joint limit)
    e_(k+1) = e_k − a_k t_k,   d_(k+1) = d_k − m_k t_k

so that together the joints make the turn and the move asked for, less what none could take.  ``N_k`` is
``1 + w²·mean|m_k|²`` over the joint ranges (``mean``): a constant of the arm, so nothing is divided online.

:func:`simulate` closes the loop on the light robot as the spiking network does: the comparison sees the arm
``delay_s`` late, each joint's goal leads its angle by its share, and the joint follows its goal with gain κ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import Chain, lever_lengths, workspace
from reaching_benchmark.geometry.rotations import from_lift_pitch_yaw, lift_pitch_yaw, rotation_angle, turn_between, wrap
from reaching_benchmark.limits import (TURN_BOUND, bounded, faded, first_time_within, goal_matrix, limit_room, offset,  # noqa: F401
                                       step_caps, step_joints, travel, travel_position, turn)

from ..config import IkDesign
from ..neurons.ring_layout import RingMap

#: the largest share of every joint, rad: ``lead_indices`` ring indices of its own ring (the shared step caps)
share_caps = step_caps

#: a share is never divided by less than this
SMALLEST_NORMALISER = 1e-9
#: poses the mean of |m_k|² is taken over, and their seed
LEVER_SAMPLES = 4000
LEVER_SEED = 20261009


@dataclass(frozen=True)
class Levers:
    """Constants of an arm's levers, each (n,)."""

    #: ``ρ_k = L_k / reach``: the longest lever of joint k over the reach
    ratios: np.ndarray
    #: mean of |m_k|² over poses drawn evenly from the joint ranges
    mean_square: np.ndarray


_LEVERS: Dict[Chain, Levers] = {}


def levers(chain: Chain) -> Levers:
    """The lever constants of an arm (worked out once, from its geometry alone)."""

    if chain not in _LEVERS:
        rng = np.random.default_rng([LEVER_SEED, chain.n])
        squares = np.mean([np.sum(kinematics.moves(chain, chain.random_pose(rng)) ** 2, axis=1) for _ in range(LEVER_SAMPLES)], axis=0)
        _LEVERS[chain] = Levers(lever_lengths(chain) / workspace(chain).radius, squares)
    return _LEVERS[chain]








def normalisers(design: IkDesign, lever: Levers, move: Optional[np.ndarray] = None) -> np.ndarray:
    """``N_k``: what each joint's share is divided by."""

    beta = design.move_weight
    if design.normaliser == "none":
        return np.ones(lever.ratios.size)
    if design.normaliser == "bound":
        return 1.0 + beta * lever.ratios ** 2
    if design.normaliser == "mean":
        return 1.0 + beta * lever.mean_square
    if move is None:
        raise ValueError("the exact normaliser needs the joints' moves")
    return 1.0 + beta * np.sum(np.asarray(move, dtype=float) ** 2, axis=1)


def ring_maps(chain: Chain, ring_size: int = 200) -> List[RingMap]:
    return [RingMap.for_joint(joint, int(ring_size)) for joint in chain.joints]








def pose_cascade(axes: np.ndarray, move: np.ndarray, turn_: Sequence[float], offset_: Sequence[float], design: IkDesign,
                 lever: Levers, caps: Optional[Sequence[float]] = None, room: Optional[np.ndarray] = None,
                 noise: float = 0.0, dead_zone: float = 0.0, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Shares ``t`` (n,) and what is left before each joint and after the last, ``rests`` (n + 1, 6): turn, then offset."""

    a, m = np.asarray(axes, dtype=float), np.asarray(move, dtype=float)
    n = a.shape[0]
    alpha, beta = design.turn_weight, design.move_weight
    norms = np.maximum(normalisers(design, lever, m), SMALLEST_NORMALISER)
    e = np.asarray(turn_, dtype=float).copy() if alpha else np.zeros(3)
    d = np.asarray(offset_, dtype=float).copy() if beta else np.zeros(3)
    shares, rests = np.zeros(n), [np.concatenate([e, d])]
    for k in range(n):
        share = float(alpha * (a[k] @ e) + beta * (m[k] @ d)) / float(norms[k])
        if noise:
            share += float((rng or np.random.default_rng(0)).normal(0.0, noise))
        if caps is not None:
            share = float(np.clip(share, -caps[k], caps[k]))
        if room is not None:
            share = faded(share, caps[k] if caps is not None else abs(share), room[k, 0], room[k, 1], design.limit_fade)
        if abs(share) < dead_zone:
            share = 0.0
        shares[k] = share
        if design.cascade:
            if alpha:
                e = e - a[k] * share
            if beta:
                d = d - m[k] * share
        rests.append(np.concatenate([e, d]))
    return shares, np.array(rests)


def left_size(rest: np.ndarray, design: IkDesign) -> float:
    """Size of what is left of turn and offset, the offset weighed with the balance (rad)."""

    rest = np.asarray(rest, dtype=float)
    return float(np.sqrt(design.turn_weight * np.sum(rest[:3] ** 2) + design.move_weight * np.sum(rest[3:] ** 2)))












@dataclass
class Trace:
    """One closed-loop trial of the exact rule."""

    t: np.ndarray                 # (ticks,) s
    q: np.ndarray                 # (ticks, n)
    error_deg: np.ndarray         # (ticks,) true orientation against the final goal
    error_reach: np.ndarray       # (ticks,) true position against the final goal, reaches
    shares: np.ndarray            # (ticks, n) rad
    left: np.ndarray              # (ticks,) what is left after the last joint
    start_error_deg: float
    start_error_reach: float
    path_deg: float
    path_reach: float
    settled_s: Optional[float]
    at_limit: bool
    #: (ticks,) orientation the turn was computed from, against the true one (deg)
    estimate_error_deg: Optional[np.ndarray] = None

    @property
    def final_error_deg(self) -> float:
        return float(self.error_deg[-1])

    @property
    def final_error_reach(self) -> float:
        return float(self.error_reach[-1])

    def within(self, design: IkDesign) -> np.ndarray:
        ok = np.ones(self.t.size, dtype=bool)
        if design.orientation:
            ok &= self.error_deg <= design.reach_deg
        if design.position:
            ok &= self.error_reach <= design.reach_reach
        return ok

    def reached(self, design: IkDesign) -> bool:
        return bool(self.within(design)[-1])

    def time_to(self, design: IkDesign) -> float:
        return first_time_within(self.t, self.within(design))


#: ``rule(axes, move, turn, offset, design, lever, caps, room) -> shares``: another way to hand out turn and offset
Rule = Callable[..., np.ndarray]


def simulate(chain: Chain, q0: Sequence[float], goal_lpy: Sequence[float], goal_position: Sequence[float],
             design: Optional[IkDesign] = None, *, ring_size: int = 200, dt: float = 0.05, delay_s: float = 0.25,
             noise: float = 0.0, dead_zone: float = 0.0, motor_scale: float = 1.0,
             push: Optional[Tuple[float, int, float]] = None,
             goal_change: Optional[Tuple[float, Sequence[float], Sequence[float]]] = None,
             rng: Optional[np.random.Generator] = None, stop_when_settled: bool = True, bound_errors: bool = True,
             rule: Optional[Rule] = None, orientation: Any = None,
             lock: Optional[Tuple[float, int]] = None, informed: bool = False) -> Trace:
    """Close the loop on the light robot.

    ``delay_s``: the comparison sees the arm as it was that long ago.  ``noise``: s.d. added to every share (rad);
    ``dead_zone``: shares below it move nothing.  ``motor_scale``: a joint moves this much of what it is told.
    ``push``: ``(time, joint, angle)``; ``goal_change``: ``(time, new lift-pitch-yaw, new position)``.
    ``rule``: a standard method in place of the cascade (rules/references.py: DLS IK, Jacobian transpose).
    ``orientation``: where the turn's orientation comes from (rules/observer_rule.py): an object with
    ``reset(q)`` and ``observe(seen_q, seen_qdot, q, qdot, dt) -> (pose the forward network shows, R)``;
    ``None``: the delayed exact matrix (the old method, without readout noise).
    ``lock``: ``(time, joint)``, the joint jammed from that time on (the joint-loss benchmark). With ``informed`` the
    rule knows: the joint's room is 0 both ways, so the cascade gives it no share and hands its part of the error to
    the joints after it, and DLS leaves its column out. Without ``informed`` the rule keeps handing it a share.
    """

    design = design or IkDesign()
    rng = rng or np.random.default_rng(0)
    n = chain.n
    radius = workspace(chain).radius
    lever = levers(chain)
    caps = share_caps(chain, design, ring_size)
    q = np.asarray(q0, dtype=float).copy()
    history = [q.copy() for _ in range(int(round(delay_s / dt)) + 1)]
    qdot = np.zeros(n)
    rates = [qdot.copy() for _ in history]
    final_lpy, final_position = np.asarray(goal_lpy, dtype=float), np.asarray(goal_position, dtype=float)
    written_lpy = np.array(lift_pitch_yaw(chain.rotation(q)))
    written_position = kinematics.position(chain, q)
    turn_bound = TURN_BOUND if bound_errors else None
    offset_bound = 1.0 / design.position_scale if bound_errors else None
    if orientation is not None:
        orientation.reset(q)
    rows: List[Tuple[Any, ...]] = []
    previous_rotation, previous_position = chain.rotation(q), kinematics.position(chain, q)
    start_deg = float(np.degrees(rotation_angle(previous_rotation, goal_matrix(final_lpy))))
    start_reach = float(np.linalg.norm(final_position - previous_position) / radius)
    path_deg, path_reach, quiet, settled = 0.0, 0.0, 0, None
    locked: Optional[int] = None
    for step in range(int(round(design.trial_s / dt))):
        now = step * dt
        if goal_change is not None and now >= goal_change[0]:
            final_lpy, final_position = np.asarray(goal_change[1], dtype=float), np.asarray(goal_change[2], dtype=float)
            goal_change = None
        if push is not None and now >= push[0]:
            q = q.copy()
            q[int(push[1])] += float(push[2])
            q = step_joints(chain, q, np.zeros(n), dt)
            push = None
        if lock is not None and locked is None and now >= float(lock[0]) - 1e-9:
            locked = int(lock[1])
        written_lpy = travel(written_lpy, final_lpy, design.goal_speed * dt)
        written_position = travel_position(written_position, final_position, design.position_speed * radius * dt)
        seen = history[0]
        if orientation is None:
            seen_rotation = chain.rotation(seen)
        else:
            seen, seen_rotation = orientation.observe(seen, rates[0], q, qdot, dt)
        axes, move = kinematics.joint_axes(chain, seen), kinematics.moves(chain, seen)
        turn_ = bounded(turn(seen_rotation, written_lpy), turn_bound) if design.orientation else np.zeros(3)
        away = bounded((written_position - kinematics.position(chain, seen)) / radius, offset_bound) if design.position else np.zeros(3)
        room = limit_room(chain, q, design, ring_size)
        if locked is not None and informed:
            room[locked] = 0.0
        if rule is not None:
            shares = np.asarray(rule(axes, move, turn_, away, design, lever, caps, room), dtype=float).copy()
            if noise:
                shares = np.clip(shares + rng.normal(0.0, noise, n), -caps, caps)
            if locked is not None and informed:
                shares[locked] = 0.0
            shares[np.abs(shares) < dead_zone] = 0.0
            rest = np.concatenate([turn_ - axes.T @ shares, away - move.T @ shares])
        else:
            shares, rests = pose_cascade(axes, move, turn_, away, design, lever, caps, room, noise, dead_zone, rng)
            rest = rests[-1]
        before = q
        velocity = motor_scale * design.kappa * shares
        if locked is not None:
            velocity = np.array(velocity, dtype=float)
            velocity[locked] = 0.0
        q = step_joints(chain, q, velocity, dt)
        qdot = wrap(q - before) / dt
        history = history[1:] + [q.copy()]
        rates = rates[1:] + [qdot.copy()]
        rotation, position = chain.rotation(q), kinematics.position(chain, q)
        path_deg += float(np.degrees(rotation_angle(previous_rotation, rotation)))
        path_reach += float(np.linalg.norm(position - previous_position) / radius)
        previous_rotation, previous_position = rotation, position
        rows.append((now + dt, q.copy(), float(np.degrees(rotation_angle(rotation, goal_matrix(final_lpy)))),
                     float(np.linalg.norm(final_position - position) / radius), shares.copy(), left_size(rest, design),
                     float(np.degrees(rotation_angle(seen_rotation, chain.rotation(before))))))
        arrived = bool(np.all(np.abs(wrap(written_lpy - final_lpy)) < 1e-6) and np.linalg.norm(written_position - final_position) < 1e-9)
        quiet = quiet + 1 if (arrived and goal_change is None and push is None and (lock is None or locked is not None)
                              and np.all(np.abs(shares) < design.quiet_share)) else 0
        if stop_when_settled and quiet >= int(round(design.settle_s / dt)):
            settled = now + dt
            break
    at_limit = bool(np.any(limit_room(chain, q, design, ring_size) < 0.5))
    columns = list(zip(*rows))
    return Trace(np.array(columns[0]), np.array(columns[1]), np.array(columns[2]), np.array(columns[3]), np.array(columns[4]),
                 np.array(columns[5]), start_deg, start_reach, path_deg, path_reach, settled, at_limit, np.array(columns[6]))
