"""The pose inverse-kinematics rule in exact arithmetic (no neurons).

**The comparison** turns goal and current pose into six numbers in the base frame:

* the *turn* ``e`` (rad): about which axis, and how far, the hand still has to turn.  This is the
  comparison of :mod:`ring_so3.ik.rules`, ``e = ½ Σ_i r_i × g_i``;
* the *offset* ``d = (p* − p) / reach``: where, and how far, the hand still has to go, in reaches.

**What a joint does to both.**  Joint ``k`` turns everything after it about its axis ``a_k`` (a
unit vector, a column of link ``k``'s orientation).  Per radian the hand turns by ``a_k`` and its
end point goes ``m_k = a_k × l_k / reach`` (the axis crossed with the lever from the joint to the
end point; ``|m_k|`` is at most ``ρ_k = L_k / reach``, the lever's largest length over the reach).

**The cascade** hands both down the arm.  With the balance ``w`` (a reach of offset counts as
``w`` radians of turn)::

    e_1 = e,  d_1 = d
    t_k = clip( (a_k · e_k + w² m_k · d_k) / N_k , ±cap_k )      (nothing toward a joint limit that is reached)
    e_(k+1) = e_k − a_k t_k,   d_(k+1) = d_k − m_k t_k

so that ``Σ_k a_k t_k = e − e_(n+1)`` and ``Σ_k m_k t_k = d − d_(n+1)``: together the joints make
the turn and the move that are asked for, less what none of them could take.  ``N_k`` is what
the share is divided by (:data:`ring_se3.ik.design.NORMALISERS`).  The share that removes most of
what is left is ``N_k = 1 + w² |m_k|²`` (``exact``), which changes with the pose.  ``mean`` puts
the mean of ``|m_k|²`` over the joint ranges in its place and ``bound`` its largest value
``ρ_k²``: constants of the chain, so that nothing has to be divided.

For the task ``position`` the turn is left out of the share (and of what is handed on), for
``orientation`` the offset: the latter is the cascade of the SO-3 repo.

:func:`simulate` closes the loop with the light robot as the spiking network does: each joint's
goal leads its angle by its share, and the joint follows its goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ring_so3.ik import rules as so3
from ring_so3.kinematics import Chain, lift_pitch_yaw, rotation_angle

from .. import kinematics
from ..chains import lever_lengths, workspace
from .design import PoseIkDesign

#: a share is never divided by less than this (a chain whose last joint does not move the end point)
SMALLEST_NORMALISER = 1e-9


@dataclass(frozen=True)
class Levers:
    """What the cascade knows about a chain's levers, each (n,): constants of the chain."""

    #: ``ρ_k = L_k / reach``: the longest lever of joint k over the reach (1 for the first joint)
    ratios: np.ndarray
    #: the mean of ``|m_k|²`` over poses drawn evenly from the joint ranges (at most ``ρ_k²``)
    mean_square: np.ndarray


#: poses the mean of ``|m_k|²`` is taken over, and the seed they are drawn with
LEVER_SAMPLES = 4000
LEVER_SEED = 20261009
_LEVERS: Dict[Chain, Levers] = {}


def lever_ratios(chain: Chain) -> np.ndarray:
    """``ρ_k = L_k / reach``, k = 1 .. n: the longest lever of joint k over the reach (1 for the first joint)."""

    return lever_lengths(chain) / workspace(chain).radius


def levers(chain: Chain) -> Levers:
    """The lever constants of a chain (worked out once per chain, from its geometry alone)."""

    if chain not in _LEVERS:
        rng = np.random.default_rng([LEVER_SEED, chain.n])
        squares = np.mean([np.sum(moves(chain, chain.random_pose(rng)) ** 2, axis=1) for _ in range(LEVER_SAMPLES)], axis=0)
        _LEVERS[chain] = Levers(lever_ratios(chain), squares)
    return _LEVERS[chain]


def moves(chain: Chain, q: Sequence[float]) -> np.ndarray:
    """``m_k = a_k × l_k / reach``, ``(n, 3)``: how far the end point goes, in reaches, per radian of joint k."""

    return kinematics.jacobian(chain, q).T / workspace(chain).radius


def offset(chain: Chain, q: Sequence[float], goal_position: Sequence[float]) -> np.ndarray:
    """``(p* − p) / reach``: where the hand still has to go (``goal_position`` in metres, base frame)."""

    return (np.asarray(goal_position, dtype=float) - kinematics.position(chain, q)) / workspace(chain).radius


def normalisers(design: PoseIkDesign, lever: Levers, move: Optional[np.ndarray] = None) -> np.ndarray:
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


def pose_cascade(axes: np.ndarray, move: np.ndarray, turn: Sequence[float], offset_: Sequence[float], design: PoseIkDesign,
                 lever: Levers, caps: Optional[Sequence[float]] = None, room: Optional[np.ndarray] = None,
                 noise: float = 0.0, dead_zone: float = 0.0, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Shares ``t`` (n,) and what is left before each joint and after the last, ``rests`` (n + 1, 6): turn, then offset.

    ``caps``: the largest share each joint takes.  ``room``: ``(n, 2)`` factors in [0, 1] on a negative and on a
    positive share (0 at a joint limit in that direction).  ``noise``: standard deviation added to every share
    before it is capped; ``dead_zone``: shares below it are nothing (and are not handed on).
    """

    a, m = np.asarray(axes, dtype=float), np.asarray(move, dtype=float)
    n = a.shape[0]
    alpha, beta = design.turn_weight, design.move_weight
    norms = np.maximum(normalisers(design, lever, m), SMALLEST_NORMALISER)
    e = np.asarray(turn, dtype=float).copy() if alpha else np.zeros(3)
    d = np.asarray(offset_, dtype=float).copy() if beta else np.zeros(3)
    shares, rests = np.zeros(n), [np.concatenate([e, d])]
    for k in range(n):
        share = float(alpha * (a[k] @ e) + beta * (m[k] @ d)) / float(norms[k])
        if noise:
            share += float((rng or np.random.default_rng(0)).normal(0.0, noise))
        if caps is not None:
            share = float(np.clip(share, -caps[k], caps[k]))
        if room is not None:
            share = so3.faded(share, caps[k] if caps is not None else abs(share), room[k, 0], room[k, 1], design.limit_fade)
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


def left_size(rest: np.ndarray, design: PoseIkDesign) -> float:
    """Length of what is left of turn and offset, the offset weighed with the balance (rad)."""

    rest = np.asarray(rest, dtype=float)
    return float(np.sqrt(design.turn_weight * np.sum(rest[:3] ** 2) + design.move_weight * np.sum(rest[3:] ** 2)))


def travel_position(current: np.ndarray, goal: Sequence[float], step: float) -> np.ndarray:
    """Move a point toward a goal by at most ``step`` on the straight line between them."""

    difference = np.asarray(goal, dtype=float) - current
    distance = float(np.linalg.norm(difference))
    return current + difference * (min(1.0, float(step) / distance) if distance > 0 else 0.0)


def bounded(vector: np.ndarray, bound: Optional[float]) -> np.ndarray:
    """Every component held within ``±bound`` (a population that is full); ``None``: as it is."""

    return vector if bound is None else np.clip(vector, -float(bound), float(bound))


@dataclass
class Trace:
    """One closed-loop trial of the exact rule."""

    t: np.ndarray                 # (ticks,) s
    q: np.ndarray                 # (ticks, n) joint angles
    error_deg: np.ndarray         # (ticks,) true orientation against the final goal
    error_reach: np.ndarray       # (ticks,) true position against the final goal, in reaches
    shares: np.ndarray            # (ticks, n) rad
    left: np.ndarray              # (ticks,) size of what is left after the last joint (rad; offset weighed with the balance)
    start_error_deg: float
    start_error_reach: float
    path_deg: float               # how far the hand turned in all
    path_reach: float             # how far the end point travelled in all, in reaches
    settled_s: Optional[float]    # when all shares had been quiet for the settle time (None: ran to the end)
    at_limit: bool                # a bounded joint ended at a limit (inside the zone where its share fades)

    @property
    def final_error_deg(self) -> float:
        return float(self.error_deg[-1])

    @property
    def final_error_reach(self) -> float:
        return float(self.error_reach[-1])

    def within(self, design: PoseIkDesign) -> np.ndarray:
        """Per tick: is the hand within the marks of the task?"""

        ok = np.ones(self.t.size, dtype=bool)
        if design.orientation:
            ok &= self.error_deg <= design.reach_deg
        if design.position:
            ok &= self.error_reach <= design.reach_reach
        return ok

    def reached(self, design: PoseIkDesign) -> bool:
        return bool(self.within(design)[-1])

    def time_to(self, design: PoseIkDesign) -> float:
        """The time from which the hand stays within the marks (NaN: it does not at the end)."""

        return first_time_within(self.t, self.within(design))


def first_time_within(t: np.ndarray, ok: np.ndarray) -> float:
    """The first time from which ``ok`` stays true to the end (NaN when it is false at the end)."""

    ok = np.asarray(ok, dtype=bool)
    if ok.size == 0 or not ok[-1]:
        return float("nan")
    outside = np.flatnonzero(~ok)
    return float(np.asarray(t)[outside[-1] + 1 if outside.size else 0])


Rule = Callable[..., np.ndarray]


def simulate(chain: Chain, q0: Sequence[float], goal_lpy: Sequence[float], goal_position: Sequence[float],
             design: Optional[PoseIkDesign] = None, *, ring_size: int = 200, dt: float = 0.05, delay_s: float = 0.25,
             noise: float = 0.0, dead_zone: float = 0.0, readout_bias: Optional[np.ndarray] = None,
             position_bias: Optional[Sequence[float]] = None, motor_scale: float = 1.0,
             push: Optional[Tuple[float, int, float]] = None, goal_change: Optional[Tuple[float, Sequence[float], Sequence[float]]] = None,
             rng: Optional[np.random.Generator] = None, stop_when_settled: bool = True, bound_errors: bool = True,
             rule: Optional[Rule] = None) -> Trace:
    """Close the loop on the light robot.

    ``goal_lpy``: lift, pitch, yaw of the goal; ``goal_position``: its position in metres, base frame.
    ``delay_s``: the comparison sees the arm as it was that long ago (the forward network's delay).
    ``noise``: standard deviation added to every share (rad); ``dead_zone``: shares below it move nothing.
    ``readout_bias``: a fixed 3 x 3 rotation by which the decoded orientation is off; ``position_bias``: a
    fixed vector (in reaches) by which the decoded position is off.  ``motor_scale``: the joint moves this much
    of what it is told.  ``push``: ``(time, joint, angle)`` displaces one joint; ``goal_change``:
    ``(time, new lift-pitch-yaw, new position)``.  ``bound_errors``: turn and offset are held within what their
    populations hold (0.5 rad and ``1 / position_scale`` per component).
    ``rule``: another way to hand out turn and offset (:mod:`ring_se3.standard.ik`):
    ``rule(axes, move, turn, offset, design, lever, caps, room)`` returns every joint's share.
    """

    design = design or PoseIkDesign()
    rng = rng or np.random.default_rng(0)
    n = chain.n
    radius = workspace(chain).radius
    lever = levers(chain)
    caps = so3.share_caps(chain, design, ring_size)
    q = np.asarray(q0, dtype=float).copy()
    history = [q.copy() for _ in range(int(round(delay_s / dt)) + 1)]
    final_lpy, final_position = np.asarray(goal_lpy, dtype=float), np.asarray(goal_position, dtype=float)
    written_lpy = np.array(lift_pitch_yaw(chain.rotation(q)))            # the goal starts where the hand is
    written_position = kinematics.position(chain, q)
    bias = np.eye(3) if readout_bias is None else np.asarray(readout_bias, dtype=float)
    shift = np.zeros(3) if position_bias is None else np.asarray(position_bias, dtype=float)
    turn_bound = 0.5 if bound_errors else None
    offset_bound = 1.0 / design.position_scale if bound_errors else None
    rows: List[Tuple[Any, ...]] = []
    previous_rotation, previous_position = chain.rotation(q), kinematics.position(chain, q)
    start_deg = float(np.degrees(rotation_angle(previous_rotation, so3.goal_matrix(*final_lpy))))
    start_reach = float(np.linalg.norm(final_position - previous_position) / radius)
    path_deg, path_reach, quiet, settled = 0.0, 0.0, 0, None
    for step in range(int(round(design.trial_s / dt))):
        now = step * dt
        if goal_change is not None and now >= goal_change[0]:
            final_lpy, final_position = np.asarray(goal_change[1], dtype=float), np.asarray(goal_change[2], dtype=float)
            goal_change = None
        if push is not None and now >= push[0]:
            q = q.copy()
            q[int(push[1])] += float(push[2])
            q = so3.step_joints(chain, q, np.zeros(n), dt)
            push = None
        written_lpy = so3.travel(written_lpy, final_lpy, design.goal_speed * dt)
        written_position = travel_position(written_position, final_position, design.position_speed * radius * dt)
        seen = history[0]
        axes, move = so3.joint_axes(chain, seen), moves(chain, seen)
        seen_rotation = bias @ chain.rotation(seen)
        turn = bounded(so3.turn(seen_rotation, written_lpy, design.comparison), turn_bound) if design.orientation else np.zeros(3)
        away = bounded((written_position - kinematics.position(chain, seen)) / radius - shift, offset_bound) if design.position else np.zeros(3)
        room = so3.limit_room(chain, q, design, ring_size)
        if rule is not None:
            shares = np.asarray(rule(axes, move, turn, away, design, lever, caps, room), dtype=float).copy()
            if noise:
                shares = np.clip(shares + rng.normal(0.0, noise, n), -caps, caps)
            shares[np.abs(shares) < dead_zone] = 0.0
            rest = np.concatenate([turn - axes.T @ shares, away - move.T @ shares])
        else:
            shares, rests = pose_cascade(axes, move, turn, away, design, lever, caps, room, noise, dead_zone, rng)
            rest = rests[-1]
        q = so3.step_joints(chain, q, motor_scale * design.kappa * shares, dt)
        history = history[1:] + [q.copy()]
        rotation, position = chain.rotation(q), kinematics.position(chain, q)
        path_deg += float(np.degrees(rotation_angle(previous_rotation, rotation)))
        path_reach += float(np.linalg.norm(position - previous_position) / radius)
        previous_rotation, previous_position = rotation, position
        rows.append((now + dt, q.copy(), float(np.degrees(rotation_angle(rotation, so3.goal_matrix(*final_lpy)))),
                     float(np.linalg.norm(final_position - position) / radius), shares.copy(), left_size(rest, design)))
        arrived = bool(np.all(np.abs(so3.wrap(written_lpy - final_lpy)) < 1e-6) and np.linalg.norm(written_position - final_position) < 1e-9)
        quiet = quiet + 1 if (arrived and goal_change is None and push is None and np.all(np.abs(shares) < design.quiet_share)) else 0
        if stop_when_settled and quiet >= int(round(design.settle_s / dt)):
            settled = now + dt
            break
    at_limit = bool(np.any(so3.limit_room(chain, q, design, ring_size) < 0.5))          # a joint ended within reach of a limit
    columns = list(zip(*rows))
    return Trace(np.array(columns[0]), np.array(columns[1]), np.array(columns[2]), np.array(columns[3]), np.array(columns[4]),
                 np.array(columns[5]), start_deg, start_reach, path_deg, path_reach, settled, at_limit)


__all__ = ["LEVER_SAMPLES", "LEVER_SEED", "Levers", "SMALLEST_NORMALISER", "Trace", "bounded", "first_time_within", "left_size", "lever_ratios", "levers",
           "moves", "normalisers", "offset",
           "pose_cascade", "simulate", "travel_position"]
