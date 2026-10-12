"""The closed loop: the light robot follows each joint's goal ring, and the network sees where the robot is.

Every tick:

1. the goal that is written into the network travels toward the final goal: lift, pitch and yaw on
   the goal rings as in the SO-3 repo, the position on the goal sheets along a straight line;
2. every joint moves with a speed of ``kappa`` times the distance from its own angle to the
   angle its goal ring held at the end of the last tick (no dynamics: the light robot);
3. the joint angles and the written goal are handed to the network, which is simulated for one tick.

A trial ends when the written goal has arrived and the hand has turned less than ``rest_deg`` and
moved less than ``rest_reach`` of the reach within the last ``settle_s``, or after ``trial_s``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ring_so3.ik import rules as so3
from ring_so3.kinematics import Chain, lift_pitch_yaw, rotation_angle
from ring_so3.robots import LightRobot

from .. import kinematics
from ..chains import workspace
from . import rules
from .design import PoseIkDesign


@dataclass
class Trial:
    t: np.ndarray                    # (ticks,) s
    q: np.ndarray                    # (ticks, n)
    error_deg: np.ndarray            # true orientation against the final goal
    error_reach: np.ndarray          # true position against the final goal, in reaches
    shares: np.ndarray               # (ticks, n) decoded shares, rad
    lead: np.ndarray                 # (ticks, n) lead of each goal ring over its state ring, indices
    left: np.ndarray                 # (ticks,) size of what the arm could not take (rad; offset weighed with the balance)
    written: np.ndarray              # (ticks, 6) the goal as written: lift, pitch, yaw, then position (m)
    decoded_error_deg: np.ndarray    # decoded orientation against the true one
    decoded_error_reach: np.ndarray  # position read from the forward sheets against the true one, in reaches
    goal_error_reach: np.ndarray     # position the goal sheets hold against the one written on them, in reaches (NaN without)
    start_error_deg: float
    start_error_reach: float
    path_deg: float
    path_reach: float
    settled_s: Optional[float]
    at_limit: bool
    #: Gazebo only: how far every joint was from the target it was sent, per tick, rad
    lag: Optional[np.ndarray] = None

    @property
    def final_error_deg(self) -> float:
        return float(self.error_deg[-1])

    @property
    def final_error_reach(self) -> float:
        return float(self.error_reach[-1])

    def within(self, design: PoseIkDesign) -> np.ndarray:
        ok = np.ones(self.t.size, dtype=bool)
        if design.orientation:
            ok &= self.error_deg <= design.reach_deg
        if design.position:
            ok &= self.error_reach <= design.reach_reach
        return ok

    def reached(self, design: PoseIkDesign) -> bool:
        return bool(self.within(design)[-1])

    def time_to(self, design: PoseIkDesign) -> float:
        """The time from which the hand stays within the marks of the task (NaN: it is not within them at the end)."""

        return rules.first_time_within(self.t, self.within(design))


class PoseIkSession:
    def __init__(self, network: Any, q: Optional[Sequence[float]] = None, motor_scale: float = 1.0, robot: Any = None) -> None:
        """``robot``: a :class:`ring_so3.robots.LightRobot` (the default) or ``GazeboRobot``."""

        self.network = network
        self.chain: Chain = network.chain
        self.design: PoseIkDesign = network.ik_design
        self.dt = float(network.options.tick_ms) / 1000.0
        self.ring_size = int(network.calibration.settings["ring_size"])
        self.radius = float(workspace(self.chain).radius)
        self.robot = robot if robot is not None else LightRobot(self.chain, q, motor_scale)
        self.written_lpy = np.array(lift_pitch_yaw(self.chain.rotation(self.q)))
        self.written_position = kinematics.position(self.chain, self.q)
        self.goal_lpy = self.written_lpy.copy()
        self.goal_position = self.written_position.copy()
        self.follow = True
        self.last: Any = None
        #: another way to turn the comparison into joint targets (a standard method): ``targets = method(session)``
        self.method: Any = None
        self.method_shares: Optional[np.ndarray] = None

    @property
    def q(self) -> np.ndarray:
        return self.robot.q

    @q.setter
    def q(self, value: Sequence[float]) -> None:
        self.robot.reset(value)

    def arrived(self) -> bool:
        """Has the written goal arrived at the final one?"""

        return bool(np.all(np.abs(so3.wrap(self.written_lpy - self.goal_lpy)) < 1e-6)
                    and np.linalg.norm(self.written_position - self.goal_position) < 1e-9)

    def give(self, goal_lpy: Sequence[float], goal_position: Sequence[float]) -> None:
        self.goal_lpy = np.asarray(goal_lpy, dtype=float).copy()
        self.goal_position = np.asarray(goal_position, dtype=float).copy()

    # -- one tick ------------------------------------------------------------
    def targets(self) -> np.ndarray:
        """Where every joint is told to go: the angle its goal ring holds (or what ``method`` says)."""

        if self.method is not None:
            return np.asarray(self.method(self), dtype=float)
        if self.last is None or not self.follow:
            return self.q.copy()
        return np.where(np.isfinite(self.last.joint_goal), self.last.joint_goal, self.q)

    def _travel(self) -> None:
        self.written_lpy = so3.travel(self.written_lpy, self.goal_lpy, self.design.goal_speed * self.dt)
        self.written_position = rules.travel_position(self.written_position, self.goal_position,
                                                      self.design.position_speed * self.radius * self.dt)

    def tick(self):
        self._travel()
        self.robot.follow(self.targets(), 1.0 / self.design.kappa, self.dt)
        self.last = self.network.step_ik(self.q, self.written_lpy, self.written_position)
        return self.last

    def warm_up(self, ticks: int = 20) -> None:
        for _ in range(int(ticks)):
            self.hold()

    def hold(self, move: bool = True):
        """One tick with the goal kept on the hand: nothing is asked of the arm.  The written goal travels to where
        the hand is at its usual speed.  ``move=False``: the robot has already made its step of this tick."""

        if move:
            self.robot.follow(self.q.copy(), self.dt, self.dt, exact=True)
        self.give(lift_pitch_yaw(self.chain.rotation(self.q)), kinematics.position(self.chain, self.q))
        self._travel()
        self.last = self.network.step_ik(self.q, self.written_lpy, self.written_position)
        return self.last

    def place(self, q: Sequence[float], speed: Optional[float] = None, settle_ticks: int = 12) -> None:
        """Move the arm by hand to ``q``, let the written goal arrive at the hand, then let the network settle."""

        target = np.asarray(q, dtype=float)
        speed = float(speed if speed is not None else self.robot.place_speed)
        for _ in range(2400):
            difference = self.robot.difference(target)
            there = np.max(np.abs(difference)) < self.robot.tolerance
            if there and self.arrived():
                break
            if not there:
                self.robot.follow(self.q + np.clip(difference, -speed * self.dt, speed * self.dt), self.dt, self.dt, exact=True)
            else:
                self.robot.follow(target, self.dt, self.dt, exact=True)
            self.hold(move=False)
        for _ in range(int(settle_ticks)):
            self.hold()

    # -- one trial -----------------------------------------------------------
    def reach(self, goal_lpy: Sequence[float], goal_position: Sequence[float], push: Optional[Tuple[float, int, float]] = None,
              goal_change: Optional[Tuple[float, Sequence[float], Sequence[float]]] = None, stop_when_settled: bool = True) -> Trial:
        """From where the arm is to the pose ``goal_lpy`` (lift, pitch, yaw), ``goal_position`` (m).
        ``push``: ``(time, joint, angle)``; ``goal_change``: ``(time, new lift-pitch-yaw, new position)``."""

        design, chain, radius = self.design, self.chain, self.radius
        self.give(goal_lpy, goal_position)
        final = so3.goal_matrix(*self.goal_lpy)
        previous_rotation, previous_position = chain.rotation(self.q), kinematics.position(chain, self.q)
        start_deg = float(np.degrees(rotation_angle(previous_rotation, final)))
        start_reach = float(np.linalg.norm(self.goal_position - previous_position) / radius)
        rows: Dict[str, List[Any]] = {key: [] for key in ("t", "q", "deg", "reach", "shares", "lead", "left", "written", "decoded_deg",
                                                          "decoded_reach", "goal_reach", "lag")}
        path_deg, path_reach, quiet, settled = 0.0, 0.0, 0, None
        recent: List[Tuple[np.ndarray, np.ndarray]] = []
        window = int(round(design.settle_s / self.dt))
        for step in range(int(round(design.trial_s / self.dt))):
            now = step * self.dt
            if goal_change is not None and now >= goal_change[0]:
                self.give(goal_change[1], goal_change[2])
                final, goal_change = so3.goal_matrix(*self.goal_lpy), None
            if push is not None and now >= push[0]:
                self.robot.displace(int(push[1]), float(push[2]))
                push = None
            tick = self.tick()
            rotation, position = chain.rotation(self.q), kinematics.position(chain, self.q)
            path_deg += float(np.degrees(rotation_angle(previous_rotation, rotation)))
            path_reach += float(np.linalg.norm(position - previous_position) / radius)
            previous_rotation, previous_position = rotation, position
            rows["t"].append(now + self.dt)
            rows["q"].append(self.q.copy())
            rows["deg"].append(float(np.degrees(rotation_angle(rotation, final))))
            rows["reach"].append(float(np.linalg.norm(self.goal_position - position) / radius))
            rows["shares"].append(tick.shares.copy())
            rows["lead"].append(tick.lead.copy())
            rows["left"].append(rules.left_size(tick.left, design))
            rows["written"].append(np.concatenate([self.written_lpy, self.written_position]))
            rows["decoded_deg"].append(float(np.degrees(rotation_angle(tick.tick.tool_rotation(), rotation))))
            rows["decoded_reach"].append(float(np.linalg.norm(tick.tick.position - position) / radius))
            rows["goal_reach"].append(float(np.linalg.norm(tick.goal_position - self.written_position) / radius))
            if hasattr(self.robot, "lag"):
                rows["lag"].append(np.array(self.robot.lag, dtype=float))
            quiet = quiet + 1 if (self.arrived() and goal_change is None and push is None) else 0
            recent.append((rotation, position))
            recent = recent[-(window + 1):]
            at_rest = (len(recent) > window and np.degrees(rotation_angle(recent[0][0], rotation)) < design.rest_deg
                       and np.linalg.norm(recent[0][1] - position) / radius < design.rest_reach)
            if stop_when_settled and quiet >= window and at_rest:
                settled = now + self.dt
                break
        at_limit = bool(np.any(so3.limit_room(chain, self.q, design, self.ring_size) < 0.5))
        return Trial(np.array(rows["t"]), np.array(rows["q"]), np.array(rows["deg"]), np.array(rows["reach"]), np.array(rows["shares"]),
                     np.array(rows["lead"]), np.array(rows["left"]), np.array(rows["written"]), np.array(rows["decoded_deg"]),
                     np.array(rows["decoded_reach"]), np.array(rows["goal_reach"]), start_deg, start_reach, path_deg, path_reach, settled, at_limit,
                     np.array(rows["lag"]) if rows["lag"] else None)


def trial_row(trial: Trial, design: PoseIkDesign, every: int = 4) -> Dict[str, Any]:
    """A JSON-ready record of one trial (the time courses thinned to every ``every``-th tick)."""

    row = {
        "start_error_deg": trial.start_error_deg, "start_error_reach": trial.start_error_reach,
        "final_error_deg": trial.final_error_deg, "final_error_reach": trial.final_error_reach,
        "reached": trial.reached(design), "time_to_reach_s": trial.time_to(design),
        "path_deg": trial.path_deg, "path_reach": trial.path_reach,
        "path_ratio": trial.path_reach / trial.start_error_reach if trial.start_error_reach > 1e-9 else float("nan"),
        "turn_ratio": trial.path_deg / trial.start_error_deg if trial.start_error_deg > 1e-9 else float("nan"),
        "settled_s": trial.settled_s, "duration_s": float(trial.t[-1]), "at_limit": bool(trial.at_limit),
        "final_q": trial.q[-1].tolist(),
        "decoded_error_median_deg": float(np.median(trial.decoded_error_deg)),
        "decoded_error_median_reach": float(np.nanmedian(trial.decoded_error_reach)),
        "decoded_error_final_reach": float(np.nanmean(trial.decoded_error_reach[-10:])),
        "goal_error_final_reach": float(np.nanmean(trial.goal_error_reach[-10:])) if np.any(np.isfinite(trial.goal_error_reach[-10:])) else None,
        "largest_lead": float(np.nanmax(np.abs(trial.lead))) if np.any(np.isfinite(trial.lead)) else float("nan"),
        "t": trial.t[::every].tolist(), "error_deg": trial.error_deg[::every].tolist(), "error_reach": trial.error_reach[::every].tolist(),
        "left": trial.left[::every].tolist(), "shares": trial.shares[::every].tolist(), "lead": trial.lead[::every].tolist(),
        "q": trial.q[::every].tolist(),
    }
    if trial.lag is not None:                       # Gazebo: the joints are behind the targets they were sent
        row.update(lag_median_rad=float(np.median(np.max(np.abs(trial.lag), axis=1))), lag_final_rad=[float(v) for v in trial.lag[-1]])
    return row


__all__ = ["PoseIkSession", "Trial", "trial_row"]
