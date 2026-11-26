"""The closed loop every controller is measured in: one task, one plant, one way to end and score a trial.

Every 50 ms tick:

1. the written goal travels toward the final goal (lift, pitch, yaw at ``goal_speed``; position along a straight line
   at ``position_speed`` reaches/s): every controller is given this same travelling goal;
2. the controller says where every joint should go (:meth:`ClosedLoop.targets`), and the plant moves every joint
   toward its target as if to arrive after 1/κ (the light robot), or sends the targets to the arm's trajectory
   controller for one tick of physics (Gazebo);
3. the controller sees the plant's new joint angles (:meth:`ClosedLoop.advance`: a spiking network runs one tick
   here; a controller that reads the joints directly does nothing).

A trial ends when the written goal has arrived and the hand turned less than ``rest_deg`` and moved less than
``rest_reach`` within ``settle_s``, or after ``trial_s``. It counts as reached when, at its last sample, the hand is
within ``reach_deg`` of the goal orientation and ``reach_reach`` reaches of the goal position (only what the task asks).

A controller is a subclass that overrides :meth:`ClosedLoop.targets` and, if it keeps state, the other hooks
(:meth:`advance`, :meth:`after_place`, :meth:`record_tick`, :meth:`disable_controller`, :meth:`restore_controller`).
:class:`RuleLoop` is the controller that reads the measured joint angles and the true geometry every tick and turns
a rule's joint steps into targets (DLS IK, the cascade rule without spikes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import kinematics
from .geometry.chains import Chain, workspace
from .geometry.rotations import lift_pitch_yaw, rotation_angle, wrap
from .limits import (STEP_RESOLUTION, TURN_BOUND, bounded, first_time_within, goal_matrix, limit_room, offset, step_caps,
                     travel, travel_position, turn)
from .task import IkDesign

TICK_S = 0.05


@dataclass
class Trial:
    t: np.ndarray                    # (ticks,) s
    q: np.ndarray                    # (ticks, n)
    error_deg: np.ndarray            # true orientation against the final goal
    error_reach: np.ndarray          # true position against the final goal, reaches
    written: np.ndarray              # (ticks, 6) the written goal: lift, pitch, yaw, position (m)
    start_error_deg: float
    start_error_reach: float
    path_deg: float
    path_reach: float
    settled_s: Optional[float]
    at_limit: bool
    lag: Optional[np.ndarray] = None  # Gazebo: how far each joint was from its target, per tick (rad)
    #: a disabled joint: joint, condition, onset_s, onset_index (the first row it was disabled in), q_onset (the angles
    #: then) and the lesion's description
    disabled: Optional[Dict[str, Any]] = None
    #: per-tick records only some controllers have (a spiking network's decoded shares, errors, ...), by name
    extras: Dict[str, np.ndarray] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name != "extras" and name in self.__dict__.get("extras", {}):
            return self.__dict__["extras"][name]
        raise AttributeError(name)

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
        """The time from which the hand stays within the marks (NaN: not within them at the end)."""

        return first_time_within(self.t, self.within(design))

    def onset(self, onset_s: float) -> Tuple[int, np.ndarray]:
        """The first row recorded after ``onset_s``, and the angles just before it. In a disabled trial these are where
        the joint was frozen; in an intact one, where it would have been."""

        if self.disabled is not None:
            return int(self.disabled["onset_index"]), np.asarray(self.disabled["q_onset"], dtype=float)
        index = int(np.searchsorted(self.t, float(onset_s) + 1e-9, side="left"))
        index = min(max(index, 0), len(self.t))
        return index, (self.q[index - 1] if index > 0 else self.q[0]).copy()


class ClosedLoop:
    """The task loop around one controller and one plant (``robot``: a light robot or the TIAGo in Gazebo)."""

    def __init__(self, chain: Chain, design: IkDesign, robot: Any, dt: float = TICK_S, resolution: int = STEP_RESOLUTION) -> None:
        self.chain = chain
        self.design = design
        self.dt = float(dt)
        #: the unit of the step caps and limit tapers (:mod:`reaching_benchmark.limits`)
        self.ring_size = int(resolution)
        self.radius = float(workspace(chain).radius)
        self.robot = robot
        self.written_lpy = np.array(lift_pitch_yaw(self.chain.rotation(self.q)))
        self.written_position = kinematics.position(self.chain, self.q)
        self.goal_lpy = self.written_lpy.copy()
        self.goal_position = self.written_position.copy()
        self.last: Any = None
        #: what a disabled joint changed in the running trial, undone when it ends
        self._locked = False
        self._lesion: Any = None

    # ---- the plant and the goal ----------------------------------------------------------------------------------
    @property
    def q(self) -> np.ndarray:
        return self.robot.q

    @q.setter
    def q(self, value: Sequence[float]) -> None:
        self.robot.reset(value)

    def arrived(self) -> bool:
        """Has the written goal arrived at the final one?"""

        return bool(np.all(np.abs(wrap(self.written_lpy - self.goal_lpy)) < 1e-6)
                    and np.linalg.norm(self.written_position - self.goal_position) < 1e-9)

    def give(self, goal_lpy: Sequence[float], goal_position: Sequence[float]) -> None:
        self.goal_lpy = np.asarray(goal_lpy, dtype=float).copy()
        self.goal_position = np.asarray(goal_position, dtype=float).copy()

    def _travel(self) -> None:
        self.written_lpy = travel(self.written_lpy, self.goal_lpy, self.design.goal_speed * self.dt)
        self.written_position = travel_position(self.written_position, self.goal_position, self.design.position_speed * self.radius * self.dt)

    # ---- the controller's hooks --------------------------------------------------------------------------------
    def targets(self) -> np.ndarray:
        """Where every joint is told to go this tick."""

        raise NotImplementedError

    def advance(self, ahead: bool = True) -> Any:
        """After the plant moved: the controller sees the new joint angles (a network tick). ``ahead=False``: the arm was
        moved by hand (placing). Returns what the controller reports for the tick, kept as ``last``."""

        return None

    def after_place(self, settle_ticks: int) -> int:
        """After the arm was placed by hand, before it settles: how many ticks to settle."""

        return int(settle_ticks)

    def disable_controller(self, joint: int, site: str) -> Any:
        """Silence part of the controller for one joint (a lesion); returns a handle with ``describe()``."""

        raise ValueError("this controller has no %s neurons to silence" % site)

    def restore_controller(self, lesion: Any) -> None:
        pass

    def record_tick(self, rows: Dict[str, List[Any]], tick: Any, rotation: np.ndarray, position: np.ndarray) -> None:
        """Add this controller's own per-tick records to ``rows`` (kept in ``Trial.extras``)."""

    # ---- the loop ----------------------------------------------------------------------------------------------
    def tick(self) -> Any:
        self._travel()
        self.robot.follow(self.targets(), 1.0 / self.design.kappa, self.dt)
        self.last = self.advance()
        return self.last

    def warm_up(self, ticks: int = 20) -> None:
        for _ in range(int(ticks)):
            self.hold()

    def hold(self, move: bool = True) -> Any:
        """One tick with the goal kept on the hand.  ``move=False``: the robot has already made its step."""

        if move:
            self.robot.follow(self.q.copy(), self.dt, self.dt, exact=True)
        self.give(lift_pitch_yaw(self.chain.rotation(self.q)), kinematics.position(self.chain, self.q))
        self._travel()
        self.last = self.advance(ahead=False)
        return self.last

    def place(self, q: Sequence[float], speed: Optional[float] = None, settle_ticks: int = 12) -> None:
        """Move the arm by hand to ``q``, let the written goal arrive at the hand, let the controller settle."""

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
        for _ in range(self.after_place(settle_ticks)):
            self.hold()

    def reach(self, goal_lpy: Sequence[float], goal_position: Sequence[float], push: Optional[Tuple[float, int, float]] = None,
              goal_change: Optional[Tuple[float, Sequence[float], Sequence[float]]] = None, stop_when_settled: bool = True,
              on_tick: Optional[Callable[[Any], None]] = None, disable: Any = None) -> Trial:
        """From where the arm is to the pose ``goal_lpy`` (lift, pitch, yaw), ``goal_position`` (m).
        ``push``: ``(time, joint, angle)``; ``goal_change``: ``(time, new lift-pitch-yaw, new position)``;
        ``disable``: a joint disabled from a time on (``onset_s``, ``joint``, ``condition``, ``locks``, ``site``),
        undone when the trial ends."""

        try:
            return self._reach(goal_lpy, goal_position, push, goal_change, stop_when_settled, on_tick, disable)
        finally:
            self._undo_disable()

    def _undo_disable(self) -> None:
        """Unjam the robot and give a lesion's neurons back (after every trial, so the next one starts intact)."""

        if self._locked:
            self.robot.unlock()
            self._locked = False
        if self._lesion is not None:
            self.restore_controller(self._lesion)
            self._lesion = None

    def _reach(self, goal_lpy: Sequence[float], goal_position: Sequence[float], push: Optional[Tuple[float, int, float]],
               goal_change: Optional[Tuple[float, Sequence[float], Sequence[float]]], stop_when_settled: bool,
               on_tick: Optional[Callable[[Any], None]], disable: Any) -> Trial:
        design, chain, radius = self.design, self.chain, self.radius
        self.give(goal_lpy, goal_position)
        final = goal_matrix(self.goal_lpy)
        previous_rotation, previous_position = chain.rotation(self.q), kinematics.position(chain, self.q)
        start_deg = float(np.degrees(rotation_angle(previous_rotation, final)))
        start_reach = float(np.linalg.norm(self.goal_position - previous_position) / radius)
        rows: Dict[str, List[Any]] = {key: [] for key in ("t", "q", "deg", "reach", "written", "lag")}
        extra: Dict[str, List[Any]] = {}
        path_deg, path_reach, quiet, settled = 0.0, 0.0, 0, None
        recent: List[Tuple[np.ndarray, np.ndarray]] = []
        window = int(round(design.settle_s / self.dt))
        pending, disabled = disable, None
        for step in range(int(round(design.trial_s / self.dt))):
            now = step * self.dt
            if goal_change is not None and now >= goal_change[0]:
                self.give(goal_change[1], goal_change[2])
                final, goal_change = goal_matrix(self.goal_lpy), None
            if push is not None and now >= push[0]:
                self.robot.displace(int(push[1]), float(push[2]))
                push = None
            if pending is not None and now >= pending.onset_s - 1e-9:
                disabled = {"joint": int(pending.joint), "condition": pending.condition, "onset_s": now,
                            "onset_index": len(rows["t"]), "q_onset": self.q.copy(), "lesion": None}
                if pending.locks:
                    self.robot.lock(pending.joint)
                    self._locked = True
                if pending.site is not None:
                    self._lesion = self.disable_controller(pending.joint, pending.site)
                    disabled["lesion"] = self._lesion.describe()
                pending = None
            tick = self.tick()
            if on_tick is not None:
                on_tick(tick)
            rotation, position = chain.rotation(self.q), kinematics.position(chain, self.q)
            path_deg += float(np.degrees(rotation_angle(previous_rotation, rotation)))
            path_reach += float(np.linalg.norm(position - previous_position) / radius)
            previous_rotation, previous_position = rotation, position
            rows["t"].append(now + self.dt)
            rows["q"].append(self.q.copy())
            rows["deg"].append(float(np.degrees(rotation_angle(rotation, final))))
            rows["reach"].append(float(np.linalg.norm(self.goal_position - position) / radius))
            rows["written"].append(np.concatenate([self.written_lpy, self.written_position]))
            self.record_tick(extra, tick, rotation, position)
            if hasattr(self.robot, "lag"):
                rows["lag"].append(np.array(self.robot.lag, dtype=float))
            quiet = quiet + 1 if (self.arrived() and goal_change is None and push is None and pending is None) else 0
            recent = (recent + [(rotation, position)])[-(window + 1):]
            at_rest = (len(recent) > window and np.degrees(rotation_angle(recent[0][0], rotation)) < design.rest_deg
                       and np.linalg.norm(recent[0][1] - position) / radius < design.rest_reach)
            if stop_when_settled and quiet >= window and at_rest:
                settled = now + self.dt
                break
        at_limit = bool(np.any(limit_room(chain, self.q, design, self.ring_size) < 0.5))
        return Trial(np.array(rows["t"]), np.array(rows["q"]), np.array(rows["deg"]), np.array(rows["reach"]),
                     np.array(rows["written"]), start_deg, start_reach, path_deg, path_reach, settled, at_limit,
                     np.array(rows["lag"]) if rows["lag"] else None, disabled=disabled,
                     extras={key: np.array(value) for key, value in extra.items() if value})


def trial_row(trial: Trial, design: IkDesign, every: int = 4) -> Dict[str, Any]:
    """A JSON-ready record of one trial, the part every controller has (time courses thinned to every ``every``-th tick)."""

    row: Dict[str, Any] = {
        "start_error_deg": trial.start_error_deg, "start_error_reach": trial.start_error_reach,
        "final_error_deg": trial.final_error_deg, "final_error_reach": trial.final_error_reach,
        "reached": trial.reached(design), "time_to_reach_s": trial.time_to(design),
        "path_deg": trial.path_deg, "path_reach": trial.path_reach,
        "path_ratio": trial.path_reach / trial.start_error_reach if trial.start_error_reach > 1e-9 else float("nan"),
        "turn_ratio": trial.path_deg / trial.start_error_deg if trial.start_error_deg > 1e-9 else float("nan"),
        "settled_s": trial.settled_s, "duration_s": float(trial.t[-1]), "at_limit": bool(trial.at_limit),
        "final_q": trial.q[-1].tolist(),
        "jitter_rad_s": float(np.mean(np.abs(np.diff(trial.q, n=2, axis=0)))) / 0.05 if len(trial.q) > 2 else float("nan"),
        "t": trial.t[::every].tolist(), "error_deg": trial.error_deg[::every].tolist(), "error_reach": trial.error_reach[::every].tolist(),
        "q": trial.q[::every].tolist(),
    }
    if trial.lag is not None:
        row.update(lag_median_rad=float(np.median(np.max(np.abs(trial.lag), axis=1))))
    return row


def rule_targets(loop: ClosedLoop, shares: Callable[..., np.ndarray], gain: float) -> np.ndarray:
    """Joint targets from a rule that reads the measured joint angles and the true geometry: the turn and offset to the
    written goal (bounded as for every controller), the joints' axes and moves, the step caps and limit room go in;
    the rule's joint steps come out; the targets are the angles plus the steps times ``gain`` / κ."""

    chain, design, q = loop.chain, loop.design, loop.q
    turn_ = bounded(turn(chain.rotation(q), loop.written_lpy), TURN_BOUND) if design.orientation else np.zeros(3)
    away = bounded(offset(chain, q, loop.written_position), 1.0 / design.position_scale) if design.position else np.zeros(3)
    caps = step_caps(chain, design, loop.ring_size)
    room = limit_room(chain, q, design, loop.ring_size)
    axes, move = kinematics.joint_axes(chain, q), kinematics.moves(chain, q)
    loop.method_shares = np.asarray(shares(axes, move, turn_, away, design, caps, room), dtype=float)
    return q + loop.method_shares * gain / design.kappa


class RuleLoop(ClosedLoop):
    """A controller with no state of its own: every tick, a rule turns the measured joint angles and the true geometry
    into joint steps (``shares(axes, move, turn, offset, design, caps, room) -> steps``), sent as targets with ``gain``."""

    def __init__(self, chain: Chain, design: IkDesign, robot: Any, shares: Callable[..., np.ndarray], gain: float = 1.0,
                 dt: float = TICK_S) -> None:
        super().__init__(chain, design, robot, dt)
        self.shares = shares
        self.gain = float(gain)
        self.method_shares: Optional[np.ndarray] = None

    def targets(self) -> np.ndarray:
        return rule_targets(self, self.shares, self.gain)

    def record_tick(self, rows: Dict[str, List[Any]], tick: Any, rotation: np.ndarray, position: np.ndarray) -> None:
        rows.setdefault("shares", []).append(np.array(self.method_shares, dtype=float).copy())
