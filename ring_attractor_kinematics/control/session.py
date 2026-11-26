"""The ring-attractor kinematics model in the shared closed loop (:mod:`reaching_benchmark.closed_loop`): a robot follows
each joint's written goal ring, and the network sees where the robot is.

Every 50 ms tick:

1. the goal written into the network travels toward the final goal (lift, pitch, yaw on the goal rings;
   the position on the goal sheets along a straight line);
2. every joint moves toward the angle its written goal ring held at the end of the last tick, with gain κ;
3. joint angles and written goal go into the network, which runs for one tick.

A trial ends when the written goal has arrived and the hand turned less than ``rest_deg`` and moved less
than ``rest_reach`` within ``settle_s``, or after ``trial_s`` (the shared loop's rule, the same for every model).

A reach can disable one joint part-way (:class:`Disable`, the joint-loss benchmark): the robot's joint jams, or the
joint's motor or allocation neurons are silenced. Everything is undone when the trial ends.

:class:`ForwardSession` steps the forward network alone with the moving robot (held-pose and motion measurements).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reaching_benchmark import closed_loop
from reaching_benchmark.closed_loop import ClosedLoop, Trial
from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.rotations import lift_pitch_yaw, nearest_rotation, orientation_error_deg, rotation_angle, wrap
from reaching_benchmark.plants.light_robot import LightRobot, MovingRobot

from ..config import IkDesign
from ..rules.cascade_rule import left_size

#: the forward network shows the last joint about this late, and each earlier joint 10 ms later still: with the
#: ``lead`` method every joint is written this far ahead
LEAD_BASE_S = 0.14
LEAD_PER_JOINT_S = 0.01


def lead_times(joints: int) -> np.ndarray:
    return LEAD_BASE_S + LEAD_PER_JOINT_S * (int(joints) - 1 - np.arange(int(joints)))


#: how a joint can be disabled part-way through a reach (see :class:`Disable`)
DISABLE_CONDITIONS = ("lock", "motor", "allocation", "allocation_lock")


@dataclass(frozen=True)
class Disable:
    """One joint (counting from 0) disabled from ``onset_s`` into a reach to its end:

    * ``lock`` (actuator lock): the robot's joint jams (:meth:`LightRobot.lock`). The network is untouched, sees the
      joint's true, unmoving angle, and keeps handing it a share;
    * ``motor`` (motor-neuron loss): the joint's gain pair and written goal ring are silenced
      (:meth:`PoseIkNetwork.lesion_joint`). Its target goes missing and the robot holds the joint; the cascade still
      hands it a share;
    * ``allocation`` (allocation-neuron loss): the joint's two share halves are silenced. Its share reads 0, so it holds,
      and the joints after it are handed the whole remainder;
    * ``allocation_lock``: ``allocation`` with the actuator jammed as well.
    """

    onset_s: float
    joint: int
    condition: str

    def __post_init__(self) -> None:
        if self.condition not in DISABLE_CONDITIONS:
            raise ValueError("condition must be one of %r" % (DISABLE_CONDITIONS,))

    @property
    def locks(self) -> bool:
        return self.condition in ("lock", "allocation_lock")

    @property
    def site(self) -> Optional[str]:
        """The network lesion this condition makes, if any."""

        return {"motor": "motor", "allocation": "allocation", "allocation_lock": "allocation"}.get(self.condition)


class PoseIkSession(ClosedLoop):
    """The model's network as the controller of the shared closed loop."""

    def __init__(self, network: Any, q: Optional[Sequence[float]] = None, motor_scale: float = 1.0, robot: Any = None) -> None:
        """``robot``: a :class:`LightRobot` (default) or a ``GazeboRobot``."""

        self.network = network
        chain = network.chain
        robot = robot if robot is not None else LightRobot(chain, q, motor_scale)
        super().__init__(chain, network.ik_design, robot, float(network.options.tick_ms) / 1000.0,
                         int(network.calibrations.ring_size))
        self.follow = True
        #: another way to turn the comparison into joint targets (rules/references.py): ``targets = method(session)``
        self.method: Optional[Callable[["PoseIkSession"], np.ndarray]] = None
        self.method_shares: Optional[np.ndarray] = None
        self.lead = lead_times(self.chain.n) if getattr(network, "method", "matrix") == "lead" else None

    def targets(self) -> np.ndarray:
        """Where every joint is told to go: the angle its written goal ring holds (or what ``method`` says)."""

        if self.method is not None:
            return np.asarray(self.method(self), dtype=float)
        if self.last is None or not self.follow:
            return self.q.copy()
        return np.where(np.isfinite(self.last.joint_goal), self.last.joint_goal, self.q)

    def written(self) -> Optional[np.ndarray]:
        """With the ``lead`` method: the angles written ahead on the joint rings (the commanded velocity times the lead)."""

        if self.lead is None:
            return None
        ahead = self.q + self.lead * np.asarray(getattr(self.robot, "velocity", np.zeros(self.chain.n)), dtype=float)
        return np.where(self.chain.circular_mask(), wrap(ahead), self.chain.clip(ahead))

    def advance(self, ahead: bool = True) -> Any:
        """One network tick.  ``ahead=False`` (the arm moved by hand between trials): no write-ahead — a 3 rad/s move
        would put the written angle half a radian ahead, further than a ring attractor's bump can jump — and, with the
        observer method, no prediction: the observer rings are seeded from the network's own late matrix instead."""

        if getattr(self.network, "method", "matrix") == "observer":
            velocity = np.asarray(getattr(self.robot, "velocity", np.zeros(self.chain.n)), dtype=float)
            return self.network.step_ik(self.q, self.written_lpy, self.written_position, velocity=velocity, seed=not ahead)
        if self.lead is None or not ahead:
            return self.network.step_ik(self.q, self.written_lpy, self.written_position)
        return self.network.step_ik(self.q, self.written_lpy, self.written_position, written=self.written())

    def after_place(self, settle_ticks: int) -> int:
        if hasattr(self.network, "restart_observer"):
            self.network.restart_observer()
            settle_ticks = max(int(settle_ticks), int(self.network.observer_seed_ticks))
        return int(settle_ticks)

    def disable_controller(self, joint: int, site: str) -> Any:
        return self.network.lesion_joint(joint, site)

    def restore_controller(self, lesion: Any) -> None:
        self.network.restore(lesion)

    def record_tick(self, rows: Dict[str, List[Any]], tick: Any, rotation: np.ndarray, position: np.ndarray) -> None:
        def add(key: str, value: Any) -> None:
            rows.setdefault(key, []).append(value)
        add("shares", tick.shares.copy())
        add("lead", tick.lead.copy())
        add("left", left_size(tick.left, self.design))
        add("decoded_error_deg", float(np.degrees(rotation_angle(tick.forward.tool_rotation(), rotation))))
        observer = getattr(tick, "observer", None)
        turn_matrix = nearest_rotation(observer.hand) if observer is not None else tick.forward.tool_rotation()
        add("turn_error_deg", float(np.degrees(rotation_angle(turn_matrix, rotation))))
        if observer is not None:
            add("selected", observer.selected)
            add("fallback", observer.fallback)
        add("decoded_error_reach", float(np.linalg.norm(tick.forward.position - position) / self.radius))
        add("goal_error_reach", float(np.linalg.norm(tick.goal_position - self.written_position) / self.radius))


def joint_loss_fields(trial: Trial, joint: int, onset_s: float) -> Dict[str, Any]:
    """What happened from ``onset_s`` on, with ``joint`` disabled (or, in an intact trial, as the paired reference).

    * the angles at onset, and the hand's errors then;
    * ``disabled_drift_rad``: how far ``joint`` moved after onset (0 for a jammed joint; in an intact trial, what
      it would have had to move);
    * ``remaining_excursion_rad``: the summed motion of every other joint after onset;
    * ``disabled_share_median_rad``: the decoded share of ``joint`` after onset (about 0 only when its share halves
      are silenced)."""

    k = int(joint)
    index, q_onset = trial.onset(onset_s)
    after = np.vstack([q_onset[None, :], trial.q[index:]])
    steps = np.abs(wrap(np.diff(after, axis=0)))
    others = [j for j in range(after.shape[1]) if j != k]
    shares = trial.shares[index:, k]
    out = {"disabled_joint": k, "onset_s": float(onset_s), "q_onset": q_onset.tolist(),
           "error_at_onset_deg": float(trial.error_deg[index - 1]) if index > 0 else trial.start_error_deg,
           "error_at_onset_reach": float(trial.error_reach[index - 1]) if index > 0 else trial.start_error_reach,
           "disabled_drift_rad": float(np.max(np.abs(wrap(after[:, k] - q_onset[k])))),
           "remaining_excursion_rad": float(np.sum(steps[:, others])),
           "disabled_share_median_rad": float(np.median(np.abs(shares))) if shares.size else float("nan")}
    if trial.disabled is not None:
        out.update(disable_condition=trial.disabled["condition"], lesion=trial.disabled["lesion"])
    return out


def trial_row(trial: Trial, design: IkDesign, every: int = 4) -> Dict[str, Any]:
    """A JSON-ready record of one trial: the shared fields (:func:`reaching_benchmark.closed_loop.trial_row`) and the
    network's own (decoded errors, shares, leads, the observer's charts)."""

    extras = trial.extras
    row = closed_loop.trial_row(trial, design, every)
    goal_error = extras["goal_error_reach"]
    finite_goal = np.isfinite(goal_error[-10:])
    lead = extras["lead"]
    row.update({
        "decoded_error_median_deg": float(np.median(extras["decoded_error_deg"])),
        "decoded_error_median_reach": float(np.nanmedian(extras["decoded_error_reach"])),
        "goal_error_final_reach": float(np.nanmean(goal_error[-10:])) if finite_goal.any() else None,
        "largest_lead": float(np.nanmax(np.abs(lead))) if np.any(np.isfinite(lead)) else float("nan"),
        "shares": extras["shares"][::every].tolist(), "lead": lead[::every].tolist(),
    })
    turn_error = extras.get("turn_error_deg")
    if turn_error is not None and len(turn_error):
        speed = np.max(np.abs(np.diff(trial.q, axis=0, prepend=trial.q[:1])), axis=1) / 0.05
        moving = speed > 0.05
        row.update(turn_error_median_deg=float(np.nanmedian(turn_error)),
                   turn_error_moving_deg=float(np.nanmedian(turn_error[moving])) if moving.any() else float("nan"))
    if extras.get("selected") is not None:
        row.update(observer_fallback_share=float(np.mean(extras["fallback"])), observer_switches=int(np.sum(np.diff(extras["selected"]) != 0)))
    return row


class ForwardSession:
    """The forward network and the moving robot stepped together; measures held poses and motion."""

    HOLD_TICKS = 20          # 1 s
    MEASURE_TICKS = 10       # the readout is averaged over the last 0.5 s of the hold
    MAX_MOVE_TICKS = 400

    def __init__(self, network: Any, robot: MovingRobot) -> None:
        self.network = network
        self.robot = robot
        self.dt_s = float(network.options.tick_ms) / 1000.0

    def tick(self) -> Any:
        self.robot.step(self.dt_s)
        return self.network.step(self.robot.q)

    def warm_up(self, ticks: int = 20) -> None:
        for _ in range(int(ticks)):
            self.network.step(self.robot.q)

    def go_to(self, q: Sequence[float]) -> Dict[str, Any]:
        """Move to ``q``, hold it, and compare the readout averaged over the end of the hold with the truth."""

        self.robot.set_target(q)
        moved = 0
        while not self.robot.arrived() and moved < self.MAX_MOVE_TICKS:
            self.tick()
            moved += 1
        ticks = [self.tick() for _ in range(self.HOLD_TICKS)][-self.MEASURE_TICKS:]
        return held_pose_row(self.network, self.robot.q.copy(), ticks)


def held_pose_row(network: Any, q: np.ndarray, ticks: List[Any]) -> Dict[str, Any]:
    chain, radius = network.chain, network.space.radius
    true_rotation, true_position = chain.rotation(q), kinematics.position(chain, q)
    tool = np.mean([t.tool for t in ticks], axis=0)
    links = np.mean([t.links for t in ticks], axis=0)
    counts = np.sum([t.sheet_counts for t in ticks], axis=0)
    position, disagreement = network.position_from_counts(counts)
    cells = np.mean([t.cells_position for t in ticks], axis=0)
    true_links = chain.link_rotations(q)
    lift, pitch, yaw = lift_pitch_yaw(true_rotation)
    angles = ticks[-1].angles
    return {
        "q": q.tolist(),
        "orientation_error_deg": orientation_error_deg(tool, true_rotation),
        "link_error_deg": [orientation_error_deg(links[k], true_links[k]) for k in range(chain.n)],
        "position_error_reach": float(np.linalg.norm(position - true_position) / radius),
        "cells_error_reach": float(np.linalg.norm(cells - true_position) / radius),
        "disagreement_reach": (np.asarray(disagreement) / radius).tolist(),
        "readout_ring_error_deg": [float(np.degrees(wrap(a - b))) if np.isfinite(a) else float("nan")
                                   for a, b in zip(angles, (lift, pitch, yaw))],
        "pitch_deg": float(np.degrees(pitch)),
        "sheet_spikes": counts.sum(axis=1).tolist(),
    }
