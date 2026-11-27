"""The light robots: joint angles that move as told, with no dynamics (the Gazebo robot is in gazebo_robot.py).

Every robot has the same few calls, so the closed loop does not care which one runs:

* ``q``: the joint angles now;
* ``follow(targets, time_s, dt)``: every joint moves toward its target as if it should arrive after ``time_s``,
  for one tick of ``dt``.  With a joint's goal-ring angle as target and ``time_s = 1/κ`` this is "the joint
  follows its goal ring with gain κ";
* ``reset(q)``: put the arm at ``q``; ``displace(joint, angle)``: push one joint;
* ``lock(joint)`` / ``unlock()``: a jammed actuator (the joint-loss benchmark).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set

import numpy as np

from reaching_benchmark.geometry.chains import Chain
from reaching_benchmark.geometry.rotations import wrap

from ..limits import step_joints

#: the moving robot of the forward measurements: speed = APPROACH_GAIN × remaining distance, up to a limit
APPROACH_GAIN = 6.0
ARRIVED_RAD = 2.0e-3


class LightRobot:
    """A joint moves exactly as told; ``motor_scale``: it moves this much of what it is told."""

    name = "light"
    tolerance = 1e-9          # how close to a pose counts as there, rad
    place_speed = 3.0         # how fast the arm is moved by hand between trials, rad/s

    def __init__(self, chain: Chain, q: Optional[Sequence[float]] = None, motor_scale: float = 1.0) -> None:
        self.chain = chain
        self.circular = chain.circular_mask()
        self.q = np.array(q, dtype=float) if q is not None else chain.start_pose()
        self.motor_scale = float(motor_scale)
        self.target = self.q.copy()
        self.velocity = np.zeros(chain.n)
        #: joints whose actuator is jammed: they ignore what they are told (see :meth:`lock`)
        self.locked: Set[int] = set()

    def difference(self, target: Sequence[float]) -> np.ndarray:
        d = np.asarray(target, dtype=float) - self.q
        return np.where(self.circular, wrap(d), d)

    def follow(self, targets: Sequence[float], time_s: float, dt: float, exact: bool = False) -> np.ndarray:
        difference = self.difference(targets)
        difference = np.where(np.isfinite(difference), difference, 0.0)          # a silent goal ring: stay
        self.target = self.q + difference
        velocity = (1.0 if exact else self.motor_scale) * (difference / max(float(time_s), 1e-9))
        if self.locked and not exact:                                            # a jammed joint does not move
            velocity = velocity.copy()
            velocity[sorted(self.locked)] = 0.0
        self.velocity = velocity
        self.q = step_joints(self.chain, self.q, self.velocity, dt)
        return self.q

    @property
    def measured_velocity(self) -> np.ndarray:
        """How fast every joint moved in the last tick, rad/s (what a joint encoder would report)."""

        return np.array(self.velocity, dtype=float)

    def lock(self, joint: int) -> None:
        """Jam a joint's actuator. From now on it stays where it is, whatever its target, and ``velocity`` reports
        that. Only the closed loop's own steps obey the lock; moving the arm by hand (``exact``) still moves the joint."""

        self.locked.add(int(joint))

    def unlock(self) -> None:
        self.locked.clear()

    def reset(self, q: Sequence[float]) -> None:
        self.q = step_joints(self.chain, np.asarray(q, dtype=float), np.zeros(self.chain.n), 0.0)
        self.target = self.q.copy()
        self.velocity = np.zeros(self.chain.n)

    def displace(self, joint: int, angle: float) -> None:
        moved = self.q.copy()
        moved[int(joint)] += float(angle)
        self.reset(moved)

    def describe(self) -> dict:
        return {"robot": self.name, "motor_scale": self.motor_scale, "locked": sorted(self.locked)}


class MovingRobot:
    """The robot of the forward measurements: every joint approaches its target at a speed proportional to the
    remaining distance, up to ``max_speed``; joints on the whole circle take the short way round."""

    def __init__(self, chain: Chain, q: Optional[Sequence[float]] = None, max_speed: float = 1.0) -> None:
        self.chain = chain
        limits = chain.limits()
        self.lower, self.upper = limits[:, 0], limits[:, 1]
        self.circular = chain.circular_mask()
        self.q = self._legal(chain.start_pose() if q is None else np.asarray(q, dtype=float))
        self.target = self.q.copy()
        self.max_speed = float(max_speed)
        self.velocity = np.zeros(chain.n)

    def _legal(self, q: np.ndarray) -> np.ndarray:
        wrapped = np.where(self.circular, np.angle(np.exp(1j * q)), q)
        return np.where(self.circular, wrapped, np.clip(wrapped, self.lower, self.upper))

    def set_target(self, q: Sequence[float]) -> None:
        self.target = self._legal(np.asarray(q, dtype=float).reshape(-1))

    def remaining(self) -> np.ndarray:
        difference = self.target - self.q
        return np.where(self.circular, np.angle(np.exp(1j * difference)), difference)

    def arrived(self) -> bool:
        return bool(np.all(np.abs(self.remaining()) < ARRIVED_RAD))

    def step(self, dt_s: float) -> np.ndarray:
        remaining = self.remaining()
        limit = np.minimum(self.max_speed, np.array([j.max_velocity for j in self.chain.joints]))
        self.velocity = np.clip(APPROACH_GAIN * remaining, -limit, limit)
        move = np.where(np.abs(remaining) < ARRIVED_RAD, remaining, self.velocity * float(dt_s))
        self.q = self._legal(self.q + move)
        return self.q


class GoalSequence:
    """Random joint goals inside the limits, one after another, with a pause at each."""

    def __init__(self, chain: Chain, seed: int = 0, dwell_s: float = 1.0) -> None:
        self.chain = chain
        self.rng = np.random.default_rng(int(seed))
        self.dwell_s = float(dwell_s)
        self._waited = 0.0
        self.goals: List[np.ndarray] = []

    def update(self, robot: MovingRobot, dt_s: float) -> bool:
        """Give the robot a new goal once it has rested at the current one; True when it did."""

        if self.goals and not robot.arrived():
            return False
        if self.goals:
            self._waited += float(dt_s)
            if self._waited < self.dwell_s:
                return False
            self._waited = 0.0
        goal = self.chain.random_pose(self.rng)
        self.goals.append(goal)
        robot.set_target(goal)
        return True
