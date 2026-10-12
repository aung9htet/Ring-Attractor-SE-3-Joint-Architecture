"""The Gazebo robot of the pose IK: :class:`ring_so3.robots.GazeboRobot` with another way to tell that the arm is at rest.

Moving the arm to a start pose ends when it is there and still.  The SO-3 class reads "still" from the
joint velocities Gazebo reports.  With the hand and the force sensor mounted, a wrist joint reports
velocities of more than a radian per second while its angle does not change (measured: 1.9 rad/s at
``arm_6_joint`` over 2 s in which the angle moved by 0.002 rad), so that test never passes and every
start pose costs the full 2,000 ticks.  Here "still" is read from the angles themselves.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ring_so3.robots import GazeboRobot

#: the arm is still when no joint has moved by more than this (rad) over this many ticks
STILL_RAD = 0.003
STILL_TICKS = 6


class PoseGazeboRobot(GazeboRobot):
    def reset(self, q: Sequence[float], speed: float = 1.0, dt: float = 0.05, settle_s: float = 1.0) -> None:
        """Move to ``q`` at up to ``speed`` rad/s and wait until the arm is there and has stopped moving."""

        goal = np.clip(np.asarray(q, dtype=float), self.limits[:, 0], self.limits[:, 1])
        self.link.put_base_back()
        recent: List[np.ndarray] = []
        for _ in range(2000):
            difference = goal - self.q
            recent = (recent + [self.q.copy()])[-STILL_TICKS:]
            still = len(recent) == STILL_TICKS and float(np.max(np.abs(recent[-1] - recent[0]))) < STILL_RAD
            if np.max(np.abs(difference)) < self.tolerance and still:
                break
            self.follow(self.q + np.clip(difference, -speed * 4 * dt, speed * 4 * dt), 4 * dt, dt)
        for _ in range(int(round(settle_s / dt))):
            self.follow(goal, 4 * dt, dt)


__all__ = ["PoseGazeboRobot", "STILL_RAD", "STILL_TICKS"]
