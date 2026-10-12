"""The settings of the pose inverse kinematics (one place for the exact rule and the spiking network).

The orientation half is the layer of :mod:`ring_so3.ik`; a :class:`PoseIkDesign` carries the
settings of that layer under the same names, so its rules take either design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

TASKS = ("pose", "position", "orientation")
POSITION_GOALS = ("sheets", "cells")
NORMALISERS = ("mean", "bound", "exact", "none")
GOAL_RINGS = ("written", "none")


@dataclass(frozen=True)
class PoseIkDesign:
    #: what the hand is asked for: ``pose`` (orientation and position), or one of the two alone
    task: str = "pose"
    #: where the position is compared.  ``sheets``: the goal is a packet on three goal sheets, and the ramp
    #: readouts of goal sheets and forward sheets are compared (the ramp comparator of the SO-2 repo, in a plane).
    #: ``cells``: the goal is three numbers, compared with the position cells (no sheet in the loop)
    position_goal: str = "sheets"
    #: True: joint k takes its share of what the joints before it left.  False: every joint gets the whole error
    cascade: bool = True
    #: how much a position error weighs beside a turn: an error of one reach counts as this many radians
    balance: float = 3.0
    #: what a joint's share is divided by (``m_k``: how far joint k moves the end point, in reaches per radian).
    #: ``exact``: 1 + balance² · |m_k|², the share that removes most of what is left; it needs a division.
    #: ``mean``: the same with |m_k|² replaced by its mean over the joint's poses, a constant of the chain.
    #: ``bound``: with the largest value |m_k|² can take, (lever / reach)².  ``none``: 1
    normaliser: str = "mean"
    #: loop gain, 1/s: a joint moves at ``kappa`` times its share (the light robot follows its goal ring with this gain)
    kappa: float = 1.0
    #: the largest lead of a joint's goal bump over its state bump, in ring indices: caps the share a joint takes
    lead_indices: float = 8.0
    #: ring indices before a joint limit over which the share that points beyond the limit fades to nothing
    limit_taper_indices: float = 5.0
    limit_fade: str = "subtract"
    #: the goal travels to a new goal at up to these speeds: lift, pitch and yaw on the goal rings (rad/s), and
    #: the position on the goal sheets (reaches per second, on a straight line)
    goal_speed: float = 1.0
    position_speed: float = 0.3
    #: the position error is held as this times (goal - hand) / reach: an error of 1 / position_scale fills a population
    position_scale: float = 4.0
    #: each joint's goal ring: "written" (a gain pair gated by the state bump writes beside it) or "none"
    goal_ring: str = "written"
    #: ticks of goal-ring spikes summed for one reading of where a joint's goal bump is
    goal_window: int = 3
    #: the comparison of orientations (the SO(3) layer has a second one, ``angles``, which is not used here)
    comparison: str = "matrix"
    #: a trial ends when the goal has arrived and the hand has moved less than ``rest_deg`` and ``rest_reach`` (of
    #: the reach) within ``settle_s``, or after ``trial_s``.  The exact rule ends when all shares are below ``quiet_share``
    quiet_share: float = 0.005
    settle_s: float = 2.0
    trial_s: float = 25.0
    rest_deg: float = 0.5
    rest_reach: float = 0.004
    #: a goal counts as reached when, at the end of the trial, the true orientation is within ``reach_deg`` of it
    #: and the true position within ``reach_reach`` of the reach (each only where it is part of the task)
    reach_deg: float = 5.0
    reach_reach: float = 0.04

    def __post_init__(self) -> None:
        for value, allowed, name in ((self.task, TASKS, "task"), (self.position_goal, POSITION_GOALS, "position_goal"),
                                     (self.normaliser, NORMALISERS, "normaliser"), (self.goal_ring, GOAL_RINGS, "goal_ring")):
            if value not in allowed:
                raise ValueError("%s must be one of %r" % (name, allowed))
        if self.comparison != "matrix":
            raise ValueError("the pose layer compares orientations as matrices")

    @property
    def orientation(self) -> bool:
        return self.task != "position"

    @property
    def position(self) -> bool:
        return self.task != "orientation"

    @property
    def turn_weight(self) -> float:
        """Weight of a joint's turning axis in its share: 1, or 0 when orientation is not asked for."""

        return 1.0 if self.orientation else 0.0

    @property
    def move_weight(self) -> float:
        """Weight of the direction in which a joint moves the hand: balance², or 0 when position is not asked for."""

        return float(self.balance) ** 2 if self.position else 0.0

    def describe(self) -> Dict[str, object]:
        return asdict(self)


__all__ = ["GOAL_RINGS", "NORMALISERS", "POSITION_GOALS", "PoseIkDesign", "TASKS"]
