"""The reaching task every controller is measured on: what is asked, how the written goal travels, the step limits, and
when a trial ends and counts as reached.

:class:`IkDesign` is one record for every model. Some fields describe the task and its evaluation and hold for every
controller (``task``, ``balance``, ``kappa``, ``goal_speed``, ``position_speed``, ``lead_indices`` and
``limit_taper_indices`` through :mod:`reaching_benchmark.limits`, ``position_scale`` through the bounded offset,
``settle_s``, ``trial_s``, ``rest_*``, ``reach_*``); the others are read by the ring-attractor kinematics model only
(``position_goal``, ``cascade``, ``normaliser``, ``limit_fade``, ``goal_window``, ``quiet_share``). Every result file
stores the whole record as its ``design``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

TASKS = ("pose", "position", "orientation")
POSITION_GOALS = ("sheets", "cells")
NORMALISERS = ("mean", "bound", "exact", "none")


@dataclass(frozen=True)
class IkDesign:
    #: what the hand is asked for: ``pose`` (orientation and position), or one of the two alone
    task: str = "pose"
    #: where positions are compared: ``sheets`` (ramp read-outs of goal and forward sheets) or ``cells`` (numbers)
    position_goal: str = "sheets"
    #: True: joint k takes its share of what the joints before it left (the cascade).  False: all get the whole error
    cascade: bool = True
    #: balance w: an error of one reach counts as this many radians of turn
    balance: float = 3.0
    #: what a joint's share is divided by: ``mean`` (1 + w²·mean|m_k|², a constant), ``bound``, ``exact``, ``none``
    normaliser: str = "mean"
    #: loop gain κ, 1/s: a joint moves at κ times its share
    kappa: float = 1.0
    #: the largest lead of a goal bump over its joint bump, ring indices: the cap on a joint's share
    lead_indices: float = 8.0
    #: ring indices before a joint limit over which a share toward it fades to nothing
    limit_taper_indices: float = 5.0
    limit_fade: str = "subtract"
    #: the written goal travels at up to these speeds: lift, pitch, yaw (rad/s); position (reaches/s)
    goal_speed: float = 1.0
    position_speed: float = 0.3
    #: the offset is held as this times (goal − hand) / reach
    position_scale: float = 4.0
    #: ticks of goal-ring spikes summed for one reading of a joint's goal bump
    goal_window: int = 3
    #: a trial ends when the goal has arrived and the hand moved less than rest_deg and rest_reach within settle_s,
    #: or after trial_s; the exact rule ends when every share is below quiet_share
    quiet_share: float = 0.005
    settle_s: float = 2.0
    trial_s: float = 25.0
    rest_deg: float = 0.5
    rest_reach: float = 0.004
    #: a goal is reached when, at the end, orientation is within reach_deg and position within reach_reach of the reach
    reach_deg: float = 5.0
    reach_reach: float = 0.04

    def __post_init__(self) -> None:
        for value, allowed, name in ((self.task, TASKS, "task"), (self.position_goal, POSITION_GOALS, "position_goal"),
                                     (self.normaliser, NORMALISERS, "normaliser")):
            if value not in allowed:
                raise ValueError("%s must be one of %r" % (name, allowed))

    @property
    def orientation(self) -> bool:
        return self.task != "position"

    @property
    def position(self) -> bool:
        return self.task != "orientation"

    @property
    def turn_weight(self) -> float:
        """α: 1, or 0 when orientation is not asked for."""

        return 1.0 if self.orientation else 0.0

    @property
    def move_weight(self) -> float:
        """β = w², or 0 when position is not asked for."""

        return float(self.balance) ** 2 if self.position else 0.0

    def describe(self) -> Dict[str, Any]:
        return asdict(self)


def default_task(joints: int) -> str:
    """A pose has six freedoms: an arm with fewer joints is asked for the position alone."""

    return "pose" if int(joints) >= 6 else "position"
