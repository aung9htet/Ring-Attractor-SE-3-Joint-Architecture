"""Start poses and goals of the pose inverse-kinematics trials.

A goal is the pose (orientation and position of the end point) of another joint pose of the
same arm, so every goal can be reached.  The seeds are kept apart: the reported trials, the
trials used while gain and balance were set, and the exact-rule study never share a pair.

``spread``: the goal's joint pose is the start pose with every joint moved by up to that many
radians (inside its limits); ``None``: another pose drawn from the whole range of every joint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ring_so3.experiments import POSE_SEED
from ring_so3.kinematics import Chain, lift_pitch_yaw

from .. import kinematics

TEST_SEED = POSE_SEED + 51            # the reported trials
DEVELOPMENT_SEED = POSE_SEED + 61     # setting gain and balance
STUDY_SEED = POSE_SEED + 71           # the exact-rule study


@dataclass(frozen=True)
class PosePair:
    index: int
    start: np.ndarray                 # joint angles at the start
    goal_pose: np.ndarray             # a joint pose that has the goal pose (one of many)
    goal: np.ndarray                  # lift, pitch, yaw of the goal
    goal_position: np.ndarray         # position of the goal, m, base frame


def pairs(chain: Chain, count: int, seed: int = TEST_SEED, spread: Optional[float] = None) -> List[PosePair]:
    key = [int(seed), chain.n, 0 if chain.joints[0].circular else 1] + ([int(round(1000 * spread))] if spread is not None else [])
    rng = np.random.default_rng(key)
    circular = np.array([j.circular for j in chain.joints])
    out = []
    for index in range(int(count)):
        start = chain.random_pose(rng)
        if spread is None:
            goal_pose = chain.random_pose(rng)
        else:
            moved = start + rng.uniform(-float(spread), float(spread), chain.n)
            goal_pose = np.where(circular, np.angle(np.exp(1j * moved)), chain.clip(moved))
        out.append(PosePair(index, start, goal_pose, np.array(lift_pitch_yaw(chain.rotation(goal_pose))),
                            kinematics.position(chain, goal_pose)))
    return out


__all__ = ["DEVELOPMENT_SEED", "PosePair", "STUDY_SEED", "TEST_SEED", "pairs"]
