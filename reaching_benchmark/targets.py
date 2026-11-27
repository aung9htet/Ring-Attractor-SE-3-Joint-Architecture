"""Start poses and goals of every measurement, with their seeds kept apart.

A goal is the pose of another joint pose of the same arm, so every goal can be reached.  ``spread``: the
goal's joint pose is the start pose with every joint moved by up to that many radians (inside its limits);
``None``: a pose drawn from the whole range.  Calibration never uses any of these seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import Chain
from reaching_benchmark.geometry.rotations import lift_pitch_yaw

POSE_SEED = 20261005                  # held poses of the forward measurements
TEST_SEED = POSE_SEED + 51            # reaching trials
DEVELOPMENT_SEED = POSE_SEED + 61     # setting gain and balance (never reported)
STUDY_SEED = POSE_SEED + 71           # the exact-rule study
SCREEN_SEED = POSE_SEED + 81          # the Stage-2 screening of orientation sources
MOTION_SEED = POSE_SEED + 91          # open-loop motions of the estimation benchmark
TRAIN_SEED = POSE_SEED + 101          # training episodes of the learned baselines (never the reported pairs)
#: goals of the reported trials: within this many radians of the start, joint by joint
SPREAD = 0.8


@dataclass(frozen=True)
class PosePair:
    index: int
    start: np.ndarray                 # joint angles at the start
    goal_pose: np.ndarray             # a joint pose with the goal pose (one of many)
    goal: np.ndarray                  # lift, pitch, yaw of the goal
    goal_position: np.ndarray         # position of the goal, m, base frame


def pose_pairs(chain: Chain, count: int, seed: int = TEST_SEED, spread: Optional[float] = SPREAD) -> List[PosePair]:
    key = [int(seed), chain.n, 0 if chain.joints[0].circular else 1] + ([int(round(1000 * spread))] if spread is not None else [])
    rng = np.random.default_rng(key)
    circular = chain.circular_mask()
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


def pole_pairs(chain: Chain, count: int, seed: int = STUDY_SEED, within_deg: float = 15.0, spread: float = SPREAD,
               tries: int = 200000) -> List[PosePair]:
    """Pairs whose goal has |pitch| within ``within_deg`` of 90° (the lift-pitch-yaw chart's pole)."""

    rng = np.random.default_rng([int(seed), chain.n, 90])
    circular = chain.circular_mask()
    out: List[PosePair] = []
    for _ in range(int(tries)):
        if len(out) >= int(count):
            break
        start = chain.random_pose(rng)
        moved = start + rng.uniform(-float(spread), float(spread), chain.n)
        goal_pose = np.where(circular, np.angle(np.exp(1j * moved)), chain.clip(moved))
        lpy = np.array(lift_pitch_yaw(chain.rotation(goal_pose)))
        if abs(abs(np.degrees(lpy[1])) - 90.0) <= within_deg:
            out.append(PosePair(len(out), start, goal_pose, lpy, kinematics.position(chain, goal_pose)))
    return out


def disabled_joints(chain: Chain, count: int, seed: int = TEST_SEED) -> np.ndarray:
    """The joint (counting from 0) disabled in each reaching pair of the joint-loss benchmark.

    Pairs come in blocks of ``chain.n``, and each block disables every joint once, in its own seeded order. Every joint is
    therefore disabled equally often, to within one, and pair i gets the same joint whatever the count."""

    blocks = [np.random.default_rng([int(seed), chain.n, 79, block]).permutation(chain.n) for block in range(-(-int(count) // chain.n))]
    return np.concatenate(blocks)[: int(count)] if blocks else np.zeros(0, dtype=int)


def test_poses(chain: Chain, count: int, seed: int = POSE_SEED) -> List[np.ndarray]:
    rng = np.random.default_rng([int(seed), chain.n, 0 if chain.joints[0].circular else 1])
    return [chain.random_pose(rng) for _ in range(int(count))]
