#!/usr/bin/env python3
"""The shortest way into the pose IK code: the 4-joint TIAGo arm takes its hand to one position goal.

    docker/run.bash nest -- 'python3 -B scripts/ik_example.py'
    docker/run.bash nest -- 'python3 -B scripts/ik_example.py --chain tiago7'     # a pose goal: 69,036 neurons, several minutes
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import nest  # noqa: E402

from ring_se3 import kinematics  # noqa: E402
from ring_se3.chains import MODEL_NAMES  # noqa: E402
from ring_se3.ik import experiments  # noqa: E402
from ring_se3.ik.session import PoseIkSession  # noqa: E402
from ring_so3.kinematics import lift_pitch_yaw  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chain", default="tiago4", choices=MODEL_NAMES)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)

    # the pose network of the chain with the IK layer on top; the four calibrations come from artifacts/
    chain, network = experiments.build_ik(nest, args.chain, threads=args.threads)
    design = network.ik_design                               # the task: "pose" from six joints on, else "position"
    print("%s, task %s: %d neurons, %d of them the IK layer" % (chain.name, design.task, network.neuron_count(), network.ik_neuron_count()))

    start = np.full(chain.n, 0.5) if chain.joints[0].circular else chain.limits().mean(axis=1)
    session = PoseIkSession(network, start)                  # the light robot, at the start pose
    session.warm_up()                                        # 1 s: rings and packets form; the goal sits on the hand

    there = chain.clip(start + 0.5 * np.resize([1.0, -1.0], chain.n))        # the goal is the pose of this joint pose: it can be reached
    goal_lpy, goal_position = lift_pitch_yaw(chain.rotation(there)), kinematics.position(chain, there)
    print("goal: x %.3f, y %.3f, z %.3f m" % tuple(goal_position)
          + ("; lift %.0f, pitch %.0f, yaw %.0f deg" % tuple(np.degrees(goal_lpy)) if design.orientation else ""))

    trial = session.reach(goal_lpy, goal_position)           # the closed loop, until the hand rests (or 25 s)
    for i in range(9, len(trial.t), 10):
        print("t %4.1f s   %6.1f mm%s from the goal   shares (deg) %s" % (
            trial.t[i], 1000.0 * session.radius * trial.error_reach[i], ", %5.1f deg" % trial.error_deg[i] if design.orientation else "",
            np.round(np.degrees(trial.shares[i]), 1)))
    print("ended after %.1f s: %.1f mm (%.1f%% of the reach)%s from the goal, %s; joint angles %s" % (
        trial.t[-1], 1000.0 * session.radius * trial.final_error_reach, 100.0 * trial.final_error_reach,
        ", %.1f deg" % trial.final_error_deg if design.orientation else "",
        "reached" if trial.reached(design) else "not reached", np.round(trial.q[-1], 2)))


if __name__ == "__main__":
    main()
