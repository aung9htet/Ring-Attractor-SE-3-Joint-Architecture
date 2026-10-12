#!/usr/bin/env python3
"""IK test 10: the closed loop with the TIAGo in Gazebo as the robot (run inside a container started with "gazebo up").

    docker/run.bash gazebo up ik-gazebo
    docker/run.bash gazebo exec ik-gazebo -- 'python3 -B scripts/run_ik_gazebo.py --check 5 --out docs/ik/results/data/gazebo_kinematics_tiago7.json'
    docker/run.bash gazebo exec ik-gazebo -- 'python3 -B scripts/run_ik_gazebo.py --pairs 30 --out docs/ik/results/data/gazebo_tiago7_pose.json'
    docker/run.bash gazebo exec ik-gazebo -- 'python3 -B scripts/run_ik_gazebo.py --pairs 30 --method dls --out docs/ik/results/data/gazebo_tiago7_pose_dls.json'
    docker/run.bash gazebo exec ik-gazebo -- 'python3 -B scripts/run_ik_gazebo.py --chain tiago4 --pairs 20 --out docs/ik/results/data/gazebo_tiago4_position.json'
    docker/run.bash gazebo down ik-gazebo

The first ``--pairs`` test pairs of a TIAGo chain.  Physics is advanced in lock step with the network
(50 ms per tick), so the network being slower than real time does not matter.  ``--method``: ``ring``
(the spiking IK) or a method that reads the exact joint angles.  A run that its time guard cut is
finished with ``--only`` and the pairs it lists as not run.

``--check N`` runs no network: the arm is moved to N poses and the end point Gazebo has (from its
link states) is compared with the chain's kinematics at the joint angles Gazebo reports.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_so3.experiments import write_json  # noqa: E402
from ring_se3.chains import get_chain  # noqa: E402
from ring_se3.ik import experiments  # noqa: E402
from ring_se3.ik.design import POSITION_GOALS, TASKS, PoseIkDesign  # noqa: E402
from ring_se3.ik.pairs import DEVELOPMENT_SEED, TEST_SEED  # noqa: E402
from ring_se3.standard.targets import EXACT_METHODS  # noqa: E402

CHECK_SEED = 20261010          # the poses of --check
CHECK_SPREAD = 0.6             # rad about the middle of the joint ranges


def main(argv=None):
    defaults = PoseIkDesign()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chain", default="tiago7", choices=["tiago%d" % n for n in range(2, 8)])
    parser.add_argument("--task", choices=TASKS, default=None, help="default: pose from six joints on, else position")
    parser.add_argument("--position-goal", choices=POSITION_GOALS, default=defaults.position_goal)
    parser.add_argument("--method", default="ring", choices=("ring",) + EXACT_METHODS)
    parser.add_argument("--kappa", type=float, default=defaults.kappa)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--only", default=None, metavar="I,J,...")
    parser.add_argument("--max-minutes", type=float, default=100.0)
    parser.add_argument("--check", type=int, default=0, metavar="N", help="no network: compare Gazebo's end point with the chain's kinematics at N poses")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from ring_se3.gazebo.robot import PoseGazeboRobot as GazeboRobot
    from ring_se3.gazebo.link import PoseRosLink, kinematics_check

    link = PoseRosLink()
    chain = get_chain(args.chain)
    if args.check:
        rng = np.random.default_rng([CHECK_SEED, chain.n])
        middle = chain.limits().mean(axis=1)
        poses = [chain.clip(middle + rng.uniform(-CHECK_SPREAD, CHECK_SPREAD, chain.n)) for _ in range(args.check)]
        document = kinematics_check(link, GazeboRobot(chain, link), chain, poses)
        write_json(args.out, document)
        print("%s: Gazebo and the chain's kinematics differ by at most %.2f mm and %.3f deg over %d poses" % (
            chain.name, 1000.0 * (document["largest_position_error_m"] or float("nan")), document["largest_orientation_error_deg"] or float("nan"), args.check))
        print("written:", args.out)
        return 0

    import nest

    task = args.task or experiments.default_task(chain)
    design = PoseIkDesign(task=task, position_goal=args.position_goal, kappa=args.kappa)
    document = experiments.reach_run(
        nest, args.chain, args.pairs, design, DEVELOPMENT_SEED if args.development else TEST_SEED, experiments.SPREAD, (0, 1), args.max_minutes,
        only=[int(v) for v in args.only.split(",")] if args.only else None, make_robot=lambda c: GazeboRobot(c, link), method=args.method,
        checkpoint=args.out, threads=args.threads)
    write_json(args.out, document)
    s = document["summary"]
    print("%s in Gazebo, %s, %s: %d trials (%d not run)" % (args.chain, task, args.method, s["trials"], len(document["skipped"])))
    if s["trials"]:
        print("reached %.0f%% (exact rule on the light robot %.0f%%); final error %.2f deg, %.2f%% of the reach (median); within the marks after %.1f s; path %.2f x straight; missed %d (%d at a limit)" % (
            100 * s["reached"], 100 * s["exact_reached"], s["final_error_median_deg"], 100 * s["final_error_median_reach"],
            s["time_to_reach_median_s"] or float("nan"), s["path_ratio_median"] or float("nan"), s["missed"], s["missed_at_limit"]))
    print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
