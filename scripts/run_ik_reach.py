#!/usr/bin/env python3
"""IK tests 4 to 7: the arm reaches pose goals in closed loop (spiking network, light robot).

    python3 -B scripts/run_ik_reach.py --chain tiago4 --out docs/ik/results/data/reach_tiago4_position.json
    python3 -B scripts/run_ik_reach.py --chain generic7 --task pose --pairs 5 --out /out/try.json
    python3 -B scripts/run_ik_reach.py --chain tiago7 --shard 0/3 --max-minutes 100 --out /out/reach_tiago7_0.json
    python3 -B scripts/run_ik_reach.py --chain tiago4 --pairs 10 --development --out /out/try.json

A goal is the pose of a joint pose within ``--spread`` radians of the start pose, joint by joint
(``--spread 0``: any joint pose).  ``--shard i/n`` runs every n-th pair from the i-th on, so that n
runs side by side cover all pairs.  ``--development`` takes the pairs of the development seed.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_so3.experiments import write_json  # noqa: E402
from ring_se3.chains import get_chain  # noqa: E402
from ring_se3.ik import experiments  # noqa: E402
from ring_se3.ik.design import POSITION_GOALS, TASKS, PoseIkDesign  # noqa: E402
from ring_se3.ik.pairs import DEVELOPMENT_SEED, TEST_SEED  # noqa: E402


def main(argv=None):
    defaults = PoseIkDesign()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chain", default="tiago4")
    parser.add_argument("--task", choices=TASKS, default=None, help="default: pose from six joints on, else position")
    parser.add_argument("--position-goal", choices=POSITION_GOALS, default=defaults.position_goal)
    parser.add_argument("--all-at-once", action="store_true", help="no cascade: every joint gets the whole error")
    parser.add_argument("--balance", type=float, default=defaults.balance, help="a reach of offset counts as this many radians of turn")
    parser.add_argument("--kappa", type=float, default=defaults.kappa, help="the joint moves at this times the lead of its goal ring, 1/s")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--spread", type=float, default=experiments.SPREAD, help="goal within this many radians per joint (0: anywhere)")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--pair-seed", type=int, default=None)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--only", default=None, metavar="I,J,...", help="only these pairs (those an earlier run did not get to)")
    parser.add_argument("--method", default="ring", choices=("ring", "cascade", "dls", "transpose"),
                        help="ring: each joint follows its goal ring; else the network runs and the joints follow that method")
    parser.add_argument("--motor-scale", type=float, default=1.0, help="the robot moves this much of what it is told")
    parser.add_argument("--push", type=float, nargs=2, default=None, metavar=("TIME", "ANGLE"), help="displace one joint on the way")
    parser.add_argument("--goal-change", type=float, default=None, metavar="TIME", help="change the goal at this time")
    parser.add_argument("--noise-pa", type=float, default=0.0)
    parser.add_argument("--silence", type=float, default=0.0, help="share of the neurons of comparison and cascade switched off")
    parser.add_argument("--seed", type=int, default=1, help="kernel seed")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import nest

    seed = args.pair_seed if args.pair_seed is not None else (DEVELOPMENT_SEED if args.development else TEST_SEED)
    task = args.task or experiments.default_task(get_chain(args.chain))
    design = PoseIkDesign(task=task, position_goal=args.position_goal, cascade=not args.all_at_once, balance=args.balance, kappa=args.kappa)
    shard = tuple(int(v) for v in args.shard.split("/"))
    document = experiments.reach_run(nest, args.chain, args.pairs, design, seed, args.spread if args.spread > 0 else None, shard,
                                     args.max_minutes, args.motor_scale, args.push, args.goal_change, silence=args.silence,
                                     only=[int(v) for v in args.only.split(",")] if args.only else None, method=args.method,
                                     checkpoint=args.out, threads=args.threads, noise_pa=args.noise_pa, seed=args.seed)
    write_json(args.out, document)
    s = document["summary"]
    print("%s, %s: %d neurons, %.2f of real time, %d trials (%d not run)" % (args.chain, task, document["neurons"], document["speed"], s["trials"], len(document["skipped"])))
    if s["trials"]:
        print("reached %.0f%% (exact rule %.0f%%; of those the exact rule reaches: %s); final error %.2f deg, %.2f%% of the reach (median); within the marks after %.1f s; path %.2f x straight" % (
            100 * s["reached"], 100 * s["exact_reached"],
            "-" if s["reached_of_exact_reached"] is None else "%.0f%%" % (100 * s["reached_of_exact_reached"]),
            s["final_error_median_deg"], 100 * s["final_error_median_reach"], s["time_to_reach_median_s"] or float("nan"), s["path_ratio_median"] or float("nan")))
        print("missed %d, of which %d with a joint at a limit; %.0f%% of trials settled, median duration %.1f s" % (
            s["missed"], s["missed_at_limit"], 100 * s["settled"], s["duration_median_s"]))
    print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
