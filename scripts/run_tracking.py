#!/usr/bin/env python3
"""Test 7: the arm keeps moving from one random goal to the next; every 50 ms tick is logged.

    python3 -B scripts/run_tracking.py --chain tiago7 --speed 0.5 --out docs/results/data/tracking/tiago7_050.json
    python3 -B scripts/run_tracking.py --chain tiago7 --speed 0.5 --no-path-integration --out /out/anchor_only.json
    python3 -B scripts/run_tracking.py --chain tiago7 --speed 0.5 --anchor-lead 0.2 --out /out/lead.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_se3 import experiments  # noqa: E402
from run_static import add_network_arguments, network_options  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_network_arguments(parser)
    parser.add_argument("--speed", type=float, default=0.5, help="joint speed limit, rad/s")
    parser.add_argument("--goals", type=int, default=20)
    parser.add_argument("--dwell", type=float, default=1.0, help="rest at every goal, s")
    parser.add_argument("--goal-seed", type=int, default=experiments.POSE_SEED + 11)
    parser.add_argument("--lead", action="store_true", help="write each joint's angle ahead by the SO(3) readout's delay")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import nest

    design = experiments.make_design(seed=args.design_seed)
    document = experiments.tracking_run(nest, args.chain, args.speed, args.goals, args.dwell, design, goal_seed=args.goal_seed,
                                        lead=args.lead, **network_options(args))
    experiments.write_json(args.out, document)
    print("written: %s (%d ticks, %.0f s simulated in %.0f s)" % (args.out, len(document["ticks"]), document["simulated_s"], document["wall_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
