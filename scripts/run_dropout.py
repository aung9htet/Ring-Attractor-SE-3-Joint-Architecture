#!/usr/bin/env python3
"""Test 8: the anchor is switched off for a while and on again.

    python3 -B scripts/run_dropout.py --chain tiago7 --speed 0 --out docs/results/data/dropout/tiago7_rest.json        # arm at rest: does the packet stay?
    python3 -B scripts/run_dropout.py --chain tiago7 --speed 0.25 --out docs/results/data/dropout/tiago7_025.json      # arm moving: path integration alone
    python3 -B scripts/run_dropout.py --chain tiago7 --speed 0.25 --no-path-integration --out /out/no_pi.json          # the same without it

During a gap the sheets get nothing from the position cells.  Every tick is logged with its phase.
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
    parser.add_argument("--speed", type=float, default=0.25, help="joint speed limit, rad/s; 0: the arm rests during the gap")
    parser.add_argument("--gaps", type=float, nargs="+", default=[2.0, 5.0, 10.0], help="lengths of the gaps, s")
    parser.add_argument("--repeats", type=int, default=5, help="gaps of each length")
    parser.add_argument("--goal-seed", type=int, default=experiments.POSE_SEED + 12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import nest

    design = experiments.make_design(seed=args.design_seed)
    document = experiments.dropout_run(nest, args.chain, args.speed, args.gaps, args.repeats, design=design, goal_seed=args.goal_seed,
                                       **network_options(args))
    experiments.write_json(args.out, document)
    print("written: %s (%d ticks, %.0f s simulated in %.0f s)" % (args.out, len(document["ticks"]), document["simulated_s"], document["wall_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
