#!/usr/bin/env python3
"""The shift-register experiment: during a gap the network gets neither the anchor nor the joint velocities.

    python3 -B scripts/run_trail.py --chain generic3 --mode trail --out docs/results/data/trail/generic3_trail.json
    python3 -B scripts/run_trail.py --chain generic3 --mode none --out docs/results/data/trail/generic3_none.json
    python3 -B scripts/run_trail.py --chain generic3 --mode velocity --out docs/results/data/trail/generic3_velocity.json

``trail``: a shift register (after the sequence generator of Lin, Yiu and Leibold 2026) keeps where the packets
were in the last 0.5 s; when the gap starts, the velocity those places show is handed to the velocity cells and
kept.  ``none``: nothing is handed over.  ``velocity``: the joint velocities stay on, for comparison.
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
    parser.add_argument("--mode", choices=("trail", "none", "velocity"), default="trail")
    parser.add_argument("--speed", type=float, default=0.25, help="joint speed limit, rad/s")
    parser.add_argument("--gap", type=float, default=2.0, help="length of a gap, s")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import nest

    design = experiments.make_design(seed=args.design_seed)
    document = experiments.trail_run(nest, args.chain, args.speed, args.gap, args.repeats, args.mode, design=design, **network_options(args))
    experiments.write_json(args.out, document)
    print("written: %s (%d ticks, %.0f s simulated in %.0f s)" % (args.out, len(document["ticks"]), document["simulated_s"], document["wall_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
