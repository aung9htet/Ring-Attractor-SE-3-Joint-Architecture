#!/usr/bin/env python3
"""Test 4 (and its variants): held poses.  The arm moves to each pose and holds it for 1 s; position and orientation read out over the last 0.5 s are compared with the exact kinematics.

    python3 -B scripts/run_static.py --chain tiago7 --out docs/results/data/static/tiago7.json
    python3 -B scripts/run_static.py --chain generic4 --silence-sheets 0.1 --out /out/sheets10.json     # robustness (test 9)

300 random poses (uniform in the joint ranges, seed fixed).  Needs both calibrations in
artifacts/ (the SO(3) one and scripts/calibrate_pose.py).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_se3 import experiments  # noqa: E402
from ring_se3.sheet import SheetDesign  # noqa: E402


def add_network_arguments(parser):
    parser.add_argument("--chain", required=True, help="generic2..generic7, tiago2..tiago7")
    parser.add_argument("--no-relay", action="store_true", help="no link-orientation populations in the path")
    parser.add_argument("--encoder", choices=("fractional", "continuous", "corrective"), default="fractional")
    parser.add_argument("--design-seed", type=int, default=1, help="seed of the populations' thresholds")
    parser.add_argument("--seed", type=int, default=1, help="kernel seed (the encoders' Poisson generators)")
    parser.add_argument("--noise", type=float, default=0.0, help="noise current into every neuron between rings and sheets, pA (standard deviation)")
    parser.add_argument("--sheet", type=int, default=SheetDesign().size, help="neurons along one side of a sheet (needs its pose calibration)")
    parser.add_argument("--no-velocity", action="store_true", help="build no velocity path: the sheets are anchored only")
    parser.add_argument("--no-path-integration", action="store_true", help="velocity cells silenced (the velocity is still computed)")
    parser.add_argument("--anchor-lead", type=float, default=0.0, metavar="S",
                        help="anchor the sheets at position + velocity x S (the position this many seconds ahead)")
    parser.add_argument("--threads", type=int, default=4)


def network_options(args):
    return dict(relay=not args.no_relay, encoder=args.encoder, threads=args.threads, seed=args.seed, noise_pa=args.noise,
                velocity=not args.no_velocity, path_integration=not args.no_path_integration, anchor_lead_s=args.anchor_lead,
                sheet=SheetDesign(size=args.sheet))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_network_arguments(parser)
    parser.add_argument("--poses", type=int, default=300)
    parser.add_argument("--sweep-points", type=int, default=0, help="values per joint in sweeps of single joints (0: none)")
    parser.add_argument("--silence-sheets", type=float, default=0.0, help="share of the sheets' neurons switched off")
    parser.add_argument("--silence", type=float, default=0.0, help="share of the SO(3) product neurons switched off")
    parser.add_argument("--max-speed", type=float, default=1.0, help="joint speed limit of the moves between poses, rad/s")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import nest

    design = experiments.make_design(seed=args.design_seed)
    document = experiments.static_run(nest, args.chain, args.poses, args.sweep_points, design, args.silence_sheets, args.silence,
                                      max_speed=args.max_speed, **network_options(args))
    experiments.write_json(args.out, document)
    print("written: %s (%d rows, %.0f s simulated in %.0f s)" % (args.out, len(document["rows"]), document["simulated_s"], document["wall_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
