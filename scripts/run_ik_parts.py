#!/usr/bin/env python3
"""IK test 3: the comparisons and the shares of the spiking pose IK against the exact ones, arm moved by hand.

    python3 -B scripts/run_ik_parts.py --chain tiago4 --out docs/ik/results/data/parts_tiago4_position.json
    python3 -B scripts/run_ik_parts.py --chain generic7 --task pose --poses 40 --out /out/parts_generic7.json

The arm is held at a pose, a goal part of the way to another pose is written, and the turn, the
offset and every joint's share are read from the network and compared with exact arithmetic.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_so3.experiments import write_json  # noqa: E402
from ring_se3.chains import get_chain  # noqa: E402
from ring_se3.ik import experiments  # noqa: E402
from ring_se3.ik.design import POSITION_GOALS, TASKS, PoseIkDesign  # noqa: E402
from ring_se3.ik.pairs import TEST_SEED  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chain", default="tiago4")
    parser.add_argument("--task", choices=TASKS, default=None, help="default: pose from six joints on, else position")
    parser.add_argument("--position-goal", choices=POSITION_GOALS, default=PoseIkDesign().position_goal)
    parser.add_argument("--all-at-once", action="store_true", help="no cascade: every joint gets the whole error")
    parser.add_argument("--poses", type=int, default=100)
    parser.add_argument("--pair-seed", type=int, default=TEST_SEED)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import nest

    task = args.task or experiments.default_task(get_chain(args.chain))
    design = PoseIkDesign(task=task, position_goal=args.position_goal, cascade=not args.all_at_once)
    document = experiments.open_loop_run(nest, args.chain, args.poses, design, args.pair_seed, threads=args.threads)
    write_json(args.out, document)
    s = document["summary"]
    print("%s, %s: %d neurons, %.2f of real time" % (args.chain, task, document["neurons"], document["speed"]))
    if design.orientation:
        print("turn: direction error median %.1f deg, p90 %.1f deg (turn median %.2f rad); rms %.4f rad" % (
            s["turn_direction_error_median_deg"], s["turn_direction_error_p90_deg"], s["turn_size_median_rad"], s["turn_rms_rad"]))
    if design.position:
        print("offset: off by %.4f reaches in the median, rms %.4f per coordinate; direction error median %.1f deg (offset median %.3f reaches)" % (
            s["offset_error_median"], s["offset_rms"], s["offset_direction_error_median_deg"], s["offset_size_median"]))
        print("positions, error in reaches (median): forward sheets %.4f, position cells %.4f, goal sheets %.4f; ramp readouts: hand %s, goal %s" % (
            s["hand_sheet_error_median"], s["cells_error_median"], s["goal_sheet_error_median"] or float("nan"),
            "-" if s["ramp_hand_error_median"] is None else "%.4f" % s["ramp_hand_error_median"],
            "-" if s["ramp_goal_error_median"] is None else "%.4f" % s["ramp_goal_error_median"]))
    print("shares: right sign in %s of %d shares above %.2f rad; rms %.4f rad; %s rad where none is asked" % (
        "-" if s["share_right_sign"] is None else "%.0f%%" % (100 * s["share_right_sign"]), s["shares_large"], s["share_floor_rad"], s["share_rms_rad"],
        "-" if s["share_when_none_asked_rms_rad"] is None else "%.4f" % s["share_when_none_asked_rms_rad"]))
    for row in s["per_joint"]:
        print("  joint %d: rms %.4f rad, right sign %s of %d" % (
            row["joint"], row["rms_rad"], "-" if row["right_sign"] is None else "%.0f%%" % (100 * row["right_sign"]), row["large"]))
    print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
