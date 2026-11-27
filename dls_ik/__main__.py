"""``python -m dls_ik``: DLS IK on the reaching benchmark::

    python -m dls_ik reach      [--plant gazebo|light] [--arm tiago7] [--pairs 30] [--shard i/n] [--max-minutes M]
    python -m dls_ik reach --stuck [--plant gazebo|light]   one joint jammed 2 s into each reach
    python -m dls_ik summarise  [--plant gazebo|light] [--stuck]

``--plant gazebo`` runs beside a Gazebo started with ``docker/run.bash gazebo up NAME`` (results in
``results/dls_ik/reach_gazebo/``); ``--plant light`` uses the kinematic plant (``results/dls_ik/reach_kinematic/``).
DLS IK has nothing random in it, so its run files carry no seed.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from reaching_benchmark import results, summary
from reaching_benchmark.benchmark import BENCHMARKS, PLANTS, STUCK_BENCHMARKS, STUCK_ONSET_S, make_plant, run_reach, test_pairs
from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.task import IkDesign, default_task

from .controller import DESCRIPTION, make_loop

MODEL = "dls_ik"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dls_ik", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("reach", "summarise"))
    parser.add_argument("--plant", choices=PLANTS, default="gazebo")
    parser.add_argument("--arm", default="tiago7")
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--shard", default="0/1", help="i/n: run every n-th pair from the i-th on")
    parser.add_argument("--max-minutes", type=float, default=None, help="stop before the next trial after this long")
    parser.add_argument("--stuck", action="store_true",
                        help="jam one joint per pair 2 s into the reach, DLS IK not told (results in joint_loss_gazebo/ or joint_loss/)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    benchmark = (STUCK_BENCHMARKS if args.stuck else BENCHMARKS)[args.plant]
    if args.command == "reach":
        chain = get_chain(args.arm)
        design = IkDesign(task=default_task(chain.n), kappa=1.0)
        shard = tuple(int(v) for v in args.shard.split("/"))
        run = {"arm": args.arm, "pairs": args.pairs, "kappa": design.kappa, "plant": args.plant, "kind": "near"}
        if args.stuck:
            run.update(condition="lock", onset_s=STUCK_ONSET_S)
        name = results.run_name(args.arm, None, task=design.task, condition="lock" if args.stuck else None, shard=shard)
        loop = make_loop(chain, design, make_plant(args.plant, chain))
        run_reach(MODEL, benchmark, name, run, loop, test_pairs(chain, args.pairs), DESCRIPTION, shard, args.max_minutes,
                  stuck_onset_s=STUCK_ONSET_S if args.stuck else None)
    print("wrote", results.write_summary(MODEL, benchmark, summary.summarise(MODEL, benchmark)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
