#!/usr/bin/env python3
"""Test 2: the position cells and the velocity populations in the copy without spikes (NumPy only, no NEST).

    python3 -B scripts/rate_study.py --out docs/results/data/rate_study.json

For every model: 200 random poses (not the test poses) with random joint velocities, the
position and the velocity the populations hold as firing rates, against the exact ones.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_so3.experiments import write_json  # noqa: E402
from ring_se3 import kinematics  # noqa: E402
from ring_se3.chains import MODEL_NAMES, get_chain  # noqa: E402
from ring_se3.rate.network import PoseRateNetwork  # noqa: E402
from ring_se3.wiring import pose_wiring  # noqa: E402

STUDY_SEED = 99          # not the seed of the test poses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--poses", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = []
    for name in MODEL_NAMES:
        chain = get_chain(name)
        network = PoseRateNetwork(pose_wiring(chain))
        rng = np.random.default_rng([STUDY_SEED, chain.n])
        radius = network.wiring.space.radius
        position, velocity = [], []
        for _ in range(args.poses):
            q, qdot = chain.random_pose(rng), rng.uniform(-0.5, 0.5, chain.n)
            out = network.decode(q, qdot)
            position.append(100.0 * np.linalg.norm(out.position - kinematics.position(chain, q)) / radius)
            true = kinematics.velocity(chain, q, qdot)
            velocity.append(100.0 * np.linalg.norm(out.velocity - true) / max(np.linalg.norm(true), 1e-9))
        rows.append({"model": name, "poses": args.poses, "median_pct": float(np.median(position)), "p90_pct": float(np.percentile(position, 90)),
                     "velocity_median_pct": float(np.median(velocity)), "velocity_p90_pct": float(np.percentile(velocity, 90))})
        print("%-9s position cells: %.2f / %.2f %% of reach (median / p90); velocity populations: %.0f / %.0f %% of the true speed" % (
            name, rows[-1]["median_pct"], rows[-1]["p90_pct"], rows[-1]["velocity_median_pct"], rows[-1]["velocity_p90_pct"]), flush=True)
    write_json(args.out, {"experiment": "rate_study", "seed": STUDY_SEED, "rows": rows})
    print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
