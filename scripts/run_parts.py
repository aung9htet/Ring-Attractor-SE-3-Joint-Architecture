#!/usr/bin/env python3
"""Test 3: each new part of the pose network measured on its own in NEST.

    python3 -B scripts/run_parts.py --out docs/results/data/parts.json

* a packet left alone for 10 s at several places (does it stay?);
* a sheet anchored through real gate populations at random places (within one cell?);
* a free packet moved by a known velocity through the real velocity populations and cells
  (is its speed proportional to the velocity?);
* the velocity populations of a whole network against the exact velocity of the end point, for
  random poses and joint velocities.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_so3.design import Design  # noqa: E402
from ring_so3.nest import calibrate as so3_calibrate  # noqa: E402
from ring_se3 import experiments, kinematics  # noqa: E402
from ring_se3.nest import calibrate, rigs  # noqa: E402
from ring_se3.sheet import SheetDesign, SheetMap  # noqa: E402

PARTS_SEED = 4711          # not the seed of the test poses


def velocity_populations_check(nest, chain_name, poses, threads):
    """Random held poses, and at each one random joint velocities handed to the network while the arm stays put."""

    chain, design, network, session = experiments.build(nest, chain_name, threads=threads, path_integration=False)
    session.warm_up()
    rng = np.random.default_rng([PARTS_SEED, chain.n])
    rows = []
    for _ in range(int(poses)):
        session.go_to(chain.random_pose(rng))
        qdot = rng.uniform(-0.5, 0.5, chain.n)
        ticks = [network.step(session.robot.q, qdot) for _ in range(20)]
        decoded = np.mean([network.velocity_m_s(t) for t in ticks[10:]], axis=0)
        true = kinematics.velocity(chain, session.robot.q, qdot)
        rows.append({"true": true.tolist(), "decoded": decoded.tolist(), "radius_m": network.space.radius})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="fewer points (for a first look)")
    args = parser.parse_args(argv)
    import nest

    started = time.time()
    design, sheet = Design(), SheetDesign()
    mapping = SheetMap(sheet)
    base = so3_calibrate.load(design)
    pose = calibrate.load(design, sheet=sheet)
    rng = np.random.default_rng(PARTS_SEED)
    quick = args.quick

    hold = []
    for i, place in enumerate(rng.uniform(-0.9, 0.9, (3 if quick else 8, 2))):
        row = rigs.measure_hold(nest, mapping, place, 10.0, args.threads, seed=1 + i)
        row["place"] = place.tolist()
        hold.append(row)
    print("hold: drift after 10 s, cells: %s (%.0f s)" % (" ".join("%.2f" % r["drift_cells"] for r in hold), time.time() - started), flush=True)

    places = rng.uniform(-1.0, 1.0, (10 if quick else 60, 2))
    places = places[np.linalg.norm(places, axis=1) <= 1.0]
    anchor = rigs.measure_anchor(nest, design, base, pose, mapping, places, args.threads)
    errors = np.array([r["error_cells"] for r in anchor])
    print("anchor: %d places, error median %.2f, largest %.2f cells (%.0f s)" % (len(anchor), np.nanmedian(errors), np.nanmax(errors), time.time() - started), flush=True)

    top = float(pose.fit["velocity_value_at_full_speed"])
    values = []
    for size in ((0.3, 0.7) if quick else (0.15, 0.3, 0.5, 0.7, 0.9)):
        for heading in ((1, 0), (-1, 0), (0, 1), (0, -1), (0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, 0.7071), (-0.7071, -0.7071)):
            values.append(size * top * np.array(heading))
    speed = []
    for i, value in enumerate(values):
        for seed in ((1,) if quick else (1, 2)):
            row = rigs.measure_velocity_chain(nest, design, base, pose, mapping, value, args.threads, seed=seed)
            row["seed"] = seed
            speed.append(row)
    ratio = [np.dot(r["velocity"], r["wanted"]) / np.dot(r["wanted"], r["wanted"]) for r in speed]
    print("packet speed over the speed asked for: median %.2f, 10th to 90th percentile %.2f to %.2f (%d runs, %.0f s)" % (
        np.nanmedian(ratio), np.nanpercentile(ratio, 10), np.nanpercentile(ratio, 90), len(ratio), time.time() - started), flush=True)

    velocity = {name: velocity_populations_check(nest, name, 5 if quick else 30, args.threads) for name in (("generic3",) if quick else ("generic3", "tiago7"))}
    for name, rows in velocity.items():
        rel = [np.linalg.norm(np.subtract(r["decoded"], r["true"])) / max(np.linalg.norm(r["true"]), 1e-9) for r in rows]
        print("velocity populations, %s: error median %.0f%% of the true speed (%d poses, %.0f s)" % (name, 100 * np.median(rel), len(rows), time.time() - started), flush=True)

    document = {"experiment": "parts", "threads": args.threads, "seed": PARTS_SEED, "sheet": sheet.describe(),
                "cell_size": mapping.cell_size, "cells_per_unit_velocity": calibrate.cells_per_unit_velocity(mapping),
                "pose_calibration_fit": pose.fit, "curve_drive": pose.curve_drive.tolist(), "curve_speed": pose.curve_speed.tolist(),
                "curve_by_direction": pose.curve_by_direction.tolist(),
                "hold": hold, "anchor": anchor, "speed": speed, "velocity_cells": velocity, "wall_s": round(time.time() - started, 1)}
    experiments.write_json(args.out, document)
    print("written: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
