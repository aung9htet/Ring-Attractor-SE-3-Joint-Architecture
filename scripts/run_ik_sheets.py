#!/usr/bin/env python3
"""IK test 3c: the sheets as the comparison of positions needs them: level cells and the ramp readout.

    python3 -B scripts/run_ik_sheets.py --out docs/ik/results/data/sheets.json

Three sheets on their own, anchored through gate populations at 80 random places of the workspace
(not the places the ramp readout was fitted on), once without level cells and once with them:
how strong is each packet, does any go out, and how well do the ramp readouts give the packets' place?
Then, on the first 30 of the places, level cells with other gains and synapses: how even do they make
the packets, and do they make a packet's strength swing?
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_so3.design import Design  # noqa: E402
from ring_so3.experiments import write_json  # noqa: E402
from ring_so3.nest import calibrate as so3_calibrate  # noqa: E402
from ring_so3.nest import ik_calibrate as so3_ik_calibrate  # noqa: E402
from ring_se3.nest import calibrate, ik_calibrate  # noqa: E402
from ring_se3.nest.sheet import LevelDesign  # noqa: E402
from ring_se3.sheet import SheetDesign, SheetMap  # noqa: E402

SHEETS_SEED = 4712          # not the seed of anything else
#: other level cells than those that are built: (gain in pA, time constant of their synapse from the sheet in ms)
SCAN = ((100.0, 30.0), (300.0, 30.0), (100.0, 100.0), (300.0, 150.0))
#: a packet below these spikes per second counts as out, or as weak
OUT_HZ, WEAK_HZ = 100.0, 300.0


def random_places(count, seed=SHEETS_SEED, radius=1.0):
    """Places drawn evenly from the workspace, in an order in which the anchor never has far to go."""

    rng = np.random.default_rng(int(seed))
    points = rng.normal(size=(int(count), 3))
    points *= (radius * rng.uniform(size=(int(count), 1)) ** (1.0 / 3.0)) / np.linalg.norm(points, axis=1, keepdims=True)
    order, left = [0], list(range(1, int(count)))
    while left:
        nearest = min(left, key=lambda i: float(np.linalg.norm(points[i] - points[order[-1]])))
        order.append(nearest)
        left.remove(nearest)
    return points[order]


def summary(data, decoders):
    totals, centres, places = data["totals"], data["centres"], data["places"]
    read = data["rates"] @ decoders[:, 0]
    alive = np.isfinite(centres)
    error = (read - centres)[alive]
    return {
        "sheet_rate_median_hz": float(np.median(totals)), "sheet_rate_sd_hz": float(np.std(totals)),
        "sheet_rate_smallest_hz": float(np.min(totals)), "sheet_rate_largest_hz": float(np.max(totals)),
        "sheet_rate_ripple_hz": float(np.median(data["ripple"])),
        "packets": int(totals.size), "packets_out": int(np.sum(totals < OUT_HZ)), "packets_weak": int(np.sum(totals < WEAK_HZ)),
        "anchor_rms": float(np.sqrt(np.mean((centres - places)[alive] ** 2))),
        "ramp_rms": float(np.sqrt(np.mean(error ** 2))), "ramp_median": float(np.median(np.abs(error))),
        "ramp_p90": float(np.percentile(np.abs(error), 90)), "ramp_largest": float(np.max(np.abs(error))),
        "ramp_bias": float(np.mean(error)), "coordinates_read": int(error.size), "coordinates_without_spikes": int((~alive).sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--places", type=int, default=80)
    parser.add_argument("--scan-places", type=int, default=30, help="places on which other level cells are tried")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    import nest

    started = time.time()
    design, sheet = Design(), SheetDesign()
    mapping = SheetMap(sheet)
    base, pose = so3_calibrate.load(design), calibrate.load(design, sheet=sheet)
    bounded = so3_ik_calibrate.load(design).decoders["bounded"]
    ramp = ik_calibrate.load(design, sheet=sheet)
    places = random_places(args.places)
    document = {"experiment": "pose_ik_sheets", "seed": SHEETS_SEED, "threads": args.threads, "sheet": sheet.describe(),
                "level": LevelDesign().describe(), "out_hz": OUT_HZ, "weak_hz": WEAK_HZ, "ramp_calibration_fit": ramp.fit,
                "ramp_settings": ramp.settings, "places": places.tolist()}
    for name, level in (("without", None), ("with", LevelDesign())):
        data = ik_calibrate.measure(nest, design, base, pose, mapping, places, ramp.tuning(design), ramp.total_rate_hz, args.threads,
                                    hold_ticks=20, step_units=0.04, level=level, level_decoders=bounded)
        row = summary(data, ramp.decoders["ramp"])
        row.update(totals=data["totals"].tolist(), centres=data["centres"].tolist(),
                   read=(data["rates"] @ ramp.decoders["ramp"][:, 0]).tolist())
        document[name] = row
        print("%s level cells: a packet sends %.0f spikes a second in the median (sd %.0f, %.0f to %.0f); %d of %d out, %d weak; "
              "ramp readout off by %.4f of the reach (rms), largest %.4f (%.0f s)" % (
                  name, row["sheet_rate_median_hz"], row["sheet_rate_sd_hz"], row["sheet_rate_smallest_hz"], row["sheet_rate_largest_hz"],
                  row["packets_out"], row["packets"], row["packets_weak"], row["ramp_rms"], row["ramp_largest"], time.time() - started), flush=True)
    document["scan"] = []
    for gain, tau in ((None, None), (LevelDesign().gain_pa, LevelDesign().tau_ms)) + SCAN:
        level = None if gain is None else LevelDesign(gain_pa=gain, tau_ms=tau)
        data = ik_calibrate.measure(nest, design, base, pose, mapping, places[:args.scan_places], ramp.tuning(design), ramp.total_rate_hz, args.threads,
                                    hold_ticks=20, step_units=0.04, level=level, level_decoders=bounded)
        row = summary(data, ramp.decoders["ramp"])
        row.update(gain_pa=gain, tau_ms=tau)
        document["scan"].append(row)
        print("level cells %s: spikes per second sd %.0f over the places, %.0f from tick to tick; %d weak (%.0f s)" % (
            "none" if gain is None else "%.0f pA, %.0f ms" % (gain, tau), row["sheet_rate_sd_hz"], row["sheet_rate_ripple_hz"], row["packets_weak"],
            time.time() - started), flush=True)
    document["wall_s"] = round(time.time() - started, 1)
    write_json(args.out, document)
    print("written:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
