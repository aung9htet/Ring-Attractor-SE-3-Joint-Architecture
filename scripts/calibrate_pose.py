#!/usr/bin/env python3
"""Measure what the pose network needs beyond the SO(3) calibration and store it in artifacts/.

    python3 -B scripts/calibrate_pose.py            # the default design, about two minutes
    python3 -B scripts/calibrate_pose.py --out /out/pose.npz      # somewhere else (leaves artifacts/ alone)

The SO(3) calibration of the same design must exist (``artifacts/calibration_*.npz``, copied
from the SO-3 repo or written by ``scripts/calibrate.py``).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from ring_so3.design import Design  # noqa: E402
from ring_so3.nest import calibrate as so3_calibrate  # noqa: E402
from ring_se3.nest import calibrate  # noqa: E402
from ring_se3.sheet import SheetDesign  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--encoder", default="fractional")
    parser.add_argument("--seed", type=int, default=1, help="design seed of the populations")
    parser.add_argument("--sheet", type=int, default=SheetDesign().size, help="neurons along one side of a sheet")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default=None, help="write here instead of artifacts/")
    args = parser.parse_args()
    import nest

    design = Design(seed=args.seed)
    sheet = SheetDesign(size=args.sheet)
    base = so3_calibrate.load(design, design.sizes.ring, args.encoder)
    result = calibrate.run(nest, base, design, sheet, threads=args.threads)
    path = args.out or os.path.join(calibrate.ARTIFACTS, calibrate.artifact_name(design, design.sizes.ring, args.encoder, sheet))
    result.save(path)
    print("written: %s" % os.path.relpath(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
