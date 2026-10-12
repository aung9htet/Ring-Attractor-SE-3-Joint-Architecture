#!/usr/bin/env python3
"""Measure the readout that turns a packet's place into a number (the inverse kinematics needs it) and store it in artifacts/.

    python3 -B scripts/calibrate_ramp.py            # the default design, about seven minutes
    python3 -B scripts/calibrate_ramp.py --out /out/ramp.npz      # somewhere else (leaves artifacts/ alone)

Three sheets with their level cells are anchored at places on a grid; no pose of any chain is used.
The SO(3) calibration, the pose calibration and the SO(3) inverse-kinematics calibration of the same
design must exist (``artifacts/calibration_*.npz``, ``pose_*.npz``, ``ik_calibration_*.npz``).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from ring_so3.design import Design  # noqa: E402
from ring_so3.nest import calibrate as so3_calibrate  # noqa: E402
from ring_so3.nest import ik_calibrate as so3_ik_calibrate  # noqa: E402
from ring_se3.nest import calibrate, ik_calibrate  # noqa: E402
from ring_se3.sheet import SheetDesign  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--encoder", default="fractional")
    parser.add_argument("--seed", type=int, default=1, help="design seed of the populations")
    parser.add_argument("--sheet", type=int, default=SheetDesign().size, help="neurons along one side of a sheet")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default=None, help="write here instead of artifacts/")
    parser.add_argument("--force", action="store_true", help="write even when the file exists")
    args = parser.parse_args()
    import nest

    design = Design(seed=args.seed)
    sheet = SheetDesign(size=args.sheet)
    path = args.out or os.path.join(ik_calibrate.ARTIFACTS, ik_calibrate.artifact_name(design, design.sizes.ring, args.encoder, sheet))
    if os.path.exists(path) and not args.force:
        print("exists, not written again (results depend on it):", os.path.relpath(path), "- use --out for another file")
        return 1
    base = so3_calibrate.load(design, design.sizes.ring, args.encoder)
    pose = calibrate.load(design, design.sizes.ring, args.encoder, sheet)
    bounded = so3_ik_calibrate.load(design, design.sizes.ring, args.encoder).decoders["bounded"]
    result = ik_calibrate.run(nest, base, pose, bounded, design, sheet, threads=args.threads)
    result.save(path)
    print("written: %s" % os.path.relpath(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
