"""``python -m ring_attractor_kinematics calibrate NAME``: measure one calibration in NEST and store it in ``artifacts/``.

The four the network uses depend on each other in this order: value_populations → bounded_readouts → sheet_anchor →
ramp_readout (about 10, 2, 1 and 60 minutes on 8 threads).  The observer method needs two more, in this order:
``shift_cells`` (its prediction motif and the sub-gate, under a minute) and ``observer`` (its read-outs and the cue's
delay, a few minutes).  The shipped files were made with these commands.
"""

from __future__ import annotations

import argparse

import nest

from ..neurons.tuning import Tunings
from . import bounded_readouts, observer, ramp, shift_cells, sheet_anchor, value_populations
from .artifacts import ARTIFACTS, NAMES, OBSERVER_NAMES, Artifact


def add_arguments(parser: argparse.ArgumentParser, command: str) -> None:
    parser.add_argument("name", choices=NAMES + OBSERVER_NAMES + ("all",))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1, help="kernel seed of the measurement rigs")
    parser.add_argument("--directory", default=ARTIFACTS)


def main(args: argparse.Namespace, command: str) -> int:
    tunings = Tunings()
    names = NAMES + OBSERVER_NAMES if args.name == "all" else (args.name,)
    for name in names:
        if name == "value_populations":
            artifact = value_populations.run(nest, tunings, threads=args.threads, seed=args.seed)
        elif name == "bounded_readouts":
            artifact = bounded_readouts.run(nest, tunings, Artifact.load("value_populations", args.directory), args.threads, args.seed)
        elif name == "sheet_anchor":
            artifact = sheet_anchor.run(nest, tunings, Artifact.load("value_populations", args.directory), threads=args.threads, seed=args.seed)
        elif name == "shift_cells":
            artifact = shift_cells.run(nest, tunings, Artifact.load("bounded_readouts", args.directory), threads=args.threads,
                                       directory=args.directory)
        elif name == "observer":
            artifact = observer.run(nest, tunings, Artifact.load("value_populations", args.directory),
                                    Artifact.load("bounded_readouts", args.directory), Artifact.load("shift_cells", args.directory),
                                    threads=args.threads, seed=args.seed,
                                    directory=args.directory)
        else:
            artifact = ramp.run(nest, tunings, Artifact.load("value_populations", args.directory), Artifact.load("bounded_readouts", args.directory),
                                Artifact.load("sheet_anchor", args.directory), threads=args.threads, seed=args.seed)
        print("wrote", artifact.save(args.directory))
    return 0
