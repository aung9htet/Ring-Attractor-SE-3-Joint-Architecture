"""The ring-attractor kinematics (RAK) model's command line::

    python -m ring_attractor_kinematics calibrate  {value_populations | bounded_readouts | sheet_anchor | ramp_readout | shift_cells | observer | all}
    python -m ring_attractor_kinematics benchmark  NAME [options]   (benchmarks/command.py lists every NAME)
    python -m ring_attractor_kinematics summarise  NAME             (recompute results/ring_attractor_kinematics/NAME/summary.json)
    python -m ring_attractor_kinematics simulator  [--chain tiago7] [--gazebo] [--stub]

``python -m reaching_benchmark status`` shows how far every run of every model is.

Each command lives in its own module, which is loaded only when that command runs: the NEST and ROS imports of one
command never slow down or break another.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import List, Optional

#: command -> module with ``add_arguments(parser, command)`` and ``main(args, command)``
COMMANDS = {
    "calibrate": "ring_attractor_kinematics.calibration.command",
    "benchmark": "ring_attractor_kinematics.benchmarks.command",
    "summarise": "ring_attractor_kinematics.benchmarks.command",
    "simulator": "ring_attractor_kinematics.simulator.command",
}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    command = argv.pop(0)
    module = importlib.import_module(COMMANDS[command])
    parser = argparse.ArgumentParser(prog="python -m ring_attractor_kinematics %s" % command)
    module.add_arguments(parser, command)
    return int(module.main(parser.parse_args(argv), command) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
