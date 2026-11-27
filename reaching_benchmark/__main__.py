"""The shared command line::

    python -m reaching_benchmark status   [MODEL/BENCHMARK | BENCHMARK ...] [--watch S]   how far every run is (host or container)
    python -m reaching_benchmark compare  BENCHMARK                                       every model's summary of one benchmark side by side
    python -m reaching_benchmark rescore  [MODEL/BENCHMARK ...]                           score stored trials again at the current tolerance

Each model has its own command line: ``python -m ring_attractor_kinematics``, ``python -m dls_ik``,
``python -m mpc_nn_dynamics``, ``python -m pets``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import List, Optional

COMMANDS = {"status": "reaching_benchmark.status", "compare": "reaching_benchmark.compare", "rescore": "reaching_benchmark.rescore"}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    command = argv.pop(0)
    module = importlib.import_module(COMMANDS[command])
    parser = argparse.ArgumentParser(prog="python -m reaching_benchmark %s" % command)
    module.add_arguments(parser, command)
    return int(module.main(parser.parse_args(argv), command) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
