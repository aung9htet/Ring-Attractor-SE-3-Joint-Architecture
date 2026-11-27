"""Where the analysis reads and writes.

Inputs are the run files under ``results/`` that the models' benchmarks write (README, "Results: how each one was
produced"). Outputs stay inside ``analysis/``:

* ``analysis/source_data/``  data derived from ``results/`` that several figures share (``derive``, ``activity``);
* ``analysis/source_data/plotexports/``  the numbers behind every plotted panel, one CSV or JSON per panel group;
* ``analysis/figures/``  every figure as PDF, SVG and PNG.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RAK = RESULTS / "ring_attractor_kinematics"
OUT = REPO / "analysis"
FIGURES = OUT / "figures"
DATA = OUT / "source_data"
EXPORT = DATA / "plotexports"

#: SHA-256 of every input file read, written to ``source_data/plotexports/inputs_sha256.json``
HASHES: Dict[str, str] = {}


def ensure_dirs() -> None:
    for folder in (FIGURES, DATA, EXPORT):
        folder.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read(path: Path) -> Dict[str, Any]:
    """Read a JSON file and remember its hash."""

    path = Path(path)
    data = path.read_bytes()
    HASHES[relative(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def runs(folder: Path):
    """Every run file of one benchmark folder (not its summary), in name order."""

    return [p for p in sorted(Path(folder).glob("*.json")) if p.name != "summary.json"]


def write_json(path: Path, document: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")
