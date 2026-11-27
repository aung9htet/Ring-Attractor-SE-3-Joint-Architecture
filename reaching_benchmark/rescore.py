"""``python -m reaching_benchmark rescore [MODEL/BENCHMARK ...]``: score every stored closed-loop trial again at the
current tolerances (:class:`reaching_benchmark.task.IkDesign`), without running it again.

The tolerances only judge a trial; no controller and no trial ending reads them, so a trial run under other tolerances
is the same trial. Reached is decided at the last sample from the stored final errors, exactly. The time from which the
hand stays within tolerance is read from the stored time course (every 4th tick, 0.2 s) with the last sample added, so
it is at most 0.2 s later than the full-rate time and can miss a brief exit between two stored samples. The file's
``design`` takes the new tolerances, and ``scoring`` records what was done. Exact-loop twins and the joint-loss
reachability check are computed again, exactly, by ``python -m ring_attractor_kinematics benchmark twins``; then every
summary is recomputed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import numpy as np

from . import results
from .limits import first_time_within
from .progress import Progress
from .task import IkDesign

TRACE_TICKS = 4


def rescore_row(row: Dict[str, Any], design: Dict[str, Any]) -> None:
    task = design.get("task", "pose")

    def within(deg: np.ndarray, reach: np.ndarray) -> np.ndarray:
        ok = np.ones(deg.shape, dtype=bool)
        if task != "position":
            ok &= deg <= design["reach_deg"]
        if task != "orientation":
            ok &= reach <= design["reach_reach"]
        return ok

    t = np.append(np.asarray(row["t"], float), float(row["duration_s"]))
    deg = np.append(np.asarray(row["error_deg"], float), float(row["final_error_deg"]))
    reach = np.append(np.asarray(row["error_reach"], float), float(row["final_error_reach"]))
    ok = within(deg, reach)
    row["reached"] = bool(ok[-1])
    row["time_to_reach_s"] = first_time_within(t, ok)


def rescore_document(document: Dict[str, Any]) -> bool:
    rows = document.get("rows") or []
    if "design" not in document or not rows or "error_reach" not in rows[0] or "duration_s" not in rows[0]:
        return False
    current = IkDesign()
    old = {"reach_deg": document["design"]["reach_deg"], "reach_reach": document["design"]["reach_reach"]}
    new = {"reach_deg": current.reach_deg, "reach_reach": current.reach_reach}
    if old == new and "scoring" in document:
        return False
    document["design"].update(new)
    for row in rows:
        rescore_row(row, document["design"])
    scoring = document.setdefault("scoring", {"run_with": old})
    scoring.update(scored_with=new, time_from="stored time course (every %d ticks) and the last sample" % TRACE_TICKS)
    return True


def add_arguments(parser: argparse.ArgumentParser, command: str) -> None:
    parser.add_argument("only", nargs="*", help="MODEL or MODEL/BENCHMARK (default: every run file under results/)")


def main(args: argparse.Namespace, command: str = "rescore") -> int:
    paths: List[str] = []
    for pattern in (args.only or ["*"]):
        pattern = pattern if "/" in pattern else pattern + "/*"
        paths += sorted(glob.glob(os.path.join(results.ROOT, pattern, "*.json")))
    paths = [p for p in paths if os.path.basename(p) != results.SUMMARY + ".json" and "/comparison/" not in p and "/training/" not in p]
    progress = Progress("rescore", len(paths), unit="file")
    changed = 0
    for path in paths:
        progress.start(os.path.relpath(path, results.ROOT))
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
        if rescore_document(document):
            results.write(path, document)
            changed += 1
        progress.advance()
    progress.close()
    print("rescored %d of %d run files" % (changed, len(paths)))
    return 0
