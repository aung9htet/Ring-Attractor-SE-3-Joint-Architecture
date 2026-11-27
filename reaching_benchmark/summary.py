"""The summary of a reaching benchmark for any controller: success with a Wilson 95% interval, final errors, timing, path
length, and the fraction of trials within the marks over time, per run file.

Every model's reaching summary has these fields under the same names (the ring-attractor kinematics model's summary
adds its own, such as decoded errors and exact-loop twins); :mod:`reaching_benchmark.compare` reads them side by side.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import results
from .statistics import censored, success_summary

GRID = tuple(np.arange(0.0, 25.01, 0.25))


def within_curve(rows: List[Dict[str, Any]], design: Dict[str, Any], grid: Sequence[float] = GRID) -> List[float]:
    """Fraction of trials within the marks at each time (a trial that ended stays as it ended)."""

    out = []
    for t in grid:
        inside = []
        for row in rows:
            times, deg, reach = np.array(row["t"]), np.array(row["error_deg"]), np.array(row["error_reach"])
            i = int(np.searchsorted(times, t, side="right")) - 1
            if i < 0:
                inside.append(False)
                continue
            ok = True
            if design["task"] != "position":
                ok &= deg[i] <= design["reach_deg"]
            if design["task"] != "orientation":
                ok &= reach[i] <= design["reach_reach"]
            inside.append(bool(ok))
        out.append(float(np.mean(inside)) if inside else float("nan"))
    return out


def reach_entry(rows: List[Dict[str, Any]], design: Dict[str, Any], radius_m: Optional[float] = None) -> Dict[str, Any]:
    """Success, medians and per-trial records of one run (every row of its file)."""

    limit = float(design["trial_s"])
    entry = {"trials": len(rows),
             "success": success_summary([r["reached"] for r in rows]),
             "time_median_s": float(np.median(censored([r["time_to_reach_s"] for r in rows], limit))),
             "final_error_median_deg": float(np.median([r["final_error_deg"] for r in rows])),
             "final_error_median_reach": float(np.median([r["final_error_reach"] for r in rows])),
             "path_ratio_median": float(np.nanmedian([r["path_ratio"] for r in rows])),
             "jitter_median_rad_s": float(np.nanmedian([r["jitter_rad_s"] for r in rows])),
             "within": within_curve(rows, design),
             "per_trial": {"index": [r["index"] for r in rows], "reached": [r["reached"] for r in rows],
                           "final_error_deg": [r["final_error_deg"] for r in rows],
                           "final_error_reach": [r["final_error_reach"] for r in rows],
                           "time_s": [r["time_to_reach_s"] for r in rows], "path_ratio": [r["path_ratio"] for r in rows]}}
    if radius_m is not None:
        entry["final_error_median_mm"] = 1000.0 * float(radius_m) * entry["final_error_median_reach"]
    return entry


SHARD = re.compile(r"_shard(\d+)of(\d+)$")


def summarise(model: str, benchmark: str) -> Dict[str, Any]:
    """One entry per run of ``results/<model>/<benchmark>/``, keyed by its run file's name; the shards of one run
    (``…_shard0of3``, ``…_shard1of3``, …) are put together under the name without the shard."""

    from .geometry.chains import get_chain, workspace

    runs: Dict[str, Dict[str, Any]] = {}
    for path, document in results.read_runs(model, benchmark):
        name = SHARD.sub("", os.path.splitext(os.path.basename(path))[0])
        entry = runs.setdefault(name, {"run": dict(document["run"]), "design": document["design"], "controller": document.get("controller"),
                                       "rows": [], "wall_s": 0.0, "simulated_s": 0.0, "shards": []})
        entry["rows"] += document["rows"]
        entry["wall_s"] += float(document.get("wall_s") or 0.0)
        entry["simulated_s"] += float(document.get("simulated_s") or 0.0)
        entry["shards"].append(list(document.get("shard", [0, 1])))
    out: Dict[str, Any] = {"model": model, "benchmark": benchmark, "runs": {}}
    for name, entry in runs.items():
        rows = sorted(entry["rows"], key=lambda r: r["index"])
        if len({r["index"] for r in rows}) != len(rows):
            raise ValueError("%s/%s %s: a pair appears in two shards" % (model, benchmark, name))
        radius = workspace(get_chain(entry["run"]["arm"])).radius
        out["runs"][name] = dict(reach_entry(rows, entry["design"], radius), run=entry["run"], controller=entry["controller"],
                                 wall_s=entry["wall_s"], simulated_s=entry["simulated_s"], shards=sorted(entry["shards"]))
    return out
