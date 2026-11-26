"""``python -m reaching_benchmark compare BENCHMARK``: every model's summary of one benchmark side by side.

Reads ``results/<model>/<BENCHMARK>/summary.json`` for every model, writes ``results/comparison/<BENCHMARK>.json`` and
prints a table: targets reached with a Wilson 95% interval, median final errors, median time to enter tolerance for good
(missed targets counted at 25 s) and median path ratio. Every controller is paired with the ring-attractor kinematics
model (its default variant) on the targets both ran: exact McNemar test on reached, and paired differences of the final
errors and times (other − RAK, median with a percentile bootstrap interval). A learned model's training budget
(``results/<model>/training/training.json``) is shown beside it.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from . import results
from .geometry.chains import get_chain, workspace
from .statistics import censored, mcnemar, paired

MODELS = ("ring_attractor_kinematics", "dls_ik", "mpc_nn_dynamics", "pets")
LABELS = {"ring_attractor_kinematics": "RAK", "dls_ik": "DLS IK", "mpc_nn_dynamics": "NN dynamics + MPC", "pets": "PETS"}
#: the ring-attractor kinematics model's variants (summary group suffix → label)
RAK_VARIANTS = {"": "RAK", "_lead": "RAK, predictive encoding", "_observer": "RAK, observer", "_cascade": "Cascade rule, no spikes"}
REFERENCE = "RAK"


def add_arguments(parser: argparse.ArgumentParser, command: str) -> None:
    parser.add_argument("benchmark", help="e.g. reach_gazebo")


def entries(benchmark: str) -> List[Dict[str, Any]]:
    """One entry per controller (and variant) that has a summary of ``benchmark``."""

    out = []
    for model in MODELS:
        path = results.summary_path(model, benchmark)
        if not os.path.exists(path):
            continue
        summary = results.read_summary(model, benchmark)
        training = os.path.join(results.ROOT, model, "training", "training.json")
        budget = None
        if os.path.exists(training):
            with open(training, encoding="utf-8") as stream:
                budget = {k: v for k, v in json.load(stream).items() if k in ("transitions", "episodes", "wall_s", "iterations", "device")}
        if model == "ring_attractor_kinematics":
            for key, group in sorted(summary.get("groups", {}).items()):
                suffix = key[len("%s_%s_k%g" % (group["arm"], group["task"], group["kappa"])):]
                if group.get("condition") or suffix not in RAK_VARIANTS:
                    continue
                stats = group["network"]
                out.append(dict(stats, model=model, label=RAK_VARIANTS[suffix], arm=group["arm"], run=key, training=None,
                                controller=group.get("controller")))
        else:
            for name, run in sorted(summary.get("runs", {}).items()):
                out.append(dict(run, model=model, label=LABELS[model], arm=run["run"]["arm"], run=name, training=budget))
    return out


def against(reference: Dict[str, Any], other: Dict[str, Any], limit: float = 25.0) -> Dict[str, Any]:
    a, b = reference["per_trial"], other["per_trial"]
    common = sorted(set(a["index"]) & set(b["index"]))
    pick = lambda t, name: [t[name][t["index"].index(i)] for i in common]  # noqa: E731
    return {"pairs": len(common), "reached": [int(sum(pick(a, "reached"))), int(sum(pick(b, "reached")))],
            "mcnemar": mcnemar(pick(b, "reached"), pick(a, "reached")),
            "final_error_deg": paired(pick(b, "final_error_deg"), pick(a, "final_error_deg")),
            "final_error_reach": paired(pick(b, "final_error_reach"), pick(a, "final_error_reach")),
            "time": paired(censored(pick(b, "time_s"), limit), censored(pick(a, "time_s"), limit))}


def compare(benchmark: str) -> Dict[str, Any]:
    rows = entries(benchmark)
    reference: Optional[Dict[str, Any]] = next((r for r in rows if r["label"] == REFERENCE), None)
    table = []
    for r in rows:
        radius_mm = 1000.0 * workspace(get_chain(r["arm"])).radius
        line = {"model": r["model"], "label": r["label"], "run": r["run"], "arm": r["arm"],
                "reached": r["success"]["reached"], "n": r["success"]["n"], "wilson_95": r["success"]["ci"],
                "final_error_median_deg": r["final_error_median_deg"], "final_error_median_mm": radius_mm * r["final_error_median_reach"],
                "time_median_s": r["time_median_s"], "path_ratio_median": r["path_ratio_median"],
                "controller": r.get("controller"), "training": r.get("training"), "per_trial": r["per_trial"], "within": r.get("within")}
        if reference is not None and r is not reference:
            line["against_rak"] = against(reference, r)
        table.append(line)
    return {"benchmark": benchmark, "reference": REFERENCE, "controllers": table}


def render(document: Dict[str, Any]) -> str:
    lines = ["%-28s %-34s %9s %12s %9s %10s %8s %6s" % ("controller", "run", "reached", "Wilson 95%", "deg", "mm", "time s", "path")]
    for r in document["controllers"]:
        lines.append("%-28s %-34s %5d/%-3d %5.0f–%-5.0f%% %9.2f %10.1f %8.2f %6.2f" % (
            r["label"], r["run"], r["reached"], r["n"], 100 * r["wilson_95"][0], 100 * r["wilson_95"][1], r["final_error_median_deg"],
            r["final_error_median_mm"], r["time_median_s"], r["path_ratio_median"]))
    return "\n".join(lines)


def main(args: argparse.Namespace, command: str = "compare") -> int:
    document = compare(args.benchmark)
    path = results.write(os.path.join(results.ROOT, "comparison", args.benchmark + ".json"), document, digits=6, pretty=True)
    print(render(document))
    print("wrote", path)
    return 0
