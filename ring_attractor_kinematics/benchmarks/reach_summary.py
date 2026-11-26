"""Summaries of the closed-loop benchmarks (B2–B6): success with 95% intervals, paired tests, time courses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from reaching_benchmark.statistics import censored, mcnemar, paired, success_summary
from reaching_benchmark.summary import reach_entry

from .. import results
from ..rules import observer_rule

TWINS = ("cascade", "dls", "transpose") + observer_rule.CODES


def within_curve(rows: List[Dict[str, Any]], design: Dict[str, Any], source: Optional[str] = None,
                 grid: Sequence[float] = tuple(np.arange(0.0, 25.01, 0.25))) -> List[float]:
    """Fraction of trials within the marks at each time (a trial that ended stays as it ended)."""

    out = []
    for t in grid:
        inside = []
        for row in rows:
            r = row["twins"][source] if source else row
            times, deg, reach = np.array(r["t"]), np.array(r["error_deg"]), np.array(r["error_reach"])
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


def summarise_rows(rows: List[Dict[str, Any]], design: Dict[str, Any]) -> Dict[str, Any]:
    if rows and "twins" not in rows[0]:              # the cascade rule without spikes: the shared fields only
        return {"trials": len(rows), "network": reach_entry(rows, design), "twins": {}}
    limit = float(design["trial_s"])
    network = {"success": success_summary([r["reached"] for r in rows]),
               "time_median_s": float(np.median(censored([r["time_to_reach_s"] for r in rows], limit))),
               "final_error_median_deg": float(np.median([r["final_error_deg"] for r in rows])),
               "final_error_median_reach": float(np.median([r["final_error_reach"] for r in rows])),
               "path_ratio_median": float(np.nanmedian([r["path_ratio"] for r in rows])),
               "jitter_median_rad_s": float(np.nanmedian([r["jitter_rad_s"] for r in rows])),
               "decoded_error_median_deg": float(np.median([r["decoded_error_median_deg"] for r in rows])),
               "turn_error_moving_median_deg": float(np.nanmedian([r.get("turn_error_moving_deg", np.nan) for r in rows]))
               if any("turn_error_moving_deg" in r for r in rows) else None,
               "observer_fallback_share": float(np.mean([r["observer_fallback_share"] for r in rows]))
               if all("observer_fallback_share" in r for r in rows) and rows else None,
               "within": within_curve(rows, design),
               "per_trial": {"index": [r["index"] for r in rows], "reached": [r["reached"] for r in rows],
                             "final_error_deg": [r["final_error_deg"] for r in rows], "final_error_reach": [r["final_error_reach"] for r in rows],
                             "time_s": [r["time_to_reach_s"] for r in rows], "path_ratio": [r["path_ratio"] for r in rows]}}
    methods = {}
    for method in TWINS:
        twin = [r["twins"][method] for r in rows]
        methods[method] = {"success": success_summary([t["reached"] for t in twin]),
                           "time_median_s": float(np.median(censored([t["time_s"] for t in twin], limit))),
                           "final_error_median_deg": float(np.median([t["final_error_deg"] for t in twin])),
                           "final_error_median_reach": float(np.median([t["final_error_reach"] for t in twin])),
                           "path_ratio_median": float(np.nanmedian([t["path_ratio"] for t in twin])),
                           "against_network": {"mcnemar": mcnemar([t["reached"] for t in twin], [r["reached"] for r in rows]),
                                               "time": paired(censored([t["time_s"] for t in twin], limit),
                                                              censored([r["time_to_reach_s"] for r in rows], limit)),
                                               "final_error_deg": paired([t["final_error_deg"] for t in twin], [r["final_error_deg"] for r in rows]),
                                               "final_error_reach": paired([t["final_error_reach"] for t in twin], [r["final_error_reach"] for r in rows])},
                           "within": within_curve(rows, design, method)}
    return {"trials": len(rows), "network": network, "twins": methods}


def summarise(name: str) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    for document in results.read_raw(name):
        run = document["run"]
        key = "%s_%s_k%g%s%s" % (run["arm"], document["design"]["task"], run["kappa"], "_" + run["condition"] if run.get("condition") else "",
                                 "_" + run["method"] if run.get("method", "ring") != "ring" else "") + (
            "_" + run["network_method"] if run.get("network_method", "matrix") != "matrix" else "")
        group = groups.setdefault(key, {"design": document["design"], "run": run, "rows": [], "seeds": set(),
                                        "neurons": document.get("neurons", 0), "synapses": document.get("synapses", 0),
                                        "simulated_s": 0.0, "wall_s": 0.0})
        seed = run.get("seed", 0)                    # 0: nothing random (the cascade rule without spikes)
        group["rows"] += [dict(r, seed=seed) for r in document["rows"]]
        group["seeds"].add(seed)
        group["simulated_s"] += document["simulated_s"]
        group["wall_s"] += document["wall_s"]
    out: Dict[str, Any] = {"benchmark": name, "groups": {}}
    for key, group in groups.items():
        rows = group["rows"]
        networks = {seed: summarise_rows([r for r in rows if r["seed"] == seed], group["design"])["network"] for seed in sorted(group["seeds"])}
        first = [r for r in rows if r["seed"] == min(group["seeds"])]
        out["groups"][key] = {"arm": group["run"]["arm"], "task": group["design"]["task"], "kappa": group["run"]["kappa"],
                              "condition": group["run"].get("condition"), "method": group["run"].get("method", "ring"),
                              "network_method": group["run"].get("network_method", "matrix"),
                              "neurons": group["neurons"], "synapses": group["synapses"],
                              "speed": group["simulated_s"] / group["wall_s"] if group["wall_s"] else None,
                              "seeds": sorted(group["seeds"]),
                              "success_by_seed": {seed: network["success"] for seed, network in networks.items()},
                              # every kernel seed on its own (the top-level numbers below are the first seed's), and every
                              # (seed, pair) trial together; a pair repeated at another seed is not a new target
                              "by_seed": {str(seed): {k: v for k, v in network.items() if k != "within"} for seed, network in networks.items()},
                              "all_seeds_success": success_summary([r["reached"] for r in rows]),
                              **summarise_rows(first, group["design"])}
    out["old_vs_new"] = paired_methods(out["groups"], "lead", "matrix")
    out["old_vs_observer"] = paired_methods(out["groups"], "observer", "matrix")
    out["new_vs_observer"] = paired_methods(out["groups"], "observer", "lead")
    return out


def paired_trials(a: Dict[str, List[Any]], b: Dict[str, List[Any]], limit: float = 25.0) -> Dict[str, Any]:
    """Paired tests of per-trial records ``b`` against ``a`` on the pair indices both have: counts [a, b], differences b − a."""

    common = sorted(set(a["index"]) & set(b["index"]))
    pick = lambda t, name: [t[name][t["index"].index(i)] for i in common]  # noqa: E731
    return {
        "pairs": len(common),
        "reached": [int(sum(pick(a, "reached"))), int(sum(pick(b, "reached")))],
        "mcnemar": mcnemar(pick(b, "reached"), pick(a, "reached")),
        "time": paired(censored(pick(b, "time_s"), limit), censored(pick(a, "time_s"), limit)),
        "final_error_deg": paired(pick(b, "final_error_deg"), pick(a, "final_error_deg")),
        "final_error_reach": paired(pick(b, "final_error_reach"), pick(a, "final_error_reach")),
        "path_ratio": paired(pick(b, "path_ratio"), pick(a, "path_ratio"))}


def paired_methods(groups: Dict[str, Any], method: str, reference: str) -> Dict[str, Any]:
    """Paired tests of one network method against another on the pairs both ran (keyed by the old method's group);
    each entry is [reference, method] for counts, and method − reference for the paired differences. The top level pairs
    the first kernel seed; ``by_seed`` pairs every seed both groups ran, a pair with its own seed's twin only."""

    out: Dict[str, Any] = {}
    for key, new in groups.items():
        if new.get("network_method") != method:
            continue
        base = key[: -len("_" + method)]
        old = groups.get(base if reference == "matrix" else base + "_" + reference)
        if old is None:
            continue
        out[base] = paired_trials(old["network"]["per_trial"], new["network"]["per_trial"])
        seeds = sorted(set(old.get("by_seed", {})) & set(new.get("by_seed", {})), key=int)
        out[base]["by_seed"] = {seed: paired_trials(old["by_seed"][seed]["per_trial"], new["by_seed"][seed]["per_trial"]) for seed in seeds}
    return out
