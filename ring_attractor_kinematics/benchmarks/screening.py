"""Stage-2 screening (Extended Data 2): every orientation source in the exact closed loop, and the go/no-go gate.

The loop is :func:`ring_attractor_kinematics.rules.cascade_rule.simulate` with each source of :mod:`ring_attractor_kinematics.rules.observer_rule`
in place of the turn's orientation, on the same pairs, with the forward network's lag and noise swept:

* FK delay τ ∈ {0.2, 0.3, 0.4} s, cue noise ∈ {0.8°, 1.6°} per axis, loop gain κ ∈ {1, 1.5, 2} 1/s;
* TIAGo7 and generic7, pose task, 30 pairs each; every share also gets 0.01 rad noise (as the spiking shares).

**Gate (declared before the run).**  At the nominal condition (τ = 0.3 s, 0.8°, κ = 1) and pooled over both
arms, a new source goes on to a spiking build only if, against *both* M0 and M1, it reaches at least as many
goals (no more than 2 fewer of 60), has a median path ratio no more than 10% longer, and is either ≥15% faster to
stay within the marks (median paired time) or has ≥25% less moving orientation error (median over moving ticks).
"""

from __future__ import annotations

import itertools
import multiprocessing
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.statistics import censored, mcnemar, paired, success_summary
from reaching_benchmark.targets import SCREEN_SEED, pose_pairs

from .. import results
from ..config import IkDesign
from ..rules import observer_rule
from ..rules.cascade_rule import simulate

ARMS = ("tiago7", "generic7")
DELAYS = (0.2, 0.3, 0.4)
NOISES = (0.8, 1.6)
GAINS = (1.0, 1.5, 2.0)
NOMINAL = (0.3, 0.8, 1.0)
NEW_SOURCES = ("M2", "M3", "M4", "M5", "MQ", "MB")
SHARE_NOISE = 0.01
MOVING_RAD_S = 0.05
GATE = {"faster": 0.15, "less_error": 0.25, "fewer_reached": 2, "longer_path": 0.10}


def run_trial(job: Tuple[str, int, str, float, float, float]) -> Dict[str, Any]:
    arm, index, code, delay, noise, kappa = job
    chain = get_chain(arm)
    pair = pose_pairs(chain, index + 1, SCREEN_SEED)[index]
    design = IkDesign(task="pose", kappa=kappa)
    source = observer_rule.make_source(code, chain, delay, observer_rule.SourceSettings(cue_noise_deg=noise, seed=index))
    trace = simulate(chain, pair.start, pair.goal, pair.goal_position, design, delay_s=delay, noise=SHARE_NOISE,
                     rng=np.random.default_rng(index), orientation=source)
    speed = np.max(np.abs(np.diff(np.vstack([pair.start, trace.q]), axis=0)), axis=1) / 0.05
    moving = speed > MOVING_RAD_S
    return {"arm": arm, "index": index, "source": code, "delay_s": delay, "cue_noise_deg": noise, "kappa": kappa,
            "reached": trace.reached(design), "time_s": trace.time_to(design), "trial_s": float(trace.t[-1]),
            "final_error_deg": trace.final_error_deg, "final_error_reach": trace.final_error_reach,
            "path_ratio": trace.path_reach / trace.start_error_reach if trace.start_error_reach > 1e-9 else float("nan"),
            "moving_error_deg": float(np.median(trace.estimate_error_deg[moving])) if moving.any() else float("nan"),
            "estimate_error_deg": float(np.median(trace.estimate_error_deg))}


def jobs(pairs: int, codes: Sequence[str] = observer_rule.CODES, full: bool = True) -> List[Tuple[str, int, str, float, float, float]]:
    conditions = list(itertools.product(DELAYS, NOISES, GAINS)) if full else [NOMINAL]
    return [(arm, i, code, d, n, k) for arm in ARMS for i in range(int(pairs)) for code in codes for d, n, k in conditions]


def compare(rows: List[Dict[str, Any]], code: str, reference: str, condition: Tuple[float, float, float]) -> Dict[str, Any]:
    def pick(c: str) -> List[Dict[str, Any]]:
        out = [r for r in rows if r["source"] == c and (r["delay_s"], r["cue_noise_deg"], r["kappa"]) == condition]
        return sorted(out, key=lambda r: (r["arm"], r["index"]))

    a, b = pick(code), pick(reference)
    limit = max([r["trial_s"] for r in a + b] or [25.0])
    time_a, time_b = censored([r["time_s"] for r in a], limit), censored([r["time_s"] for r in b], limit)
    moving_a, moving_b = np.array([r["moving_error_deg"] for r in a]), np.array([r["moving_error_deg"] for r in b])
    path_a, path_b = np.array([r["path_ratio"] for r in a]), np.array([r["path_ratio"] for r in b])
    return {"source": code, "reference": reference, "n": len(a),
            "reached": [int(sum(r["reached"] for r in a)), int(sum(r["reached"] for r in b))],
            "mcnemar": mcnemar([r["reached"] for r in a], [r["reached"] for r in b]),
            "time_median_s": [float(np.median(time_a)), float(np.median(time_b))], "time": paired(time_a, time_b),
            "faster": float(1.0 - np.median(time_a) / np.median(time_b)) if np.median(time_b) > 0 else 0.0,
            "moving_error_median_deg": [float(np.nanmedian(moving_a)), float(np.nanmedian(moving_b))],
            "moving_error": paired(moving_a, moving_b),
            "less_error": float(1.0 - np.nanmedian(moving_a) / np.nanmedian(moving_b)) if np.nanmedian(moving_b) > 0 else 0.0,
            "path_ratio_median": [float(np.nanmedian(path_a)), float(np.nanmedian(path_b))]}


def gate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rule": GATE, "nominal": list(NOMINAL), "sources": {}}
    for code in NEW_SOURCES:
        verdicts = []
        for reference in ("M0", "M1"):
            c = compare(rows, code, reference, NOMINAL)
            matched = (c["reached"][0] >= c["reached"][1] - GATE["fewer_reached"]
                       and c["path_ratio_median"][0] <= (1.0 + GATE["longer_path"]) * c["path_ratio_median"][1])
            better = c["faster"] >= GATE["faster"] or c["less_error"] >= GATE["less_error"]
            c.update(matched=bool(matched), better=bool(better), passes=bool(matched and better))
            verdicts.append(c)
        out["sources"][code] = {"against": verdicts, "passes": all(v["passes"] for v in verdicts)}
    out["finalists"] = [code for code, v in out["sources"].items() if v["passes"]]
    return out


def summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    table = []
    for code in observer_rule.CODES:
        for condition in sorted({(r["delay_s"], r["cue_noise_deg"], r["kappa"]) for r in rows}):
            group = [r for r in rows if r["source"] == code and (r["delay_s"], r["cue_noise_deg"], r["kappa"]) == condition]
            if not group:
                continue
            limit = max(r["trial_s"] for r in group)
            table.append({"source": code, "delay_s": condition[0], "cue_noise_deg": condition[1], "kappa": condition[2],
                          "success": success_summary([r["reached"] for r in group]),
                          "time_median_s": float(np.median(censored([r["time_s"] for r in group], limit))),
                          "moving_error_median_deg": float(np.nanmedian([r["moving_error_deg"] for r in group])),
                          "path_ratio_median": float(np.nanmedian([r["path_ratio"] for r in group]))})
    return {"table": table, "gate": gate(rows)}


def run(pairs: int = 30, processes: int = 20, full: bool = True, log=print) -> Dict[str, Any]:
    started = time.time()
    work = jobs(pairs, full=full)
    with multiprocessing.Pool(int(processes)) as pool:
        rows = pool.map(run_trial, work, chunksize=8)
    log("screening: %d trials in %.0f s" % (len(rows), time.time() - started))
    return {"benchmark": "screening", "pairs": int(pairs), "arms": list(ARMS), "delays_s": list(DELAYS), "cue_noises_deg": list(NOISES),
            "gains": list(GAINS), "share_noise_rad": SHARE_NOISE, "seed": SCREEN_SEED, "rows": rows, "summary": summary(rows),
            "wall_s": round(time.time() - started, 1)}


def run_command(args: Any) -> None:
    document = run(pairs=args.pairs or 30, processes=args.processes)
    results.write_raw("screening", "all", document)


def summarise(name: str = "screening") -> Dict[str, Any]:
    document = results.read_raw("screening")[0]
    out = summary(document["rows"])
    out.update(pairs=document["pairs"], arms=document["arms"], share_noise_rad=document["share_noise_rad"])
    return out
