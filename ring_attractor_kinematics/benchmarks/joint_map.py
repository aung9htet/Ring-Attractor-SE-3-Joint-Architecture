"""The joint-loss map in the exact loop (no neurons): every joint disabled in turn on every pair (supplement).

For the 30 TIAGo7 reaching pairs (the ``reach`` and ``joint_loss`` pairs), each joint is jammed in turn, at the start
of the reach (0 s) or part-way through it (2 s, as in ``joint_loss``). This is done for the pose task (the paper's)
and for the position task (where the arm keeps spare joints). Each case runs in the exact loop with the spiking
network's lag (0.3 s) and share noise (0.01 rad), under four rules:

* the cascade, uninformed (``cascade_lock``): it keeps handing the jammed joint a share, like ``lock`` and ``motor``;
* the cascade, informed (``cascade_informed``): the joint gets no share, and the joints after it are handed the whole
  remainder, like ``allocation``;
* DLS uninformed and informed (``dls_lock``, ``dls_informed``: the second leaves the joint's column out).

Each case also gets the reachability check of :func:`~ring_attractor_kinematics.benchmarks.twins.reachable_after`, run from the angles
the arm had when the joint jammed. Pairs with nothing disabled are the intact reference. ``--processes`` sets the
worker count; ``--pairs`` sets how many pairs.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from reaching_benchmark.geometry.chains import get_chain, workspace
from reaching_benchmark.progress import Progress
from reaching_benchmark.statistics import success_summary
from reaching_benchmark.targets import TEST_SEED, pose_pairs

from .. import results
from ..config import IkDesign
from ..rules.references import GAINS, RULES
from ..rules.cascade_rule import simulate
from .twins import EXACT_DELAY_S, LOSS_TWINS, SHARE_NOISE, joint_loss_twins, reachable_after, twin_entry

ARM = "tiago7"
TASKS = ("pose", "position")
ONSETS_S = (0.0, 2.0)
RULE_NAMES = tuple(name for name in LOSS_TWINS if "_" in name)        # cascade_lock, cascade_informed, dls_lock, dls_informed


def run_case(case: Tuple[str, int, int, float]) -> Dict[str, Any]:
    """One (task, pair, joint, onset): the four rules and the reachability check."""

    task, index, joint, onset_s = case
    chain = get_chain(ARM)
    design = IkDesign(task=task)
    pair = pose_pairs(chain, index + 1, TEST_SEED)[index]
    out = joint_loss_twins(chain, pair, design, onset_s, joint, RULE_NAMES)
    # where the arm was when the joint jammed: the same for every rule here (same noise stream up to the onset)
    trace = simulate(chain, pair.start, pair.goal, pair.goal_position, design, delay_s=EXACT_DELAY_S, noise=SHARE_NOISE,
                     rng=np.random.default_rng(int(pair.index)), lock=(onset_s, joint))
    before = int(np.searchsorted(trace.t, onset_s + 1e-9, side="left"))
    q_onset = trace.q[before - 1] if before > 0 else np.asarray(pair.start, dtype=float)
    return {"task": task, "index": index, "joint": joint, "onset_s": onset_s,
            "rules": {name: {key: value for key, value in entry.items() if key not in ("t", "error_deg", "error_reach")}
                      for name, entry in out.items()},
            "reachability": reachable_after(chain, q_onset, joint, pair, design)}


def run_intact(case: Tuple[str, int]) -> Dict[str, Any]:
    task, index = case
    chain = get_chain(ARM)
    design = IkDesign(task=task)
    pair = pose_pairs(chain, index + 1, TEST_SEED)[index]
    out = {}
    for method in ("cascade", "dls"):
        settings: Dict[str, Any] = dict(delay_s=EXACT_DELAY_S, noise=SHARE_NOISE, rng=np.random.default_rng(int(pair.index)))
        method_design = design
        if method in RULES:
            settings["rule"] = RULES[method]
            method_design = IkDesign(**{**design.describe(), "kappa": design.kappa * GAINS[method]})
        entry = twin_entry(simulate(chain, pair.start, pair.goal, pair.goal_position, method_design, **settings), design)
        out[method] = {key: value for key, value in entry.items() if key not in ("t", "error_deg", "error_reach")}
    return {"task": task, "index": index, "rules": out}


def run_command(args: Any) -> None:
    pairs = int(args.pairs or 30)
    chain = get_chain(ARM)
    cases = [(task, index, joint, onset) for task in TASKS for index in range(pairs) for joint in range(chain.n) for onset in ONSETS_S]
    intact_cases = [(task, index) for task in TASKS for index in range(pairs)]
    started = time.time()
    progress = Progress("joint_map", len(cases) + len(intact_cases), unit="case")
    rows: List[Dict[str, Any]] = []
    intact: List[Dict[str, Any]] = []
    with multiprocessing.Pool(int(args.processes)) as pool:
        for row in pool.imap_unordered(run_intact, intact_cases):
            intact.append(row)
            progress.advance(1, "intact pairs")
        for row in pool.imap_unordered(run_case, cases, chunksize=2):
            rows.append(row)
            progress.advance(1, "%s, pair %d, joint %d jammed at %g s" % (row["task"], row["index"], row["joint"] + 1, row["onset_s"]))
    progress.close()
    rows.sort(key=lambda r: (r["task"], r["index"], r["joint"], r["onset_s"]))
    intact.sort(key=lambda r: (r["task"], r["index"]))
    results.write_raw("joint_map", "all", {"benchmark": "joint_map", "arm": ARM, "pairs": pairs, "tasks": list(TASKS),
                                           "onsets_s": list(ONSETS_S), "rules": list(RULE_NAMES), "delay_s": EXACT_DELAY_S,
                                           "share_noise_rad": SHARE_NOISE, "wall_s": time.time() - started,
                                           "radius_m": workspace(chain).radius, "intact": intact, "rows": rows})


def summarise(name: str = "joint_map") -> Dict[str, Any]:
    document = results.read_raw_part(name, "all")
    if not document:
        return {"benchmark": name}
    out: Dict[str, Any] = {"benchmark": name, "pairs": document["pairs"], "tasks": {}}
    for task in document["tasks"]:
        intact = [r for r in document["intact"] if r["task"] == task]
        entry: Dict[str, Any] = {"intact": {m: success_summary([r["rules"][m]["reached"] for r in intact]) for m in ("cascade", "dls")},
                                 "onsets": {}}
        for onset in document["onsets_s"]:
            rows = [r for r in document["rows"] if r["task"] == task and r["onset_s"] == onset]
            joints = sorted({r["joint"] for r in rows})
            per_joint = {}
            for joint in joints:
                mine = [r for r in rows if r["joint"] == joint]
                per_joint[str(joint + 1)] = {"reachable": success_summary([r["reachability"]["reachable"] for r in mine]),
                                             **{rule: success_summary([r["rules"][rule]["reached"] for r in mine]) for rule in document["rules"]}}
            entry["onsets"]["%g" % onset] = {
                "pooled": {"reachable": success_summary([r["reachability"]["reachable"] for r in rows]),
                           **{rule: success_summary([r["rules"][rule]["reached"] for r in rows]) for rule in document["rules"]},
                           **{rule + "_of_reachable": success_summary([r["rules"][rule]["reached"] for r in rows if r["reachability"]["reachable"]])
                              for rule in document["rules"]}},
                "by_joint": per_joint}
        out["tasks"][task] = entry
    return out
