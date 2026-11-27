"""Exact-loop twins of every closed-loop trial: the same start, goal and seed without neurons.

The exact cascade, damped least squares, the Jacobian transpose, and the cascade with every orientation source
of :mod:`ring_attractor_kinematics.rules.observer_rule`, all with the forward network's lag (0.3 s) and 0.01 rad share noise.
``python -m ring_attractor_kinematics benchmark twins`` recomputes them for every stored closed-loop trial (after a change to a
source, the spiking trials need not be run again).

A joint-loss trial gets its own twins (:func:`joint_loss_twins`). It also gets a check of whether its goal could still be
reached at all with the joint frozen where the spiking trial froze it (:func:`reachable_after`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from reaching_benchmark.geometry.chains import Chain, get_chain
from reaching_benchmark.geometry.rotations import wrap
from reaching_benchmark.targets import PosePair

from .. import results
from ..config import IkDesign
from ..rules import observer_rule
from ..rules.references import GAINS, RULES
from ..rules.cascade_rule import Trace, simulate

EXACT_DELAY_S = 0.3
SHARE_NOISE = 0.01
TWINS = ("cascade", "dls", "transpose") + observer_rule.CODES
CLOSED_LOOP = ("reach_kinematic", "reach_pole", "reach_gain", "reach_disturbance", "reach_gazebo")
#: twins of a joint-loss trial: the cascade and DLS intact, and with the joint jammed from onset, the rule either
#: uninformed (``_lock``: it keeps handing the joint a share) or informed (``_informed``: it leaves the joint out)
LOSS_TWINS = ("cascade", "dls", "cascade_lock", "cascade_informed", "dls_lock", "dls_informed")
#: the reachability check: DLS on the remaining joints, no delay, no noise, for this long, from the trial's own angles at
#: onset and from a few starts nearby, the remaining joints moved by up to this much (rad)
REACH_CHECK_S = 60.0
REACH_CHECK_RESTARTS = 3
REACH_CHECK_SPREAD = 0.3


def twin_entry(trace: Trace, design: IkDesign) -> Dict[str, Any]:
    return {"reached": trace.reached(design), "time_s": trace.time_to(design), "final_error_deg": trace.final_error_deg,
            "final_error_reach": trace.final_error_reach, "duration_s": float(trace.t[-1]),
            "path_ratio": trace.path_reach / trace.start_error_reach if trace.start_error_reach > 1e-9 else float("nan"),
            "t": trace.t[::4].tolist(), "error_deg": trace.error_deg[::4].tolist(), "error_reach": trace.error_reach[::4].tolist()}


def twins(chain: Chain, pair: PosePair, design: IkDesign, push: Optional[Tuple[float, int, float]]) -> Dict[str, Dict[str, Any]]:
    """The same pair in the exact loop: the cascade, the standard methods, and the cascade with every orientation source."""

    out: Dict[str, Dict[str, Any]] = {}
    for method in TWINS:
        settings: Dict[str, Any] = dict(delay_s=EXACT_DELAY_S, noise=SHARE_NOISE, rng=np.random.default_rng(int(pair.index)), push=push)
        method_design = design
        if method in RULES:
            settings["rule"] = RULES[method]
            method_design = IkDesign(**{**design.describe(), "kappa": design.kappa * GAINS[method]})
        elif method in observer_rule.CODES:
            settings["orientation"] = observer_rule.make_source(method, chain, EXACT_DELAY_S,
                                                                 observer_rule.SourceSettings(seed=int(pair.index)))
        trace = simulate(chain, pair.start, pair.goal, pair.goal_position, method_design, **settings)
        out[method] = twin_entry(trace, design)
    return out


def joint_loss_twins(chain: Chain, pair: PosePair, design: IkDesign, onset_s: float, joint: int,
                     names: Sequence[str] = LOSS_TWINS) -> Dict[str, Dict[str, Any]]:
    """The pair in the exact loop (0.3 s lag, 0.01 rad share noise) with ``joint`` jammed from ``onset_s``. Each twin
    freezes the joint where its own path has it at onset, which need not be where the spiking trial froze it."""

    out: Dict[str, Dict[str, Any]] = {}
    for name in names:
        method, _, kind = name.partition("_")
        settings: Dict[str, Any] = dict(delay_s=EXACT_DELAY_S, noise=SHARE_NOISE, rng=np.random.default_rng(int(pair.index)))
        if kind:
            settings.update(lock=(float(onset_s), int(joint)), informed=kind == "informed")
        method_design = design
        if method in RULES:
            settings["rule"] = RULES[method]
            method_design = IkDesign(**{**design.describe(), "kappa": design.kappa * GAINS[method]})
        out[name] = twin_entry(simulate(chain, pair.start, pair.goal, pair.goal_position, method_design, **settings), design)
    return out


def reachable_after(chain: Chain, q_onset: Sequence[float], joint: int, pair: PosePair, design: IkDesign) -> Dict[str, Any]:
    """Can the joints other than ``joint`` still bring the hand to the pair's goal, with ``joint`` held at its angle in
    ``q_onset``? This is a check of the goal, not of any controller.

    DLS on the remaining joints runs without delay or noise for :data:`REACH_CHECK_S` from ``q_onset``. If that misses,
    it runs again from :data:`REACH_CHECK_RESTARTS` nearby starts (the remaining joints moved by up to
    :data:`REACH_CHECK_SPREAD`, seeded by the pair and joint). The result is a lower bound on reachability: a local
    method can miss a goal that is reachable. Returns the first success, or the try that came closest."""

    k = int(joint)
    long = IkDesign(**{**design.describe(), "trial_s": REACH_CHECK_S})
    start = np.asarray(q_onset, dtype=float)
    rng = np.random.default_rng([int(pair.index), k, 97])
    starts = [start]
    for _ in range(REACH_CHECK_RESTARTS):
        moved = start + rng.uniform(-REACH_CHECK_SPREAD, REACH_CHECK_SPREAD, chain.n)
        moved[k] = start[k]
        starts.append(np.where(chain.circular_mask(), wrap(moved), chain.clip(moved)))
    best: Optional[Dict[str, Any]] = None
    for tries, q0 in enumerate(starts, 1):
        trace = simulate(chain, q0, pair.goal, pair.goal_position, long, delay_s=0.0, noise=0.0, rule=RULES["dls"],
                         lock=(0.0, k), informed=True)
        entry = {"reachable": trace.reached(design), "tries": tries, "final_error_deg": trace.final_error_deg,
                 "final_error_reach": trace.final_error_reach}
        if entry["reachable"]:
            return entry
        distance = max(([entry["final_error_deg"] / design.reach_deg] if design.orientation else [])
                       + ([entry["final_error_reach"] / design.reach_reach] if design.position else []))
        if best is None or distance < best["distance"]:
            best = dict(entry, distance=distance)
    assert best is not None
    best.pop("distance")
    return best


def _file_twins(job: Tuple[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Every trial of one run file (each case is seeded by its own pair, so files can run in parallel)."""

    path, document = job
    chain = get_chain(document["run"]["arm"])
    design = IkDesign(**document["design"])
    for row in document["rows"]:
        pair = PosePair(row["index"], np.array(row["start"]), np.array(row["start"]), np.array(row["goal_lpy"]),
                        np.array(row["goal_position"]))
        if row.get("disabled_joint") is not None:   # a joint-loss trial: its own twins and the reachability check
            joint = int(row["disabled_joint"])
            row["twins"] = joint_loss_twins(chain, pair, design, float(row["onset_s"]), joint)
            row["reachability"] = reachable_after(chain, row["q_onset"], joint, pair, design)
        else:
            push = (2.0, row["pushed_joint"], 0.3) if row.get("pushed_joint") is not None else None
            row["twins"] = twins(chain, pair, design, push)
    return path, document


def run_command(args: Any) -> None:
    """``--condition reach,pole``: only those benchmarks (default: every closed-loop benchmark and joint loss); one
    process per run file, up to ``--processes``."""

    import multiprocessing

    from reaching_benchmark.progress import Progress

    jobs = [(path, document) for name in (args.condition.split(",") if args.condition else CLOSED_LOOP + ("joint_loss",))
            for path, document in results.read_raw_files(name)]
    progress = Progress("twins", sum(len(document["rows"]) for _, document in jobs))
    context = multiprocessing.get_context("spawn")   # fresh workers: a fork would carry NEST's state along
    with context.Pool(int(getattr(args, "processes", None) or 20)) as pool:
        for path, document in pool.imap_unordered(_file_twins, jobs):
            results.shared.write(path, document)
            progress.advance(len(document["rows"]), path.rsplit("/", 1)[-1])
    progress.close()
