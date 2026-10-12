#!/usr/bin/env python3
"""IK test 2: the pose rule itself, with exact kinematics and no neurons (NumPy only).

    python3 -B scripts/ik_study.py --out docs/ik/results/data/study.json

For every arm 60 start poses and goals (not the pairs of the reported trials).  The loop is closed as
the spiking network closes it: the comparison sees the arm 0.4 s late, every share carries noise, and
the decoded pose is off by a fixed 1 deg and 1.3% of the reach.  Compared: what a share is divided by
(the mean over the joint ranges, the largest value, the exact value), the cascade against every joint
getting the whole error, the cascade's order and a second sweep, the balance between turning and
moving, goals near and anywhere, and two standard methods on the same pairs.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_so3.experiments import write_json  # noqa: E402
from ring_so3.ik import rules as so3  # noqa: E402
from ring_se3.chains import get_chain  # noqa: E402
from ring_se3.ik import rules  # noqa: E402
from ring_se3.ik.design import PoseIkDesign  # noqa: E402
from ring_se3.ik.experiments import SPREAD, default_task  # noqa: E402
from ring_se3.ik.pairs import STUDY_SEED, pairs  # noqa: E402
from ring_se3.standard import ik as standard  # noqa: E402

MODELS = ("tiago4", "generic4", "generic7", "tiago7")
#: as the spiking network has them: delay, noise of a share (rad), and how far the decoded pose is off
BASE = dict(delay_s=0.4, noise=0.01, bias_deg=1.0, bias_reach=0.013)


def hand_first(axes, move, turn, offset, design, lever, caps, room):
    """The cascade run from the last joint to the first."""

    back = np.arange(axes.shape[0])[::-1]
    shares, _ = rules.pose_cascade(axes[back], move[back], turn, offset, design, rules.Levers(lever.ratios[back], lever.mean_square[back]),
                                   caps[back], room[back])
    out = np.zeros(axes.shape[0])
    out[back] = shares
    return out


def two_sweeps(axes, move, turn, offset, design, lever, caps, room):
    """The cascade run down the arm twice in one tick: the second time on what the first left."""

    first, rests = rules.pose_cascade(axes, move, turn, offset, design, lever, caps, room)
    second, _ = rules.pose_cascade(axes, move, rests[-1, :3], rests[-1, 3:], design, lever, caps, room)
    return np.clip(first + second, -caps, caps)


OWN_RULES = {"hand_first": hand_first, "two_sweeps": two_sweeps}


def run(job):
    name, group, label, design_settings, rule, spread, trials, settings = job
    chain = get_chain(name)
    design = PoseIkDesign(**dict(dict(task=default_task(chain)), **design_settings))
    conditions = dict(BASE, **settings)
    bias_deg, bias_reach = conditions.pop("bias_deg"), conditions.pop("bias_reach")
    rows = []
    for pair in pairs(chain, trials, STUDY_SEED, spread):
        rng = np.random.default_rng([STUDY_SEED, chain.n, pair.index])
        axis, shift = rng.normal(size=3), rng.normal(size=3)
        trace = rules.simulate(chain, pair.start, pair.goal, pair.goal_position, design, rng=rng,
                               readout_bias=so3.small_rotation(np.radians(bias_deg) * axis / np.linalg.norm(axis)),
                               position_bias=bias_reach * shift / np.linalg.norm(shift),
                               rule=dict(standard.RULES, **OWN_RULES)[rule] if rule else None, **conditions)
        rows.append((trace.reached(design), trace.final_error_deg, trace.final_error_reach, trace.time_to(design), trace.at_limit,
                     trace.path_reach / trace.start_error_reach if trace.start_error_reach > 0.1 else np.nan,
                     trace.start_error_deg, trace.start_error_reach))
    r = np.array(rows, dtype=float)
    ok = r[:, 0] > 0
    row = {
        "model": name, "task": design.task, "group": group, "label": label, "rule": rule or "cascade", "normaliser": design.normaliser,
        "cascade": design.cascade, "balance": design.balance, "kappa": design.kappa, "spread": spread, "trials": int(trials),
        "conditions": dict(conditions, bias_deg=bias_deg, bias_reach=bias_reach),
        "reached_pct": float(100.0 * ok.mean()),
        "final_median_deg": float(np.median(r[:, 1])), "final_p90_deg": float(np.percentile(r[:, 1], 90)),
        "final_median_reach": float(np.median(r[:, 2])), "final_p90_reach": float(np.percentile(r[:, 2], 90)),
        "time_to_reach_s": float(np.nanmedian(r[:, 3])) if np.any(np.isfinite(r[:, 3])) else None,
        "path_over_straight": float(np.nanmedian(r[ok, 5])) if np.any(np.isfinite(r[ok, 5])) else None,
        "missed": int((~ok).sum()), "missed_at_limit": int((~ok & (r[:, 4] > 0)).sum()),
        "start_median_deg": float(np.median(r[:, 6])), "start_median_reach": float(np.median(r[:, 7])),
    }
    line = "%-9s %-46s reached %5.1f%% | final %5.2f deg %5.2f%% of the reach | within the marks after %s s | path %s | missed %d (at a limit %d)" % (
        name, label, row["reached_pct"], row["final_median_deg"], 100 * row["final_median_reach"],
        "%4.1f" % row["time_to_reach_s"] if row["time_to_reach_s"] is not None else "   -",
        "%.2f" % row["path_over_straight"] if row["path_over_straight"] else "-", row["missed"], row["missed_at_limit"])
    return row, line


def plan(trials):
    jobs = []

    def add(name, group, label, design=None, rule=None, spread=SPREAD, **settings):
        jobs.append((name, group, label, design or {}, rule, spread, trials, settings))

    for name in MODELS:
        add(name, "rule", "cascade (as built: divided by the mean)")
        add(name, "rule", "cascade, divided by the largest value", dict(normaliser="bound"))
        add(name, "rule", "cascade, divided by the exact value", dict(normaliser="exact"))
        add(name, "rule", "all joints at once", dict(cascade=False, kappa=0.5))
        add(name, "rule", "damped least squares", rule="dls")
        add(name, "rule", "Jacobian transpose", dict(kappa=standard.GAINS["transpose"]), rule="transpose")
        add(name, "order", "cascade, from the hand to the base", rule="hand_first")
        add(name, "order", "cascade, down the arm twice per tick", rule="two_sweeps")
        for balance in (2.0, 4.0, 5.0):
            add(name, "balance", "cascade, balance %g" % balance, dict(balance=balance))
        for spread in (0.4, 1.2):
            add(name, "spread", "cascade, goals within %.1f rad" % spread, spread=spread)
        add(name, "anywhere", "cascade, goals anywhere", spread=None)
        add(name, "anywhere", "cascade divided by the exact value, goals anywhere", dict(normaliser="exact"), spread=None)
        add(name, "anywhere", "damped least squares, goals anywhere", rule="dls", spread=None)
    for name in ("generic7", "tiago7"):
        for kappa in (0.75, 1.5):
            add(name, "needs", "cascade, gain %.2f" % kappa, dict(kappa=kappa))
        for delay in (0.25, 0.55):
            add(name, "needs", "cascade, delay %.2f s" % delay, delay_s=delay)
        add(name, "needs", "cascade, share noise 0.02 rad", noise=0.02)
        add(name, "needs", "cascade, readout off by 2 deg and 2.6%", bias_deg=2.0, bias_reach=0.026)
        add(name, "needs", "cascade, nothing in the way", delay_s=0.0, noise=0.0, bias_deg=0.0, bias_reach=0.0)
        add(name, "needs", "orientation alone (the cascade of the SO-3 repo)", dict(task="orientation"))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=8, help="conditions side by side")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    started = time.time()
    rows = []
    with Pool(max(args.jobs, 1)) as pool:
        for row, line in pool.imap(run, plan(args.trials)):
            rows.append(row)
            print(line, flush=True)
    write_json(args.out, {"experiment": "pose_ik_study", "seed": STUDY_SEED, "base": BASE, "spread": SPREAD,
                          "design": PoseIkDesign().describe(), "rows": rows, "wall_s": round(time.time() - started, 1)})
    print("written: %s (%.0f s)" % (args.out, time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
