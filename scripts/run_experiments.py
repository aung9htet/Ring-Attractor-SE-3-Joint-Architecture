#!/usr/bin/env python3
"""Every run behind docs/results.md, started side by side.

    python3 -B scripts/run_experiments.py --jobs 3 --threads 6
    python3 -B scripts/run_experiments.py --list                  # print the commands, run nothing
    python3 -B scripts/run_experiments.py --only static --models tiago7 generic7      # part of it

Groups: ``calibrate`` (the pose calibrations the runs need), ``static`` (tests 4-6 and 10: the
twelve models), ``tracking`` (test 7), ``dropout`` (test 8), ``trail`` (the shift-register
experiment), ``robust`` (test 9: variants on generic4 and tiago7).  A run whose output file exists is skipped, so an interrupted session
can be started again.  Logs go to ``docs/results/data/logs/``.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "docs", "results", "data")
GROUPS = ("calibrate", "static", "tracking", "dropout", "trail", "robust")
MODELS = ["generic%d" % n for n in range(2, 8)] + ["tiago%d" % n for n in range(2, 8)]
TRACKING_MODELS = ("generic3", "generic7", "tiago3", "tiago7")
TRACKING_SPEEDS = (0.25, 0.5, 1.0)
#: name -> extra arguments of scripts/run_tracking.py (on generic3 and tiago7 at 0.5 rad/s)
TRACKING_VARIANTS = {
    "anchor_only": ["--no-path-integration"],
    "anchor_lead": ["--anchor-lead", "0.2"],
    "joint_lead": ["--lead"],
}
DROPOUT_MODELS = ("generic3", "tiago7")
#: name -> arguments of scripts/run_dropout.py
DROPOUTS = {
    "rest": ["--speed", "0", "--gaps", "2", "5", "10"],
    "move010": ["--speed", "0.1", "--gaps", "2", "5", "10"],
    "move010_no_pi": ["--speed", "0.1", "--gaps", "2", "5", "10", "--no-path-integration"],
    "move025": ["--speed", "0.25", "--gaps", "2", "5", "10"],
    "move025_no_pi": ["--speed", "0.25", "--gaps", "2", "5", "10", "--no-path-integration"],
    "move050": ["--speed", "0.5", "--gaps", "2", "5"],
    "move050_no_pi": ["--speed", "0.5", "--gaps", "2", "5", "--no-path-integration"],
}
ROBUST_MODELS = ("generic4", "tiago7")
ROBUST_POSES = 100
#: name -> extra arguments of scripts/run_static.py
VARIANTS = {
    "sheets05": ["--silence-sheets", "0.05"], "sheets10": ["--silence-sheets", "0.10"], "sheets20": ["--silence-sheets", "0.20"],
    "products05": ["--silence", "0.05"],
    "noise050": ["--noise", "50"], "noise150": ["--noise", "150"],
    "sheet32": ["--sheet", "32"], "sheet64": ["--sheet", "64"],
    "seed2": ["--design-seed", "2"], "seed3": ["--design-seed", "3"],
    "no_velocity": ["--no-velocity"],
}
#: pose calibrations the runs need: name -> arguments of scripts/calibrate_pose.py
CALIBRATIONS = {"default": [], "sheet32": ["--sheet", "32"], "sheet64": ["--sheet", "64"], "seed2": ["--seed", "2"], "seed3": ["--seed", "3"]}


def artifact(extra):
    values = {"--seed": "1", "--sheet": "48"}
    values.update(dict(zip(extra[::2], extra[1::2])))
    return os.path.join(ROOT, "artifacts", "pose_N200_f120_p200_e60_seed%s_fractional_s%s_u1p5_k3.npz" % (values["--seed"], values["--sheet"]))


def plan(threads, models):
    py = [sys.executable, "-B"]
    runs = []
    for name, extra in CALIBRATIONS.items():
        runs.append(("calibrate", "calibrate_" + name, artifact(extra), py + ["scripts/calibrate_pose.py", "--threads", "8"] + extra))
    for model in MODELS:
        out = os.path.join(DATA, "static", model + ".json")
        runs.append(("static", "static_" + model, out, py + ["scripts/run_static.py", "--chain", model, "--threads", str(threads), "--out", out]))
    for model in TRACKING_MODELS:
        for speed in TRACKING_SPEEDS:
            out = os.path.join(DATA, "tracking", "%s_%.2f.json" % (model, speed))
            runs.append(("tracking", "tracking_%s_%.2f" % (model, speed), out,
                         py + ["scripts/run_tracking.py", "--chain", model, "--speed", str(speed), "--threads", str(threads), "--out", out]))
    for model in ("generic3", "tiago7"):
        for name, extra in TRACKING_VARIANTS.items():
            out = os.path.join(DATA, "tracking_variants", "%s_%s.json" % (model, name))
            runs.append(("tracking", "tracking_%s_%s" % (model, name), out,
                         py + ["scripts/run_tracking.py", "--chain", model, "--speed", "0.5", "--threads", str(threads), "--out", out] + extra))
    for model in DROPOUT_MODELS:
        for name, extra in DROPOUTS.items():
            out = os.path.join(DATA, "dropout", "%s_%s.json" % (model, name))
            runs.append(("dropout", "dropout_%s_%s" % (model, name), out,
                         py + ["scripts/run_dropout.py", "--chain", model, "--threads", str(threads), "--out", out] + extra))
    for model in DROPOUT_MODELS:
        for mode in ("trail", "none", "velocity"):
            out = os.path.join(DATA, "trail", "%s_%s.json" % (model, mode))
            runs.append(("trail", "trail_%s_%s" % (model, mode), out,
                         py + ["scripts/run_trail.py", "--chain", model, "--mode", mode, "--threads", str(threads), "--out", out]))
    for model in ROBUST_MODELS:
        for name, extra in VARIANTS.items():
            out = os.path.join(DATA, "robust", "%s_%s.json" % (model, name))
            runs.append(("robust", "robust_%s_%s" % (model, name), out,
                         py + ["scripts/run_static.py", "--chain", model, "--poses", str(ROBUST_POSES), "--threads", str(threads), "--out", out] + extra))
    if models:
        runs = [r for r in runs if r[0] == "calibrate" or any(m in r[1].split("_") for m in models)]
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=int, default=3, help="runs side by side")
    parser.add_argument("--threads", type=int, default=6, help="NEST threads per run")
    parser.add_argument("--only", nargs="*", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--models", nargs="*", default=[], help="only runs of these models")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="start no further run after this many minutes (the ones that are running finish); start again for the rest")
    parser.add_argument("--list", action="store_true", help="print the commands and stop")
    args = parser.parse_args(argv)

    runs = [r for r in plan(args.threads, args.models) if r[0] in args.only]
    if args.list:
        for group, name, out, command in runs:
            print("%-10s %s" % (group, " ".join(command[1:])))
        return 0
    logs = os.path.join(DATA, "logs")
    os.makedirs(logs, exist_ok=True)
    started = time.time()

    def execute(run):
        group, name, out, command = run
        if os.path.exists(out):
            return name, "exists"
        if args.max_minutes is not None and time.time() - started > 60.0 * args.max_minutes:
            print("%-32s not started: past --max-minutes" % name, flush=True)
            return name, "deferred"
        t0 = time.time()
        with open(os.path.join(logs, name + ".log"), "w", encoding="utf-8") as log:
            code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        status = "ok" if code == 0 and os.path.exists(out) else "FAILED (see docs/results/data/logs/%s.log)" % name
        print("%-32s %s  %.0f s  (%.0f s since start)" % (name, status, time.time() - t0, time.time() - started), flush=True)
        return name, status

    failed = []
    # calibrations first, one after another (an existing file is never written again); then everything else side by side
    for run in [r for r in runs if r[0] == "calibrate"]:
        failed += [n for n, s in [execute(run)] if s.startswith("FAILED")]
    rest = [r for r in runs if r[0] != "calibrate"]
    rest.sort(key=lambda r: -max([int(c) for c in r[1] if c in "234567"] or [0]))       # the long ones (most joints) first
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        for name, status in pool.map(execute, rest):
            if status.startswith("FAILED"):
                failed.append(name)
    print("done in %.0f s; %d runs, %d failed%s" % (time.time() - started, len(runs), len(failed), ": " + ", ".join(failed) if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
