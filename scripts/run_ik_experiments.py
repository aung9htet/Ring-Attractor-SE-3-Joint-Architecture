#!/usr/bin/env python3
"""Every run behind docs/ik/results.md, in batches.

    python3 -B scripts/run_ik_experiments.py --list
    python3 -B scripts/run_ik_experiments.py --batch reach_pose --jobs 2 --threads 12
    python3 -B scripts/run_ik_experiments.py --batch reach_position parts --jobs 4 --threads 6

Batches: ``study`` (the rule with exact kinematics; no NEST), ``parts`` (comparisons and shares in spikes,
arm held by hand), ``reach_position`` (4-joint arms, position goals), ``reach_pose`` (7-joint arms, pose
goals), ``variants`` (position compared on cells instead of sheets; every joint gets the whole error),
``disturbances`` (motor too weak or too strong, a push, a goal that changes).  A run whose output file
exists is skipped, so an interrupted batch is finished by starting it again; a closed-loop run that was
cut by ``--max-minutes`` is finished by a second start, which runs the pairs that are left.
Logs go to ``docs/ik/results/data/logs/``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "docs", "ik", "results", "data")
REACH, PARTS = "scripts/run_ik_reach.py", "scripts/run_ik_parts.py"
#: batch -> [(name of the output file, script, arguments)]
BATCHES = {
    "study": [("study", "scripts/ik_study.py", [])],
    "parts": [
        ("parts_tiago4_position", PARTS, ["--chain", "tiago4", "--poses", "60"]),
        ("parts_generic7_pose", PARTS, ["--chain", "generic7", "--poses", "40"]),
        ("parts_tiago7_pose", PARTS, ["--chain", "tiago7", "--poses", "40"]),
    ],
    "reach_position": [
        ("reach_tiago4_position", REACH, ["--chain", "tiago4", "--pairs", "40"]),
        ("reach_generic4_position", REACH, ["--chain", "generic4", "--pairs", "40"]),
    ],
    "reach_pose": [
        ("reach_generic7_pose", REACH, ["--chain", "generic7", "--pairs", "30"]),
        ("reach_tiago7_pose", REACH, ["--chain", "tiago7", "--pairs", "30"]),
    ],
    "variants": [
        ("reach_tiago4_position_cells", REACH, ["--chain", "tiago4", "--pairs", "40", "--position-goal", "cells"]),
        ("reach_tiago4_position_all_at_once", REACH, ["--chain", "tiago4", "--pairs", "40", "--all-at-once", "--kappa", "0.5"]),
        ("reach_tiago7_pose_cells", REACH, ["--chain", "tiago7", "--pairs", "20", "--position-goal", "cells"]),
        ("reach_tiago7_orientation", REACH, ["--chain", "tiago7", "--pairs", "20", "--task", "orientation"]),
    ],
    "disturbances": [
        ("reach_tiago4_position_motor070", REACH, ["--chain", "tiago4", "--pairs", "20", "--motor-scale", "0.7"]),
        ("reach_tiago4_position_motor130", REACH, ["--chain", "tiago4", "--pairs", "20", "--motor-scale", "1.3"]),
        ("reach_tiago4_position_push", REACH, ["--chain", "tiago4", "--pairs", "20", "--push", "4.0", "0.3"]),
        ("reach_tiago4_position_goal_change", REACH, ["--chain", "tiago4", "--pairs", "20", "--goal-change", "4.0"]),
        ("reach_tiago4_position_anywhere", REACH, ["--chain", "tiago4", "--pairs", "20", "--spread", "0"]),
    ],
}


def left_over(path):
    """Pairs a closed-loop run did not get to (it was cut by its time guard); ``None`` when there is no file."""

    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as stream:
        return [int(i) for i in json.load(stream).get("skipped", [])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch", nargs="*", choices=sorted(BATCHES), default=sorted(BATCHES))
    parser.add_argument("--jobs", type=int, default=2, help="runs side by side")
    parser.add_argument("--threads", type=int, default=12, help="NEST threads per run")
    parser.add_argument("--max-minutes", type=float, default=None, help="time guard of every closed-loop run")
    parser.add_argument("--list", action="store_true", help="print the commands and stop")
    args = parser.parse_args(argv)

    runs = []
    for batch in args.batch:
        for name, script, extra in BATCHES[batch]:
            out = os.path.join(DATA, name + ".json")
            command = [sys.executable, "-B", script] + extra + ["--out", out]
            if script != "scripts/ik_study.py":
                command += ["--threads", str(args.threads)]
            if script == REACH and args.max_minutes is not None:
                command += ["--max-minutes", str(args.max_minutes)]
            runs.append((batch, name, out, command, script))
    if args.list:
        for batch, name, out, command, _ in runs:
            print("%-15s %s" % (batch, " ".join(command[1:]).replace(ROOT + os.sep, "")))
        return 0
    logs = os.path.join(DATA, "logs")
    os.makedirs(logs, exist_ok=True)
    started = time.time()

    def execute(run):
        batch, name, out, command, script = run
        skipped = left_over(out)
        if skipped is not None and (script != REACH or not skipped):
            return name, "exists"
        if skipped:                                   # finish a run that its time guard cut: the pairs that are left, into a second file
            out = out[:-len(".json")] + "_rest.json"
            if os.path.exists(out):
                return name, "exists"
            command = command[:command.index("--out")] + ["--out", out] + command[command.index("--out") + 2:] + ["--only", ",".join(str(i) for i in skipped)]
        t0 = time.time()
        with open(os.path.join(logs, os.path.basename(out)[:-len(".json")] + ".log"), "w", encoding="utf-8") as log:
            code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        status = "ok" if code == 0 and os.path.exists(out) else "FAILED (see docs/ik/results/data/logs/)"
        print("%-40s %s  %.0f s  (%.0f s since start)" % (os.path.basename(out), status, time.time() - t0, time.time() - started), flush=True)
        return name, status

    failed = []
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        for name, status in pool.map(execute, runs):
            if status.startswith("FAILED"):
                failed.append(name)
    print("done in %.0f s; %d runs, %d failed%s" % (time.time() - started, len(runs), len(failed), ": " + ", ".join(failed) if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
