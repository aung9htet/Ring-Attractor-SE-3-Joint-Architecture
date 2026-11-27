"""``python -m reaching_benchmark status [MODEL/BENCHMARK | BENCHMARK ...] [--watch S]``: how far every run is.

Reads the run files the benchmarks write after every trial (``results/<model>/<benchmark>/<run>.json``). For each run it
prints one bar: done of planned, reached so far, wall time, the time left at the run's own pace, and how long ago the
file was last written. A run whose last write is much older than a trial normally takes (about 8 min for the TIAGo arm
with the spiking network) has probably stopped; running the same command again resumes it.

Standard library only, so it also runs on the host::

    python3 -m reaching_benchmark status --watch 60
    python3 -m reaching_benchmark status joint_loss dls_ik/reach_gazebo
    python3 -m reaching_benchmark status --stuck --watch 60     (the stuck-joint runs of Fig. 5d, one line per controller)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from .progress import bar, clock

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def add_arguments(parser: argparse.ArgumentParser, command: str) -> None:
    parser.add_argument("names", nargs="*", help="MODEL/BENCHMARK or BENCHMARK (every model's) to show (default: all)")
    parser.add_argument("--watch", type=float, default=None, help="print again every this many seconds (Ctrl-C stops)")
    parser.add_argument("--stuck", action="store_true", help="the stuck-joint runs (docker/run_tiago_stuck.bash): one line per "
                                                             "controller, done / running / stopped / waiting")


def planned_rows(document: Dict[str, Any], done: int) -> int:
    """Trials the run will make: written by the runner, or worked out from the run and its shard (older files)."""

    if document.get("planned") is not None:
        return int(document["planned"])
    run = document.get("run") or {}
    pairs, shard = run.get("pairs"), document.get("shard")
    if pairs and shard and len(shard) == 2:
        return len(range(int(shard[0]), int(pairs), int(shard[1])))
    return int(pairs) if pairs else done


def describe(benchmark: str, path: str, document: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    now = time.time() if now is None else now
    rows = document.get("rows")
    entry: Dict[str, Any] = {"benchmark": benchmark, "part": os.path.splitext(os.path.basename(path))[0],
                             "wall_s": document.get("wall_s"), "age_s": now - os.path.getmtime(path), "reached": None,
                             "skipped": len(document.get("skipped") or [])}
    if not isinstance(rows, list):                 # one-shot documents (estimation, cost, ...): done once written
        entry.update(done=1, planned=1)
        return entry
    entry.update(done=len(rows), planned=planned_rows(document, len(rows)))
    if any("reached" in row for row in rows):
        entry["reached"] = sum(1 for row in rows if row.get("reached") is True)
    return entry


def folders(root: str, names: Sequence[str]) -> List[str]:
    """``model/benchmark`` folders under ``root`` that hold run files, those named (``MODEL/BENCHMARK`` or ``BENCHMARK``)."""

    out = []
    for model in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if not os.path.isdir(os.path.join(root, model)):
            continue
        for benchmark in sorted(os.listdir(os.path.join(root, model))):
            if not os.path.isdir(os.path.join(root, model, benchmark)):
                continue
            key = model + "/" + benchmark
            if not names or key in names or benchmark in names:
                out.append(key)
    return out


def collect(root: str, names: Sequence[str]) -> List[Dict[str, Any]]:
    out = []
    for key in folders(root, names):
        for path in sorted(glob.glob(os.path.join(root, key, "*.json"))):
            if os.path.basename(path) == "summary.json":
                continue
            try:
                with open(path, encoding="utf-8") as stream:
                    document = json.load(stream)
            except (OSError, ValueError):
                continue                           # being replaced right now: shown next time
            out.append(describe(key, path, document))
    return out


def render(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return "no run files yet"
    lines = []
    width = max(len(e["part"]) for e in entries)
    bench_width = max(len(e["benchmark"]) for e in entries)
    left_total = 0.0
    for e in entries:
        done, planned = int(e["done"]), max(int(e["planned"]), 1)
        per = e["wall_s"] / done if done and e.get("wall_s") else None
        left = per * max(planned - done, 0) if per is not None else None
        if left is not None:
            left_total += left
        state = "done" if done >= planned else "left " + clock(left)
        if e["skipped"] and done < planned:
            state += " (%d skipped: run again to resume)" % e["skipped"]
        reached = " reached %2d" % e["reached"] if e["reached"] is not None else ""
        lines.append("%-*s %-*s %s %3d/%-3d%s  wall %s  %s  written %s ago" % (
            bench_width, e["benchmark"], width, e["part"], bar(done / planned, 16), done, planned, reached, clock(e.get("wall_s")), state,
            clock(e["age_s"])))
    done_all = sum(int(e["done"]) for e in entries)
    planned_all = sum(max(int(e["planned"]), 1) for e in entries)
    lines.append("total: %d/%d items; %s of running time left if the unfinished runs ran one after another" % (
        done_all, planned_all, clock(left_total)))
    return "\n".join(lines)


#: the stuck-joint runs of docker/run_tiago_stuck.bash: (label, model, run file stem, phase)
STUCK_PLAN = [("RAK model", "ring_attractor_kinematics", "tiago7_seed1_lock", 1),
              ("RAK, predictive encoding", "ring_attractor_kinematics", "tiago7_seed1_predictive_encoding_lock", 1),
              ("DLS IK", "dls_ik", "tiago7_lock", 1), ("CCD IK", "ccd_ik", "tiago7_lock", 1),
              ("NN dynamics + MPC", "mpc_nn_dynamics", "tiago7_seed1_lock", 2), ("PETS", "pets", "tiago7_seed1_lock", 2)]
STUCK_PAIRS = 30


def stuck_entry(root: str, model: str, stem: str, now: float) -> Dict[str, Any]:
    """One controller's stuck-joint run, its shards put together: done, reached, the time it has taken and has left (its
    shards run side by side, so the slowest shard sets it), when it was last written and the onset it was run with."""

    folder = os.path.join(root, model, "joint_loss_gazebo")
    paths = [path for path in sorted(glob.glob(os.path.join(folder, stem + "*.json")))
             if os.path.basename(path)[len(stem):] in (".json",) or os.path.basename(path)[len(stem):].startswith("_shard")]
    entry: Dict[str, Any] = {"done": 0, "reached": 0, "left_s": 0.0, "wall_s": 0.0, "age_s": None, "onsets": set(), "per_s": None}
    for path in paths:
        try:
            with open(path, encoding="utf-8") as stream:
                document = json.load(stream)
        except (OSError, ValueError):
            continue
        rows = document.get("rows") or []
        done, planned = len(rows), planned_rows(document, len(rows))
        wall = float(document.get("wall_s") or 0.0)
        per = wall / done if done else None
        entry["done"] += done
        entry["reached"] += sum(1 for row in rows if row.get("reached") is True)
        entry["wall_s"] = max(entry["wall_s"], wall)
        if per is not None:
            entry["per_s"] = max(entry["per_s"] or 0.0, per)
            entry["left_s"] = max(entry["left_s"], per * max(planned - done, 0))
        age = now - os.path.getmtime(path)
        entry["age_s"] = age if entry["age_s"] is None else min(entry["age_s"], age)
        onset = (document.get("run") or {}).get("onset_s")
        if onset is not None:
            entry["onsets"].add(float(onset))
    if entry["done"] >= STUCK_PAIRS:
        entry["state"] = "done"
    elif not paths:
        entry["state"] = "waiting"
    elif entry["age_s"] is not None and entry["age_s"] < max(3.0 * (entry["per_s"] or 0.0), 20 * 60.0):
        entry["state"] = "running"
    else:
        entry["state"] = "stopped"
    return entry


def render_stuck(root: str, now: Optional[float] = None) -> str:
    now = time.time() if now is None else now
    lines = ["Fig. 5d: one joint stuck, every controller in Gazebo (30 targets each)", ""]
    phase_left = {1: 0.0, 2: 0.0}
    unknown = False
    for label, model, stem, phase in STUCK_PLAN:
        e = stuck_entry(root, model, stem, now)
        onset = ", ".join("%g s" % o for o in sorted(e["onsets"])) or "-"
        if e["state"] == "waiting":
            detail = "not started (phase %d)" % phase
            unknown = True
        elif e["state"] == "done":
            detail = "took %s" % clock(e["wall_s"])
        else:
            detail = "%s left, last trial %s ago" % (clock(e["left_s"]) if e["per_s"] else "?", clock(e["age_s"]))
            if e["state"] == "stopped":
                detail += "  (stopped: run the script again to resume)"
        phase_left[phase] = max(phase_left[phase], e["left_s"] if e["state"] != "done" else 0.0)
        reached = "reached %2d" % e["reached"] if e["done"] else "          "
        lines.append("%-25s %-8s %s %2d/%d  %s  stuck at %-5s %s" % (label, e["state"], bar(e["done"] / STUCK_PAIRS, 16), e["done"],
                                                                     STUCK_PAIRS, reached, onset, detail))
    lines.append("")
    left = phase_left[1] + phase_left[2]
    lines.append("phase 1 (RAK models, DLS IK, CCD IK) left: %s; phase 2 (NN dynamics + MPC, PETS) %s" % (
        clock(phase_left[1]), "left: " + clock(phase_left[2]) if not unknown or phase_left[2] else "starts after phase 1 (about 1 h)"))
    if not unknown:
        lines.append("finished at about %s" % time.strftime("%H:%M", time.localtime(now + left)))
    return "\n".join(lines)


def main(args: argparse.Namespace, command: str = "status") -> int:
    root = ROOT
    shown = 0                                      # lines printed last time (--watch redraws over them, in place)
    while True:
        text = render_stuck(root) if getattr(args, "stuck", False) else render(collect(root, args.names))
        block = time.strftime("%Y-%m-%d %H:%M:%S") + "  " + root + "\n" + text
        if args.watch and sys.stdout.isatty() and shown:
            sys.stdout.write("\033[%dF\033[J" % shown)     # up to the first line printed last time, clear below
        print(block, flush=True)
        shown = block.count("\n") + 1
        if not args.watch:
            return 0
        try:
            time.sleep(float(args.watch))
        except KeyboardInterrupt:
            return 0
