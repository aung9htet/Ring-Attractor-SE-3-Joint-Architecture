"""``python -m reaching_benchmark status [MODEL/BENCHMARK | BENCHMARK ...] [--watch S]``: how far every run is.

Reads the run files the benchmarks write after every trial (``results/<model>/<benchmark>/<run>.json``). For each run it
prints one bar: done of planned, reached so far, wall time, the time left at the run's own pace, and how long ago the
file was last written. A run whose last write is much older than a trial normally takes (about 8 min for the TIAGo arm
with the spiking network) has probably stopped; running the same command again resumes it.

Standard library only, so it also runs on the host::

    python3 -m reaching_benchmark status --watch 60
    python3 -m reaching_benchmark status joint_loss dls_ik/reach_gazebo
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


def main(args: argparse.Namespace, command: str = "status") -> int:
    root = ROOT
    while True:
        text = render(collect(root, args.names))
        if args.watch and sys.stdout.isatty():
            sys.stdout.write("\033[2J\033[H")
        print(time.strftime("%Y-%m-%d %H:%M:%S") + "  " + root)
        print(text, flush=True)
        if not args.watch:
            return 0
        try:
            time.sleep(float(args.watch))
        except KeyboardInterrupt:
            return 0
