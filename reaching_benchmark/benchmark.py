"""Running the reaching benchmark for any controller: the same pairs, the same placement, one run file per run.

:func:`run_reach` puts the arm at each pair's start, lets the controller settle, gives it the pair's goal and records the
trial (:func:`reaching_benchmark.closed_loop.trial_row`). The run file ``results/<model>/<benchmark>/<run>.json`` is
written after every trial; running the same command again goes on where a stopped run left off. Every run shows the
shared progress bar.

The plants: ``light`` (the kinematic plant, no dynamics) and ``gazebo`` (the TIAGo in Gazebo, run beside a Gazebo
started with ``docker/run.bash gazebo up``); see :func:`make_plant`.
"""

from __future__ import annotations

import importlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import results
from .closed_loop import ClosedLoop, Trial, trial_row
from .geometry import kinematics
from .geometry.chains import Chain
from .geometry.rotations import rotation_angle, wrap
from .limits import goal_matrix
from .progress import Progress, say, trial_detail
from .targets import TEST_SEED, PosePair, pose_pairs
from .task import IkDesign

PLANTS = ("light", "gazebo")
#: the benchmark folder of each plant
BENCHMARKS = {"light": "reach_kinematic", "gazebo": "reach_gazebo"}


def make_plant(name: str, chain: Chain) -> Any:
    """``light``: the kinematic plant. ``gazebo``: the TIAGo in Gazebo (needs ROS and a running Gazebo)."""

    if name == "light":
        from .plants.light_robot import LightRobot
        return LightRobot(chain)
    if name == "gazebo":
        gazebo = importlib.import_module("reaching_benchmark.plants.gazebo_robot")
        return gazebo.GazeboRobot(chain, gazebo.RosLink())
    raise ValueError("plant must be one of %r" % (PLANTS,))


def test_pairs(chain: Chain, count: int = 30) -> List[PosePair]:
    """The reported reaching pairs (every model is measured on these)."""

    return pose_pairs(chain, count, TEST_SEED)


def tick_progress(progress: Progress, loop: ClosedLoop, pair: PosePair) -> Callable[[Any], None]:
    """``on_tick`` for :meth:`ClosedLoop.reach`: trial time and the hand's errors on the progress bar."""

    chain, design, goal = loop.chain, loop.design, goal_matrix(pair.goal)
    ticks = [0]

    def on_tick(_: Any) -> None:
        ticks[0] += 1
        q = loop.q
        degrees = float(np.degrees(rotation_angle(chain.rotation(q), goal))) if design.orientation else None
        mm = 1000.0 * float(np.linalg.norm(kinematics.position(chain, q) - pair.goal_position)) if design.position else None
        progress.update(trial_detail(ticks[0] * loop.dt, design.trial_s, degrees, mm))
    return on_tick


def run_reach(model: str, benchmark: str, name: str, run: Dict[str, Any], loop: ClosedLoop, pairs: List[PosePair],
              controller: Dict[str, Any], shard: Tuple[int, int] = (0, 1), max_minutes: Optional[float] = None,
              row_of: Optional[Callable[[Trial, IkDesign], Dict[str, Any]]] = None, warm_up_ticks: int = 20,
              log: Optional[Callable[[str], None]] = None) -> str:
    """Run (or resume) one run: every pair of ``pairs`` in this shard, checkpointed after each trial.

    ``controller``: what the controller reads, is given and learns (stored in the run file, shown in the comparison).
    ``row_of``: the trial's record (default :func:`trial_row`). Returns the run file's path."""

    started = time.time()
    design = loop.design
    earlier = results.read_run(model, benchmark, name)
    rows: List[Dict[str, Any]] = list(earlier.get("rows", []))
    done = {row["index"] for row in rows}
    mine = [pair for pair in pairs if pair.index % shard[1] == shard[0]]
    chosen = [pair for pair in mine if pair.index not in done]
    path = results.run_path(model, benchmark, name)
    if not chosen:
        say("%s/%s %s: all %d pairs done" % (model, benchmark, name, len(rows)))
        return path
    if rows:
        say("%s/%s %s: resuming after %d pairs" % (model, benchmark, name, len(rows)))
    loop.warm_up(warm_up_ticks)
    progress = Progress("%s %s" % (benchmark, name), len(mine), len(rows), earlier.get("wall_s", 0.0), sum(bool(r["reached"]) for r in rows))
    simulated = float(earlier.get("simulated_s", 0.0))
    skipped: List[int] = []
    position = -1
    for position, pair in enumerate(chosen):
        if max_minutes is not None and (time.time() - started) / 60.0 > max_minutes:
            skipped = [p.index for p in chosen[position:]]
            break
        if hasattr(loop.robot, "link"):
            loop.robot.link.put_base_back()
        loop.place(pair.start)
        progress.start("%d" % pair.index)
        trial = loop.reach(pair.goal, pair.goal_position, on_tick=tick_progress(progress, loop, pair))
        simulated += float(trial.t[-1])
        row = (row_of or trial_row)(trial, design)
        row.update(index=pair.index, start=pair.start.tolist(), goal_lpy=pair.goal.tolist(), goal_position=pair.goal_position.tolist(),
                   start_off_rad=float(np.max(np.abs(wrap(trial.q[0] - pair.start)))))
        rows.append(row)
        rows.sort(key=lambda r: r["index"])
        line = "%s/%s %s pair %d: %.0f° %.0f%% → %.1f° %.1f%% (%s) in %.1f s | %d done, %.0f min" % (
            model, benchmark, name, pair.index, row["start_error_deg"], 100 * row["start_error_reach"], row["final_error_deg"],
            100 * row["final_error_reach"], "reached" if row["reached"] else "missed", row["duration_s"], len(rows),
            (time.time() - started) / 60.0)
        progress.finish(bool(row["reached"]), line)
        if log is not None:
            log(line)
        document = {"model": model, "benchmark": benchmark, "run": run, "design": design.describe(), "controller": controller,
                    "shard": list(shard), "planned": len(mine), "simulated_s": simulated,
                    "wall_s": earlier.get("wall_s", 0.0) + time.time() - started, "robot": loop.robot.describe(), "rows": rows,
                    "skipped": skipped + [p.index for p in chosen[position + 1:]]}
        results.write_run(model, benchmark, name, document)
    progress.close()
    return path
