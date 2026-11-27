"""The simulator's main loop: a pose goal set in the browser, a method that takes the hand there, a robot that moves.

* **Task**: pose, position or orientation; positions compared on the sheets or as numbers (each a rebuild).
* **Methods**: ``ring`` is the spiking network (each joint follows its written goal ring); ``cascade`` the same
  rule in exact arithmetic; ``dls`` and ``transpose`` standard methods.  With any method but ``ring`` the network
  still runs and shows its rings and sheets, but the joints follow the method.
* **Robots**: ``light`` (no dynamics), and ``gazebo`` when started beside a running Gazebo (TIAGo arms only).
* **Runs**: a run starts when a goal is given; its distance from the goal over time is kept, so methods can be
  compared on the same start and goal ("same start and goal, next method").
"""

from __future__ import annotations

import queue
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from reaching_benchmark.targets import SPREAD
from reaching_benchmark.plants.light_robot import LightRobot
from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import CHAINS, MODEL_NAMES, Chain, get_chain, workspace
from reaching_benchmark.geometry.rotations import lift_pitch_yaw, nearest_rotation, orientation_error_deg, rotation_angle, wrap

from ..config import POSITION_GOALS, TASKS, IkDesign, default_task
from ..control.session import PoseIkSession
from ..neurons.sheet_layout import PLANES
from ..rules.references import exact_targets
from ..rules.cascade_rule import goal_matrix, share_caps
from .server import SimState
from .stub import StubNetwork

METHODS = ("ring", "cascade", "dls", "transpose")
METHOD_LABELS = {"ring": "Ring IK (spiking network, cascade)", "cascade": "Exact cascade (no neurons)",
                 "dls": "Damped least squares", "transpose": "Jacobian transpose"}
ROBOT_LABELS = {"light": "Light robot (no dynamics)", "gazebo": "TIAGo in Gazebo"}
KEPT_RUNS = 6


def start_pose(chain: Chain) -> np.ndarray:
    """The middle of the joint ranges, or every circular joint bent by half a radian (a stretched arm cannot move its
    hand along itself)."""

    return np.full(chain.n, 0.5) if chain.joints[0].circular else chain.limits().mean(axis=1)


class Simulator:
    def __init__(self, make_network: Callable[[Chain, IkDesign], Any], chain_name: str = "tiago4", state: Optional[SimState] = None,
                 task: Optional[str] = None, position_goal: str = "sheets", method: str = "ring", seed: int = 0,
                 realtime: bool = True, speed: float = 1.0, make_robot: Optional[Callable[[str, Chain], Any]] = None,
                 robots: tuple = ("light",), robot: str = "light") -> None:
        self.make_network = make_network
        self.make_robot = make_robot
        self.robots = tuple(robots)
        self.robot_name = robot if robot in self.robots else "light"
        self.state = state or SimState()
        self.task = task
        self.position_goal = position_goal
        self.method = method
        self.realtime = bool(realtime)
        self.speed = float(speed)
        self.running = True
        self._wall_per_tick = 0.0
        self.rng = np.random.default_rng(int(seed))
        self.load(chain_name)

    # -- building ------------------------------------------------------------
    def load(self, chain_name: str, keep_pose: Optional[np.ndarray] = None, task: Optional[str] = None) -> None:
        self.chain = get_chain(chain_name)
        self.space = workspace(self.chain)
        if self.robot_name == "gazebo" and not self.chain.name.startswith("tiago"):
            self.robot_name = "light"
        if task is not None:
            self.task = task
        design = IkDesign(task=self.task or default_task(self.chain.n), position_goal=self.position_goal)
        started = time.time()
        self.network = self.make_network(self.chain, design)
        self.build_s = time.time() - started
        self.design: IkDesign = self.network.ik_design
        self.dt_s = float(self.network.options.tick_ms) / 1000.0
        self.session = PoseIkSession(self.network, robot=self._robot(self.robot_name, keep_pose))
        self._apply_method()
        self.mode = "hold"                 # hold: nothing asked; ik: reaching; manual: sliders; placing: back to a start
        self.manual_target = self.session.q.copy()
        self.runs: List[Dict[str, Any]] = []
        self.run: Optional[Dict[str, Any]] = None
        self.pending: Optional[Dict[str, Any]] = None
        self._previous = (self.chain.rotation(self.session.q), kinematics.position(self.chain, self.session.q))
        self.state.publish("config", self.config())

    def _robot(self, name: str, q: Optional[np.ndarray] = None) -> Any:
        if name == "gazebo" and self.make_robot is not None:
            return self.make_robot(name, self.chain)
        return LightRobot(self.chain, q if q is not None else start_pose(self.chain))

    def _apply_method(self) -> None:
        self.session.method = None if self.method == "ring" else exact_targets(self.method)
        self.session.method_shares = None

    def config(self) -> Dict[str, Any]:
        mapping = self.network.mapping
        return {
            "chain": self.chain.name, "description": self.chain.description, "chains": MODEL_NAMES,
            "joints": [{"name": j.name, "lower": j.lower, "upper": j.upper, "circular": j.circular, "axis": j.axis} for j in self.chain.joints],
            "neurons": int(self.network.neuron_count()), "ik_neurons": int(self.network.ik_neuron_count()),
            "ring_size": int(self.session.ring_size), "tick_ms": float(self.network.options.tick_ms), "build_s": round(self.build_s, 2),
            "radius": float(self.space.radius), "centre": [float(v) for v in self.space.centre],
            "reach": float(self.space.radius + np.linalg.norm(self.space.centre)),
            "sheet": {"size": int(mapping.n), "extent": float(mapping.design.extent)}, "planes": [name for name, _, _ in PLANES],
            "stub": isinstance(self.network, StubNetwork), "ik": True,
            "task": self.design.task, "tasks": list(TASKS), "position_goal": self.design.position_goal, "position_goals": list(POSITION_GOALS),
            "on_sheets": bool(self.design.position and self.design.position_goal == "sheets"),
            "balance": float(self.design.balance), "reach_deg": float(self.design.reach_deg), "reach_reach": float(self.design.reach_reach),
            "caps": [round(float(c), 4) for c in share_caps(self.chain, self.design, self.session.ring_size)],
            "methods": [{"name": m, "label": METHOD_LABELS[m]} for m in METHODS],
            "robot": self.robot_name, "robots": [{"name": r, "label": ROBOT_LABELS.get(r, r)} for r in ("light", "gazebo")],
            "robots_there": list(self.robots), "gazebo_chains": [name for name in MODEL_NAMES if name.startswith("tiago")],
        }

    # -- commands ------------------------------------------------------------
    def _legal_position(self, position: Any) -> np.ndarray:
        """A goal position inside the workspace (the ball the sheets cover)."""

        scaled = self.space.scaled(np.asarray(position, dtype=float))
        length = float(np.linalg.norm(scaled))
        return self.space.metres(scaled / length if length > 1.0 else scaled)

    def handle(self, command: Dict[str, Any]) -> None:
        name, session = command.get("command"), self.session
        if name == "goal":
            lpy, position = session.goal_lpy.copy(), session.goal_position.copy()
            if isinstance(command.get("matrix"), list) and len(command["matrix"]) == 9:
                lpy = np.array(lift_pitch_yaw(nearest_rotation(np.array(command["matrix"], dtype=float).reshape(3, 3))))
            elif isinstance(command.get("lpy"), list) and len(command["lpy"]) == 3:
                values = [float(v) for v in command["lpy"]]
                lpy = np.array([float(wrap(values[0])), min(max(values[1], -np.pi / 2), np.pi / 2), float(wrap(values[2]))])
            if isinstance(command.get("position"), list) and len(command["position"]) == 3:
                position = self._legal_position([float(v) for v in command["position"]])
            self._give(lpy, position, new_run=bool(command.get("commit", True)))
        elif name == "random_goal":
            if command.get("near", True):
                moved = session.q + self.rng.uniform(-SPREAD, SPREAD, self.chain.n)
                pose = np.where(self.chain.circular_mask(), wrap(moved), self.chain.clip(moved))
            else:
                pose = self.chain.random_pose(self.rng)
            self._give(np.array(lift_pitch_yaw(self.chain.rotation(pose))), kinematics.position(self.chain, pose), new_run=True)
        elif name == "method" and command.get("method") in METHODS:
            self.method = command["method"]
            self._apply_method()
        elif name == "robot" and command.get("robot") in self.robots:
            self._switch_robot(command["robot"])
        elif name == "task" and command.get("task") in TASKS and command["task"] != self.design.task:
            self._rebuild(task=command["task"])
        elif name == "position_goal" and command.get("on") in POSITION_GOALS and command["on"] != self.position_goal:
            self.position_goal = command["on"]
            self._rebuild()
        elif name == "goal_anchor":
            self.network.set_goal_anchor(bool(command.get("on")))
        elif name == "replay":
            self._replay(command.get("method"))
        elif name == "target":
            q = command.get("q")
            if isinstance(q, list) and len(q) == self.chain.n:
                self._end_run()
                self.mode = "manual"
                self.manual_target = self.chain.clip(q) if not self.chain.joints[0].circular else np.array([float(v) for v in q])
        elif name == "stop":
            self._end_run()
            self.mode = "hold"
        elif name == "speed":
            self.speed = min(max(float(command.get("speed", self.speed)), 0.05), 3.0)
        elif name == "chain" and command.get("chain") in CHAINS:
            self.task = None
            self.load(command["chain"])
        elif name == "quit":
            self.running = False

    def _switch_robot(self, wanted: str) -> None:
        if wanted == self.robot_name or (wanted == "gazebo" and not self.chain.name.startswith("tiago")):
            return
        self._end_run()
        self.robot_name = wanted
        self.session.robot = self._robot(wanted, self.session.q.copy())       # the light robot takes over where Gazebo's arm is
        self.mode = "hold"
        self.manual_target = self.session.q.copy()
        self._previous = (self.chain.rotation(self.session.q), kinematics.position(self.chain, self.session.q))
        self.state.publish("config", self.config())

    def _replay(self, method: Optional[str]) -> None:
        source = self.run or (self.runs[-1] if self.runs else None)
        if source is None:
            return
        self._end_run()
        self.method = method if method in METHODS else METHODS[(METHODS.index(self.method) + 1) % len(METHODS)]
        self._apply_method()
        self.pending = {"start": np.array(source["start"]), "goal_lpy": np.array(source["goal_lpy"]), "goal_position": np.array(source["goal_position"])}
        self.mode = "placing"

    def _rebuild(self, task: Optional[str] = None) -> None:
        runs, pose = self.runs, self.session.q.copy()
        self.load(self.chain.name, keep_pose=pose, task=task)
        self.runs = runs

    def _give(self, goal_lpy: np.ndarray, goal_position: np.ndarray, new_run: bool) -> None:
        session = self.session
        session.give(goal_lpy, goal_position)
        if new_run or self.mode != "ik" or self.run is None:
            self._end_run()
            rotation, position = self.chain.rotation(session.q), kinematics.position(self.chain, session.q)
            self.run = {"id": (self.runs[-1]["id"] + 1) if self.runs else 1, "method": self.method, "task": self.design.task, "robot": self.robot_name,
                        "start": session.q.copy().tolist(), "goal_lpy": session.goal_lpy.tolist(), "goal_position": session.goal_position.tolist(),
                        "t": [], "deg": [], "mm": [], "path": 0.0, "reached_s": None,
                        "start_deg": float(np.degrees(rotation_angle(rotation, goal_matrix(session.goal_lpy)))),
                        "start_mm": float(1000.0 * np.linalg.norm(session.goal_position - position))}
            self._previous = (rotation, position)
        else:                                     # a drag in progress: the same run, the goal moves
            self.run["goal_lpy"], self.run["goal_position"] = session.goal_lpy.tolist(), session.goal_position.tolist()
        self.mode = "ik"

    def _end_run(self) -> None:
        if self.run is not None and self.run["t"]:
            self.runs = (self.runs + [self.run])[-KEPT_RUNS:]
        self.run = None

    # -- one tick ------------------------------------------------------------
    def _step(self) -> Any:
        session = self.session
        if self.mode == "ik":
            return session.tick()
        if self.mode == "placing" and self.pending is not None:
            difference = session.robot.difference(self.pending["start"])
            there = np.max(np.abs(difference)) < session.robot.tolerance
            if there and session.arrived():
                pending, self.pending = self.pending, None
                self._give(pending["goal_lpy"], pending["goal_position"], new_run=True)
                return session.tick()
            limit = 3.0 * self.dt_s
            session.robot.follow(session.q + np.clip(difference, -limit, limit) if not there else self.pending["start"], self.dt_s, self.dt_s, exact=True)
            return session.hold(move=False)
        if self.mode == "manual":
            difference = session.robot.difference(self.manual_target)
            session.robot.follow(session.q + np.clip(difference, -self.speed * self.dt_s, self.speed * self.dt_s), self.dt_s, self.dt_s, exact=True)
            return session.hold(move=False)
        return session.hold()

    def tick(self) -> Dict[str, Any]:
        started = time.time()
        while True:
            try:
                self.handle(self.state.commands.get_nowait())
            except queue.Empty:
                break
        snapshot = self.snapshot(self._step())
        elapsed = time.time() - started
        self._wall_per_tick = 0.9 * self._wall_per_tick + 0.1 * elapsed if self._wall_per_tick else elapsed
        rate = self.dt_s / max(self._wall_per_tick, 1e-6)
        snapshot["rate"] = round(min(rate, 1.0) if self.realtime else rate, 2)
        self.state.publish("tick", snapshot)
        if self.realtime:
            spare = self.dt_s - (time.time() - started)
            if spare > 0:
                time.sleep(spare)
        return snapshot

    def _within(self, deg: float, mm: float) -> bool:
        design = self.design
        return ((not design.orientation or deg <= design.reach_deg)
                and (not design.position or mm <= 1000.0 * design.reach_reach * self.space.radius))

    def _run_row(self, run: Dict[str, Any], live: bool) -> Dict[str, Any]:
        every = max(len(run["t"]) // 150, 1)
        return {"id": run["id"], "method": run["method"], "task": run["task"], "robot": run.get("robot", "light"), "live": live,
                "start_deg": round(run["start_deg"], 1), "start_mm": round(run["start_mm"], 1), "reached_s": run["reached_s"],
                "final_deg": round(run["deg"][-1], 2) if run["deg"] else None, "final_mm": round(run["mm"][-1], 1) if run["mm"] else None,
                "path_ratio": round(run["path"] / run["start_mm"], 2) if run["start_mm"] > 20.0 else None,
                "t": run["t"][::every], "deg": run["deg"][::every], "mm": run["mm"][::every]}

    def _record_run(self, goal_deg: float, goal_mm: float, position: np.ndarray) -> None:
        if self.run is None or self.mode != "ik":
            return
        run = self.run
        run["path"] += float(1000.0 * np.linalg.norm(position - self._previous[1]))
        run["t"].append(round(len(run["t"]) * self.dt_s + self.dt_s, 3))
        run["deg"].append(round(goal_deg, 2))
        run["mm"].append(round(goal_mm, 1))
        if not self._within(goal_deg, goal_mm):
            run["reached_s"] = None               # reached means: within the marks and staying there
        elif run["reached_s"] is None:
            run["reached_s"] = run["t"][-1]

    def snapshot(self, result: Any) -> Dict[str, Any]:
        session, chain, tick, design = self.session, self.chain, result.forward, self.design
        mapping, space = self.network.mapping, self.space
        q = session.q
        frames = chain.frames(q)
        true = [rotation for _, rotation in frames[1:]]
        position = kinematics.position(chain, q)
        raw = list(tick.links) + [tick.tool]
        decoded = [nearest_rotation(m) for m in raw]
        goal = goal_matrix(session.goal_lpy)
        goal_deg = float(np.degrees(rotation_angle(true[-1], goal)))
        goal_mm = float(1000.0 * np.linalg.norm(session.goal_position - position))
        self._record_run(goal_deg, goal_mm, position)
        self._previous = (true[-1], position)

        def number(v: Any) -> Any:
            return None if v is None or not np.isfinite(v) else round(float(v), 4)

        def r3(values: Any) -> List[float]:
            return [round(float(v), 4) for v in values]

        def r4(values: Any) -> List[Any]:
            return [number(v) for v in values]

        def sheet_rows(counts: np.ndarray) -> List[List[List[int]]]:
            return [[[int(k), int(counts[i][k])] for k in np.flatnonzero(counts[i])] for i in range(len(PLANES))]

        def cells(point: np.ndarray) -> List[Any]:
            s = space.scaled(point)
            return [r4(mapping.to_cells(s[[a, b]])) for _, a, b in PLANES]

        shares = result.shares if session.method is None or session.method_shares is None else session.method_shares
        scaled = space.scaled(position)
        sheet_error = float(np.linalg.norm(tick.position - position))
        gazebo = None
        if self.robot_name == "gazebo":
            link = session.robot.link
            gazebo = {"sim_time": round(float(link.sim_time_s()), 3), "frames": int(link.frames()), "lag": r3(session.robot.lag),
                      "overshoot_ms": round(1000.0 * float(link.overshoot_s), 2)}
        return {
            "t": round(tick.t_ms / 1000.0, 3), "chain": chain.name, "mode": self.mode, "method": self.method, "speed": self.speed,
            "robot": self.robot_name, "gazebo": gazebo,
            "q": r3(q), "target": r3(self.manual_target if self.mode == "manual" else session.robot.target),
            "points": [r3(origin) for origin, _ in frames],
            "true": [r3(m.reshape(9)) for m in true], "decoded": [r3(m.reshape(9)) for m in decoded],
            "error": round(orientation_error_deg(raw[-1], true[-1]), 3),
            "lpy_true": r3(lift_pitch_yaw(true[-1])), "lpy_ring": r4(tick.angles),
            "pos_true": r3(position), "pos": r4(tick.position), "pos_cells": r4(tick.cells_position),
            "pos_error_mm": None if not np.isfinite(sheet_error) else round(1000.0 * sheet_error, 1),
            "rings": [[int(c) for c in row] for row in self.network.ring_counts()],
            "written": [round(float(ring.target_index), 2) if np.isfinite(ring.target_index) else None for ring in self.network.rings],
            "out_rings": [[int(c) for c in row] for row in self.network.readout_ring_counts()],
            "sheets": sheet_rows(tick.sheet_counts), "goal_sheets": sheet_rows(result.goal_sheet_counts),
            "sheet_true": [r4(mapping.to_cells(scaled[[a, b]])) for _, a, b in PLANES],
            "sheet_goal": cells(session.written_position), "sheet_final": cells(session.goal_position),
            "ik": {
                "on": self.mode == "ik", "task": design.task, "goal": r4(session.goal_lpy), "written": r4(session.written_lpy),
                "goal_position": r3(session.goal_position), "written_position": r3(session.written_position),
                "goal_matrix": r3(goal.reshape(9)), "goal_error": round(goal_deg, 3), "goal_error_mm": round(goal_mm, 1),
                "turn": r4(result.turn), "offset_mm": r4(1000.0 * space.radius * result.offset),
                "true_offset_mm": r3(1000.0 * (session.written_position - position)),
                "shares": r4(shares), "network_shares": r4(result.shares), "lead": r4(result.lead),
                "left_deg": number(np.degrees(np.linalg.norm(result.left[:3]))),
                "left_mm": number(1000.0 * space.radius * np.linalg.norm(result.left[3:])),
                "goal_held": r4(result.goal_position), "ramp_goal": r4(result.ramp_goal), "ramp_hand": r4(result.ramp_hand),
                "goal_anchor": bool(getattr(self.network, "goal_anchor_on", False)),
                "goal_rings": [[int(c) for c in row] for row in self.network.goal_counts()],
                "joint_goal_rings": [[int(c) for c in row] for row in self.network.joint_goal_counts()],
            },
            "runs": [self._run_row(run, False) for run in self.runs] + ([self._run_row(self.run, True)] if self.run is not None and self.run["t"] else []),
        }

    def run_loop(self, max_ticks: Optional[int] = None) -> None:
        done = 0
        while self.running and (max_ticks is None or done < max_ticks):
            self.tick()
            done += 1
