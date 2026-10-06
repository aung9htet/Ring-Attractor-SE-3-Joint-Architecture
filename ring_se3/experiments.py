"""The measurements behind docs/results.md: one function per kind of run, each returning a JSON-ready dict."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ring_so3.design import Design
from ring_so3.experiments import POSE_SEED, make_design, sweep_poses, test_poses, write_json
from ring_so3.kinematics import Chain, lift_pitch_yaw, orientation_error_deg
from ring_so3.sim.robot import GoalSequence, SimRobot

from . import kinematics
from .chains import get_chain
from .session import PoseSession, pose_row
from .sheet import SheetDesign


def start_pose(chain: Chain) -> np.ndarray:
    return np.zeros(chain.n) if chain.joints[0].circular else chain.limits().mean(axis=1)


def build(backend: Any, chain_name: str, design: Optional[Design] = None, max_speed: float = 1.0, lead: bool = False,
          sheet: Optional[SheetDesign] = None, **options: Any):
    """A pose network for ``chain_name`` with the stored calibrations of its design, and a session on it."""

    from ring_so3.nest import calibrate as so3_calibrate
    from .nest import calibrate as pose_calibrate
    from .nest.network import PoseNetwork, PoseOptions

    chain = get_chain(chain_name)
    design = design or Design()
    sheet = sheet or SheetDesign()
    pose_options = PoseOptions(sheet=sheet, **options)
    calibration = so3_calibrate.load(design, design.sizes.ring, pose_options.encoder)
    pose_calibration = pose_calibrate.load(design, design.sizes.ring, pose_options.encoder, sheet)
    network = PoseNetwork(backend, chain, calibration, pose_calibration, design, pose_options)
    session = PoseSession(network, SimRobot(chain, start_pose(chain), max_speed=max_speed), lead=lead)
    return chain, design, network, session


def _meta(chain: Chain, design: Design, network: Any, started: float) -> Dict[str, Any]:
    o = network.pose_options
    wall = time.time() - started
    sim_s = network.t_ms / 1000.0
    return {
        "chain": chain.name, "joints": chain.n, "design": design.describe(), "sheet": o.sheet.describe(),
        "radius_m": network.space.radius, "centre_m": network.space.centre.tolist(),
        "options": {"mode": o.mode, "relay": o.relay, "encoder": o.encoder, "threads": o.threads, "seed": o.seed,
                    "tick_ms": o.tick_ms, "readout_tau_ms": o.readout_tau_ms, "noise_pa": o.noise_pa, "velocity": o.velocity,
                    "anchor_lead_s": o.anchor_lead_s, "anchor": o.anchor, "path_integration": o.path_integration},
        "neurons": network.neuron_count(), "so3_neurons": network.so3_neuron_count(),
        "products": len(network.products), "pose_products": len(network.muls), "pose_sums": len(network.relays),
        "synapses": int(network.backend.GetKernelStatus("num_connections")),
        "simulated_s": round(sim_s, 2), "wall_s": round(wall, 1), "speed": round(sim_s / wall, 3) if wall > 0 else None,
        "calibration_fit": network.calibration.fit, "pose_calibration_fit": network.pose_calibration.fit,
    }


def static_run(backend: Any, chain_name: str, poses: int = 300, sweep_points: int = 0, design: Optional[Design] = None,
               silence_sheets: float = 0.0, silence: float = 0.0, pose_seed: int = POSE_SEED, max_speed: float = 1.0,
               sheet: Optional[SheetDesign] = None, log=print, **options: Any) -> Dict[str, Any]:
    """Random held poses (and, with ``sweep_points``, a sweep of every joint).  ``silence_sheets`` / ``silence``:
    share of the sheets' / the product populations' neurons switched off."""

    started = time.time()
    chain, design, network, session = build(backend, chain_name, design, max_speed, sheet=sheet, **options)
    dead_sheet = network.silence_sheets(silence_sheets, seed=3) if silence_sheets > 0.0 else 0
    dead = network.silence(silence, seed=3) if silence > 0.0 else 0
    session.warm_up()
    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(test_poses(chain, poses, pose_seed)):
        row = pose_row(chain, network, session.go_to(q))
        row.update(kind="random", index=i)
        rows.append(row)
        if log is not None and (i + 1) % 50 == 0:
            errors = np.array([r["position_error_m"] for r in rows]) / network.space.radius * 100.0
            log("%s: %d poses, position error median %.2f%% p90 %.2f%% of reach, orientation median %.2f deg, %.0f s" % (
                chain_name, i + 1, np.nanmedian(errors), np.nanpercentile(errors, 90),
                np.median([r["error_deg"] for r in rows]), time.time() - started))
    for i, item in enumerate(sweep_poses(chain, sweep_points, pose_seed) if sweep_points > 0 else []):
        row = pose_row(chain, network, session.go_to(item["q"]))
        row.update(kind="sweep", index=i, sweep_joint=item["joint"])
        rows.append(row)
    document = _meta(chain, design, network, started)
    document.update(experiment="static", pose_seed=int(pose_seed), max_speed=float(max_speed), silenced_neurons=int(dead),
                    silenced_sheet_neurons=int(dead_sheet), silence=float(silence), silence_sheets=float(silence_sheets), rows=rows)
    return document


def _tick_row(chain: Chain, network: Any, session: PoseSession, tick: Any, tick_window: int = 1) -> Dict[str, Any]:
    q = session.robot.q.copy()
    true = kinematics.position(chain, q)
    rotation = chain.rotation(q)
    now, _ = network.position_from_counts(tick.sheet_counts)
    return {
        "t": tick.t_ms / 1000.0, "q": q.tolist(), "moving": bool(not session.robot.arrived()),
        "joint_speed": float(np.max(np.abs(session.robot.velocity))),
        "true_position": true.tolist(), "position": tick.position.tolist(), "position_tick": now.tolist(),
        "cells_position": tick.cells_position.tolist(),
        "position_error_m": kinematics.position_error(tick.position, true),
        "tick_error_m": kinematics.position_error(now, true),
        "cells_error_m": kinematics.position_error(tick.cells_position, true),
        "true_velocity": kinematics.velocity(chain, q, session.robot.velocity).tolist(),
        "velocity": network.velocity_m_s(tick).tolist(),
        "disagreement_m": (tick.disagreement * network.space.radius).tolist(),
        "sheet_spikes": tick.sheet_counts.sum(axis=1).tolist(),
        "error_deg": orientation_error_deg(tick.tool, rotation),
        "true_lift_pitch_yaw": list(lift_pitch_yaw(rotation)), "ring_lift_pitch_yaw": tick.angles.tolist(),
        "anchor": bool(network.anchor_on), "path_integration": bool(network.path_integration_on),
    }


def tracking_run(backend: Any, chain_name: str, speed: float = 0.5, goals: int = 20, dwell_s: float = 1.0,
                 design: Optional[Design] = None, goal_seed: int = POSE_SEED + 11, lead: bool = False,
                 sheet: Optional[SheetDesign] = None, log=print, **options: Any) -> Dict[str, Any]:
    """The arm keeps moving from goal to goal; every tick is logged."""

    started = time.time()
    chain, design, network, session = build(backend, chain_name, design, max_speed=speed, lead=lead, sheet=sheet, **options)
    session.warm_up()
    sequence = GoalSequence(chain, goal_seed, dwell_s)
    ticks: List[Dict[str, Any]] = []
    reached = 0
    while reached <= int(goals):
        if sequence.update(session.robot, session.dt_s):
            reached += 1
            if reached > int(goals):
                break
        row = _tick_row(chain, network, session, session.tick())
        row["goal"] = reached
        ticks.append(row)
    document = _meta(chain, design, network, started)
    document.update(experiment="tracking", speed=float(speed), goals=int(goals), dwell_s=float(dwell_s),
                    goal_seed=int(goal_seed), lead=bool(lead), ticks=ticks)
    if log is not None:
        errors = np.array([t["position_error_m"] for t in ticks]) / network.space.radius * 100.0
        moving = np.array([t["moving"] for t in ticks])
        log("%s at %.2f rad/s: %d ticks, position error median %.2f%% of reach while moving, %.2f%% at rest" % (
            chain_name, speed, len(ticks), np.nanmedian(errors[moving]) if moving.any() else float("nan"),
            np.nanmedian(errors[~moving]) if (~moving).any() else float("nan")))
    return document


def dropout_run(backend: Any, chain_name: str, speed: float = 0.5, gaps_s: Sequence[float] = (2.0, 5.0, 10.0), repeats: int = 5,
                anchored_s: float = 4.0, after_s: float = 3.0, design: Optional[Design] = None, goal_seed: int = POSE_SEED + 12,
                sheet: Optional[SheetDesign] = None, log=print, **options: Any) -> Dict[str, Any]:
    """The anchor is switched off for a while and on again, over and over.

    ``speed`` 0: the arm goes to a random pose and rests there during the gap (does the packet stay?).
    ``speed`` > 0: the arm keeps moving from goal to goal through the gap (does path integration carry the packet?).
    Every tick is logged with ``phase`` (``before``, ``gap``, ``after``), ``gap`` (index) and ``gap_s``.
    """

    started = time.time()
    moving_arm = float(speed) > 0.0
    chain, design, network, session = build(backend, chain_name, design, max_speed=speed if moving_arm else 1.0, sheet=sheet, **options)
    session.warm_up()
    sequence = GoalSequence(chain, goal_seed, dwell_s=0.2 if moving_arm else 1.0e9)
    dt = session.dt_s
    ticks: List[Dict[str, Any]] = []
    gap_index = 0

    def run(seconds: float, phase: str, gap_s: float) -> None:
        for _ in range(int(round(seconds / dt))):
            if moving_arm:
                sequence.update(session.robot, dt)
            row = _tick_row(chain, network, session, session.tick())
            row.update(phase=phase, gap=gap_index, gap_s=float(gap_s))
            ticks.append(row)

    for gap_s in gaps_s:
        for _ in range(int(repeats)):
            if not moving_arm:
                session.robot.set_target(sequence.next_goal())
                while not session.robot.arrived():
                    session.tick()
            run(anchored_s, "before", gap_s)
            network.set_anchor(False)
            run(gap_s, "gap", gap_s)
            network.set_anchor(True)
            run(after_s, "after", gap_s)
            gap_index += 1
        if log is not None:
            ends = [t for t in ticks if t["phase"] == "gap" and t["gap_s"] == gap_s]
            last = {}
            for t in ends:
                last[t["gap"]] = t["position_error_m"]
            values = np.array(list(last.values())) / network.space.radius * 100.0
            log("%s, speed %.2f, anchor off for %.0f s: error at the end of the gap median %.2f%% of reach (%d gaps), %.0f s" % (
                chain_name, speed, gap_s, np.nanmedian(values), len(values), time.time() - started))
    document = _meta(chain, design, network, started)
    document.update(experiment="dropout", speed=float(speed), gaps_s=[float(g) for g in gaps_s], repeats=int(repeats),
                    anchored_s=float(anchored_s), after_s=float(after_s), goal_seed=int(goal_seed), ticks=ticks)
    return document


TRAIL_TICKS = 10          # the shift register keeps the packets' places of the last 0.5 s


def trail_run(backend: Any, chain_name: str, speed: float = 0.25, gap_s: float = 2.0, repeats: int = 10, mode: str = "trail",
              anchored_s: float = 4.0, after_s: float = 2.0, design: Optional[Design] = None, goal_seed: int = POSE_SEED + 13,
              sheet: Optional[SheetDesign] = None, log=print, **options: Any) -> Dict[str, Any]:
    """The shift-register experiment (after Lin, Yiu and Leibold 2026): during a gap the network gets neither the
    anchor nor the joint velocities.

    ``mode`` ``"trail"``: a shift register keeps where the packets were in the last ``TRAIL_TICKS`` ticks; when the
    gap starts, the velocity those places show is handed to the velocity cells and kept for the whole gap.
    ``"none"``: nothing is handed over (the packets stay).  ``"velocity"``: the joint velocities stay on (path
    integration as in :func:`dropout_run`), for comparison on the same goals.
    """

    if mode not in ("trail", "none", "velocity"):
        raise ValueError("mode must be 'trail', 'none' or 'velocity'")
    started = time.time()
    chain, design, network, session = build(backend, chain_name, design, max_speed=speed, sheet=sheet, trail=True, **options)
    session.warm_up()
    sequence = GoalSequence(chain, goal_seed, dwell_s=0.2)
    dt = session.dt_s
    register: List[np.ndarray] = []          # the shift register: newest place last
    ticks: List[Dict[str, Any]] = []
    gap_index = 0

    def run(seconds: float, phase: str, blind: bool) -> None:
        for _ in range(int(round(seconds / dt))):
            sequence.update(session.robot, dt)
            session.robot.step(dt)
            tick = network.step(session.robot.q, np.zeros(chain.n) if blind else session.robot.velocity)
            if np.all(np.isfinite(tick.position)):
                register.append(tick.position.copy())
                del register[:-TRAIL_TICKS]
            row = _tick_row(chain, network, session, tick)
            row.update(phase=phase, gap=gap_index, gap_s=float(gap_s))
            ticks.append(row)

    for _ in range(int(repeats)):
        run(anchored_s, "before", False)
        estimate = np.zeros(3)
        if len(register) == TRAIL_TICKS:
            t = np.arange(TRAIL_TICKS) * dt
            estimate = np.array([np.polyfit(t, np.array(register)[:, a], 1)[0] for a in range(3)])
        network.set_anchor(False)
        if mode == "trail":
            network.set_trail_velocity(estimate)
        run(gap_s, "gap", mode != "velocity")
        network.set_trail_velocity(None)
        network.set_anchor(True)
        run(after_s, "after", False)
        ticks[-1]["trail_velocity"] = estimate.tolist()
        gap_index += 1
    document = _meta(chain, design, network, started)
    document.update(experiment="trail", mode=mode, speed=float(speed), gaps_s=[float(gap_s)], repeats=int(repeats),
                    trail_ticks=TRAIL_TICKS, goal_seed=int(goal_seed), ticks=ticks)
    if log is not None:
        last = {}
        for t in ticks:
            if t["phase"] == "gap":
                last[t["gap"]] = t["position_error_m"]
        values = np.array([v for v in last.values() if v is not None]) / network.space.radius * 100.0
        log("%s, %s: error at the end of a %.0f s gap median %.1f%% of reach (%d gaps)" % (chain_name, mode, gap_s, np.nanmedian(values), len(values)))
    return document


__all__ = ["POSE_SEED", "TRAIL_TICKS", "build", "dropout_run", "trail_run", "make_design", "start_pose", "static_run", "test_poses", "tracking_run", "write_json"]
