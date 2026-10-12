"""The measurements of the pose inverse kinematics in NEST, each returning a JSON-ready dict."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ring_so3.design import Design
from ring_so3.experiments import write_json
from ring_so3.ik import rules as so3
from ring_so3.ik.experiments import angle_between_deg
from ring_so3.kinematics import Chain, lift_pitch_yaw

from .. import kinematics
from ..chains import get_chain
from ..sheet import SheetDesign
from . import rules
from .design import PoseIkDesign
from .pairs import TEST_SEED, pairs

MOVE_SPEED = 3.0          # rad/s: how fast arm and goal are moved between two open-loop measurements
HOLD_TICKS = 24
MEASURE_TICKS = 8
#: goals of the reported trials: the pose of a joint pose within this many radians of the start pose, joint by joint
SPREAD = 0.8


def default_task(chain: Chain) -> str:
    """A pose has six freedoms: an arm with fewer joints is asked for the position alone."""

    return "pose" if chain.n >= 6 else "position"


def build_ik(backend: Any, chain_name: str, ik_design: Optional[PoseIkDesign] = None, design: Optional[Design] = None,
             goal_ring_settings: Any = None, sheet: Optional[SheetDesign] = None, **options: Any):
    """A pose network with the inverse-kinematics layer for ``chain_name``, with the stored calibrations of its design."""

    from ring_so3.nest import calibrate as so3_calibrate
    from ring_so3.nest import ik_calibrate as so3_ik_calibrate
    from ..nest import calibrate as pose_calibrate
    from ..nest import ik_calibrate
    from ..nest.ik_network import PoseIkNetwork, ik_options

    chain = get_chain(chain_name)
    design = design or Design()
    sheet = sheet or SheetDesign()
    ik_design = ik_design or PoseIkDesign(task=default_task(chain))
    pose_options = ik_options(sheet=sheet, **options)
    ring, encoder = design.sizes.ring, pose_options.encoder
    on_sheets = ik_design.position and ik_design.position_goal == "sheets"
    network = PoseIkNetwork(backend, chain, so3_calibrate.load(design, ring, encoder), pose_calibrate.load(design, ring, encoder, sheet),
                            so3_ik_calibrate.load(design, ring, encoder),
                            ik_calibrate.load(design, ring, encoder, sheet) if on_sheets else None,
                            design, pose_options, ik_design, goal_ring_settings)
    return chain, network


def meta(chain: Chain, network: Any, started: float) -> Dict[str, Any]:
    o = network.pose_options
    wall, sim_s = time.time() - started, network.t_ms / 1000.0
    return {"chain": chain.name, "joints": chain.n, "design": network.design.describe(), "ik_design": network.ik_design.describe(),
            "sheet": o.sheet.describe(), "level": o.level_design.describe() if o.level else None,
            "radius_m": network.space.radius, "centre_m": network.space.centre.tolist(),
            "options": {"encoder": o.encoder, "threads": o.threads, "seed": o.seed, "tick_ms": o.tick_ms,
                        "readout_tau_ms": o.readout_tau_ms, "noise_pa": o.noise_pa, "velocity": o.velocity, "level": o.level},
            "neurons": network.neuron_count(), "ik_neurons": network.ik_neuron_count(), "ik_wiring": network.ik_wiring.summary(),
            "normalisers": network.ik_wiring.norms.tolist(), "caps": network.ik_wiring.caps.tolist(),
            "synapses": int(network.backend.GetKernelStatus("num_connections")),
            "simulated_s": round(sim_s, 2), "wall_s": round(wall, 1), "speed": round(sim_s / wall, 3) if wall > 0 else None,
            "ik_calibration_fit": network.ik_calibration.fit,
            "ramp_calibration_fit": network.ramp_calibration.fit if network.ramp_calibration is not None else None}


def _toward(current: np.ndarray, target: np.ndarray, step: float, circular: Optional[np.ndarray] = None) -> np.ndarray:
    difference = target - current
    if circular is not None:
        difference = np.where(circular, so3.wrap(difference), difference)
    return current + np.clip(difference, -step, step)


def open_loop_run(backend: Any, chain_name: str, count: int = 100, ik_design: Optional[PoseIkDesign] = None,
                  pair_seed: int = TEST_SEED, fraction: Sequence[float] = (0.05, 0.4), spread: Optional[float] = SPREAD, log=print,
                  **options: Any) -> Dict[str, Any]:
    """The comparisons and the shares in spikes: the arm is held at a pose, a goal part of the way to another pose is
    written, and turn, offset and shares are read and compared with the exact ones."""

    started = time.time()
    chain, network = build_ik(backend, chain_name, ik_design, **options)
    design = network.ik_design
    dt = network.options.tick_ms / 1000.0
    radius = network.space.radius
    circular = np.array([j.circular for j in chain.joints])
    rng = np.random.default_rng([int(pair_seed), 99])
    q = np.zeros(chain.n) if chain.joints[0].circular else chain.limits().mean(axis=1)
    goal_lpy, goal_position = np.array(lift_pitch_yaw(chain.rotation(q))), kinematics.position(chain, q)
    for _ in range(20):
        network.step_ik(q, goal_lpy, goal_position)
    rows: List[Dict[str, Any]] = []
    for pair in pairs(chain, count, pair_seed, spread):
        f = float(rng.uniform(*fraction))
        step_to = np.where(circular, so3.wrap(pair.goal_pose - pair.start), pair.goal_pose - pair.start)
        part = pair.start + f * step_to
        target_lpy, target_position = np.array(lift_pitch_yaw(chain.rotation(part))), kinematics.position(chain, part)
        moved = 0
        while moved < 400 and (np.max(np.abs(np.where(circular, so3.wrap(pair.start - q), pair.start - q))) > 1e-9
                               or np.max(np.abs(so3.wrap(target_lpy - goal_lpy))) > 1e-9 or np.linalg.norm(target_position - goal_position) > 1e-9):
            q = _toward(q, pair.start, MOVE_SPEED * dt, circular)
            goal_lpy = so3.travel(goal_lpy, target_lpy, MOVE_SPEED * dt)
            goal_position = rules.travel_position(goal_position, target_position, MOVE_SPEED * radius * dt)
            network.step_ik(q, goal_lpy, goal_position)
            moved += 1
        ticks = [network.step_ik(q, goal_lpy, goal_position) for _ in range(HOLD_TICKS)][-MEASURE_TICKS:]
        turn, offset = np.mean([t.turn for t in ticks], axis=0), np.mean([t.offset for t in ticks], axis=0)
        shares = np.mean([t.shares for t in ticks], axis=0)
        true_turn = so3.turn(chain.rotation(q), goal_lpy, design.comparison) if design.orientation else np.zeros(3)
        true_offset = rules.offset(chain, q, goal_position) if design.position else np.zeros(3)
        wired_turn, wired_offset, wired_shares, wired_left = network.exact(q, goal_lpy, goal_position)
        true_position = kinematics.position(chain, q)
        rows.append({
            "index": pair.index, "q": q.tolist(), "goal_lpy": goal_lpy.tolist(), "goal_position": goal_position.tolist(), "fraction": f,
            "turn": turn.tolist(), "true_turn": true_turn.tolist(), "wired_turn": wired_turn.tolist(),
            "turn_direction_error_deg": angle_between_deg(turn, true_turn) if design.orientation else None,
            "offset": offset.tolist(), "true_offset": true_offset.tolist(), "wired_offset": wired_offset.tolist(),
            "offset_direction_error_deg": angle_between_deg(offset, true_offset) if design.position else None,
            "shares": shares.tolist(), "wired_shares": wired_shares.tolist(),
            "share_sd": np.std([t.shares for t in ticks], axis=0).tolist(), "offset_sd": np.std([t.offset for t in ticks], axis=0).tolist(),
            "left": np.mean([t.left for t in ticks], axis=0).tolist(), "wired_left": wired_left.tolist(),
            #: the two positions as the ramp readouts hold them, against the place of the packets and against the truth (reaches)
            "ramp_goal_error": (np.linalg.norm(np.mean([t.ramp_goal for t in ticks], axis=0) - goal_position) / radius),
            "ramp_hand_error": (np.linalg.norm(np.mean([t.ramp_hand for t in ticks], axis=0) - true_position) / radius),
            "hand_sheet_error": float(np.linalg.norm(np.nanmean([t.tick.position for t in ticks], axis=0) - true_position) / radius),
            "goal_sheet_error": float(np.linalg.norm(np.nanmean([t.goal_position for t in ticks], axis=0) - goal_position) / radius),
            "cells_error": float(np.linalg.norm(np.mean([t.tick.cells_position for t in ticks], axis=0) - true_position) / radius),
            "sheet_spikes": np.mean([t.tick.sheet_counts.sum(axis=1) for t in ticks], axis=0).tolist(),
            "goal_sheet_spikes": np.mean([t.goal_sheet_counts.sum(axis=1) for t in ticks], axis=0).tolist(),
            "room": so3.limit_room(chain, q, design, int(network.calibration.settings["ring_size"])).tolist(),
        })
        if log is not None and len(rows) % 20 == 0:
            s = open_loop_summary(rows, design)
            log("%s: %d poses, offset off by %.4f reaches (rms), shares off by %.4f rad (rms), %.0f s" % (
                chain_name, len(rows), s["offset_rms"] or float("nan"), s["share_rms_rad"], time.time() - started))
    document = meta(chain, network, started)
    document.update(experiment="pose_ik_open_loop", pair_seed=int(pair_seed), fraction=list(fraction), spread=spread, rows=rows,
                    summary=open_loop_summary(rows, design))
    return document


def exact_trace(chain: Chain, pair: Any, design: PoseIkDesign, **options: Any) -> rules.Trace:
    """The exact rule on the same pair, with the delay, noise and bounds the spiking network has about."""

    settings = dict(delay_s=0.3, noise=0.01, rng=np.random.default_rng(int(pair.index)))
    settings.update(options)
    return rules.simulate(chain, pair.start, pair.goal, pair.goal_position, design, **settings)


def reach_run(backend: Any, chain_name: str, count: int = 100, ik_design: Optional[PoseIkDesign] = None, pair_seed: int = TEST_SEED,
              spread: Optional[float] = SPREAD, shard: Sequence[int] = (0, 1), max_minutes: Optional[float] = None,
              motor_scale: float = 1.0, push: Optional[Sequence[float]] = None, goal_change_s: Optional[float] = None,
              goal_ring_settings: Any = None, silence: float = 0.0, only: Optional[Sequence[int]] = None, log=print,
              make_robot: Any = None, method: str = "ring", checkpoint: Optional[str] = None, **options: Any) -> Dict[str, Any]:
    """Closed-loop trials on start and goal pairs.  ``shard``: ``(i, n)`` runs every n-th pair from the i-th on.
    ``max_minutes``: stop before the next trial when this much time has passed (the pairs that were not run are
    listed).  ``push``: ``(time, angle)``: one joint, chosen per pair, is displaced.  ``goal_change_s``: at that
    time the goal becomes the next pair's goal.  ``make_robot(chain)``: another robot than the light one (Gazebo).
    ``method``: ``ring`` (each joint follows its goal ring) or one of
    :data:`ring_se3.standard.targets.EXACT_METHODS` (the network runs, the joints follow the method).
    ``checkpoint``: a file that is written again after every trial."""

    from .session import PoseIkSession, trial_row

    started = time.time()
    chain, network = build_ik(backend, chain_name, ik_design, goal_ring_settings=goal_ring_settings, **options)
    design = network.ik_design
    dead = network.silence_ik(silence, seed=3) if silence > 0.0 else 0
    session = PoseIkSession(network, motor_scale=motor_scale, robot=make_robot(chain) if make_robot is not None else None)
    if method != "ring":
        from ..standard.targets import exact_targets

        session.method = exact_targets(method)
    session.warm_up()
    in_gazebo = hasattr(session.robot, "link")
    start_base = session.robot.link.base_pose() if in_gazebo else None
    every = pairs(chain, count, pair_seed, spread)
    chosen = [pair for pair in every if pair.index % int(shard[1]) == int(shard[0])]
    if only is not None:                      # the pairs an earlier run did not get to
        chosen = [pair for pair in every if pair.index in set(int(i) for i in only)]
    rng = np.random.default_rng([int(pair_seed), 77])
    pushed = {pair.index: int(rng.integers(chain.n)) for pair in every}
    rows: List[Dict[str, Any]] = []
    skipped: List[int] = []

    def document_now(not_run: Sequence[int]) -> Dict[str, Any]:
        out = meta(chain, network, started)
        out.update(experiment="pose_ik_reach", pair_seed=int(pair_seed), pairs=int(count), spread=spread, shard=[int(shard[0]), int(shard[1])],
                   motor_scale=float(motor_scale), push=list(push) if push is not None else None, goal_change_s=goal_change_s,
                   goal_ring_settings=network.joint_goal_rings[0].settings.describe() if network.joint_goal_rings else None,
                   silence=float(silence), silenced_neurons=int(dead), method=method, robot=session.robot.describe(),
                   skipped=list(not_run), rows=rows, summary=reach_summary(rows))
        if in_gazebo:
            out.update(base_at_start=list(start_base) if start_base else None)
        return out

    for position, pair in enumerate(chosen):
        if max_minutes is not None and (time.time() - started) / 60.0 > float(max_minutes):
            skipped.append(pair.index)
            continue
        if in_gazebo:
            session.robot.link.put_base_back()
        session.place(pair.start)
        push_now = (float(push[0]), pushed[pair.index], float(push[1])) if push is not None else None
        after = every[(pair.index + 1) % len(every)]
        change = (float(goal_change_s), after.goal, after.goal_position) if goal_change_s is not None else None
        trial = session.reach(pair.goal, pair.goal_position, push=push_now, goal_change=change)
        row = trial_row(trial, design)
        exact = exact_trace(chain, pair, design, motor_scale=motor_scale, push=push_now, goal_change=change)
        row.update(index=pair.index, start=pair.start.tolist(), goal_lpy=(after if change else pair).goal.tolist(),
                   goal_position=(after if change else pair).goal_position.tolist(),
                   exact_final_error_deg=exact.final_error_deg, exact_final_error_reach=exact.final_error_reach,
                   exact_reached=exact.reached(design), exact_time_to_reach_s=exact.time_to(design),
                   exact_path_ratio=exact.path_reach / exact.start_error_reach if exact.start_error_reach > 1e-9 else float("nan"),
                   exact_at_limit=bool(exact.at_limit), pushed_joint=pushed[pair.index] if push is not None else None,
                   start_off_rad=float(np.max(np.abs(so3.wrap(trial.q[0] - pair.start)))))
        if in_gazebo:
            row.update(base=list(session.robot.link.base_pose() or ()))
        rows.append(row)
        if log is not None:
            log("%s pair %d: from %.0f deg, %.0f%% of the reach to %.1f deg, %.1f%% (%s), reached after %.1f s, %.1f s; exact rule %.1f deg, %.1f%% | %d done, %.0f%% reached, %.0f s" % (
                chain_name, pair.index, row["start_error_deg"], 100 * row["start_error_reach"], row["final_error_deg"], 100 * row["final_error_reach"],
                "at a limit" if row["at_limit"] else "free", row["time_to_reach_s"], row["duration_s"], row["exact_final_error_deg"],
                100 * row["exact_final_error_reach"], len(rows), 100 * np.mean([r["reached"] for r in rows]), time.time() - started))
        if checkpoint:
            write_json(checkpoint + ".part", document_now(skipped + [later.index for later in chosen[position + 1:]]))
            os.replace(checkpoint + ".part", checkpoint)
    return document_now(skipped)


def _median(values: Sequence[Any]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(finite)) if finite else None


def _percentile(values: Sequence[Any], q: float) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.percentile(finite, q)) if finite else None


def reach_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"trials": 0}
    reached = [r for r in rows if r["reached"]]
    exact = [r for r in rows if r["exact_reached"]]
    return {
        "trials": len(rows),
        "reached": float(np.mean([r["reached"] for r in rows])),
        "exact_reached": float(np.mean([r["exact_reached"] for r in rows])),
        "reached_of_exact_reached": float(np.mean([r["reached"] for r in exact])) if exact else None,
        "final_error_median_deg": _median([r["final_error_deg"] for r in rows]),
        "final_error_p90_deg": _percentile([r["final_error_deg"] for r in rows], 90),
        "final_error_median_reach": _median([r["final_error_reach"] for r in rows]),
        "final_error_p90_reach": _percentile([r["final_error_reach"] for r in rows], 90),
        "final_error_reached_median_deg": _median([r["final_error_deg"] for r in reached]),
        "final_error_reached_median_reach": _median([r["final_error_reach"] for r in reached]),
        "time_to_reach_median_s": _median([r["time_to_reach_s"] for r in rows]),
        "path_ratio_median": _median([r["path_ratio"] for r in reached if r["start_error_reach"] > 0.1]),
        "exact_time_to_reach_median_s": _median([r["exact_time_to_reach_s"] for r in rows]),
        "exact_path_ratio_median": _median([r["exact_path_ratio"] for r in exact if r["start_error_reach"] > 0.1]),
        "exact_final_error_median_deg": _median([r["exact_final_error_deg"] for r in rows]),
        "exact_final_error_median_reach": _median([r["exact_final_error_reach"] for r in rows]),
        "missed": int(len(rows) - len(reached)),
        "missed_at_limit": int(sum(1 for r in rows if not r["reached"] and r["at_limit"])),
        "settled": float(np.mean([r["settled_s"] is not None for r in rows])),
        "duration_median_s": _median([r["duration_s"] for r in rows]),
        "decoded_error_median_deg": _median([r["decoded_error_median_deg"] for r in rows]),
        "decoded_error_median_reach": _median([r["decoded_error_median_reach"] for r in rows]),
        "start_error_median_deg": _median([r["start_error_deg"] for r in rows]),
        "start_error_median_reach": _median([r["start_error_reach"] for r in rows]),
    }


def open_loop_summary(rows: Sequence[Dict[str, Any]], design: PoseIkDesign, share_floor: float = 0.05) -> Dict[str, Any]:
    shares, wired_shares = np.array([r["shares"] for r in rows]), np.array([r["wired_shares"] for r in rows])
    large = np.abs(wired_shares) > share_floor
    quiet = np.abs(wired_shares) < 0.01
    out: Dict[str, Any] = {
        "poses": len(rows), "share_floor_rad": share_floor, "shares_large": int(large.sum()),
        "share_right_sign": float(np.mean(np.sign(shares[large]) == np.sign(wired_shares[large]))) if large.any() else None,
        "share_rms_rad": float(np.sqrt(np.mean((shares - wired_shares) ** 2))),
        "share_when_none_asked_rms_rad": float(np.sqrt(np.mean(shares[quiet] ** 2))) if quiet.any() else None,
        "share_sd_rad": float(np.mean([r["share_sd"] for r in rows])),
        "per_joint": [{"joint": k + 1, "large": int(large[:, k].sum()),
                       "right_sign": float(np.mean(np.sign(shares[large[:, k], k]) == np.sign(wired_shares[large[:, k], k]))) if large[:, k].any() else None,
                       "rms_rad": float(np.sqrt(np.mean((shares[:, k] - wired_shares[:, k]) ** 2)))} for k in range(shares.shape[1])],
        "turn_rms_rad": None, "turn_direction_error_median_deg": None, "turn_direction_error_p90_deg": None, "turn_size_median_rad": None,
        "offset_rms": None, "offset_direction_error_median_deg": None, "offset_direction_error_p90_deg": None, "offset_size_median": None,
    }
    if design.orientation:
        turn, true = np.array([r["turn"] for r in rows]), np.array([r["true_turn"] for r in rows])
        direction = np.array([r["turn_direction_error_deg"] for r in rows], dtype=float)
        out.update(turn_rms_rad=float(np.sqrt(np.mean((turn - np.clip(true, -0.5, 0.5)) ** 2))),
                   turn_direction_error_median_deg=float(np.nanmedian(direction)), turn_direction_error_p90_deg=float(np.nanpercentile(direction, 90)),
                   turn_size_median_rad=float(np.median(np.linalg.norm(true, axis=1))))
    if design.position:
        offset, true = np.array([r["offset"] for r in rows]), np.array([r["true_offset"] for r in rows])
        bound = 1.0 / design.position_scale
        direction = np.array([r["offset_direction_error_deg"] for r in rows], dtype=float)
        out.update(offset_rms=float(np.sqrt(np.mean((offset - np.clip(true, -bound, bound)) ** 2))),
                   offset_error_median=float(np.median(np.linalg.norm(offset - np.clip(true, -bound, bound), axis=1))),
                   offset_direction_error_median_deg=float(np.nanmedian(direction)), offset_direction_error_p90_deg=float(np.nanpercentile(direction, 90)),
                   offset_size_median=float(np.median(np.linalg.norm(true, axis=1))),
                   offset_sd=float(np.mean([r["offset_sd"] for r in rows])),
                   hand_sheet_error_median=_median([r["hand_sheet_error"] for r in rows]), cells_error_median=_median([r["cells_error"] for r in rows]),
                   goal_sheet_error_median=_median([r["goal_sheet_error"] for r in rows]),
                   ramp_goal_error_median=_median([r["ramp_goal_error"] for r in rows]), ramp_hand_error_median=_median([r["ramp_hand_error"] for r in rows]),
                   sheet_spikes_median=_median([v for r in rows for v in r["sheet_spikes"]]),
                   sheet_spikes_smallest=float(np.min([v for r in rows for v in r["sheet_spikes"]])),
                   goal_sheet_spikes_smallest=float(np.min([v for r in rows for v in r["goal_sheet_spikes"]])))
    return out


__all__ = ["SPREAD", "build_ik", "default_task", "exact_trace", "meta", "open_loop_run", "open_loop_summary", "reach_run", "reach_summary"]
