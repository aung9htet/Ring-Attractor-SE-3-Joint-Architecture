"""Data derived from ``results/`` that the figures share (no simulation; NumPy only).

* ``common_path_traces.npz``: the RAK model's hand-orientation estimate error at every 50 ms tick of the shared joint
  path replayed at 0.25, 0.5 and 1 rad/s, without (``plain``) and with (``lead``) predictive encoding, from
  ``results/ring_attractor_kinematics/moving_estimate/common*.json`` (Figs 2e and 4b);
* ``sensitivity_metrics.json``: held-pose error medians of every neuron-silencing mask, from
  ``held_pose_neuron_loss/`` against the intact ``held_pose/tiago7.json`` (Fig. 5f, Supp. neuron loss);
* ``closed_loop_loss.json``: the 30 TIAGo7 reaches with 10% of neurons silenced, beside the same reaches intact
  (``joint_loss/tiago7_seed1_loss0.1_all_l1.json``, ``reach_kinematic/tiago7_seed1.json``; Supp. closed-loop loss).
"""

from __future__ import annotations

import numpy as np

from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.geometry.rotations import nearest_rotation, rotation_angle
from reaching_benchmark.statistics import censored

from .common import DATA, RAK, ensure_dirs, read, relative, write_json

DT = 0.05
#: a tick counts as moving when some joint turns faster than this (rad/s)
MOVING_RAD_S = 0.05


def common_path() -> None:
    chain = get_chain("tiago7")
    arrays, metrics = {}, {}
    for speed in (0.25, 0.5, 1.0):
        name = "common%g" % speed
        doc = read(RAK / "moving_estimate" / (name + ".json"))
        q = np.array(doc["q"], dtype=float)
        assert doc["dt"] == DT
        speed_t = np.max(np.abs(np.diff(q, axis=0, prepend=q[:1])), axis=1) / DT
        moving = speed_t > MOVING_RAD_S
        rotations = np.array([chain.rotation(v) for v in q])
        positions = np.array([chain.frames(v)[-1][0] for v in q])
        prefix = name.replace(".", "")
        arrays.update({prefix + "_t_s": DT * np.arange(len(q)), prefix + "_q": q, prefix + "_moving": moving,
                       prefix + "_speed_rad_s": speed_t, prefix + "_true_position_m": positions})
        metrics[name] = {"speed_rad_s": speed, "ticks": len(q), "moving_ticks": int(moving.sum())}
        for method in ("plain", "lead"):
            run = doc[method]
            estimates = np.array([nearest_rotation(m) for m in np.array(run["tool"], dtype=float)])
            errors = np.array([np.degrees(rotation_angle(a, b)) for a, b in zip(estimates, rotations)])
            position_mm = 1000 * np.linalg.norm(np.array(run["position"], dtype=float) - positions, axis=1)
            arrays[prefix + "_" + method + "_orientation_deg"] = errors
            arrays[prefix + "_" + method + "_position_mm"] = position_mm
            metrics[name][method] = {"median_deg": float(np.median(errors)), "moving_median_deg": float(np.median(errors[moving])),
                                     "moving_p90_deg": float(np.percentile(errors[moving], 90))}
    np.savez_compressed(DATA / "common_path_traces.npz", **arrays)
    write_json(DATA / "common_path_metrics.json", {
        "source": "results/ring_attractor_kinematics/moving_estimate/common{0.25,0.5,1}.json",
        "orientation_estimate": "recorded forward-network tool matrix, nearest proper rotation (SVD), against the supplied joints",
        "moving": "max |q[t] - q[t-1]| / 0.05 s > %g rad/s" % MOVING_RAD_S, "paths": metrics})


def sensitivity() -> None:
    base_doc = read(RAK / "held_pose" / "tiago7.json")
    radius_mm = 1000 * base_doc["radius_m"]
    base_q = np.array([r["q"] for r in base_doc["rows"]])
    base_ori = np.array([r["orientation_error_deg"] for r in base_doc["rows"]])
    base_pos = np.array([r["position_error_reach"] for r in base_doc["rows"]])
    runs = []
    for path in sorted((RAK / "held_pose_neuron_loss").glob("tiago7_loss*.json")):
        doc = read(path)
        q = np.array([r["q"] for r in doc["rows"]])
        assert len(q) == 30 and np.array_equal(q, base_q), path   # the same 30 poses as the intact run
        ori = np.array([r["orientation_error_deg"] for r in doc["rows"]])
        pos = np.array([r["position_error_reach"] for r in doc["rows"]])
        runs.append({"file": relative(path), "lesion": doc["lesion"], "poses": 30,
                     "orientation_median_deg": float(np.median(ori)), "position_median_reach": float(np.median(pos)),
                     "position_median_mm": float(np.median(pos) * radius_mm),
                     "paired_orientation_difference_median_deg": float(np.median(ori - base_ori)),
                     "paired_position_difference_median_reach": float(np.median(pos - base_pos))})
    assert len(runs) == 21, len(runs)
    write_json(DATA / "sensitivity_metrics.json", {
        "baseline_file": relative(RAK / "held_pose" / "tiago7.json"),
        "baseline": {"poses": 30, "orientation_median_deg": float(np.median(base_ori)),
                     "position_median_reach": float(np.median(base_pos)), "position_median_mm": float(np.median(base_pos) * radius_mm)},
        "runs": runs,
        "aggregation": "each mask's median over the same 30 poses; masks are never pooled into 90 poses"})


def closed_loop_loss() -> None:
    radius_mm = 805.74

    def records(path):
        doc = read(path)
        rows = sorted(doc["rows"], key=lambda r: r["index"])
        out = [{"target_index": r["index"], "reached": bool(r["reached"]), "orientation_deg": r["final_error_deg"],
                "position_mm": r["final_error_reach"] * radius_mm, "path_ratio": r["path_ratio"],
                "time_s_censored": float(censored([r["time_to_reach_s"]], 25.0)[0])} for r in rows]
        return doc, {"reached_count": sum(r["reached"] for r in out), "n_targets": len(out),
                     "median_orientation_deg": float(np.median([r["orientation_deg"] for r in out])),
                     "median_position_mm": float(np.median([r["position_mm"] for r in out])), "per_target": out}

    lesion_path = RAK / "joint_loss" / "tiago7_seed1_loss0.1_all_l1.json"
    nominal_path = RAK / "reach_kinematic" / "tiago7_seed1.json"
    lesion_doc, lesion = records(lesion_path)
    _, nominal = records(nominal_path)
    assert [r["target_index"] for r in lesion["per_target"]] == [r["target_index"] for r in nominal["per_target"]] == list(range(30))
    write_json(DATA / "closed_loop_loss.json", {
        "source_file": relative(lesion_path), "nominal_comparator_file": relative(nominal_path),
        "n_targets": 30, "kernel_seed": 1, "lesion_seed": 1, "lesion": lesion, "nominal": nominal,
        "silencing": lesion_doc.get("lesion") or lesion_doc.get("run", {}).get("lesion"),
        "interpretation": "One fixed diffuse silencing mask from network construction across the same 30 targets: a "
                          "sensitivity measurement, not recovery or joint removal."})


def main() -> None:
    ensure_dirs()
    common_path()
    sensitivity()
    closed_loop_loss()
    print("derived data written to", relative(DATA))
