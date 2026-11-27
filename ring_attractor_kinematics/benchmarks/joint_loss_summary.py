"""Summary of the joint-loss benchmark: does the hand still reach the goal when one joint is disabled part-way?

Groups are one condition at one kernel seed: ``intact``, ``lock``, ``motor``, ``allocation`` (and ``allocation_lock``,
``loss…``). Everything is about the hand (end effector) and pools the disabled joints. The per-joint counts are kept
for the supplement.

* ``success``: reached at the end, with a Wilson 95% interval. ``success_reachable`` counts only the pairs whose goal
  the remaining joints could still reach with the joint frozen where this trial froze it (``reachability``, a lower
  bound from DLS);
* ``course``: the hand's orientation (deg) and position (mm) error against time, as the 25th, 50th and 75th percentile
  over pairs. A trial that ended keeps its last error;
* ``against_intact``: the same pairs at the same seed with nothing disabled. The intact rows come from this benchmark's
  ``intact`` run, or else from ``reach`` (TIAGo7, matrix method) at that seed; for ``gazebo_joint_loss``, from the
  ``gazebo`` benchmark's spiking run. Gives McNemar on success, paired final
  errors and times, and the extra motion of the remaining joints after onset;
* ``absorption``: the paired intact trial's final angles with the disabled joint put back where it stood at onset. That
  is where the hand would end if the other joints had not made up for the lost one. ``would_miss`` counts the pairs
  where that counterfactual hand is outside the marks; ``absorbed`` counts those this condition still reached.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import get_chain, workspace
from reaching_benchmark.geometry.rotations import rotation_angle
from reaching_benchmark.statistics import censored, mcnemar, paired, success_summary

from .. import results
from ..rules.cascade_rule import goal_matrix
from .reach_summary import within_curve
from .twins import LOSS_TWINS

GRID = np.arange(0.0, 25.001, 0.25)
DISABLED = ("lock", "motor", "allocation", "allocation_lock")


def course(rows: Sequence[Dict[str, Any]], key: str, scale: float = 1.0, grid: np.ndarray = GRID) -> Dict[str, List[float]]:
    """Quartiles over pairs of ``row[key]`` (a thinned time course) on ``grid``; a trial that ended keeps its last value."""

    if not rows:
        return {"t": grid.tolist(), "p25": [], "median": [], "p75": []}
    values = np.array([np.interp(grid, np.asarray(r["t"], dtype=float), scale * np.asarray(r[key], dtype=float)) for r in rows])
    return {"t": grid.tolist(), "p25": np.percentile(values, 25, axis=0).tolist(), "median": np.median(values, axis=0).tolist(),
            "p75": np.percentile(values, 75, axis=0).tolist()}


def counterfactual(chain: Any, intact: Dict[str, Any], joint: int, onset_q: Sequence[float]) -> Tuple[float, float]:
    """Orientation (deg) and position (reaches) error of the intact trial's final angles with ``joint`` at ``onset_q``."""

    q = np.asarray(intact["final_q"], dtype=float).copy()
    q[int(joint)] = float(onset_q[int(joint)])
    degrees = float(np.degrees(rotation_angle(chain.rotation(q), goal_matrix(intact["goal_lpy"]))))
    reach = float(np.linalg.norm(np.asarray(intact["goal_position"]) - kinematics.position(chain, q)) / workspace(chain).radius)
    return degrees, reach


def onset_angles(row: Dict[str, Any], onset_s: float) -> np.ndarray:
    """The angles at onset: stored by this benchmark, else read off the thinned path (an intact ``reach`` row)."""

    if row.get("q_onset") is not None:
        return np.asarray(row["q_onset"], dtype=float)
    t, q = np.asarray(row["t"], dtype=float), np.asarray(row["q"], dtype=float)
    return np.array([np.interp(onset_s, t, q[:, j]) for j in range(q.shape[1])])


def within_marks(design: Dict[str, Any], degrees: float, reach: float) -> bool:
    ok = True
    if design["task"] != "position":
        ok &= degrees <= design["reach_deg"]
    if design["task"] != "orientation":
        ok &= reach <= design["reach_reach"]
    return bool(ok)


def group_summary(rows: List[Dict[str, Any]], design: Dict[str, Any], radius_mm: float) -> Dict[str, Any]:
    limit = float(design["trial_s"])
    reachable = [r for r in rows if (r.get("reachability") or {}).get("reachable")]
    joints = sorted({int(r["disabled_joint"]) for r in rows if r.get("disabled_joint") is not None})
    per_trial: Dict[str, List[Any]] = {key: [r.get(key) for r in rows] for key in (
        "seed", "index", "disabled_joint", "reached", "final_error_deg", "final_error_reach", "time_to_reach_s",
        "remaining_excursion_rad", "disabled_drift_rad")}
    per_trial["reachable"] = [bool((r.get("reachability") or {}).get("reachable")) for r in rows]
    return {
        "pairs": len(rows),
        "success": success_summary([r["reached"] for r in rows]),
        "success_reachable": success_summary([r["reached"] for r in reachable]),
        "reachable": success_summary([bool((r.get("reachability") or {}).get("reachable")) for r in rows]),
        "time_median_s": float(np.median(censored([r["time_to_reach_s"] for r in rows], limit))) if rows else None,
        "final_error_median_deg": float(np.median([r["final_error_deg"] for r in rows])) if rows else None,
        "final_error_median_mm": float(radius_mm * np.median([r["final_error_reach"] for r in rows])) if rows else None,
        "remaining_excursion_median_rad": float(np.median([r["remaining_excursion_rad"] for r in rows])) if rows else None,
        "disabled_drift_median_rad": float(np.median([r["disabled_drift_rad"] for r in rows])) if rows else None,
        "disabled_share_median_rad": float(np.nanmedian([r["disabled_share_median_rad"] for r in rows])) if rows else None,
        "within": within_curve(rows, design, grid=tuple(GRID)),
        "course_deg": course(rows, "error_deg"),
        "course_mm": course(rows, "error_reach", radius_mm),
        "by_joint": {str(k + 1): success_summary([r["reached"] for r in rows if int(r["disabled_joint"]) == k]) for k in joints},
        "twins": {name: success_summary([r["twins"][name]["reached"] for r in rows]) for name in LOSS_TWINS if rows and name in rows[0]["twins"]},
        "per_trial": per_trial,
    }


#: where a joint-loss benchmark finds its intact rows when it has no ``intact`` run of its own
INTACT_FALLBACK = {"joint_loss": "reach_kinematic", "joint_loss_gazebo": "reach_gazebo"}


def intact_reference(groups: Dict[Tuple[str, int], Dict[str, Any]], seed: int, run: Dict[str, Any], task: str,
                     name: str = "joint_loss") -> Tuple[List[Dict[str, Any]], str]:
    """The intact rows for a seed: this benchmark's ``intact`` run, else the fallback benchmark's spiking run (same arm,
    task, κ and network method, no condition)."""

    method = run.get("network_method", "matrix")
    if ("intact", seed, method) in groups:
        return groups[("intact", seed, method)]["rows"], "%s intact" % name
    fallback = INTACT_FALLBACK.get(name, "reach_kinematic")
    rows: List[Dict[str, Any]] = []
    for document in results.read_raw(fallback):
        other = document["run"]
        if (other["arm"] == run["arm"] and int(other["seed"]) == seed and float(other["kappa"]) == float(run["kappa"])
                and document["design"]["task"] == task and other.get("network_method", "matrix") == method
                and other.get("method", "ring") == "ring" and not other.get("condition")):
            rows += document["rows"]
    return rows, fallback


def against_intact(chain: Any, rows: List[Dict[str, Any]], intact: List[Dict[str, Any]], design: Dict[str, Any], radius_mm: float,
                   onset_s: float, joint_disabled: bool) -> Dict[str, Any]:
    """Paired with the intact trials; ``joint_disabled``: also the absorption counterfactual (not for diffuse loss)."""

    by_index = {int(r["index"]): r for r in intact}
    pairs = [(by_index[int(r["index"])], r) for r in rows if int(r["index"]) in by_index
             and np.allclose(by_index[int(r["index"])]["start"], r["start"], atol=1e-6)]
    if not pairs:
        return {"pairs": 0}
    limit = float(design["trial_s"])
    base, cond = [a for a, _ in pairs], [b for _, b in pairs]
    out: Dict[str, Any] = {
        "pairs": len(pairs),
        "reached": [int(sum(bool(a["reached"]) for a in base)), int(sum(bool(b["reached"]) for b in cond))],
        "mcnemar": mcnemar([b["reached"] for b in cond], [a["reached"] for a in base]),
        "final_error_deg": paired([b["final_error_deg"] for b in cond], [a["final_error_deg"] for a in base]),
        "final_error_mm": paired([radius_mm * b["final_error_reach"] for b in cond], [radius_mm * a["final_error_reach"] for a in base]),
        "time": paired(censored([b["time_to_reach_s"] for b in cond], limit), censored([a["time_to_reach_s"] for a in base], limit)),
    }
    excursion = [(b["remaining_excursion_rad"], a.get("remaining_excursion_rad")) for a, b in pairs if a.get("remaining_excursion_rad") is not None]
    if excursion:
        out["remaining_excursion_rad"] = paired([b for b, _ in excursion], [a for _, a in excursion])
    if not joint_disabled:
        return out
    would_miss, absorbed, cf_deg, cf_mm = 0, 0, [], []
    for a, b in pairs:
        degrees, reach = counterfactual(chain, a, int(b["disabled_joint"]), onset_angles(a, onset_s))
        cf_deg.append(degrees)
        cf_mm.append(radius_mm * reach)
        if not within_marks(design, degrees, reach):
            would_miss += 1
            absorbed += int(bool(b["reached"]))
    out["absorption"] = {"would_miss": would_miss, "absorbed": absorbed,
                         "absorbed_share": success_summary([True] * absorbed + [False] * (would_miss - absorbed)),
                         "counterfactual_error_median_deg": float(np.median(cf_deg)), "counterfactual_error_median_mm": float(np.median(cf_mm)),
                         "counterfactual_error_deg": cf_deg, "counterfactual_error_mm": cf_mm}
    return out


def summarise(name: str = "joint_loss") -> Dict[str, Any]:
    groups: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for document in results.read_raw(name):
        run = document["run"]
        condition = "loss" if run["condition"].startswith("loss") else run["condition"]
        key = (run["condition"] if condition == "loss" else condition, int(run["seed"]), run.get("network_method", "matrix"))
        group = groups.setdefault(key, {"run": run, "design": document["design"], "rows": [], "lesion": document.get("lesion"),
                                        "neurons": document["neurons"], "simulated_s": 0.0, "wall_s": 0.0})
        group["rows"] += [dict(r, seed=int(run["seed"])) for r in document["rows"]]
        group["simulated_s"] += document["simulated_s"]
        group["wall_s"] += document["wall_s"]
    out: Dict[str, Any] = {"benchmark": name, "groups": {}}
    for (condition, seed, method), group in sorted(groups.items()):
        run, design = group["run"], group["design"]
        chain = get_chain(run["arm"])
        radius_mm = 1000.0 * workspace(chain).radius
        rows = sorted(group["rows"], key=lambda r: r["index"])
        entry = {"condition": condition, "seed": seed, "arm": run["arm"], "task": design["task"], "onset_s": run.get("onset_s"),
                 "lesion": group["lesion"], "neurons": group["neurons"],
                 "speed": group["simulated_s"] / group["wall_s"] if group["wall_s"] else None,
                 **group_summary(rows, design, radius_mm)}
        if condition != "intact":
            intact, source = intact_reference(groups, seed, run, design["task"], name)
            entry["against_intact"] = dict(against_intact(chain, rows, intact, design, radius_mm, float(run.get("onset_s") or 2.0),
                                                          condition in DISABLED), reference=source)
        variant = "" if method == "matrix" else "_" + {"lead": "predictive_encoding"}.get(method, method)
        out["groups"]["%s_seed%d%s" % (condition, seed, variant)] = dict(entry, network_method=method)
    return out
