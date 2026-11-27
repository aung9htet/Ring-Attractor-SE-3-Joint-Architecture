"""B1 analysis: every orientation source replayed on the spiking network's recorded output.

For every path the network's lag ``τ`` is measured first (the shift of the true joint path that best explains the
decoded matrix).  Each source then gets, per tick, the decoded matrix of that tick as its cue (M1: of the run
with joint lead; M2: the readout-ring angles), the joint path ``τ`` late for its axes, and the joint velocities.

Reported per source and path: median and 90th-percentile error (all ticks and moving ticks), the source's own
lag, coverage (share of ticks within 5°), error against |pitch| (pole path), and recovery after 300 ms cue drops.
The built observer (``observer``) is not replayed: its own estimate, the matrix its gates passed, is scored the same
way (no cue drops: its cue is the network's).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.geometry.rotations import rotation_angle

from .. import results
from ..rules import observer_rule

MOVING_RAD_S = 0.05
COVERAGE_DEG = 5.0
DROP_TICKS = 6                  # 300 ms
DROP_EVERY = 60                 # one drop every 3 s
PITCH_BINS = (0.0, 30.0, 60.0, 75.0, 85.0, 90.01)
MAX_LAG_TICKS = 16


def lag_ticks(chain: Any, path: np.ndarray, estimates: np.ndarray) -> int:
    """The lag (ticks) at which the estimates best match the true orientation (median angle)."""

    best, best_lag = np.inf, 0
    for lag in range(0, MAX_LAG_TICKS + 1):
        errors = [rotation_angle(estimates[t], chain.rotation(path[t - lag])) for t in range(MAX_LAG_TICKS, len(path))]
        if np.median(errors) < best:
            best, best_lag = float(np.median(errors)), lag
    return best_lag


def replay_source(code: str, chain: Any, path: np.ndarray, document: Dict[str, Any], delay_s: float,
                  cue_drop: Any = None) -> Dict[str, np.ndarray]:
    run = document["lead"] if code == "M1" else document["plain"]
    source = observer_rule.make_source(code, chain, delay_s)
    cues = np.array(run["tool"])
    angles = np.array(run["angles"], dtype=float) if code == "M2" else None
    return observer_rule.open_loop(source, chain, path, cues=cues, ring_angles=angles, cue_drop=cue_drop)


def summarise(name: str = "moving_estimate") -> Dict[str, Any]:
    chain = get_chain("tiago7")
    out: Dict[str, Any] = {"benchmark": "moving_estimate", "paths": {}}
    for document in results.read_raw("moving_estimate"):
        path = np.array(document["q"])
        decoded = np.array(document["plain"]["tool"])
        tau_ticks = lag_ticks(chain, path, np.array([observer_rule.nearest_rotation(m) for m in decoded]))
        delay_s = 0.05 * tau_ticks
        speed = np.max(np.abs(np.diff(path, axis=0, prepend=path[:1])), axis=1) / 0.05
        moving = speed > MOVING_RAD_S
        sources: Dict[str, Any] = {}
        for code in observer_rule.CODES:
            replay = replay_source(code, chain, path, document, delay_s)
            errors = replay["error_deg"]
            entry = {"error_deg": errors[::2].tolist(), "median_deg": float(np.median(errors)), "p90_deg": float(np.percentile(errors, 90)),
                     "moving_median_deg": float(np.median(errors[moving])) if moving.any() else None,
                     "moving_p90_deg": float(np.percentile(errors[moving], 90)) if moving.any() else None,
                     "coverage": float(np.mean(errors <= COVERAGE_DEG)), "lag_s": 0.05 * lag_ticks(chain, path, replay["estimates"])}
            if document["path"] == "pole" and code == "M5":
                entry["chart"] = replay["chart"][::2].tolist()
            if document["path"] == "pole":
                pitch = np.abs(replay["pitch_deg"])
                entry["by_pitch"] = [{"from_deg": a, "to_deg": b, "median_deg": float(np.median(errors[(pitch >= a) & (pitch < b)]))
                                      if np.any((pitch >= a) & (pitch < b)) else None,
                                      "coverage": float(np.mean(errors[(pitch >= a) & (pitch < b)] <= COVERAGE_DEG))
                                      if np.any((pitch >= a) & (pitch < b)) else None} for a, b in zip(PITCH_BINS[:-1], PITCH_BINS[1:])]
            if document["path"] == "speed0.5":
                recoveries, peaks, traces = [], [], []
                for start in range(DROP_EVERY, len(path) - DROP_EVERY, DROP_EVERY):
                    dropped = replay_source(code, chain, path, document, delay_s, cue_drop=(start, start + DROP_TICKS))["error_deg"]
                    base = float(np.median(errors))
                    after = dropped[start + DROP_TICKS:]
                    back = np.flatnonzero(after <= max(1.5 * base, base + 1.0))
                    recoveries.append(0.05 * float(back[0]) if back.size else float("nan"))
                    peaks.append(float(np.max(dropped[start:start + DROP_TICKS + 10])))
                    traces.append(dropped[start - 4:start + DROP_TICKS + 20].tolist())
                entry.update(drop_recovery_median_s=float(np.nanmedian(recoveries)), drop_peak_median_deg=float(np.median(peaks)),
                             drop_trace_mean_deg=np.mean(traces, axis=0).tolist())
            sources[code] = entry
        if "observer" in document:
            hands = [observer_rule.nearest_rotation(np.array(m)) for m in document["observer"]["hand"]]
            errors = np.array([np.degrees(rotation_angle(m, chain.rotation(q))) for m, q in zip(hands, path)])
            entry = {"error_deg": errors[::2].tolist(), "median_deg": float(np.median(errors)), "p90_deg": float(np.percentile(errors, 90)),
                     "moving_median_deg": float(np.median(errors[moving])) if moving.any() else None,
                     "moving_p90_deg": float(np.percentile(errors[moving], 90)) if moving.any() else None,
                     "coverage": float(np.mean(errors <= COVERAGE_DEG)), "lag_s": 0.05 * lag_ticks(chain, path, np.array(hands)),
                     "fallback_share": float(np.mean(document["observer"]["fallback"])),
                     "switches": int(np.sum(np.diff(document["observer"]["selected"]) != 0))}
            if document["path"] == "pole":
                pitch = np.abs(replay["pitch_deg"])
                entry["by_pitch"] = [{"from_deg": a, "to_deg": b, "median_deg": float(np.median(errors[(pitch >= a) & (pitch < b)]))
                                      if np.any((pitch >= a) & (pitch < b)) else None,
                                      "coverage": float(np.mean(errors[(pitch >= a) & (pitch < b)] <= COVERAGE_DEG))
                                      if np.any((pitch >= a) & (pitch < b)) else None} for a, b in zip(PITCH_BINS[:-1], PITCH_BINS[1:])]
            sources["observer"] = entry
        out["paths"][document["path"]] = {"ticks": int(len(path)), "network_lag_s": delay_s, "pitch_deg": replay["pitch_deg"][::2].tolist(),
                                          "speed_rad_s": speed[::2].tolist(), "sources": sources,
                                          "hand_speed_m_s": _hand_speed(chain, path)[::2].tolist(),
                                          "turn_rate_deg_s": _turn_rate(chain, path)[::2].tolist(),
                                          "position_error_m": _position_error(chain, path, np.array(document["plain"]["position"]))[::2].tolist()}
    return out


def _hand_speed(chain: Any, path: np.ndarray) -> np.ndarray:
    points = np.array([chain.frames(q)[-1][0] for q in path])
    return np.linalg.norm(np.diff(points, axis=0, prepend=points[:1]), axis=1) / 0.05


def _turn_rate(chain: Any, path: np.ndarray) -> np.ndarray:
    rotations = [chain.rotation(q) for q in path]
    return np.array([0.0] + [np.degrees(rotation_angle(a, b)) / 0.05 for a, b in zip(rotations[:-1], rotations[1:])])


def _position_error(chain: Any, path: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    points = np.array([chain.frames(q)[-1][0] for q in path])
    return np.linalg.norm(decoded - points, axis=1)
