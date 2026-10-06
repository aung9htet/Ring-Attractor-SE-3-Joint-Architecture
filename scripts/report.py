#!/usr/bin/env python3
"""Tables and figures of docs/results.md from the data in docs/results/data/.

    python3 -B scripts/report.py                       # writes docs/results/tables.md and docs/results/figures/
    python3 -B scripts/report.py --no-figures

Nothing here simulates; it only reads what the run scripts wrote.  Every number quoted in
README.md and docs/results.md comes from the tables written here (scripts/check_doc_numbers.py
checks that).  Position errors are given as a share of the arm's reach (the radius of its
workspace) and in millimetres.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_se3.chains import MODEL_NAMES, get_chain, workspace  # noqa: E402
from ring_se3 import kinematics  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
#: "similar", fixed before the runs: share of the reach, percent
PASS_MEDIAN, PASS_P90 = 2.0, 4.0
ORIENTATION_MEDIAN, ORIENTATION_P90 = 5.0, 10.0
HOLD_MARK, RECOVER_MARK_S, DELAY_MARK_S = 2.0, 0.5, 0.5
INK, GREY, LIGHT = "#1c2330", "#8a94a3", "#d9dee6"
SHEETS, CELLS, FLOOR, ERROR, GREEN = "#2a78d6", "#1baf7a", "#8a94a3", "#eb6834", "#1baf7a"
VARIANT_LABELS = {
    "default": "as built", "sheets05": "5% of the sheets' neurons off", "sheets10": "10% of the sheets' neurons off",
    "sheets20": "20% of the sheets' neurons off", "products05": "5% of the SO(3) product neurons off",
    "noise050": "noise current 50 pA", "noise150": "noise current 150 pA",
    "sheet32": "sheets of 32 x 32", "sheet64": "sheets of 64 x 64", "seed2": "other neurons (seed 2)", "seed3": "other neurons (seed 3)",
    "no_velocity": "no velocity path (anchor only)",
}
TRACKING_LABELS = {
    "default": "as built (anchor and path integration)", "anchor_only": "anchor only (velocity cells silent)",
    "anchor_lead": "anchor given the position 0.2 s ahead (position + velocity x 0.2 s)",
    "joint_lead": "joint angles written ahead (the SO(3) option)",
}
DROPOUT_LABELS = {"rest": ("arm at rest", 0.0, True), "move010": ("arm moving, 0.1 rad/s", 0.1, True), "move010_no_pi": ("arm moving, 0.1 rad/s", 0.1, False),
                  "move025": ("arm moving, 0.25 rad/s", 0.25, True), "move025_no_pi": ("arm moving, 0.25 rad/s", 0.25, False),
                  "move050": ("arm moving, 0.5 rad/s", 0.5, True), "move050_no_pi": ("arm moving, 0.5 rad/s", 0.5, False)}


# -- reading -------------------------------------------------------------------
def read(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def load(data):
    out = {"static": {}, "tracking": {}, "tracking_variants": {}, "dropout": {}, "robust": {}}
    for path in sorted(glob.glob(os.path.join(data, "static", "*.json"))):
        out["static"][os.path.basename(path)[:-5]] = read(path)
    for path in sorted(glob.glob(os.path.join(data, "tracking", "*.json"))):
        model, speed = os.path.basename(path)[:-5].rsplit("_", 1)
        out["tracking"][(model, float(speed))] = read(path)
    out["trail"] = {}
    for group in ("tracking_variants", "dropout", "robust", "trail"):
        for path in sorted(glob.glob(os.path.join(data, group, "*.json"))):
            model, variant = os.path.basename(path)[:-5].split("_", 1)
            out[group][(model, variant)] = read(path)
    for name in ("parts", "speed", "so3_reference", "rate_study"):
        path = os.path.join(data, name + ".json")
        out[name] = read(path) if os.path.exists(path) else None
    if out["parts"]:              # a run that left the sheet has no velocity: stored as null
        for row in out["parts"]["speed"]:
            row["velocity"] = [float("nan") if v is None else v for v in row["velocity"]]
        for row in out["parts"]["hold"] + out["parts"]["anchor"]:
            for key, value in list(row.items()):
                if value is None:
                    row[key] = float("nan")
    return out


def rows_of(document, kind="random"):
    return [r for r in document["rows"] if r.get("kind") == kind]


# -- statistics ----------------------------------------------------------------
def clean(values):
    return np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)


def stats(values):
    a = clean(values)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(a)), float(np.percentile(a, 90)), float(np.max(a))


def median_interval(values, seed=0, draws=2000):
    a = clean(values)
    if a.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.median(a[rng.integers(0, a.size, (draws, a.size))], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def fmt(value, digits=2):
    return "-" if value is None or not np.isfinite(value) else "%.*f" % (digits, value)


def triple(values, digits=2):
    return " / ".join(fmt(v, digits) for v in stats(values))


def percent(rows, key, radius=None):
    """Errors of ``rows`` under ``key`` (m) as percent of the reach."""

    return np.array([np.nan if r[key] is None else 100.0 * r[key] / (radius or r["radius_m"]) for r in rows], dtype=float)


def similar(values):
    median, p90, _ = stats(values)
    return bool(median <= PASS_MEDIAN and p90 <= PASS_P90)


def yes(flag):
    return "yes" if flag else "NO"


# -- tracking --------------------------------------------------------------------
def tracking_summary(document, key="position"):
    """Delay and error of one tracking run, from the position read from the sheets (``key``: ``position``,
    ``position_tick`` or ``cells_position``)."""

    ticks, radius, dt = document["ticks"], document["radius_m"], 0.05
    true = np.array([t["true_position"] for t in ticks], dtype=float)
    decoded = np.array([[np.nan if v is None else v for v in t[key]] for t in ticks], dtype=float)
    moving = np.array([t["moving"] for t in ticks])
    error = 100.0 * np.linalg.norm(decoded - true, axis=1) / radius

    def lagged(lag):
        index = np.flatnonzero(moving)
        index = index[index >= lag]
        return np.nanmedian(np.linalg.norm(decoded[index] - true[index - lag], axis=1)) if index.size else float("nan")

    lags = list(range(0, 21))
    medians = [lagged(lag) for lag in lags]
    best = int(np.nanargmin(medians))
    delay = best * dt
    if 0 < best < len(lags) - 1:                          # refine between ticks with a parabola through the three lowest
        y0, y1, y2 = medians[best - 1], medians[best], medians[best + 1]
        denominator = y0 - 2 * y1 + y2
        if denominator > 1e-12:
            delay = (best + 0.5 * (y0 - y2) / denominator) * dt
    # the arm rests 1 s (20 ticks) at every goal: the reading at the end of that rest is "1 s after the stop"
    stops = [i for i in range(1, len(ticks)) if moving[i - 1] and not moving[i]]
    after = [error[i + 19] for i in stops if i + 19 < len(ticks) and not moving[i:i + 20].any()]
    speed = np.linalg.norm(np.array([t["true_velocity"] for t in ticks], dtype=float), axis=1) / radius
    return {
        "ticks": len(ticks), "delay_s": float(delay), "error_moving": stats(error[moving]), "error_rest": stats(error[~moving]),
        "after_stop": stats(after), "lagged_moving": 100.0 * float(medians[best]) / radius,
        "speed_moving": float(np.median(speed[moving])) if moving.any() else float("nan"),
        "orientation_moving": stats([t["error_deg"] for t, m in zip(ticks, moving) if m]),
    }


def velocity_error(document):
    """Error of the decoded velocity as a share of the true speed, over the moving ticks with a speed worth the name."""

    out = []
    for t in document["ticks"]:
        true, decoded = np.array(t["true_velocity"], dtype=float), np.array([np.nan if v is None else v for v in t["velocity"]], dtype=float)
        if t["moving"] and np.linalg.norm(true) > 0.02 * document["radius_m"]:
            out.append(np.linalg.norm(decoded - true) / np.linalg.norm(true))
    return stats([100.0 * v for v in out])


# -- dropout ---------------------------------------------------------------------
def dropout_summary(document):
    """Per gap: where the packet and the true end point were at the start and at the end, and how long the
    readout took to come back after the anchor was switched on again."""

    radius = document["radius_m"]
    gaps = {}
    for t in document["ticks"]:
        gaps.setdefault(t["gap"], {"before": [], "gap": [], "after": []})[t["phase"]].append(t)
    # "back": at the level the readout has with the anchor on in this very run (the arm moves in the same way
    # before, during and after a gap): the 90th percentile of the single-tick errors before the gaps
    usual = clean([100.0 * t["tick_error_m"] / radius for t in document["ticks"] if t["phase"] == "before" and t["tick_error_m"] is not None])
    level = float(np.percentile(usual, 90)) if usual.size else PASS_P90
    out = []
    for index, phases in sorted(gaps.items()):
        gap, after = phases["gap"], phases["after"]
        if not gap:
            continue
        def position(t, key="position_tick"):
            return np.array([np.nan if v is None else v for v in t[key]], dtype=float)
        alive = all(sum(t["sheet_spikes"][i] for t in gap[-5:]) > 0 for i in range(3))        # every sheet still has its packet
        start = np.nanmean([position(t) for t in phases["before"][-5:]], axis=0) if phases["before"] else position(gap[0])
        end = np.nanmean([position(t) for t in gap[-5:]], axis=0)
        true_start, true_end = np.array(gap[0]["true_position"]), np.array(gap[-1]["true_position"])
        path = float(np.sum(np.linalg.norm(np.diff([t["true_position"] for t in gap], axis=0), axis=1)))
        errors = [np.nan if t["tick_error_m"] is None else 100.0 * t["tick_error_m"] / radius for t in after]
        recover = float("nan")
        for i in range(len(errors) - 4):
            if all(np.isfinite(e) and e <= level for e in errors[i:i + 5]):
                recover = i * 0.05
                break
        out.append({
            "gap_s": gap[0]["gap_s"], "alive": bool(alive),
            "packet_moved": 100.0 * float(np.linalg.norm(end - start)) / radius,
            "true_moved": 100.0 * float(np.linalg.norm(true_end - true_start)) / radius,
            "path": 100.0 * path / radius,
            "error_end": 100.0 * float(np.linalg.norm(end - true_end)) / radius,
            "error_start": 100.0 * float(np.linalg.norm(start - true_start)) / radius,
            "recover_s": recover, "level": level,
            "curve": [np.nan if t["tick_error_m"] is None else 100.0 * t["tick_error_m"] / radius for t in gap],
        })
    return out


# -- tables ---------------------------------------------------------------------
def table_parts(data):
    parts = data["parts"]
    if not parts:
        return None
    cell = parts["cell_size"]
    lines = ["| Part | Measured on | Result | Pass mark | Met |", "| --- | --- | --- | --- | --- |"]
    hold = parts["hold"]
    drift = np.array([r["drift_cells"] for r in hold], dtype=float)
    drift_pct = 100.0 * drift * cell
    lines.append("| a packet left alone | %d places, 10 s each, no anchor, no velocity | alive in %d of %d; moved %s cells (median / p90 / largest), that is %s%% of the reach | at most %.0f%% of the reach in 10 s | %s |" % (
        len(hold), sum(1 for r in hold if r["alive"]), len(hold), triple(drift), triple(drift_pct), HOLD_MARK, yes(np.nanmax(drift_pct) <= HOLD_MARK)))
    anchor = parts["anchor"]
    errors = np.array([r["error_cells"] for r in anchor], dtype=float)
    lines.append("| a sheet anchored through gate populations | %d random places in the workspace, held 1 s each | packet off by %s cells (median / p90 / largest), that is %s%% of the reach; from one 50 ms tick alone %s cells in the median | within 1 cell | %s |" % (
        len(anchor), triple(errors), triple(100.0 * errors * cell), fmt(np.nanmedian([r["tick_error_cells"] for r in anchor])), yes(np.nanmax(errors) <= 1.0)))
    speed = [r for r in parts["speed"] if np.all(np.isfinite(r["velocity"]))]
    ratio = np.array([np.dot(r["velocity"], r["wanted"]) / np.dot(r["wanted"], r["wanted"]) for r in speed])
    off = np.array([100.0 * np.linalg.norm(np.subtract(r["velocity"], r["wanted"])) / np.linalg.norm(r["wanted"]) for r in speed])
    lines.append("| a free packet moved by a known velocity | %d runs of 3 s: 8 directions, speeds from 15%% to 90%% of the top speed, two starts each (%d more left the sheet or fell silent) | speed along the asked direction over the asked speed: median %s, 10th to 90th percentile %s to %s; velocity off by %s%% of the asked speed (median / p90 / largest) | proportional to within 10%% | %s |" % (
        len(speed), len(parts["speed"]) - len(speed), fmt(np.median(ratio)), fmt(np.percentile(ratio, 10)), fmt(np.percentile(ratio, 90)), triple(off, 0),
        yes(np.percentile(off, 90) <= 10.0)))
    for name, rows in parts["velocity_cells"].items():
        rel = [100.0 * np.linalg.norm(np.subtract(r["decoded"], r["true"])) / max(np.linalg.norm(r["true"]), 1e-9) for r in rows]
        lines.append("| velocity populations of %s | %d random poses, random joint velocities up to 0.5 rad/s handed in at rest | velocity of the end point off by %s%% of the true speed (median / p90 / largest) | (none set) | |" % (
            name, len(rows), triple(rel, 0)))
    fit = parts["pose_calibration_fit"]
    note = ("One cell is %.3f of the reach. Calibration: a gate population gives its value to %.4f and its square to %.4f (root mean square); "
            "a bias cell fires at %.2f Hz; the packet's top speed is a velocity value of %.2f, that is %.2f of the reach per second.") % (
        cell, fit["gate_rms"], fit["gate_square_rms"], fit["bias_rate_hz"], fit["velocity_value_at_full_speed"],
        fit["velocity_value_at_full_speed"] * parts["cells_per_unit_velocity"] * cell)
    curve = ["| Drive on one side above the resting drive (share of the full drive) | " + " | ".join("%.3f" % v for v in parts["curve_drive"]) + " |",
             "| " + " | ".join(["---"] + ["---:"] * len(parts["curve_drive"])) + " |",
             "| Packet speed, mean of four directions (cells/s) | " + " | ".join("%.2f" % v for v in parts["curve_speed"]) + " |"]
    by = np.array(parts["curve_by_direction"], dtype=float)
    curve.append("| Slowest and fastest of the four directions | " + " | ".join("%.1f - %.1f" % (lo, hi) for lo, hi in zip(by.min(axis=0), by.max(axis=0))) + " |")
    return "\n".join(lines) + "\n\n" + note + "\n\nThe speed curve of the calibration (a sheet on its own):\n\n" + "\n".join(curve)


def table_models(data):
    lines = ["| Model | Joints | Reach (m) | Neurons of the SO(3) part | Products for the velocity | Sum populations | Neurons in all | Synapses |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in MODEL_NAMES:
        d = data["static"].get(name)
        if not d:
            continue
        lines.append("| %s | %d | %.3f | %s | %d | %d | %s | %s |" % (
            name, d["joints"], d["radius_m"], format(d["so3_neurons"], ","), d["pose_products"], d["pose_sums"],
            format(d["neurons"], ","), format(d["synapses"], ",")))
    return "\n".join(lines) if len(lines) > 2 else None


def table_static(data):
    lines = ["| Model | Poses | Position error, median / p90 / max (% of reach) | The same in mm | 95% interval of the median (%) | Within 4% of reach | Similar | Sheets silent |",
             "| --- | ---: | --- | --- | --- | ---: | --- | ---: |"]
    for name in MODEL_NAMES:
        d = data["static"].get(name)
        if not d:
            continue
        rows = rows_of(d)
        pct = percent(rows, "position_error_m")
        mm = [None if r["position_error_m"] is None else 1000.0 * r["position_error_m"] for r in rows]
        lo, hi = median_interval(pct)
        silent = sum(1 for v in pct if not np.isfinite(v))
        lines.append("| %s | %d | %s | %s | %s - %s | %.0f%% | %s | %d |" % (
            name, len(rows), triple(pct), triple(mm, 1), fmt(lo), fmt(hi), 100.0 * np.mean(clean(pct) <= PASS_P90), yes(similar(pct) and silent == 0), silent))
    return "\n".join(lines) if len(lines) > 2 else None


def table_sources(data):
    lines = ["| Model | Position cells alone, median / p90 (% of reach) | Sheets | Exact position of the angles the joint rings hold | Median of (sheets - cells) on the same poses | Sheets closer |",
             "| --- | --- | --- | --- | ---: | ---: |"]
    for name in MODEL_NAMES:
        d = data["static"].get(name)
        if not d:
            continue
        rows = rows_of(d)
        sheets, cells, floor = percent(rows, "position_error_m"), percent(rows, "cells_error_m"), percent(rows, "held_position_error_m")
        ok = np.isfinite(sheets) & np.isfinite(cells)
        difference = sheets[ok] - cells[ok]
        lines.append("| %s | %s / %s | %s / %s | %s / %s | %+.2f | %.0f%% |" % (
            name, fmt(stats(cells)[0]), fmt(stats(cells)[1]), fmt(stats(sheets)[0]), fmt(stats(sheets)[1]),
            fmt(stats(floor)[0]), fmt(stats(floor)[1]), float(np.median(difference)), 100.0 * np.mean(difference < 0)))
    return "\n".join(lines) if len(lines) > 2 else None


def table_agreement(data):
    lines = ["| Model | Difference of the two copies of a coordinate, median / p90 / max (% of reach) | Error of one sheet alone in its plane, median / p90 (% of reach): xy | xz | yz | Spikes of a sheet in the 0.5 s read, median |",
             "| --- | --- | --- | --- | --- | ---: |"]
    for name in MODEL_NAMES:
        d = data["static"].get(name)
        if not d:
            continue
        rows = rows_of(d)
        radius = d["radius_m"]
        difference = [100.0 * abs(v) / radius for r in rows for v in r["disagreement_m"] if v is not None]
        each = []
        for i in range(3):
            values = [100.0 * r["sheet_error_m"][i] / radius for r in rows if r["sheet_error_m"][i] is not None]
            each.append("%s / %s" % (fmt(stats(values)[0]), fmt(stats(values)[1])))
        lines.append("| %s | %s | %s | %s | %s | %.0f |" % (name, triple(difference), each[0], each[1], each[2],
                                                         float(np.median([v for r in rows for v in r["sheet_spikes"]]))))
    return "\n".join(lines) if len(lines) > 2 else None


def table_orientation(data):
    reference = (data["so3_reference"] or {}).get("models", {})
    lines = ["| Model | Orientation error here, median / p90 / max (deg) | In the SO-3 repo, same poses | Within the SO(3) mark (median <= 5, p90 <= 10) |", "| --- | --- | --- | --- |"]
    for name in MODEL_NAMES:
        d = data["static"].get(name)
        if not d:
            continue
        errors = [r["error_deg"] for r in rows_of(d)]
        ref = reference.get(name)
        median, p90, _ = stats(errors)
        lines.append("| %s | %s | %s | %s |" % (name, triple(errors), "%.2f / %.2f / %.2f" % (ref["median"], ref["p90"], ref["max"]) if ref else "-",
                                               yes(median <= ORIENTATION_MEDIAN and p90 <= ORIENTATION_P90)))
    return "\n".join(lines) if len(lines) > 2 else None


def held_p90(data, model):
    d = data["static"].get(model)
    return stats(percent(rows_of(d), "position_error_m"))[1] if d else float("nan")


def table_tracking(data):
    if not data["tracking"]:
        return None, {}
    lines = ["| Model | Joint speed limit (rad/s) | End point's speed while moving, median (reach/s) | Delay (s) | Position error while moving, median / p90 (% of reach) | At rest, median / p90 | 1 s after a stop, median / max | Held poses of the same model, p90 | Delay <= 0.5 s | Back at the held-pose level 1 s after a stop |",
             "| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |"]
    summaries = {}
    for (model, speed), d in sorted(data["tracking"].items()):
        s = tracking_summary(d)
        summaries[(model, speed)] = s
        p90 = held_p90(data, model)
        lines.append("| %s | %.2f | %.2f | %.2f | %s / %s | %s / %s | %s / %s | %s | %s | %s |" % (
            model, speed, s["speed_moving"], s["delay_s"], fmt(s["error_moving"][0]), fmt(s["error_moving"][1]),
            fmt(s["error_rest"][0]), fmt(s["error_rest"][1]), fmt(s["after_stop"][0]), fmt(s["after_stop"][2]), fmt(p90),
            yes(s["delay_s"] <= DELAY_MARK_S), yes(s["after_stop"][0] <= p90) if np.isfinite(p90) else "-"))
    return "\n".join(lines), summaries


def table_tracking_variants(data):
    if not data["tracking_variants"]:
        return None
    lines = ["| Model | Variant | Delay (s) | Position error while moving, median / p90 (% of reach) | The same against the true position one delay earlier, median | At rest, median / p90 | Position cells alone while moving, median | Orientation error while moving, median (deg) |",
             "| --- | --- | ---: | --- | ---: | --- | ---: | ---: |"]
    for model in ("generic3", "tiago7"):
        runs = [("default", data["tracking"].get((model, 0.5)))] + [(v, data["tracking_variants"].get((model, v))) for v in ("anchor_only", "anchor_lead", "joint_lead")]
        for variant, d in runs:
            if not d:
                continue
            s, c = tracking_summary(d), tracking_summary(d, "cells_position")
            lines.append("| %s | %s | %.2f | %s / %s | %s | %s / %s | %s | %s |" % (
                model, TRACKING_LABELS[variant], s["delay_s"], fmt(s["error_moving"][0]), fmt(s["error_moving"][1]), fmt(s["lagged_moving"]),
                fmt(s["error_rest"][0]), fmt(s["error_rest"][1]), fmt(c["error_moving"][0]), fmt(s["orientation_moving"][0])))
    return "\n".join(lines) if len(lines) > 2 else None


def table_velocity(data):
    if not data["tracking"]:
        return None
    lines = ["| Model | Joint speed limit (rad/s) | Velocity read from the velocity populations, off by (% of the true speed), median / p90 |", "| --- | ---: | --- |"]
    for (model, speed), d in sorted(data["tracking"].items()):
        v = velocity_error(d)
        lines.append("| %s | %.2f | %s / %s |" % (model, speed, fmt(v[0], 0), fmt(v[1], 0)))
    return "\n".join(lines)


def dropouts(data):
    return {key: dropout_summary(d) for key, d in data["dropout"].items()}


def table_hold(data, summaries):
    lines = ["| Model | Anchor off for (s) | Gaps | Packets alive at the end | Packet moved, median / max (% of reach) | Within 2% of reach | Level of the readout with the anchor on, p90 of single ticks (% of reach) | Back at that level after (s), median / max | Back within 0.5 s |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |"]
    for (model, name), gaps in sorted(summaries.items()):
        if name != "rest":
            continue
        for gap_s in sorted({g["gap_s"] for g in gaps}):
            rows = [g for g in gaps if g["gap_s"] == gap_s]
            moved = [g["packet_moved"] for g in rows]
            recover = [g["recover_s"] for g in rows]
            lines.append("| %s | %.0f | %d | %d | %s / %s | %d of %d | %s | %s / %s | %s |" % (
                model, gap_s, len(rows), sum(g["alive"] for g in rows), fmt(stats(moved)[0]), fmt(stats(moved)[2]),
                sum(1 for v in moved if v <= HOLD_MARK), len(rows), fmt(rows[0]["level"]), fmt(stats(recover)[0]), fmt(stats(recover)[2]),
                yes(len(clean(recover)) == len(rows) and stats(recover)[2] <= RECOVER_MARK_S)))
    return "\n".join(lines) if len(lines) > 2 else None


def table_dead_reckoning(data, summaries):
    lines = ["| Model | Arm | Anchor off for (s) | Path integration | Gaps | End point moved meanwhile, median (% of reach) | Error at the end of the gap, median / max (% of reach) | Error over how far the end point moved, median | Packets alive | Back at the level with the anchor on after (s), median / max |",
             "| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for (model, name), gaps in sorted(summaries.items(), key=lambda kv: (kv[0][0], DROPOUT_LABELS[kv[0][1]][1], not DROPOUT_LABELS[kv[0][1]][2])):
        label, speed, pi = DROPOUT_LABELS[name]
        if name == "rest":
            continue
        for gap_s in sorted({g["gap_s"] for g in gaps}):
            rows = [g for g in gaps if g["gap_s"] == gap_s]
            error = [g["error_end"] for g in rows]
            share = [g["error_end"] / g["true_moved"] for g in rows if g["true_moved"] > 1.0]
            recover = [g["recover_s"] for g in rows]
            lines.append("| %s | %s | %.0f | %s | %d | %s | %s / %s | %s | %d | %s / %s |" % (
                model, label, gap_s, "on" if pi else "off", len(rows), fmt(np.median([g["true_moved"] for g in rows]), 1),
                fmt(stats(error)[0], 1), fmt(stats(error)[2], 1), fmt(np.median(share)) if share else "-", sum(g["alive"] for g in rows),
                fmt(stats(recover)[0]), fmt(stats(recover)[2])))
    return "\n".join(lines) if len(lines) > 2 else None


def table_trail(data):
    if not data["trail"]:
        return None
    labels = {"none": "nothing (the packets get no input)", "trail": "the velocity the shift register shows, kept for the whole gap",
              "velocity": "the joint velocities (path integration as in section 8)"}
    lines = ["| Model | Handed to the velocity cells during the gap | Gaps | End point moved meanwhile, median (% of reach) | Error at the end of the gap, median / max (% of reach) | Error over how far the end point moved, median |",
             "| --- | --- | ---: | ---: | --- | ---: |"]
    for (model, mode), d in sorted(data["trail"].items(), key=lambda kv: (kv[0][0], ("none", "trail", "velocity").index(kv[0][1]))):
        gaps = dropout_summary(d)
        error = [g["error_end"] for g in gaps]
        share = [g["error_end"] / g["true_moved"] for g in gaps if g["true_moved"] > 1.0]
        lines.append("| %s | %s | %d | %s | %s / %s | %s |" % (model, labels[mode], len(gaps), fmt(np.median([g["true_moved"] for g in gaps]), 1),
                                                          fmt(stats(error)[0], 1), fmt(stats(error)[2], 1), fmt(np.median(share)) if share else "-"))
    return "\n".join(lines)


def table_robust(data):
    if not data["robust"]:
        return None
    lines = ["| Model | Variant | Poses | Position error, median / p90 / max (% of reach) | Median of (variant - as built) on the same poses | Sheets silent | Similar | Orientation error, median (deg) | Neurons |",
             "| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: |"]
    for model in ("generic4", "tiago7"):
        base = data["static"].get(model)
        variants = [("default", base)] + [(v, data["robust"].get((model, v))) for v in VARIANT_LABELS if v != "default"]
        base_rows = {r["index"]: r for r in rows_of(base)} if base else {}
        for variant, d in variants:
            if not d:
                continue
            rows = rows_of(d)[:100]
            pct = percent(rows, "position_error_m")
            difference = [100.0 * (r["position_error_m"] - base_rows[r["index"]]["position_error_m"]) / r["radius_m"] for r in rows
                          if r["index"] in base_rows and r["position_error_m"] is not None and base_rows[r["index"]]["position_error_m"] is not None]
            silent = sum(1 for v in pct if not np.isfinite(v))
            lines.append("| %s | %s | %d | %s | %s | %d | %s | %s | %s |" % (
                model, VARIANT_LABELS[variant], len(rows), triple(pct), "0" if variant == "default" else "%+.2f" % np.median(difference) if difference else "-",
                silent, yes(similar(pct) and silent == 0), fmt(np.median([r["error_deg"] for r in rows])), format(d["neurons"], ",")))
    return "\n".join(lines) if len(lines) > 2 else None


def table_speed(data):
    if not data["speed"]:
        return None
    rows = data["speed"]["rows"]
    threads = sorted({r["threads"] for r in rows})
    lines = ["| Model | Neurons | Of them the SO(3) part | Synapses | " + " | ".join("Simulated time per wall-clock time, %d threads" % t for t in threads) + " |",
             "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in threads) + " |"]
    for name in MODEL_NAMES:
        mine = {r["threads"]: r for r in rows if r["model"] == name}
        if not mine:
            continue
        first = mine[threads[0]]
        lines.append("| %s | %s | %s | %.1f million | %s |" % (name, format(first["neurons"], ","), format(first["so3_neurons"], ","),
                                                             first["synapses"] / 1e6, " | ".join("%.2f" % mine[t]["speed"] if t in mine else "-" for t in threads)))
    return "\n".join(lines) + "\n\nMachine: %s, %d hardware threads; nothing else was running." % (data["speed"]["cpu"], data["speed"]["cores"])


def table_rate(data):
    if not data["rate_study"]:
        return None
    lines = ["| Model | Position cells in the copy without spikes, median / p90 (% of reach) | Position cells in the spiking network |", "| --- | --- | --- |"]
    for row in data["rate_study"]["rows"]:
        d = data["static"].get(row["model"])
        spiking = "-"
        if d:
            cells = percent(rows_of(d), "cells_error_m")
            spiking = "%s / %s" % (fmt(stats(cells)[0]), fmt(stats(cells)[1]))
        lines.append("| %s | %.2f / %.2f | %s |" % (row["model"], row["median_pct"], row["p90_pct"], spiking))
    return "\n".join(lines)


def table_marks(data, tracking, drops):
    lines = ["| Test | Pass mark | Outcome | Met |", "| --- | --- | --- | --- |"]

    def row(test, mark, outcome, met):
        lines.append("| %s | %s | %s | %s |" % (test, mark, outcome, "-" if met is None else yes(met)))

    row("1. Maths", "position equal to 1e-9; velocity to 1e-6", "asserted by `tests/test_kinematics.py`", True)
    parts = data["parts"]
    if parts:
        cell = parts["cell_size"]
        anchor = np.array([r["error_cells"] for r in parts["anchor"]], dtype=float)
        row("3. Parts: anchored packet", "within 1 cell", "largest %.2f cells" % np.nanmax(anchor), np.nanmax(anchor) <= 1.0)
        speed = [r for r in parts["speed"] if np.all(np.isfinite(r["velocity"]))]
        off = np.array([100.0 * np.linalg.norm(np.subtract(r["velocity"], r["wanted"])) / np.linalg.norm(r["wanted"]) for r in speed])
        row("3. Parts: packet speed", "proportional to the velocity, to within 10%", "off by %.0f%% in the median, %.0f%% at the 90th percentile" % (np.median(off), np.percentile(off, 90)),
            np.percentile(off, 90) <= 10.0)
    models = [m for m in MODEL_NAMES if m in data["static"]]
    if models:
        good = [m for m in models if similar(percent(rows_of(data["static"][m]), "position_error_m"))]
        medians = [stats(percent(rows_of(data["static"][m]), "position_error_m"))[0] for m in models]
        p90s = [stats(percent(rows_of(data["static"][m]), "position_error_m"))[1] for m in models]
        row("4. Held poses: position", "median at most 2% of the reach and p90 at most 4%, every model",
            "%d of %d models; medians %.2f%% to %.2f%%, largest p90 %.2f%%" % (len(good), len(models), min(medians), max(medians), max(p90s)), len(good) == len(models))
        fine = [m for m in models if stats([r["error_deg"] for r in rows_of(data["static"][m])])[0] <= ORIENTATION_MEDIAN
                and stats([r["error_deg"] for r in rows_of(data["static"][m])])[1] <= ORIENTATION_P90]
        row("10. Orientation unchanged", "within the SO(3) marks (median at most 5 deg, p90 at most 10 deg)",
            "%d of %d models; largest median %.2f deg" % (len(fine), len(models), max(stats([r["error_deg"] for r in rows_of(data["static"][m])])[0] for m in models)),
            len(fine) == len(models))
    half = {k: s for k, s in tracking.items() if abs(k[1] - 0.5) < 1e-9}
    if half:
        delays = [s["delay_s"] for s in half.values()]
        row("7. Moving arm: delay", "at most 0.5 s at 0.5 rad/s", "%.2f to %.2f s (%d models)" % (min(delays), max(delays), len(delays)), max(delays) <= DELAY_MARK_S)
    if tracking:
        back = [s["after_stop"][0] <= held_p90(data, k[0]) for k, s in tracking.items() if np.isfinite(held_p90(data, k[0]))]
        if back:
            row("7. Moving arm: after a stop", "back at the held-pose level 1 s after a stop", "%d of %d sessions" % (sum(back), len(back)), all(back))
    rest = [g for (model, name), gaps in drops.items() if name == "rest" for g in gaps]
    if rest:
        ten = [g for g in rest if g["gap_s"] == max(x["gap_s"] for x in rest)]
        moved = [g["packet_moved"] for g in ten]
        row("8a. Packet at rest without anchor", "moves at most 2%% of the reach in %.0f s" % ten[0]["gap_s"],
            "median %.2f%%, largest %.2f%% (%d gaps); all alive: %s" % (np.median(moved), np.max(moved), len(ten), "yes" if all(g["alive"] for g in ten) else "no"),
            max(moved) <= HOLD_MARK and all(g["alive"] for g in ten))
    every = [g for gaps in drops.values() for g in gaps]
    if every:
        recover = [g["recover_s"] for g in every]
        late = sum(1 for v in recover if not np.isfinite(v) or v > RECOVER_MARK_S)
        row("8c. Anchor back on", "readout back at its level with the anchor on within 0.5 s", "median %.2f s, largest %s s; later than 0.5 s or not within the 3 s logged: %d of %d gaps" % (
            np.nanmedian(recover), fmt(np.nanmax(recover)), late, len(recover)), late == 0)
    return "\n".join(lines)


# -- figures ---------------------------------------------------------------------
def _style(plt):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
                         "xtick.color": GREY, "ytick.color": GREY, "text.color": INK, "axes.labelcolor": INK,
                         "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.8, "legend.frameon": False})


def figure_error_by_joints(plt, data, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, family, title in zip(axes, ("generic", "tiago"), ("Generic chains", "TIAGo arm, first n joints")):
        joints = [n for n in range(2, 8) if family + str(n) in data["static"]]
        for key, color, label in (("held_position_error_m", FLOOR, "exact position of the angles the rings hold"),
                                  ("cells_error_m", CELLS, "position cells"), ("position_error_m", SHEETS, "read from the sheets")):
            median, p90 = [], []
            for n in joints:
                s = stats(percent(rows_of(data["static"][family + str(n)]), key))
                median.append(s[0]), p90.append(s[1])
            ax.plot(joints, median, "-o", color=color, lw=2, ms=6, label=label + ", median")
            ax.plot(joints, p90, "--", color=color, lw=1.2, label="90th percentile" if key == "position_error_m" else None)
        ax.axhline(PASS_MEDIAN, color=ERROR, lw=1, ls=":")
        ax.text(joints[0] if joints else 2, PASS_MEDIAN - 0.14, "mark for the median: 2%", color=ERROR, fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("joints")
        ax.set_xticks(list(range(2, 8)))
        ax.set_ylim(0, 2.7)
    axes[0].set_ylabel("position error (% of the reach)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.17))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_tracking(plt, data, path, model="tiago7", speed=0.5):
    d = data["tracking"].get((model, speed))
    if not d:
        return False
    ticks = d["ticks"][:800]
    t = np.array([x["t"] for x in ticks]) - ticks[0]["t"]
    true = np.array([x["true_position"] for x in ticks])
    decoded = np.array([[np.nan if v is None else v for v in x["position"]] for x in ticks], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 7.0), sharex=True)
    for a, name in enumerate("xyz"):
        axes[a].plot(t, true[:, a], color=INK, lw=1.6, label="true")
        axes[a].plot(t, decoded[:, a], color=SHEETS, lw=1.4, label="read from the sheets")
        axes[a].set_ylabel(name + " (m)")
    axes[0].legend(ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    axes[3].plot(t, 100.0 * np.linalg.norm(decoded - true, axis=1) / d["radius_m"], color=ERROR, lw=1.2)
    axes[3].set_ylabel("position error\n(% of the reach)")
    axes[3].set_ylim(0, None)
    axes[3].set_xlabel("time (s)")
    fig.suptitle("%s keeps moving from goal to goal at up to %.2f rad/s per joint" % (model, speed), x=0.01, y=0.99, ha="left", fontsize=10.5, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_dropout(plt, data, drops, path):
    models = [m for m in ("generic3", "tiago7") if any(k[0] == m for k in drops)]
    if not models:
        return False
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.6), sharey=True, squeeze=False)
    styles = (("rest", GREY, "-", "arm at rest"), ("move025", SHEETS, "-", "arm moving at 0.25 rad/s, path integration"),
              ("move025_no_pi", ERROR, "--", "arm moving at 0.25 rad/s, no path integration"))
    for ax, model in zip(axes[0], models):
        for name, color, ls, label in styles:
            gaps = [g for g in drops.get((model, name), []) if g["gap_s"] == 5.0]
            if not gaps:
                continue
            curves = np.array([g["curve"][:100] for g in gaps], dtype=float)
            ax.plot(np.arange(curves.shape[1]) * 0.05, np.nanmedian(curves, axis=0), color=color, ls=ls, lw=2, label="%s (%d gaps)" % (label, len(gaps)))
        ax.set_title(model, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("time since the anchor was switched off (s)")
        ax.set_ylim(0, None)
    axes[0][0].set_ylabel("position error, median of the gaps\n(% of the reach)")
    axes[0][-1].legend(fontsize=8.5, loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_speed_curve(plt, data, path):
    parts = data["parts"]
    if not parts:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    drive, by = np.array(parts["curve_drive"]), np.array(parts["curve_by_direction"], dtype=float)
    for row in by:
        axes[0].plot(drive, row, color=LIGHT, lw=1.2)
    axes[0].plot([], [], color=LIGHT, lw=1.2, label="each of the four directions")
    axes[0].plot(drive, parts["curve_speed"], "-o", color=SHEETS, lw=2, ms=5, label="mean")
    axes[0].set_xlabel("drive on one side above the resting drive\n(share of the full drive)")
    axes[0].set_ylabel("packet speed (cells/s)")
    axes[0].set_title("Calibration: a sheet on its own", loc="left", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8.5, loc="upper left")
    speed = [r for r in parts["speed"] if np.all(np.isfinite(r["velocity"]))]
    wanted = np.array([np.linalg.norm(r["wanted"]) for r in speed])
    got = np.array([np.dot(r["velocity"], r["wanted"]) / np.linalg.norm(r["wanted"]) for r in speed])
    top = max(float(wanted.max()), float(got.max())) * 1.05
    axes[1].plot([0, top], [0, top], color=GREY, lw=1, ls=":")
    axes[1].fill_between([0, top], [0, 0.9 * top], [0, 1.1 * top], color=LIGHT, lw=0, alpha=0.6, label="within 10% (the pass mark)")
    straight = np.array([min(abs(v) for v in r["wanted"]) < 1e-9 for r in speed])
    axes[1].plot(wanted[straight], got[straight], "o", color=SHEETS, ms=5, label="along one coordinate")
    axes[1].plot(wanted[~straight], got[~straight], "s", color=CELLS, ms=5, label="diagonal")
    axes[1].set_xlabel("speed asked for (cells/s)")
    axes[1].set_ylabel("packet speed along that direction (cells/s)")
    axes[1].set_title("Test: through the velocity populations and cells", loc="left", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_robust(plt, data, path):
    if not data["robust"]:
        return False
    models = [m for m in ("generic4", "tiago7") if any(k[0] == m for k in data["robust"])]
    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 4.2), sharey=True, squeeze=False)
    order = [v for v in VARIANT_LABELS]
    for ax, model in zip(axes[0], models):
        labels, medians, p90s = [], [], []
        for variant in order:
            d = data["static"].get(model) if variant == "default" else data["robust"].get((model, variant))
            if not d:
                continue
            s = stats(percent(rows_of(d)[:100], "position_error_m"))
            labels.append(VARIANT_LABELS[variant]), medians.append(s[0]), p90s.append(s[1])
        y = np.arange(len(labels))[::-1]
        ax.barh(y, medians, color=SHEETS, height=0.62, label="median")
        ax.plot(p90s, y, "|", color=INK, ms=11, mew=2, label="90th percentile")
        ax.axvline(PASS_MEDIAN, color=ERROR, lw=1, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_title(model, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("position error (% of the reach)")
        ax.grid(axis="y", visible=False)
    axes[0][-1].legend(fontsize=8.5, loc="lower right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_cost(plt, data, path):
    models = [m for m in MODEL_NAMES if m.startswith("tiago") and m in data["static"]]
    if not models:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    so3 = np.array([data["static"][m]["so3_neurons"] for m in models], dtype=float)
    sheets = np.array([3 * data["static"][m]["sheet"]["size"] ** 2 for m in models], dtype=float)
    total = np.array([data["static"][m]["neurons"] for m in models], dtype=float)
    velocity = total - so3 - sheets
    x = np.arange(len(models))
    ax.bar(x, so3, color=GREY, width=0.62, label="SO(3) part: rings, products, link orientations", edgecolor="white", linewidth=1.5)
    ax.bar(x, velocity, bottom=so3, color=CELLS, width=0.62, label="position and velocity populations", edgecolor="white", linewidth=1.5)
    ax.bar(x, sheets, bottom=so3 + velocity, color=SHEETS, width=0.62, label="three sheets", edgecolor="white", linewidth=1.5)
    for i, value in enumerate(total):
        ax.text(i, value + 400, format(int(value), ","), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("tiago", "") for m in models])
    ax.set_xlabel("joints (TIAGo arm)")
    ax.set_ylabel("neurons")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# -- build -----------------------------------------------------------------------
def build(data, out_dir, figures=True):
    os.makedirs(out_dir, exist_ok=True)
    text = ["# Tables of the results", "",
            "Written by `scripts/report.py` from `docs/results/data/`; do not edit by hand. Position error is the distance between the",
            "decoded and the true end point, as a share of the arm's reach (the radius of its workspace, section 1) and in mm.",
            "Median / p90 / max: half of the poses are at or below the median, 90% at or below p90. \"Similar\", fixed before the runs:",
            "median at most 2% of the reach and p90 at most 4%. Reading and figures: [`../results.md`](../results.md).", ""]

    def section(title, body, note=None):
        if body:
            text.extend(["## " + title, ""] + ([note, ""] if note else []) + [body, ""])

    tracking_table, tracking = table_tracking(data)
    drops = dropouts(data)
    section("0. The parts on their own (test 3)", table_parts(data), "Each new part measured in NEST on its own (`scripts/run_parts.py`).")
    section("1. The models", table_models(data))
    section("2. Held poses: position (test 4)", table_static(data),
            "The arm moves to the pose and rests 1 s; the position is the centre of the sheets' spikes of the last 0.5 s.")
    section("3. Where the position is read (test 5)", table_sources(data),
            "The same poses. *Position cells alone*: the three populations that hold the position before the sheets. *Exact position of the "
            "angles the joint rings hold*: what an exact calculation gives from the rings' own activity; no readout of those rings can do better.")
    section("4. Agreement of the sheets (test 6)", table_agreement(data), "Each coordinate is held by two sheets.")
    section("5. Orientation in the larger network (test 10)", table_orientation(data))
    section("6. A moving arm (test 7)", tracking_table,
            "The arm goes from one random goal to the next and rests 1 s at each. The position is read from the sheets' spikes of the last two ticks. "
            "*Delay*: the shift in time for which the readout matches the true position best while the arm moves.")
    section("6b. A moving arm: variants at 0.5 rad/s", table_tracking_variants(data))
    section("6c. The velocity of the end point", table_velocity(data), "Read from the three velocity populations in the sessions of section 6, on ticks with the end point faster than 2% of the reach per second.")
    section("7. Without the anchor, arm at rest (test 8a, 8c)", table_hold(data, drops),
            "The arm rests at a random pose, the anchor is switched off for a while and on again (five times per length). *Packet moved*: the distance between the readout just before the gap and at its end.")
    section("8. Without the anchor, arm moving (test 8b, 8c)", table_dead_reckoning(data, drops),
            "The arm keeps moving from goal to goal while the anchor is off (five gaps per length). With path integration off the packets get nothing and should stay; "
            "*error over how far the end point moved* is then about 1, and below 1 is what path integration gains.")
    section("8b. The shift-register experiment", table_trail(data),
            "The arm moves at up to 0.25 rad/s; for 2 s the network gets neither the anchor nor the joint velocities. A shift register outside the network keeps the "
            "position read from the sheets in the last 10 ticks (0.5 s); the velocity a straight line through them shows is handed to the velocity cells when the gap starts.")
    section("9. Variants and robustness (test 9)", table_robust(data), "The first 100 random poses of the static test.")
    section("10. Size and speed", table_speed(data))
    section("11. Copy without spikes (test 2)", table_rate(data), "The position cells only; the sheets have no copy without spikes.")
    section("12. The pass marks", table_marks(data, tracking, drops), "Set before the tests were run (docs/tests.md). Tests without a mark (2, 5, 6, 8b, 9) are reported in their sections.")
    path = os.path.join(out_dir, "tables.md")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(text))
    written = []
    if figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _style(plt)
        folder = os.path.join(out_dir, "figures")
        os.makedirs(folder, exist_ok=True)
        if data["static"]:
            figure_error_by_joints(plt, data, os.path.join(folder, "error_by_joints.png"))
            written.append("error_by_joints.png")
        for name, made in (("tracking_tiago7.png", lambda p: figure_tracking(plt, data, p)), ("dropout.png", lambda p: figure_dropout(plt, data, drops, p)),
                           ("speed_curve.png", lambda p: figure_speed_curve(plt, data, p)), ("robustness.png", lambda p: figure_robust(plt, data, p)),
                           ("cost.png", lambda p: figure_cost(plt, data, p))):
            if made(os.path.join(folder, name)):
                written.append(name)
    return path, written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=os.path.join(ROOT, "docs", "results", "data"))
    parser.add_argument("--out", default=os.path.join(ROOT, "docs", "results"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    path, written = build(load(args.data), args.out, figures=not args.no_figures)
    with open(path, encoding="utf-8") as stream:
        lines = sum(1 for _ in stream)
    print("written: %s (%d lines), %d figures" % (os.path.relpath(path, ROOT), lines, len(written)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
