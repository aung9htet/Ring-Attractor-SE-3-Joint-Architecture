#!/usr/bin/env python3
"""Tables and figures of docs/ik/results.md from the data in docs/ik/results/data/.

    python3 -B scripts/ik_report.py                    # writes docs/ik/results/tables.md and docs/ik/results/figures/
    python3 -B scripts/ik_report.py --no-figures

Nothing here simulates; it only reads what the run scripts wrote.  Every number quoted in
docs/ik/results.md and in the inverse-kinematics section of README.md comes from the tables written
here (scripts/check_doc_numbers.py checks that).  Position errors are given as a share of the arm's
reach (the radius of its workspace) and in millimetres.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ring_so3.design import Sizes  # noqa: E402
from ring_se3.chains import get_chain  # noqa: E402
from ring_se3.ik import rules  # noqa: E402
from ring_se3.ik.design import PoseIkDesign  # noqa: E402
from ring_se3.ik.experiments import reach_summary  # noqa: E402
from ring_se3.ik.wiring import pose_ik_wiring  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "docs", "ik", "results", "data")
#: poses, and their seed, over which a share is compared with the share that removes most (section 8)
RELAX_POSES, RELAX_SEED = 4000, 20261010
#: the marks, fixed before the test pairs were run (docs/ik/tests.md)
MARK_REACHED, MARK_OF_EXACT = 0.85, 0.90
MARK_FINAL_DEG, MARK_FINAL_REACH = 3.0, 0.03
MARK_PATH, MARK_TIME_S = 1.5, 10.0
MARK_TURN_DIRECTION_DEG, MARK_OFFSET_REACH, MARK_SHARE_SIGN = 10.0, 0.02, 0.95
#: test 10 (Gazebo), fixed on 2026-10-10 before any run in Gazebo: the end point Gazebo has against the chain's
#: kinematics; the goals the ring IK reaches in Gazebo, of those it reaches on the light robot
MARK_GAZEBO_KINEMATICS_M, MARK_GAZEBO_OF_LIGHT = 0.001, 0.80
#: a joint that ends this far from the target it was sent did not get there (the arm was stopped), rad
STOPPED_RAD = 0.05
#: the four runs that carry the marks: (file, arm, task)
MAIN = (("reach_tiago4_position", "tiago4", "position"), ("reach_generic4_position", "generic4", "position"),
        ("reach_generic7_pose", "generic7", "pose"), ("reach_tiago7_pose", "tiago7", "pose"))
PARTS = (("parts_tiago4_position", "tiago4", "position"), ("parts_generic7_pose", "generic7", "pose"), ("parts_tiago7_pose", "tiago7", "pose"))
VARIANTS = (("reach_tiago4_position", "tiago4, position", "as built: positions compared on the sheets"),
            ("reach_tiago4_position_cells", "tiago4, position", "positions compared as numbers (no sheet in the loop)"),
            ("reach_tiago4_position_all_at_once", "tiago4, position", "every joint gets the whole error (no cascade), gain 0.5"),
            ("reach_tiago7_pose", "tiago7, pose", "as built: positions compared on the sheets"),
            ("reach_tiago7_pose_cells", "tiago7, pose", "positions compared as numbers (no sheet in the loop)"),
            ("reach_tiago7_orientation", "tiago7, orientation alone", "the cascade of the SO-3 repo in this network"))
DISTURBANCES = (("reach_tiago4_position", "nothing in the way"), ("reach_tiago4_position_motor070", "the robot moves 30% less than told"),
                ("reach_tiago4_position_motor130", "the robot moves 30% more than told"),
                ("reach_tiago4_position_push", "one joint pushed 0.3 rad after 4 s"),
                ("reach_tiago4_position_goal_change", "the goal changes after 4 s"),
                ("reach_tiago4_position_anywhere", "goals anywhere in the joint ranges"))
#: test 10: (Gazebo file, the light robot's file with the same pairs, arm, task, label of the Gazebo run)
GAZEBO = (("gazebo_tiago7_pose", "reach_tiago7_pose", "tiago7", "pose", "Gazebo, ring IK"),
          ("gazebo_tiago7_pose_dls", "reach_tiago7_pose", "tiago7", "pose", "Gazebo, damped least squares"),
          ("gazebo_tiago4_position", "reach_tiago4_position", "tiago4", "position", "Gazebo, ring IK"))
GAZEBO_CHECKS = (("gazebo_kinematics_tiago7", "tiago7"), ("gazebo_kinematics_tiago4", "tiago4"))
#: the arms the network was built for (they are not names of the method)
ARMS = {"tiago4": "TIAGo arm, 4 joints", "tiago7": "TIAGo arm, 7 joints", "generic4": "generic arm, 4 joints", "generic7": "generic arm, 7 joints"}
INK, GREY, LIGHT, SURFACE = "#1c2330", "#8a94a3", "#d9dee6", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"            # categorical slots 1 to 3, always in this order


# -- reading -------------------------------------------------------------------
def read(name):
    path = os.path.join(DATA, name + ".json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def read_spikes():
    path = os.path.join(ROOT, "docs", "ik", "spikes", "figures.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def reach(name):
    """A closed-loop run, with the pairs a second start finished (``<name>_rest.json``) put back in."""

    document = read(name)
    if document is None:
        return None
    rest = read(name + "_rest")
    if rest is not None:
        document = dict(document, rows=sorted(document["rows"] + rest["rows"], key=lambda r: r["index"]), skipped=rest.get("skipped", []))
    document["summary"] = reach_summary(document["rows"])
    return document


def load():
    names = {name for group in (MAIN, PARTS) for name, _, _ in group} | {name for name, _, _ in VARIANTS} | {name for name, _ in DISTURBANCES}
    out = {name: (reach(name) if name.startswith("reach_") else read(name)) for name in sorted(names)}
    out["study"], out["sheets"], out["speed"] = read("study"), read("sheets"), read("speed")
    for name, _, _, _, _ in GAZEBO:
        out[name] = reach(name)
    for name, _ in GAZEBO_CHECKS:
        out[name] = read(name)
    return out


# -- formatting ----------------------------------------------------------------
def fmt(value, digits=2, unit=""):
    return "–" if value is None or not np.isfinite(value) else ("%." + str(digits) + "f") % value + unit


def pct(value, digits=0):
    return "–" if value is None or not np.isfinite(value) else ("%." + str(digits) + "f%%") % (100.0 * value)


def reach_pct(value, digits=2):
    """A length in reaches as a percentage of the reach."""

    return pct(value, digits)


def met(flag):
    return "met" if flag else "**missed**"


def has(task, what):
    return task != ("position" if what == "orientation" else "orientation")


# -- tables --------------------------------------------------------------------
def table_marks(data):
    lines = ["## 0. The marks, set before the test pairs were run", "",
             "| Test | Mark | Result | |", "| --- | --- | --- | --- |"]
    parts = [(arm, task, data[name]["summary"]) for name, arm, task in PARTS if data.get(name)]
    turns = [s["turn_direction_error_median_deg"] for _, task, s in parts if has(task, "orientation")]
    offsets = [s["offset_error_median"] for _, task, s in parts if has(task, "position")]
    signs = [s["share_right_sign"] for _, _, s in parts if s["share_right_sign"] is not None]
    if turns:
        lines.append("| 3a | direction of the turn within %.0f° (median) | %.1f° on the worse of %d runs | %s |" % (
            MARK_TURN_DIRECTION_DEG, max(turns), len(turns), met(max(turns) <= MARK_TURN_DIRECTION_DEG)))
    if offsets:
        lines.append("| 3a | the offset within %.0f%% of the reach (median) | %s on the worst of %d runs | %s |" % (
            100 * MARK_OFFSET_REACH, reach_pct(max(offsets)), len(offsets), met(max(offsets) <= MARK_OFFSET_REACH)))
    if signs:
        lines.append("| 3b | right sign in %.0f%% of the shares above 0.05 rad | %s on the worst of %d runs | %s |" % (
            100 * MARK_SHARE_SIGN, pct(min(signs)), len(signs), met(min(signs) >= MARK_SHARE_SIGN)))
    for name, arm, task in MAIN:
        document = data.get(name)
        if not document or not document["summary"]["trials"]:
            lines.append("| 4 | %s, %s: not run | | |" % (arm, task))
            continue
        s = document["summary"]
        lines.append("| 4 | %s, %s: %.0f%% of the goals reached | %s of %d | %s |" % (arm, task, 100 * MARK_REACHED, pct(s["reached"]), s["trials"], met(s["reached"] >= MARK_REACHED)))
        if s["reached_of_exact_reached"] is not None:
            lines.append("| 4 | %s: %.0f%% of the goals the exact rule reaches | %s | %s |" % (arm, 100 * MARK_OF_EXACT, pct(s["reached_of_exact_reached"]), met(s["reached_of_exact_reached"] >= MARK_OF_EXACT)))
        final_reach, final_deg = s["final_error_reached_median_reach"], s["final_error_reached_median_deg"]
        if has(task, "orientation"):
            lines.append("| 4 | %s: final error of reached goals %.0f° and %.0f%% of the reach (medians) | %s°, %s | %s |" % (
                arm, MARK_FINAL_DEG, 100 * MARK_FINAL_REACH, fmt(final_deg), reach_pct(final_reach),
                met(final_deg is not None and final_deg <= MARK_FINAL_DEG and final_reach <= MARK_FINAL_REACH)))
        else:
            lines.append("| 4 | %s: final error of reached goals %.0f%% of the reach (median) | %s | %s |" % (
                arm, 100 * MARK_FINAL_REACH, reach_pct(final_reach), met(final_reach is not None and final_reach <= MARK_FINAL_REACH)))
        lines.append("| 5 | %s: path at most %.1f times the straight line (median) | %s | %s |" % (
            arm, MARK_PATH, fmt(s["path_ratio_median"]), met(s["path_ratio_median"] is not None and s["path_ratio_median"] <= MARK_PATH)))
        lines.append("| 5 | %s: within the marks after at most %.0f s (median) | %s | %s |" % (
            arm, MARK_TIME_S, fmt(s["time_to_reach_median_s"], 1, " s"), met(s["time_to_reach_median_s"] is not None and s["time_to_reach_median_s"] <= MARK_TIME_S)))
    checks = [(arm, data[name]) for name, arm in GAZEBO_CHECKS if data.get(name) and data[name]["largest_position_error_m"] is not None]
    if checks:
        worst = max(d["largest_position_error_m"] for _, d in checks)
        lines.append("| 10a | Gazebo's end point within %.0f mm of the chain's kinematics | %.2f mm at most, %s | %s |" % (
            1000 * MARK_GAZEBO_KINEMATICS_M, 1000 * worst, ", ".join("%s at %d poses" % (arm, len(d["rows"])) for arm, d in checks),
            met(worst <= MARK_GAZEBO_KINEMATICS_M)))
    both = gazebo_of_light(data, *GAZEBO[0][:2])
    if both is not None:
        lines.append("| 10b | tiago7 in Gazebo, ring IK: %.0f%% of the goals it reaches on the light robot | %d of %d (%s) | %s |" % (
            100 * MARK_GAZEBO_OF_LIGHT, both[0], both[1], pct(both[0] / both[1]) if both[1] else "–", met(both[1] > 0 and both[0] / both[1] >= MARK_GAZEBO_OF_LIGHT)))
    return lines


def gazebo_of_light(data, gazebo_name, light_name):
    """``(reached in Gazebo, reached on the light robot)`` over the pairs both ran, counting only goals the light robot reaches."""

    gazebo, light = data.get(gazebo_name), data.get(light_name)
    if not gazebo or not light or not gazebo["rows"]:
        return None
    reached = {r["index"]: bool(r["reached"]) for r in light["rows"]}
    rows = [r for r in gazebo["rows"] if reached.get(r["index"])]
    return sum(bool(r["reached"]) for r in rows), len(rows)


def study_row(row):
    position, turning = has(row["task"], "position"), has(row["task"], "orientation")
    return "| %s | %s | %s | %s | %s | %s | %s | %d (%d) |" % (
        row["model"], row["label"], pct(row["reached_pct"] / 100.0),
        reach_pct(row["final_median_reach"]) if position else "–", fmt(row["final_median_deg"], 2, "°") if turning else "–",
        fmt(row["time_to_reach_s"], 1, " s"), fmt(row["path_over_straight"]) if position else "–", row["missed"], row["missed_at_limit"])


def table_study(data):
    study = data.get("study")
    if not study:
        return ["## 1. The rule itself (exact kinematics, no neurons)", "", "Not run."]
    base, rows = study["base"], study["rows"]
    head = ["| Arm | Rule | Reached | Final position error, median | Final orientation error, median | Within the marks after | Path / straight | Missed (at a joint limit) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines = ["## 1. The rule itself (exact kinematics, no neurons)", "",
             "%d start and goal pairs per arm (seed %d, not the test pairs); a goal is the pose of a joint pose within %.1f rad of the start, joint by joint, "
             "unless a row says otherwise. Delay %.1f s, share noise %.2f rad, decoded pose off by %.0f° and %.1f%% of the reach. Balance %g, loop gain %g per second "
             "(all joints at once and Jacobian transpose: 0.5). tiago4 and generic4 are asked for the position alone, generic7 and tiago7 for the pose. "
             "Reached: within %.0f° and %.0f%% of the reach at the end of %.0f s. The path is that of the whole %.0f s, with the hand's wandering under the "
             "share noise after it has arrived." % (
                 rows[0]["trials"], study["seed"], study["spread"], base["delay_s"], base["noise"], base["bias_deg"], 100 * base["bias_reach"],
                 study["design"]["balance"], study["design"]["kappa"], study["design"]["reach_deg"], 100 * study["design"]["reach_reach"], study["design"]["trial_s"],
                 study["design"]["trial_s"]), ""]
    for group, title in (("rule", "What a share is divided by, the cascade, and two standard methods:"),
                         ("order", "The order of the cascade, and a second sweep:"),
                         ("balance", "The balance (how many radians of turn one reach of offset counts as):"),
                         ("spread", "How far the goal is:"), ("anywhere", "Goals anywhere in the joint ranges:"),
                         ("needs", "What the rule needs (7 joints):")):
        chosen = [r for r in rows if r["group"] == group]
        if chosen:
            lines += [title, ""] + head + [study_row(r) for r in chosen] + [""]
    return lines[:-1]


def table_sheets(data):
    sheets = data.get("sheets")
    lines = ["## 2. The sheets as the comparison needs them: level cells and the ramp readout (test 3c)", ""]
    if not sheets:
        return lines + ["Not run."]
    fit, level = sheets["ramp_calibration_fit"], sheets["level"]
    lines += ["Three sheets on their own, anchored through gate populations at %d random places of the workspace (%d packets; none of the places the readout "
              "was fitted on). Level cells: set point %.0f spikes per second, gain %.0f pA, synapse %.0f ms. A packet counts as out below %.0f spikes per second, "
              "as weak below %.0f." % (len(sheets["places"]), sheets["with"]["packets"], level["set_rate_hz"], level["gain_pa"], level["tau_ms"],
                                       sheets["out_hz"], sheets["weak_hz"]), "",
              "| Sheets | Spikes per second of a packet, median | sd | smallest | largest | Packets out | Packets weak | Packet off its anchor, rms |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, label in (("without", "without level cells (as in the forward network)"), ("with", "with level cells (as in the IK network)")):
        s = sheets[key]
        lines.append("| %s | %.0f | %.0f | %.0f | %.0f | %d of %d | %d of %d | %s |" % (
            label, s["sheet_rate_median_hz"], s["sheet_rate_sd_hz"], s["sheet_rate_smallest_hz"], s["sheet_rate_largest_hz"],
            s["packets_out"], s["packets"], s["packets_weak"], s["packets"], reach_pct(s["anchor_rms"])))
    lines += ["", "The ramp readout (%d neurons per coordinate, fitted on sheets with level cells): the place it gives against the centre of the packets' spikes, "
              "%% of the reach." % sheets["ramp_settings"]["neurons"], "",
              "| Sheets | Coordinates read | rms | median | 90th percentile | largest | mean (bias) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, label in (("without", "without level cells"), ("with", "with level cells")):
        s = sheets[key]
        lines.append("| %s | %d%s | %s | %s | %s | %s | %s |" % (
            label, s["coordinates_read"], " (%d more without a spike)" % s["coordinates_without_spikes"] if s["coordinates_without_spikes"] else "",
            reach_pct(s["ramp_rms"]), reach_pct(s["ramp_median"]), reach_pct(s["ramp_p90"]), reach_pct(s["ramp_largest"]), reach_pct(s["ramp_bias"])))
    if sheets.get("scan"):
        lines += ["", "Other level cells, on the first %d of the places (%d packets). \"From tick to tick\": the standard deviation of a packet's spikes per second "
                  "over the ten 50 ms ticks of a measurement, median over the packets." % (sheets["scan"][0]["packets"] // 3, sheets["scan"][0]["packets"]), "",
                  "| Level cells: gain, synapse from the sheet | Spikes per second of a packet, median | sd over the places | sd from tick to tick | Packets weak | Ramp readout, rms |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        built = sheets["level"]
        for row in sheets["scan"]:
            label = "none" if row["gain_pa"] is None else "%.0f pA, %.0f ms%s" % (
                row["gain_pa"], row["tau_ms"], " (as built)" if (row["gain_pa"], row["tau_ms"]) == (built["gain_pa"], built["tau_ms"]) else "")
            lines.append("| %s | %.0f | %.0f | %.0f | %d of %d | %s |" % (label, row["sheet_rate_median_hz"], row["sheet_rate_sd_hz"], row["sheet_rate_ripple_hz"],
                                                               row["packets_weak"], row["packets"], reach_pct(row["ramp_rms"])))
    follow = (read_spikes() or {}).get("follow")
    if follow:
        lines += ["", "From `docs/ik/spikes/figures.json` (three sheets with level cells, `scripts/ik_spike_census.py`): read from the spikes of one 50 ms tick, a "
                  "coordinate scatters by %s of the reach from tick to tick; read over ten ticks it is %s from the place. While the anchored place travels at "
                  "%.1f reaches per second the packets are %.2f s behind it and the ramp readouts %.2f s." % (
                      reach_pct(follow["readout_sd_from_tick_to_tick"], 1), reach_pct(follow["readouts_off_at_rest"], 1), follow["speed_units_per_s"],
                      follow["packets_behind_s"], follow["readouts_behind_s"])]
    lines += ["", "In the calibration itself (%d measurements on a grid, each also with %s pA added to every sheet neuron): fit %s (rms); on %d measurements between "
              "the grid's places %s (rms), largest %s." % (
                  fit["places"], " and ".join("%+.0f" % v for v in sheets["ramp_settings"]["offsets_pa"] if v), reach_pct(fit["ramp_rms"]),
                  fit["check_places"], reach_pct(fit["check_rms"]), reach_pct(fit["check_largest"]))]
    return lines


def table_parts(data):
    lines = ["## 3. Comparisons and shares in spikes, arm held by hand (tests 3a, 3b)", "",
             "The arm is held at a pose; the goal is part of the way to a pose within 0.8 rad per joint. Marks: direction of the turn within %.0f° (median); the offset "
             "within %.0f%% of the reach (median); right sign in %.0f%% of the shares above 0.05 rad." % (MARK_TURN_DIRECTION_DEG, 100 * MARK_OFFSET_REACH, 100 * MARK_SHARE_SIGN), "",
             "| Arm | Task | Poses | Neurons | Turn: direction error, median | p90 | Offset: error, median (% of reach) | direction error, median | Shares above 0.05 rad | Right sign | Share error, rms | Share where none is asked, rms |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    found = [(arm, task, data[name]) for name, arm, task in PARTS if data.get(name)]
    for arm, task, document in found:
        s = document["summary"]
        lines.append("| %s | %s | %d | %s | %s | %s | %s | %s | %d | %s | %s | %s |" % (
            arm, task, s["poses"], format(document["neurons"], ","), fmt(s["turn_direction_error_median_deg"], 1, "°"), fmt(s["turn_direction_error_p90_deg"], 1, "°"),
            reach_pct(s.get("offset_error_median")), fmt(s["offset_direction_error_median_deg"], 1, "°"), s["shares_large"], pct(s["share_right_sign"]),
            fmt(s["share_rms_rad"], 4, " rad"), fmt(s["share_when_none_asked_rms_rad"], 4, " rad")))
    if not found:
        return lines[:2] + ["Not run."]
    lines += ["", "Where a position is read, in the same runs: error against the true position of the hand (or against the goal that was written), median, % of the reach.", "",
              "| Arm | Position cells | Forward sheets (centre of the spikes) | Ramp readout of the forward sheets | Goal sheets (centre of the spikes) | Ramp readout of the goal sheets | Fewest spikes of a forward sheet per tick |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, task, document in found:
        s = document["summary"]
        if has(task, "position"):
            lines.append("| %s | %s | %s | %s | %s | %s | %.1f |" % (arm, reach_pct(s["cells_error_median"]), reach_pct(s["hand_sheet_error_median"]),
                                                               reach_pct(s["ramp_hand_error_median"]), reach_pct(s["goal_sheet_error_median"]),
                                                               reach_pct(s["ramp_goal_error_median"]), s["sheet_spikes_smallest"]))
    lines += ["", "Share error along the arm (rms, rad): does the cascade get worse from joint to joint?", ""]
    for arm, task, document in found:
        lines.append("- %s: %s" % (arm, ", ".join("joint %d %.3f" % (row["joint"], row["rms_rad"]) for row in document["summary"]["per_joint"])))
    return lines


REACH_HEAD = ["| Run | Trials | Reached | Exact rule reaches | Reached, of those the exact rule reaches | Final position error, median / p90 | In mm, median | Final orientation error, median / p90 | Within the marks after, median | Path / straight, median | Missed (with a joint at a limit) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]


def reach_row(label, document, task):
    s = document["summary"]
    if not s["trials"]:
        return "| %s | 0 | | | | | | | | | |" % label
    position, turning = has(task, "position"), has(task, "orientation")
    note = " (%d pairs not run)" % len(document["skipped"]) if document.get("skipped") else ""
    return "| %s | %d%s | %s | %s | %s | %s | %s | %s | %s | %s | %d (%d) |" % (
        label, s["trials"], note, pct(s["reached"]), pct(s["exact_reached"]), pct(s["reached_of_exact_reached"]),
        "%s / %s" % (reach_pct(s["final_error_median_reach"]), reach_pct(s["final_error_p90_reach"])) if position else "–",
        fmt(1000.0 * document["radius_m"] * s["final_error_median_reach"], 1) if position else "–",
        "%s / %s" % (fmt(s["final_error_median_deg"], 2, "°"), fmt(s["final_error_p90_deg"], 2, "°")) if turning else "–",
        fmt(s["time_to_reach_median_s"], 1, " s"), fmt(s["path_ratio_median"]) if position else "–", s["missed"], s["missed_at_limit"])


def table_reach(data):
    lines = ["## 4. Reaching and trajectory in closed loop (tests 4, 5)", "",
             "Spiking network, light robot, start and goal pairs of the test seed; a goal is the pose of a joint pose within 0.8 rad of the start, joint by joint. "
             "Reached: at the end of the trial the true position is within %.0f%% of the reach of the goal and, where the task has an orientation, the true "
             "orientation within %.0f°." % (100 * PoseIkDesign().reach_reach, PoseIkDesign().reach_deg), ""] + REACH_HEAD
    found = [(name, arm, task, data[name]) for name, arm, task in MAIN if data.get(name)]
    lines += [reach_row("%s, %s" % (arm, task), document, task) for _, arm, task, document in found]
    if not found:
        return lines[:2] + ["Not run."]
    speed = {row["model"]: row for row in (data.get("speed") or {}).get("rows", [])}
    threads = sorted({str(row["threads"]) for row in speed.values()})
    lines += ["", "| Arm | Neurons | Of them the IK layer | Synapses | Simulated time per wall-clock time (%s threads, nothing else on the machine) | Built in | Trial length, median | Start: from the goal, median | Position read from the forward sheets off by, median | Decoded orientation off by, median | Exact rule: final error, median | Exact rule: within the marks after |"
              % ("/".join(threads) or "–"),
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, arm, task, document in found:
        s = document["summary"]
        if not s["trials"]:
            continue
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            arm, format(document["neurons"], ","), format(document["ik_neurons"], ","), format(document["synapses"], ","),
            fmt(speed[arm]["speed"]) if arm in speed else "–", fmt(speed[arm]["build_s"], 1, " s") if arm in speed else "–",
            fmt(s["duration_median_s"], 1, " s"),
            reach_pct(s["start_error_median_reach"], 0) + (", %s" % fmt(s["start_error_median_deg"], 0, "°") if has(task, "orientation") else ""),
            reach_pct(s["decoded_error_median_reach"]), fmt(s["decoded_error_median_deg"], 2, "°"),
            reach_pct(s["exact_final_error_median_reach"]) + (", %s" % fmt(s["exact_final_error_median_deg"], 2, "°") if has(task, "orientation") else ""),
            fmt(s["exact_time_to_reach_median_s"], 1, " s")))
    return lines


def table_variants(data):
    lines = ["## 5. Variants on the same pairs (tests 6, 7)", ""] + ["| Arm, task | Variant " + REACH_HEAD[0][len("| Run "):], "| --- " + REACH_HEAD[1]]
    found = False
    for name, arm, label in VARIANTS:
        document = data.get(name)
        if document:
            found = True
            lines.append("| %s " % arm + reach_row(label, document, document["ik_design"]["task"]))
    paired = []
    for a, b, text in (("reach_tiago4_position", "reach_tiago4_position_cells", "tiago4, position: sheets against numbers"),
                       ("reach_tiago7_pose", "reach_tiago7_pose_cells", "tiago7, pose: sheets against numbers"),
                       ("reach_tiago4_position", "reach_tiago4_position_all_at_once", "tiago4, position: cascade against all joints at once")):
        if data.get(a) and data.get(b):
            first = {r["index"]: r["final_error_reach"] for r in data[a]["rows"]}
            second = {r["index"]: r["final_error_reach"] for r in data[b]["rows"]}
            both = sorted(set(first) & set(second))
            if both:
                closer = np.mean([first[i] < second[i] for i in both])
                paired.append("- %s, the same %d pairs: the first is closer to the goal at the end in %s of them (median difference %s of the reach)." % (
                    text, len(both), pct(closer), reach_pct(float(np.median([first[i] - second[i] for i in both])))))
    return (lines + ([""] + paired if paired else [])) if found else lines[:2] + ["Not run."]


def table_disturbances(data):
    disturbed = [data[name] for name, _ in DISTURBANCES[1:] if data.get(name)]
    pairs = max([len(d["rows"]) + len(d["skipped"]) for d in disturbed], default=0)
    lines = ["## 6. Disturbances (test 8)", "", "tiago4, position goals, the first %d test pairs." % pairs, ""] + ["| Condition " + REACH_HEAD[0][len("| Run "):], REACH_HEAD[1]]
    base = data.get(DISTURBANCES[0][0])
    found = False
    for name, label in DISTURBANCES:
        document = data.get(name)
        if not document:
            continue
        if name == DISTURBANCES[0][0]:                    # the same pairs as the others
            if not disturbed:
                continue
            document = dict(document, rows=[r for r in document["rows"] if r["index"] < pairs], skipped=[])
            document["summary"] = reach_summary(document["rows"])
        found = True
        lines.append(reach_row(label, document, "position"))
    if base and found:
        reference = {r["index"] for r in base["rows"] if r["reached"] and r["index"] < pairs}
        notes = []
        for name, label in DISTURBANCES[1:5]:
            document = data.get(name)
            if document and reference:
                same = [r for r in document["rows"] if r["index"] in reference]
                if same:
                    notes.append("%s: %d of %d" % (label, sum(r["reached"] for r in same), len(same)))
        if notes:
            lines += ["", "Of the goals reached with nothing in the way, reached under the disturbance: " + "; ".join(notes) + "."]
    return lines if found else lines[:2] + ["Not run."]


def table_ends(data):
    lines = ["## 7. How the trials ended (test 9)", "", "| Arm | Ended | Trials | Reached | Exact rule reaches | Final position error, median | Final orientation error, median |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    found = False
    for name, arm, task in MAIN:
        document = data.get(name)
        if not document or not document["rows"]:
            continue
        found = True
        for flag, label in ((True, "with a joint at a limit"), (False, "with no joint at a limit")):
            rows = [r for r in document["rows"] if bool(r["at_limit"]) == flag]
            if rows:
                lines.append("| %s | %s | %d | %s | %s | %s | %s |" % (
                    arm, label, len(rows), pct(np.mean([r["reached"] for r in rows])), pct(np.mean([r["exact_reached"] for r in rows])),
                    reach_pct(float(np.median([r["final_error_reach"] for r in rows]))),
                    fmt(float(np.median([r["final_error_deg"] for r in rows])), 2, "°") if has(task, "orientation") else "–"))
    missed = []
    for name, arm, task in MAIN:
        document = data.get(name)
        for r in (document["rows"] if document else []):
            if not r["reached"]:
                missed.append("- %s, pair %d: ended %s of the reach%s from the goal%s; the exact rule ended %s%s from it." % (
                    arm, r["index"], reach_pct(r["final_error_reach"], 1), ", %s" % fmt(r["final_error_deg"], 1, "°") if has(task, "orientation") else "",
                    ", a joint at a limit" if r["at_limit"] else "", reach_pct(r["exact_final_error_reach"], 1),
                    ", %s" % fmt(r["exact_final_error_deg"], 1, "°") if has(task, "orientation") else ""))
    if found:
        lines += ["", "The goals that were missed:" if missed else "No goal was missed.", ""] + missed
    return lines if found else lines[:2] + ["Not run."]


def table_gazebo(data):
    lines = ["## 9. Gazebo as the robot (test 10)", ""]
    found = [(name, light, arm, task, label, data[name]) for name, light, arm, task, label in GAZEBO if data.get(name) and data[name]["rows"]]
    checks = [(arm, data[name]) for name, arm in GAZEBO_CHECKS if data.get(name)]
    if not found and not checks:
        return lines + ["Not run."]
    lines += ["The TIAGo arm in Gazebo with its own trajectory controller and gravity. Physics advances 50 ms for every tick of the network. A joint is told the "
              "angle its goal ring holds (ring IK) or the angle the method gives (damped least squares, from the exact joint angles). The pairs are the first "
              "test pairs of each arm, the same as on the light robot.", ""]
    for arm, document in checks:
        known = [r for r in document["rows"] if r["position_error_m"] is not None]
        lines.append("- %s held at %d poses: the end point from Gazebo's link states (`%s` in `%s`) and the chain's kinematics at the joint angles Gazebo reports differ by "
                     "%s mm at most, the orientations by %s°." % (arm, len(document["rows"]), document["end_link"], document["base_link"],
                                                                 fmt(1000 * document["largest_position_error_m"] if known else None),
                                                                 fmt(document["largest_orientation_error_deg"] if known else None, 3)))
    if not found:
        return lines
    lines += [""] + ["| Arm, task | Robot, method " + REACH_HEAD[0][len("| Run "):], "| --- " + REACH_HEAD[1]]
    shown = set()
    for name, light_name, arm, task, label, document in found:
        light = data.get(light_name)
        indices = {r["index"] for r in document["rows"]}
        if light and (light_name, tuple(sorted(indices))) not in shown:
            shown.add((light_name, tuple(sorted(indices))))
            same = dict(light, rows=[r for r in light["rows"] if r["index"] in indices], skipped=[])
            same["summary"] = reach_summary(same["rows"])
            lines.append("| %s, %s " % (arm, task) + reach_row("light robot, ring IK (the same pairs)", same, task))
        lines.append("| %s, %s " % (arm, task) + reach_row(label, document, task))
    lines += ["", "| Run | A joint behind its target, median | Trials in which a joint ended more than %.0f° from its target | The base moved by at most | "
              "Reached, of the goals the ring IK reaches on the light robot |" % np.degrees(STOPPED_RAD), "| --- | ---: | ---: | ---: | ---: |"]
    for name, light_name, arm, task, label, document in found:
        rows = document["rows"]
        lags = [r["lag_median_rad"] for r in rows if r.get("lag_median_rad") is not None]
        stopped = [r for r in rows if r.get("lag_final_rad") and max(abs(v) for v in r["lag_final_rad"]) > STOPPED_RAD]
        start = document.get("base_at_start")
        moved = [float(np.hypot(r["base"][0] - start[0], r["base"][1] - start[1])) for r in rows if start and r.get("base")]
        both = gazebo_of_light(data, name, light_name)
        lines.append("| %s, %s: %s | %s | %d of %d | %s | %s |" % (
            arm, task, label, fmt(float(np.degrees(np.median(lags))) if lags else None, 2, "°"), len(stopped), len(rows),
            fmt(1000 * max(moved) if moved else None, 1, " mm"), "%d of %d" % both if both else "–"))
    return lines


def table_sizes(data):
    from draw_ik_architecture import neurons

    sizes = Sizes()
    lines = ["## 8. What the layer is made of", "", "Counted from the wiring the network is built from. A product population has %d neurons, a sum or a share half %d, "
             "a ring %d, a sheet %d, a ramp readout 160." % (sizes.product, sizes.entry, sizes.ring, 48 * 48), "",
             "| Arm, task | Goal orientation | Turn | Offset | Levers | Per joint: direction, share, what is left (products + sums or halves) | Products in all | Sums | Halves | Neurons: pose network | IK layer | Together |",
             "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, task in (("tiago4", "position"), ("tiago4", "pose"), ("generic7", "pose"), ("tiago7", "pose"), ("tiago7", "position")):
        chain = get_chain(arm)
        wiring = pose_ik_wiring(chain, design=PoseIkDesign(task=task))
        parts = {(p["part"], p["joint"]): p for p in wiring.parts()}

        def cell(key):
            p = parts.get(key)
            return "–" if p is None else "%d + %d" % (p["products"], p["sums"] + p["halves"])

        joints = "; ".join("%d: %s" % (k + 1, ", ".join(cell((part, k)) for part in ("direction", "share", "rest"))) for k in range(chain.n))
        forward, layer = neurons(chain, wiring, sizes)
        lines.append("| %s, %s | %s | %s | %s | %s | %s | %d | %d | %d | %s | %s | %s |" % (
            arm, task, cell(("goal", -1)), cell(("turn", -1)), cell(("offset", -1)), cell(("levers", -1)), joints,
            len(wiring.muls), len(wiring.relays), len(wiring.halves), format(forward, ","), format(layer, ","), format(forward + layer, ",")))
    lines += ["", "What every joint's share is divided by (1 + balance² x the mean of |m|² over the joint ranges; balance %g):" % PoseIkDesign().balance, ""]
    for arm in ("tiago4", "generic4", "generic7", "tiago7"):
        wiring = pose_ik_wiring(get_chain(arm), design=PoseIkDesign(task="position"))
        lines.append("- %s: %s; the largest lever over the reach: %s" % (arm, ", ".join("%.2f" % v for v in wiring.norms), ", ".join("%.2f" % v for v in wiring.lever.ratios)))
    lines += ["", "The share against the share that removes most (1 + balance² x |m|², the exact value, over the constant it is divided by), over %d poses "
              "drawn evenly from the joint ranges (seed %d, not that of the mean). 1: the best share; above 2: after this joint more is left than before it." % (
                  RELAX_POSES, RELAX_SEED), "",
              "| Arm | Joint | Smallest | Median | Largest | Poses above 2 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ("tiago4", "generic4", "generic7", "tiago7"):
        chain = get_chain(arm)
        design = PoseIkDesign()
        norms = rules.normalisers(design, rules.levers(chain))
        rng = np.random.default_rng([RELAX_SEED, chain.n])
        squares = np.array([np.sum(rules.moves(chain, chain.random_pose(rng)) ** 2, axis=1) for _ in range(RELAX_POSES)])
        factor = (1.0 + design.move_weight * squares) / norms
        for k in range(chain.n):
            lines.append("| %s | %d | %.2f | %.2f | %.2f | %.1f%% |" % (arm, k + 1, factor[:, k].min(), np.median(factor[:, k]), factor[:, k].max(),
                                                                 100.0 * np.mean(factor[:, k] > 2.0)))
    return lines


# -- figures -------------------------------------------------------------------
def _style(plt):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                         "xtick.color": GREY, "ytick.color": GREY, "text.color": INK, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                         "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def courses(document, key, horizon_s=20.0, step_s=0.2):
    """Every trial's time course on one time base; a trial that ended early keeps its last value (the hand rests)."""

    t = np.arange(step_s, horizon_s + 1e-9, step_s)
    out = []
    for row in document["rows"]:
        values = np.interp(t, row["t"], row[key], right=row[key][-1])
        out.append(values)
    return t, np.array(out)


def figure_error_over_time(plt, data, path):
    found = [(arm, task, data[name]) for name, arm, task in MAIN if data.get(name) and data[name]["rows"]]
    if not found:
        return False
    fig, axes = plt.subplots(2, len(found), figsize=(3.1 * len(found), 5.2), sharex=True, squeeze=False)
    design = PoseIkDesign()
    for column, (arm, task, document) in enumerate(found):
        for row, (key, scale, mark, unit, label) in enumerate((("error_reach", 100.0, 100 * design.reach_reach, "%", "from the goal\n(% of the reach)"),
                                                               ("error_deg", 1.0, design.reach_deg, "°", "turned away from\nthe goal (deg)"))):
            ax = axes[row][column]
            if row == 1 and not has(task, "orientation"):
                ax.axis("off")
                ax.text(0.5, 0.5, "position goal only:\nthe orientation is not asked for", transform=ax.transAxes, ha="center", va="center", color=GREY, fontsize=9)
                continue
            t, values = courses(document, key)
            values = np.maximum(scale * values, 0.05)
            median, low, high = np.median(values, axis=0), np.percentile(values, 25, axis=0), np.percentile(values, 75, axis=0)
            ax.fill_between(t, low, high, color=BLUE, alpha=0.13, lw=0)
            ax.plot(t, median, color=BLUE, lw=2, solid_capstyle="round")
            ax.axhline(mark, color=GREY, lw=1, ls=(0, (4, 3)))
            ax.text(t[-1], mark * 1.12, "mark: %g%s" % (mark, unit), ha="right", va="bottom", fontsize=8.5, color=GREY)
            ax.text(t[-1], median[-1] * 0.82, "%.1f%s" % (median[-1], unit), ha="right", va="top", fontsize=9, color=INK)
            ax.set_yscale("log")
            ax.set_ylim(0.4, 120.0)
            ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 100])
            ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100"])
            ax.minorticks_off()
            if column == 0 or (row == 1 and not has(found[column - 1][1], "orientation")):
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title("Ring IK, %s\n%s goals (%d trials)" % (ARMS.get(arm, arm), task, len(document["rows"])), fontsize=9.5)
        axes[1][column].set_xlabel("time since the goal was given (s)")
        axes[0][column].tick_params(labelbottom=not has(task, "orientation"))
        if not has(task, "orientation"):
            axes[0][column].set_xlabel("time since the goal was given (s)")
    fig.suptitle("The spiking ring IK in closed loop: distance from the goal, median over the trials (line) and the middle half of them (band)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_sheets(plt, data, path):
    sheets = data.get("sheets")
    if not sheets:
        return False
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 2.9))
    rows = (("without", "without level cells", GREY), ("with", "with level cells", BLUE))
    for y, (key, label, color) in enumerate(rows):
        totals = np.array(sheets[key]["totals"]).ravel()
        axes[0].plot(totals, y + rng.uniform(-0.22, 0.22, totals.size), "o", ms=4, color=color, alpha=0.55, mec="none")
        axes[0].plot([np.median(totals)] * 2, [y - 0.34, y + 0.34], color=INK, lw=2, solid_capstyle="round")
        error = 100.0 * (np.array(sheets[key]["read"], dtype=float) - np.array(sheets[key]["centres"], dtype=float)).ravel()
        error = error[np.isfinite(error)]
        shown = np.clip(error, -6.0, 6.0)
        axes[1].plot(shown, y + rng.uniform(-0.22, 0.22, shown.size), "o", ms=4, color=color, alpha=0.55, mec="none")
        beyond = int(np.sum(np.abs(error) > 6.0))
        if beyond:
            axes[1].text(6.0, y + 0.36, "%d beyond ±6%%" % beyond, ha="right", va="bottom", fontsize=8.5, color=INK)
    for ax, title, xlabel in ((axes[0], "How strong a packet is", "spikes per second of one sheet's packet (bar: median)"),
                              (axes[1], "What the ramp readout makes of its place", "place read minus centre of the packets' spikes (% of the reach)")):
        ax.set_yticks([0, 1])
        ax.set_yticklabels([label for _, label, _ in rows])
        ax.tick_params(axis="y", colors=INK, length=0)
        ax.set_ylim(-0.6, 1.7)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(left=0)
    axes[0].axvline(sheets["weak_hz"], color=GREY, lw=1, ls=(0, (4, 3)))
    axes[0].text(sheets["weak_hz"], 1.62, "weak below %.0f" % sheets["weak_hz"], ha="left", va="top", fontsize=8.5, color=GREY)
    axes[1].set_xlim(-6.3, 6.3)
    axes[1].axvline(0.0, color=GREY, lw=1)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_rule(plt, data, path):
    study = data.get("study")
    if not study:
        return False
    rows = [r for r in study["rows"] if r["group"] == "rule"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    labels = list(dict.fromkeys(r["label"] for r in rows))
    fig, axes = plt.subplots(1, len(models), figsize=(3.0 * len(models) + 2.6, 2.9), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        chosen = {r["label"]: r for r in rows if r["model"] == model}
        values = [chosen[label]["reached_pct"] for label in labels]
        y = np.arange(len(labels))[::-1]
        ax.barh(y, values, height=0.55, color=[BLUE if i == 0 else LIGHT for i in range(len(labels))], edgecolor=SURFACE, linewidth=1.5)
        for yi, value in zip(y, values):
            ax.text(value + 1.5, yi, "%.0f%%" % value, va="center", fontsize=8.8, color=INK)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", colors=INK, length=0)
        ax.grid(axis="y", visible=False)
        ax.set_title("%s, %s" % (ARMS.get(model, model), chosen[labels[0]]["task"]), loc="left", fontsize=9.5)
        ax.set_xlabel("goals reached (%)")
    fig.suptitle("The rule with exact kinematics: goals reached, by how every joint's share is worked out (blue: as built)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_example(plt, data, path, name="reach_tiago7_pose"):
    from matplotlib.colors import LinearSegmentedColormap

    document = data.get(name) or data.get("reach_generic7_pose")
    if not document or not document["rows"]:
        return False
    rows = [r for r in document["rows"] if r["reached"] and r["start_error_reach"] > 0.2] or document["rows"]
    row = sorted(rows, key=lambda r: r["start_error_reach"])[len(rows) // 2]            # the trial with the median distance to go
    t, shares = np.array(row["t"]), np.degrees(np.array(row["shares"]))
    design = PoseIkDesign()
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.25]})
    for ax, values, mark, unit, label in ((axes[0], 100.0 * np.array(row["error_reach"]), 100 * design.reach_reach, "%", "from the goal\n(% of the reach)"),
                                          (axes[1], np.array(row["error_deg"]), design.reach_deg, "°", "turned away from\nthe goal (deg)")):
        ax.plot(t, values, color=BLUE, lw=2, solid_capstyle="round")
        ax.axhline(mark, color=GREY, lw=1, ls=(0, (4, 3)))
        ax.text(t[0], mark + 0.03 * values.max(), "mark: %g%s" % (mark, unit), ha="left", va="bottom", fontsize=8.5, color=GREY)
        ax.text(0.995, 0.93, "at the end: %.1f%s" % (values[-1], unit), transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    top = max(float(np.abs(shares).max()), 1.0)
    colours = LinearSegmentedColormap.from_list("share", [ORANGE, "#f1f0ec", BLUE])       # two opposite hues about a neutral middle
    image = axes[2].imshow(shares.T, aspect="auto", cmap=colours, vmin=-top, vmax=top, extent=(t[0], t[-1], shares.shape[1] + 0.5, 0.5), interpolation="nearest")
    axes[2].set_yticks(np.arange(1, shares.shape[1] + 1))
    axes[2].set_ylabel("joint")
    axes[2].set_xlabel("time since the goal was given (s)")
    axes[2].grid(visible=False)
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, anchor=(0.0, 0.0), shrink=0.4)
    bar.set_label("share (deg):  orange back, blue forward", fontsize=8.5)
    bar.outline.set_visible(False)
    axes[0].set_title("One trial of the ring IK on the %s (pair %d): the hand goes %s of the reach%s" % (
        ARMS.get(document["chain"], document["chain"]), row["index"], reach_pct(row["start_error_reach"], 0),
        " and turns %s" % fmt(row["start_error_deg"], 0, "°") if document["ik_design"]["task"] != "position" else ""), loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def build(out_dir, figures=True):
    data = load()
    lines = ["# Inverse kinematics for a pose: tables", "", "Written by `scripts/ik_report.py` from `docs/ik/results/data/`. Do not edit by hand.", "",
             "Everything here is the ring IK, the spiking network of this repo, unless a row names another method. " + "; ".join(
                 "`%s`: %s" % item for item in ARMS.items()) + ". These are the arms the network was built for, not names of the method.", ""]
    for table in (table_marks, table_study, table_sheets, table_parts, table_reach, table_variants, table_disturbances, table_ends, table_sizes,
                  table_gazebo):
        lines += table(data) + [""]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tables.md"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    written = []
    if figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _style(plt)
        folder = os.path.join(out_dir, "figures")
        os.makedirs(folder, exist_ok=True)
        for name, figure in (("error_over_time.png", figure_error_over_time), ("sheets.png", figure_sheets), ("rule.png", figure_rule),
                             ("example_trial.png", figure_example)):
            if figure(plt, data, os.path.join(folder, name)):
                written.append(name)
    return data, written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "docs", "ik", "results"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    data, written = build(args.out, figures=not args.no_figures)
    missing = [name for name, document in data.items() if document is None]
    print("written: %s (%d data files read%s)%s" % (os.path.relpath(os.path.join(args.out, "tables.md")), sum(d is not None for d in data.values()),
                                                   "; not there yet: " + ", ".join(missing) if missing else "",
                                                   "; figures: " + ", ".join(written) if written else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
