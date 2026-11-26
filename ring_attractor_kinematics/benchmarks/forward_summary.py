"""Summary of the held-pose measurements (Fig. 2a–b), and of the same poses under diffuse neuron loss."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from .. import results


def held_pose_entry(document: Dict[str, Any]) -> Dict[str, Any]:
    rows = document["rows"]
    orientation = np.array([r["orientation_error_deg"] for r in rows])
    position = np.array([r["position_error_reach"] for r in rows])
    return {
        "joints": document["joints"], "radius_m": document["radius_m"], "neurons": document["neurons"], "poses": len(rows),
        "orientation_error_deg": orientation.tolist(), "position_error_reach": position.tolist(),
        "orientation_median_deg": float(np.nanmedian(orientation)), "orientation_p90_deg": float(np.nanpercentile(orientation, 90)),
        "position_median_reach": float(np.nanmedian(position)), "position_p90_reach": float(np.nanpercentile(position, 90)),
        "cells_median_reach": float(np.nanmedian([r["cells_error_reach"] for r in rows])),
        "speed": document["simulated_s"] / document["wall_s"]}


def summarise(name: str = "held_pose") -> Dict[str, Any]:
    """``held_pose``: every arm's held poses. ``held_pose_neuron_loss``: the same poses under diffuse neuron loss, each
    beside its intact arm (read from ``held_pose``)."""

    arms = {document["arm"]: held_pose_entry(document) for _, document in results.read_raw_files("held_pose")}
    if name == "held_pose":
        return {"benchmark": "held_pose", "arms": arms}
    loss: Dict[str, Any] = {}
    for path, document in results.read_raw_files("held_pose_neuron_loss"):
        part = os.path.splitext(os.path.basename(path))[0]
        loss[part] = dict(held_pose_entry(document), arm=document["arm"], lesion=document["lesion"])
    return {"benchmark": "held_pose_neuron_loss", "loss": loss, "loss_table": loss_table(arms, loss)}


def loss_table(arms: Dict[str, Any], loss: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One line per (arm, classes, fraction): the medians over lesion seeds of the per-run median errors, beside the
    intact arm's own."""

    keys = sorted({(e["arm"], ",".join(e["lesion"]["classes"]), e["lesion"]["fraction"]) for e in loss.values()})
    table = []
    for arm, classes, fraction in keys:
        runs = [e for e in loss.values() if e["arm"] == arm and ",".join(e["lesion"]["classes"]) == classes
                and e["lesion"]["fraction"] == fraction]
        intact = arms.get(arm, {})
        table.append({"arm": arm, "classes": classes, "fraction": fraction, "lesion_seeds": sorted(e["lesion"]["lesion_seed"] for e in runs),
                      "silenced_median": float(np.median([e["lesion"]["silenced"] for e in runs])),
                      "orientation_median_deg": float(np.median([e["orientation_median_deg"] for e in runs])),
                      "position_median_reach": float(np.median([e["position_median_reach"] for e in runs])),
                      "intact_orientation_median_deg": intact.get("orientation_median_deg"),
                      "intact_position_median_reach": intact.get("position_median_reach")})
    return table
