"""Where results live, one folder per model and benchmark::

    results/<model>/<benchmark>/<run>.json      every trial of one run, written after every trial
    results/<model>/<benchmark>/summary.json    computed from the run files of that folder only

``model`` is the package that produced the run (``ring_attractor_kinematics``, ``dls_ik``, ``mpc_nn_dynamics``,
``pets``). A run file is named by what was run (:func:`run_name`); running the same command again resumes it.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
SUMMARY = "summary"


def round_floats(value: Any, digits: int = 5) -> Any:
    """JSON-ready copy with floats rounded (NaN and inf become None)."""

    if isinstance(value, float):
        return round(value, digits) if np.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [round_floats(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return round_floats(value.item(), digits)
    if isinstance(value, np.ndarray):
        return round_floats(value.tolist(), digits)
    return value


def folder(model: str, benchmark: str) -> str:
    return os.path.join(ROOT, model, benchmark)


def run_path(model: str, benchmark: str, run: str) -> str:
    return os.path.join(folder(model, benchmark), run + ".json")


def summary_path(model: str, benchmark: str) -> str:
    return os.path.join(folder(model, benchmark), SUMMARY + ".json")


def write(path: str, document: Dict[str, Any], digits: int = 5, pretty: bool = False) -> str:
    """Write atomically (a stopped run never leaves half a file)."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".part"
    with open(temporary, "w", encoding="utf-8") as stream:
        if pretty:
            json.dump(round_floats(document, digits), stream, indent=1, sort_keys=True)
        else:
            json.dump(round_floats(document, digits), stream, separators=(",", ":"))
    os.replace(temporary, path)
    return path


def write_run(model: str, benchmark: str, run: str, document: Dict[str, Any]) -> str:
    return write(run_path(model, benchmark, run), document)


def read_run(model: str, benchmark: str, run: str) -> Dict[str, Any]:
    """One run file, or an empty dict when it does not exist yet."""

    path = run_path(model, benchmark, run)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def read_runs(model: str, benchmark: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Every run file of a folder (not its summary), in name order, with its path."""

    out = []
    for path in sorted(glob.glob(os.path.join(folder(model, benchmark), "*.json"))):
        if os.path.basename(path) == SUMMARY + ".json":
            continue
        with open(path, encoding="utf-8") as stream:
            out.append((path, json.load(stream)))
    return out


def write_summary(model: str, benchmark: str, document: Dict[str, Any]) -> str:
    return write(summary_path(model, benchmark), document, digits=6, pretty=True)


def read_summary(model: str, benchmark: str) -> Dict[str, Any]:
    with open(summary_path(model, benchmark), encoding="utf-8") as stream:
        return json.load(stream)


def run_name(arm: str, seed: Optional[int], task: Optional[str] = None, kappa: float = 1.0, variant: Optional[str] = None,
             condition: Optional[str] = None, shard: Tuple[int, int] = (0, 1)) -> str:
    """``<arm>[_<task if not pose>][_kappa<κ> if not 1][_seed<s>][_<variant>][_<condition>][_shard<i>of<n> if sharded]``;
    ``seed=None`` for a controller with nothing random in it (DLS IK, CCD IK)."""

    parts = [arm]
    if task and task != "pose":
        parts.append(task)
    if float(kappa) != 1.0:
        parts.append("kappa%g" % float(kappa))
    if seed is not None:
        parts.append("seed%d" % int(seed))
    parts += [p for p in (variant, condition) if p]
    if int(shard[1]) > 1:
        parts.append("shard%dof%d" % (int(shard[0]), int(shard[1])))
    return "_".join(parts)
