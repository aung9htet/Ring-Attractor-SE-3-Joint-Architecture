"""The ring-attractor kinematics model's results: ``results/ring_attractor_kinematics/<benchmark>/`` (see
:mod:`reaching_benchmark.results`). The benchmarks of this package call these with a benchmark name and a run name."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from reaching_benchmark import results as shared

MODEL = "ring_attractor_kinematics"


def raw_path(benchmark: str, part: str) -> str:
    return shared.run_path(MODEL, benchmark, part)


def write_raw(benchmark: str, part: str, document: Dict[str, Any]) -> str:
    return shared.write_run(MODEL, benchmark, part, document)


def read_raw_part(benchmark: str, part: str) -> Dict[str, Any]:
    return shared.read_run(MODEL, benchmark, part)


def read_raw_files(benchmark: str) -> List[Tuple[str, Dict[str, Any]]]:
    return shared.read_runs(MODEL, benchmark)


def read_raw(benchmark: str) -> List[Dict[str, Any]]:
    return [document for _, document in shared.read_runs(MODEL, benchmark)]


def write_summary(benchmark: str, document: Dict[str, Any]) -> str:
    return shared.write_summary(MODEL, benchmark, document)


def read_summary(benchmark: str) -> Dict[str, Any]:
    return shared.read_summary(MODEL, benchmark)
