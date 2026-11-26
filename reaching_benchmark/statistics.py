"""Paired statistics for the benchmarks: every trial of one method has a twin (same start, goal, seed) in the other.

* success: exact McNemar test on the discordant pairs, and a Wilson 95% interval per method;
* errors and times: Wilcoxon signed-rank test on the paired differences, and a bootstrap 95% interval of the
  median difference.  Failures stay in every denominator: a time that does not exist is counted as the trial
  length (the hand was not within the marks), never dropped.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

BOOTSTRAP = 4000


def wilson(successes: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, Any]:
    """Exact two-sided McNemar test: ``a`` and ``b`` are paired successes."""

    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_a + only_b
    p = float(min(1.0, 2.0 * stats.binom.cdf(min(only_a, only_b), n, 0.5))) if n else 1.0
    return {"only_first": only_a, "only_second": only_b, "p": p}


def bootstrap_median(values: Sequence[float], seed: int = 0) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.median(v[rng.integers(0, v.size, (BOOTSTRAP, v.size))], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def paired(first: Sequence[float], second: Sequence[float], seed: int = 0) -> Dict[str, Any]:
    """Median of ``first − second`` with a bootstrap 95% interval, and the Wilcoxon signed-rank p."""

    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    if d.size == 0:
        return {"n": 0, "median_difference": None, "ci": [None, None], "p": None}
    low, high = bootstrap_median(d, seed)
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) and d.size >= 5 else 1.0
    return {"n": int(d.size), "median_difference": float(np.median(d)), "ci": [low, high], "p": p}


def success_summary(reached: Sequence[bool]) -> Dict[str, Any]:
    r = np.asarray(reached, dtype=bool)
    low, high = wilson(int(r.sum()), int(r.size))
    return {"n": int(r.size), "reached": int(r.sum()), "rate": float(r.mean()) if r.size else float("nan"), "ci": [low, high]}


def censored(times: Sequence[Optional[float]], limit: float) -> np.ndarray:
    """Times with the missing ones (never within the marks) counted as ``limit``."""

    return np.array([limit if t is None or not np.isfinite(t) else float(t) for t in times])
