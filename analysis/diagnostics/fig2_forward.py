"""Fig. 2: the forward network is accurate at rest and late in motion."""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..common import RAK, read
from reaching_benchmark.geometry.chains import MODEL_NAMES
from .style import DOUBLE, INK, MM, MUTED, colour, panel, recessive_grid, save, setup


def _binned(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> Tuple[List[float], List[float], List[float], List[float]]:
    centres, medians, low, high = [], [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        inside = (x >= a) & (x < b) & np.isfinite(y)
        if inside.sum() >= 5:
            centres.append(float(np.median(x[inside])))
            medians.append(float(np.median(y[inside])))
            low.append(float(np.percentile(y[inside], 25)))
            high.append(float(np.percentile(y[inside], 75)))
    return centres, medians, low, high


def _strip(ax, values: List[np.ndarray], labels: List[str], ylabel: str) -> None:
    rng = np.random.default_rng(0)
    for i, v in enumerate(values):
        v = v[np.isfinite(v)]
        ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=2.0, color=colour("network"), alpha=0.45, linewidths=0)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, linewidth=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    recessive_grid(ax)


def draw() -> None:
    setup()
    forward = read(RAK / "held_pose" / "summary.json")["arms"]
    estimation = read(RAK / "moving_estimate" / "summary.json")["paths"]
    arms = [a for a in MODEL_NAMES if a in forward]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 52 * MM), gridspec_kw={"wspace": 0.5, "width_ratios": [1.25, 1.25, 1, 1]})
    names = [a.replace("generic", "g").replace("tiago", "T") for a in arms]
    _strip(axes[0], [np.array(forward[a]["orientation_error_deg"], dtype=float) for a in arms], names, "orientation error (°)")
    axes[0].set_title("Held pose: orientation", loc="left")
    _strip(axes[1], [100 * np.array(forward[a]["position_error_reach"], dtype=float) for a in arms], names, "position error (% of reach)")
    axes[1].set_title("Held pose: position", loc="left")
    speeds = [p for p in ("speed0.25", "speed0.5", "speed1") if p in estimation]
    lag = float(np.median([estimation[p]["network_lag_s"] for p in speeds]))
    hand = np.concatenate([np.array(estimation[p]["hand_speed_m_s"], dtype=float) for p in speeds]) * 1000.0
    position = np.concatenate([np.array(estimation[p]["position_error_m"], dtype=float) for p in speeds]) * 1000.0
    turn_rate = np.concatenate([np.array(estimation[p]["turn_rate_deg_s"], dtype=float) for p in speeds])
    orientation = np.concatenate([np.array(estimation[p]["sources"]["M0"]["error_deg"], dtype=float) for p in speeds])
    for ax, x, y, xlabel, ylabel, title, unit in (
            (axes[2], hand, position, "hand speed (mm s$^{-1}$)", "position error (mm)", "Moving: position", "mm"),
            (axes[3], turn_rate, orientation, "hand turning rate (° s$^{-1}$)", "orientation error (°)", "Moving: orientation", "°")):
        ax.scatter(x, y, s=1.2, color=colour("network"), alpha=0.18, linewidths=0, rasterized=True)
        bins = np.quantile(x[np.isfinite(x)], np.linspace(0, 1, 11))
        c, m, lo, hi = _binned(x, y, bins)
        ax.errorbar(c, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], color=INK, marker="o", markersize=2.5, linewidth=0.8,
                    elinewidth=0.5, capsize=0)
        grid = np.linspace(0, np.nanpercentile(x, 99), 50)
        ax.plot(grid, grid * lag, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.7)
        ax.text(grid[-1], grid[-1] * lag, "speed × %.2f s" % lag, fontsize=6, color=MUTED, ha="right", va="bottom")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, np.nanpercentile(y, 99) * 1.1)
        recessive_grid(ax)
    for ax, letter in zip(axes, "abcd"):
        panel(ax, letter)
    save(fig, "supp_forward")
