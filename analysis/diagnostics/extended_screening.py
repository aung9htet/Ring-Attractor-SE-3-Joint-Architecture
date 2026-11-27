"""Extended Data Fig. 2: the Stage-2 screening — every orientation source in the exact closed loop, and the gate."""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ..common import RAK, read
from ring_attractor_kinematics.benchmarks.screening import GATE, NEW_SOURCES, NOMINAL
from ring_attractor_kinematics.rules.observer_rule import CODES
from .style import DOUBLE, LABELS, MM, MUTED, colour, panel, recessive_grid, save, setup


def _table(summary: Dict[str, Any], **fixed: float) -> List[Dict[str, Any]]:
    return [row for row in summary["table"] if all(abs(row[k] - v) < 1e-9 for k, v in fixed.items())]


def draw() -> None:
    setup()
    summary = read(RAK / "screening" / "summary.json")
    gate = summary["gate"]["sources"]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 48 * MM), gridspec_kw={"wspace": 0.55, "width_ratios": [1.15, 1.15, 1, 1]})
    y = np.arange(len(NEW_SOURCES))[::-1]
    for ax, key, threshold, title, letter in ((axes[0], "less_error", GATE["less_error"], "Less moving-orientation error", "a"),
                                             (axes[1], "faster", GATE["faster"], "Faster to stay within marks", "b")):
        low = -100.0 if key == "less_error" else -20.0
        for offset, reference, marker in ((0.13, "M0", "o"), (-0.13, "M1", "s")):
            values = np.array([100 * next(c for c in gate[code]["against"] if c["reference"] == reference)[key] for code in NEW_SOURCES])
            ax.scatter(np.maximum(values, low), y + offset, s=10, marker=marker, color=[colour(c) for c in NEW_SOURCES], zorder=3,
                       edgecolors="white", linewidths=0.4)
            for value, yy in zip(values, y + offset):
                if value < low:
                    ax.annotate("← %.0f%%" % value, (low, yy), xytext=(4, -1.5), textcoords="offset points", fontsize=5, color=MUTED)
            ax.scatter([], [], s=10, marker=marker, color=MUTED, label="vs %s" % LABELS[reference])
        ax.axvline(100 * threshold, color="#1f1f1e", linewidth=0.6, linestyle=(0, (3, 2)))
        ax.axvline(0, color="#a9a7a0", linewidth=0.5)
        ax.text(100 * threshold + 2, -0.55, "gate", fontsize=6, ha="left", va="bottom")
        ax.set_xlim(low - 5, 100 if key == "less_error" else 30)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[c] for c in NEW_SOURCES] if letter == "a" else [])
        ax.set_xlabel("reduction in median %s (%%)" % ("error" if key == "less_error" else "time"))
        ax.set_title(title, loc="left")
        ax.set_ylim(-0.6, len(NEW_SOURCES) - 0.1)
        recessive_grid(ax, "x")
        panel(ax, letter, x=-0.62 if letter == "a" else -0.1)
    axes[0].legend(loc="lower left", handletextpad=0.2)
    delay, noise, kappa = NOMINAL
    for code in CODES:
        rows = sorted(_table(summary, cue_noise_deg=noise, kappa=kappa), key=lambda r: r["delay_s"])
        rows = [r for r in rows if r["source"] == code]
        axes[2].plot([r["delay_s"] for r in rows], [r["moving_error_median_deg"] for r in rows], marker="o", color=colour(code), label=code)
        rows = sorted([r for r in _table(summary, delay_s=delay, cue_noise_deg=noise) if r["source"] == code], key=lambda r: r["kappa"])
        axes[3].plot([r["kappa"] for r in rows], [r["time_median_s"] for r in rows], marker="o", color=colour(code), label=code)
    axes[2].set_xlabel("forward-network lag τ (s)")
    axes[2].set_ylabel("moving-orientation error (°)")
    axes[2].set_title("Lag", loc="left")
    axes[3].set_xlabel("loop gain κ (s$^{-1}$)")
    axes[3].set_ylabel("time to stay within marks (s)")
    axes[3].set_title("Gain", loc="left")
    axes[3].legend(ncol=2, loc="upper right", columnspacing=0.8, handlelength=1.2)
    for ax, letter in ((axes[2], "c"), (axes[3], "d")):
        recessive_grid(ax)
        panel(ax, letter)
    save(fig, "supp_screening")
