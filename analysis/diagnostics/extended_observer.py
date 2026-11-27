"""Extended Data Fig. 1: the observer close-up on the pole path — chart selection and what each chart does at the pole."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..common import RAK, read
from .style import LABELS, MM, MUTED, SINGLE, colour, panel, recessive_grid, save, setup


def draw() -> None:
    setup()
    pole = read(RAK / "moving_estimate" / "summary.json")["paths"]["pole"]
    pitch = np.array(pole["pitch_deg"], dtype=float)
    t = np.arange(pitch.size) * 0.1
    chart = np.array(pole["sources"]["M5"]["chart"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 70 * MM), sharex=True, gridspec_kw={"hspace": 0.25, "height_ratios": [1, 1.4]})
    ax = axes[0]
    ax.plot(t, pitch, color="#1f1f1e", linewidth=0.8)
    for i in np.flatnonzero(chart == 1):
        ax.axvspan(t[i] - 0.05, t[i] + 0.05, color="#e3efe3", linewidth=0, zorder=0)
    for limit in (-60, 60):
        ax.axhline(limit, color=MUTED, linewidth=0.5, linestyle=(0, (2, 2)))
    ax.set_ylabel("hand pitch (°)")
    ax.set_yticks([-90, -60, 0, 60, 90])
    ax.set_title("Pitch, and where the two-chart observer uses chart 1 (shaded)", loc="left")
    ax = axes[1]
    for code in ("M0", "M5", "MQ", "MB", "observer"):
        if code in pole["sources"]:
            ax.plot(t, np.array(pole["sources"][code]["error_deg"], dtype=float), color=colour(code), linewidth=0.7, label=LABELS[code],
                    linestyle=(0, (3, 1.5)) if code == "MB" else "solid")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("orientation error (°)")
    ax.legend(loc="upper center", ncol=3, columnspacing=0.8, handlelength=1.2, bbox_to_anchor=(0.5, -0.32), fontsize=5.5)
    for ax, letter in zip(axes, "ab"):
        recessive_grid(ax)
        panel(ax, letter, x=-0.14)
    save(fig, "supp_observer_pole")
