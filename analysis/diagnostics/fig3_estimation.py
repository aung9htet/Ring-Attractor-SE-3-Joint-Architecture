"""Fig. 3: orientation estimates during motion (B1): every source replayed on the spiking forward network's output, and the
observer method's spiking observer on the same paths."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..common import RAK, read
from ring_attractor_kinematics.rules.observer_rule import CODES
from .style import DOUBLE, LABELS, MM, colour, panel, recessive_grid, save, setup

SHOWN = ("M0", "M1", "M2", "MB", "observer")
SPEEDS = (("speed0.25", "0.25", "o"), ("speed0.5", "0.5", "s"), ("speed1", "1", "^"))


def draw() -> None:
    setup()
    paths = read(RAK / "moving_estimate" / "summary.json")["paths"]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 50 * MM), gridspec_kw={"wspace": 0.42, "width_ratios": [1.5, 1.1, 1, 1]})
    ax = axes[0]
    path = paths["speed1"]
    t = np.arange(len(path["sources"]["M0"]["error_deg"])) * 0.1
    window = t <= 12.0
    for code in SHOWN:
        if code not in path["sources"]:
            continue
        ax.plot(t[window], np.array(path["sources"][code]["error_deg"], dtype=float)[window], color=colour(code), linewidth=0.7,
                linestyle=(0, (3, 1.5)) if code == "MB" else "solid", label=LABELS[code])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("orientation error (°)")
    ax.set_title("A fast move (joints up to 1 rad s$^{-1}$)", loc="left")
    ax = axes[1]
    shown = list(CODES) + (["observer"] if "observer" in paths["speed1"]["sources"] else [])
    x = np.arange(len(shown))
    for offset, (name, label, marker) in zip((-0.22, 0.0, 0.22), SPEEDS):
        if name not in paths:
            continue
        values = [paths[name]["sources"][code]["moving_median_deg"] if code in paths[name]["sources"] else np.nan for code in shown]
        ax.scatter(x + offset, values, s=9, marker=marker, color=[colour(c) for c in shown], edgecolors="white", linewidths=0.3, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([c if c != "observer" else "spk" for c in shown])
    ax.set_yscale("log")
    ax.set_ylabel("moving error, median (°)")
    ax.set_title("By source and speed", loc="left")
    proxies = [Line2D([], [], marker=marker, linestyle="none", color="#6b6a66", markersize=3.2, label="%s rad s$^{-1}$" % label)
               for _, label, marker in SPEEDS]
    ax.legend(handles=proxies, loc="center right", handletextpad=0.1)
    ax = axes[2]
    pole = paths["pole"]["sources"]
    bins = pole["M0"]["by_pitch"]
    centres = [(b["from_deg"] + min(b["to_deg"], 90.0)) / 2.0 for b in bins]
    for code in ("M0", "M2", "M5", "MQ", "MB", "observer"):
        if code in pole:
            ax.plot(centres, [b["median_deg"] for b in pole[code]["by_pitch"]], marker="o" if code != "observer" else "^", color=colour(code),
                    linestyle=(0, (3, 1.5)) if code == "MB" else "solid")
    ax.set_yscale("log")
    ax.set_xlabel("|pitch| of the hand (°)")
    ax.set_ylabel("orientation error, median (°)")
    ax.set_title("Toward the chart pole", loc="left")
    ax = axes[3]
    drop = paths["speed0.5"]["sources"]
    for code in ("M0", "M1", "M4", "M5", "MQ"):
        trace = np.array(drop[code]["drop_trace_mean_deg"], dtype=float)
        ax.plot((np.arange(trace.size) - 4) * 0.05, trace, color=colour(code))
    ax.axvspan(0.0, 0.3, color="#ecebe7", zorder=0)
    ax.text(0.15, ax.get_ylim()[1] * 0.97, "no cue", ha="center", va="top", fontsize=6)
    ax.set_xlabel("time from cue loss (s)")
    ax.set_ylabel("orientation error, mean (°)")
    ax.set_title("A 300 ms cue loss", loc="left")
    handles = [Line2D([], [], color=colour(code), label="%s %s" % (code, LABELS[code]), linestyle=(0, (3, 1.5)) if code == "MB" else "solid")
               for code in CODES]
    if "observer" in paths["speed1"]["sources"]:
        handles.append(Line2D([], [], color=colour("observer"), marker="^", label="spk %s" % LABELS["observer"]))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.09), columnspacing=1.0, handlelength=1.4)
    for ax, letter in zip(axes, "abcd"):
        recessive_grid(ax)
        panel(ax, letter)
    save(fig, "supp_observer")
