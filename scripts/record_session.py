#!/usr/bin/env python3
"""Record a session of the simulator without a browser: a video, a figure and the log.

    python3 -B scripts/record_session.py --chain tiago4 --seconds 40 --out docs/results/videos/tiago4_goals
    python3 -B scripts/record_session.py --chain tiago4 --seconds 40 --gap 15 8 --out docs/results/videos/tiago4_gap

The arm keeps moving from one random goal to the next while the spiking network runs.
``--gap START LENGTH`` switches the anchor off START seconds into the session for LENGTH
seconds (the packets are then carried by path integration alone; ``--no-path-integration``
leaves them where they were).  Written: ``<out>.mp4`` (the arm with the true end point and
the one read from the sheets, the three sheets, position and position error over time),
``<out>.png`` (the curves of the whole session) and ``<out>.json`` (every tick).
``--stub`` uses exact kinematics with a little noise instead of the network.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_se3.chains import MODEL_NAMES  # noqa: E402
from ring_se3.sim.server import PoseSimulator, PoseStubNetwork  # noqa: E402
from simulator import network_factory  # noqa: E402

INK, GREY, LIGHT, DECODED, ERROR, SHADE = "#1c2330", "#8a94a3", "#d9dee6", "#2a78d6", "#eb6834", "#f1ead9"


def collect(simulator, seconds, gap, path_integration):
    ticks = []
    simulator.network.set_path_integration(path_integration)
    for i in range(int(round(seconds / simulator.dt_s))):
        now = i * simulator.dt_s
        if gap is not None:
            wanted = not (gap[0] <= now < gap[0] + gap[1])
            if wanted != simulator.network.anchor_on:
                simulator.network.set_anchor(wanted)
        ticks.append(simulator.tick())
    return ticks


def series(ticks, key, a=None):
    return np.array([np.nan if (t[key] is None or (a is not None and t[key][a] is None)) else (t[key] if a is None else t[key][a])
                     for t in ticks], dtype=float)


def shade(ax, gap):
    if gap is not None:
        ax.axvspan(gap[0], gap[0] + gap[1], color=SHADE, lw=0)


def figure(plt, ticks, config, title, path, gap):
    t = np.array([x["t"] for x in ticks]) - ticks[0]["t"]
    fig, axes = plt.subplots(5, 1, figsize=(8.5, 8.6), sharex=True)
    for a, name in enumerate("xyz"):
        shade(axes[a], gap)
        axes[a].plot(t, series(ticks, "pos_true", a), color=INK, lw=1.6, label="true")
        axes[a].plot(t, series(ticks, "pos", a), color=DECODED, lw=1.4, label="read from the sheets")
        axes[a].set_ylabel(name + " (m)")
    shade(axes[3], gap)
    axes[3].plot(t, series(ticks, "pos_error_mm"), color=ERROR, lw=1.3)
    axes[3].set_ylabel("position error (mm)")
    axes[3].set_ylim(0, None)
    shade(axes[4], gap)
    axes[4].plot(t, series(ticks, "error"), color=ERROR, lw=1.3)
    axes[4].set_ylabel("orientation error (deg)")
    axes[4].set_ylim(0, None)
    axes[4].set_xlabel("time (s)" + ("; shaded: anchor off" if gap is not None else ""))
    fig.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=10.5, fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.972), frameon=False)
    fig.subplots_adjust(top=0.92)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def video(plt, animation, ticks, config, title, path, gap, fps=20):
    t = np.array([x["t"] for x in ticks]) - ticks[0]["t"]
    n = config["sheet"]["size"]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    grid = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 0.25, 1.4, 1.4], hspace=0.4, wspace=0.15, left=0.02, right=0.97, top=0.92, bottom=0.08)
    view = fig.add_subplot(grid[:3, :3], projection="3d")
    sheets = [fig.add_subplot(grid[3, k]) for k in range(3)]
    panels = [fig.add_subplot(grid[a, 4:]) for a in range(4)]
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="bold", color=INK)
    points = np.array([x["points"] for x in ticks]).reshape(-1, 3)
    middle = (points.min(axis=0) + points.max(axis=0)) / 2.0
    span = max(float(np.max(points.max(axis=0) - points.min(axis=0))) / 2.0, config["radius"] * 0.4) * 1.15
    view.set_xlim(middle[0] - span, middle[0] + span)
    view.set_ylim(middle[1] - span, middle[1] + span)
    view.set_zlim(middle[2] - span, middle[2] + span)
    view.set_box_aspect((1, 1, 1))
    view.view_init(elev=22, azim=-55)
    for axis in (view.xaxis, view.yaxis, view.zaxis):
        axis.set_ticklabels([])
        axis.pane.set_facecolor("#f6f7f9")
        axis.pane.set_edgecolor(LIGHT)
    arm, = view.plot([], [], [], "-o", color=GREY, lw=4, ms=4, mfc=INK, mec=INK)
    true_point, = view.plot([], [], [], "o", color=INK, ms=7, label="true end point")
    read_point, = view.plot([], [], [], "o", mfc="none", mec=DECODED, mew=2.2, ms=13, label="read from the sheets")
    view.legend(loc="upper left", frameon=False, fontsize=9)
    label = view.text2D(0.02, 0.02, "", transform=view.transAxes, fontsize=12, color=INK)
    images, crosses = [], []
    for k, ax in enumerate(sheets):
        images.append(ax.imshow(np.zeros((n, n)), origin="lower", cmap="Greys", vmin=0, vmax=3, interpolation="nearest"))
        crosses.append(ax.plot([], [], "+", color=ERROR, ms=11, mew=1.8)[0])
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title("sheet " + config["planes"][k], fontsize=9, color=GREY)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(LIGHT)
    growing, cursors = [], []
    for a, name in enumerate("xyz"):
        ax = panels[a]
        shade(ax, gap)
        ax.plot(t, series(ticks, "pos_true", a), color=LIGHT, lw=1.2)
        growing.append((ax.plot([], [], color=INK, lw=1.6)[0], series(ticks, "pos_true", a), ax.plot([], [], color=DECODED, lw=1.5)[0], series(ticks, "pos", a)))
        ax.set_ylim(config["centre"][a] - 1.05 * config["radius"], config["centre"][a] + 1.05 * config["radius"])
        ax.set_ylabel(name + " (m)", fontsize=9)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelbottom=False)
        cursors.append(ax.axvline(0.0, color=GREY, lw=0.8))
    panels[0].plot([], [], color=INK, label="true")
    panels[0].plot([], [], color=DECODED, label="read from the sheets")
    panels[0].legend(loc="lower right", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(1.0, 1.0))
    errors = series(ticks, "pos_error_mm")
    shade(panels[3], gap)
    panels[3].set_xlim(t[0], t[-1])
    panels[3].set_ylim(0, max(30.0, float(np.nanmax(errors)) * 1.05))
    panels[3].set_ylabel("error (mm)", fontsize=9)
    panels[3].set_xlabel("time (s)" + ("; shaded: anchor off" if gap is not None else ""), fontsize=9)
    error_line, = panels[3].plot([], [], color=ERROR, lw=1.5)
    cursors.append(panels[3].axvline(0.0, color=GREY, lw=0.8))
    for ax in panels:
        ax.grid(True, color=LIGHT, lw=0.8)

    def draw(i):
        x = ticks[i]
        p = np.array(x["points"])
        arm.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
        true_point.set_data_3d([x["pos_true"][0]], [x["pos_true"][1]], [x["pos_true"][2]])
        if all(v is not None for v in x["pos"]):
            read_point.set_data_3d([x["pos"][0]], [x["pos"][1]], [x["pos"][2]])
        for k in range(3):
            image = np.zeros((n, n))
            for index, count in x["sheets"][k]:
                image[index % n, index // n] = count              # first coordinate across, second up
            images[k].set_data(image)
            crosses[k].set_data([x["sheet_true"][k][0]], [x["sheet_true"][k][1]])
        for true_line, true_values, read_line, read_values in growing:
            true_line.set_data(t[: i + 1], true_values[: i + 1])
            read_line.set_data(t[: i + 1], read_values[: i + 1])
        error_line.set_data(t[: i + 1], errors[: i + 1])
        for cursor in cursors:
            cursor.set_xdata([t[i], t[i]])
        text = "silent" if x["pos_error_mm"] is None else "position error %.0f mm" % x["pos_error_mm"]
        label.set_text(text + ("   anchor off" if not x["anchor"] else "") + ("   (moving)" if x["moving"] else "   (at rest)"))
        return []

    writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "24"])
    with writer.saving(fig, path, dpi=100):
        for i in range(len(ticks)):
            draw(i)
            writer.grab_frame()
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chain", default="tiago4", choices=MODEL_NAMES)
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--speed", type=float, default=0.25, help="joint speed limit, rad/s")
    parser.add_argument("--seed", type=int, default=3, help="seed of the random goals")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gap", type=float, nargs=2, default=None, metavar=("START", "LENGTH"), help="anchor off for LENGTH s from START s")
    parser.add_argument("--no-path-integration", action="store_true")
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--out", required=True, help="path without extension")
    args = parser.parse_args(argv)

    make_network = (lambda chain: PoseStubNetwork(chain)) if args.stub else network_factory(args.threads)
    simulator = PoseSimulator(make_network, args.chain, speed=args.speed, mode="goals", seed=args.seed, realtime=False)
    for _ in range(20):                                              # let the rings and the packets form
        simulator.network.step(simulator.robot.q)
    ticks = collect(simulator, args.seconds, args.gap, not args.no_path_integration)
    config = simulator.config()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    keep = ("t", "q", "moving", "error", "pos_true", "pos", "pos_cells", "pos_error_mm", "cells_error_mm", "anchor", "pi", "disagreement_mm")
    with open(args.out + ".json", "w", encoding="utf-8") as stream:
        json.dump({"chain": args.chain, "speed": args.speed, "seed": args.seed, "stub": bool(args.stub), "gap": args.gap,
                   "path_integration": not args.no_path_integration, "neurons": config["neurons"], "radius_m": config["radius"],
                   "ticks": [{k: x[k] for k in keep} for x in ticks]}, stream, separators=(",", ":"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
                         "xtick.color": GREY, "ytick.color": GREY, "text.color": INK, "axes.labelcolor": INK,
                         "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.8})
    title = "%s keeps moving from goal to goal at up to %.2f rad/s%s" % (
        args.chain, args.speed, " (stand-in network)" if args.stub else ", %s spiking neurons" % format(config["neurons"], ","))
    if args.gap is not None:
        title += "; anchor off for %.0f s%s" % (args.gap[1], "" if not args.no_path_integration else ", no path integration")
    figure(plt, ticks, config, title, args.out + ".png", args.gap)
    if not args.no_video:
        video(plt, animation, ticks, config, title, args.out + ".mp4", args.gap)
    errors = series(ticks, "pos_error_mm")
    anchored = np.array([x["anchor"] for x in ticks])
    print("written: %s.{json,png%s} | %d ticks | position error median %.0f mm anchored%s" % (
        args.out, "" if args.no_video else ",mp4", len(ticks), np.nanmedian(errors[anchored]),
        ", %.0f mm at the end of the gap" % errors[~anchored][-1] if (~anchored).any() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
