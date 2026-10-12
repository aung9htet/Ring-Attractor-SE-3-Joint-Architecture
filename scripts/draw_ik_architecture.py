#!/usr/bin/env python3
"""Draw the pose inverse-kinematics layer as it is built (docs/ik/figures/architecture.png), from the wiring of a chain.

    python3 -B scripts/draw_ik_architecture.py                      # tiago4 with a pose goal, the picture of the docs
    python3 -B scripts/draw_ik_architecture.py --chain tiago7 --out docs/ik/figures/architecture_tiago7.png
    python3 -B scripts/draw_ik_architecture.py --chain tiago4 --task position --out /out/ik_position.png

The numbers in the picture (products, sums and neurons of every part) are counted from the
same wiring the NEST network is built from (ring_se3/ik/wiring.py).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_so3.design import Sizes  # noqa: E402
from ring_se3.chains import get_chain  # noqa: E402
from ring_se3.ik.design import POSITION_GOALS, TASKS, PoseIkDesign  # noqa: E402
from ring_se3.ik.wiring import pose_ik_wiring  # noqa: E402
from ring_se3.nest.calibrate import BIAS_CELLS  # noqa: E402
from ring_se3.nest.ik_calibrate import RAMP_NEURONS  # noqa: E402
from ring_se3.sheet import SheetDesign  # noqa: E402

INK, GREY, LIGHT = "#1c2330", "#8a94a3", "#eef1f5"
BLUE, GREEN, ORANGE, PURPLE = "#2a78d6", "#1baf7a", "#eb6834", "#7559c7"


def neurons(chain, wiring, sizes=None, sheet=None):
    """Neurons of the pose network as the inverse kinematics builds it (no velocity path, level cells on its sheets),
    and of everything the inverse-kinematics layer adds."""

    sizes, sheet = sizes or Sizes(), sheet or SheetDesign()
    design, so3, pose = wiring.design, wiring.pose.so3, wiring.pose
    per_sheet = sheet.size ** 2 + sizes.entry                                   # a sheet and its level cells
    forward = chain.n * (sizes.ring + sizes.feature) + len(so3.products) * sizes.product + len(so3.entries) * sizes.entry
    forward += 3 * sizes.output_ring + sizes.norm
    forward += len(pose.muls) * sizes.product + len(pose.relays) * sizes.entry + 3 * sizes.entry + BIAS_CELLS + 3 * per_sheet
    layer = len(wiring.muls) * sizes.product + (len(wiring.relays) + len(wiring.halves)) * sizes.entry
    layer += chain.n * 3 * sizes.ring                                           # per joint: a goal ring and the two sides of its gain pair
    if design.orientation:
        layer += 3 * (sizes.ring + sizes.feature)
    if design.position:
        layer += 6 * sizes.entry                                                # the goal's three numbers: a source and a gate each
        if design.position_goal == "sheets":
            layer += 3 * per_sheet + BIAS_CELLS + 6 * RAMP_NEURONS
    return forward, layer


def box(ax, x, y, w, h, text, color, sub=None, size=9.5):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=LIGHT, ec=color, lw=1.6))
    lines = 0 if not sub else sub.count("\n") + 1
    ax.text(x, y + 0.11 * lines, text, ha="center", va="center", fontsize=size, color=INK, fontweight="bold")
    if sub:
        ax.text(x, y - 0.16, sub, ha="center", va="center", fontsize=7.8, color=GREY, linespacing=1.25)


def ring(ax, x, y, r, color, bump, ghost=None):
    """A ring of dots with a bump at ``bump``; ``ghost``: where another ring's bump is (drawn as a tick outside)."""

    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        near = abs(np.angle(np.exp(1j * (a - bump)))) < 0.6
        ax.plot(x + r * np.cos(a), y + r * np.sin(a), "o", ms=5.3 if near else 3.8, color=color if near else "white", mec=color, mew=1)
    if ghost is not None:
        ax.plot([x + 1.25 * r * np.cos(ghost), x + 1.5 * r * np.cos(ghost)], [y + 1.25 * r * np.sin(ghost), y + 1.5 * r * np.sin(ghost)],
                color=BLUE, lw=2, solid_capstyle="round")


def sheet(ax, x, y, size, color, at, other=None):
    """A sheet with a packet at ``at`` (in units of its side, from its middle); ``other``: where another sheet's packet is."""

    from matplotlib.patches import Circle, Rectangle

    ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, fc="white", ec=color, lw=1.4))
    for i in range(1, 6):
        ax.plot([x - size / 2 + size * i / 6] * 2, [y - size / 2, y + size / 2], color=LIGHT, lw=0.7)
        ax.plot([x - size / 2, x + size / 2], [y - size / 2 + size * i / 6] * 2, color=LIGHT, lw=0.7)
    for radius, alpha in ((0.2, 0.25), (0.13, 0.55), (0.07, 1.0)):
        ax.add_patch(Circle((x + at[0] * size, y + at[1] * size), radius * size, fc=color, ec="none", alpha=alpha))
    if other is not None:
        ax.plot(x + other[0] * size, y + other[1] * size, "+", ms=7, mew=1.6, color=PURPLE)


def arrow(ax, a, b, color=GREY, rad=0.0, style="-|>", lw=1.3, dash=None):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=2, shrinkB=2,
                                                    connectionstyle="arc3,rad=%g" % rad, linestyle=dash or "-"))


def label(ax, x, y, text, color=GREY, size=7.8, **kwargs):
    ax.text(x, y, text, ha=kwargs.pop("ha", "center"), va=kwargs.pop("va", "center"), fontsize=size, color=color, linespacing=1.25, **kwargs)


def counted(products, sums=0, halves=0, sizes=None, none="no product"):
    sizes = sizes or Sizes()
    lines = ["%d products x %d" % (products, sizes.product) if products else none]
    if sums:
        lines.append("%d sums x %d" % (sums, sizes.entry))
    if halves:
        lines.append("%d halves x %d" % (halves, sizes.entry))
    return "\n".join(lines)


def draw(chain, design, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes, sheets = Sizes(), SheetDesign()
    wiring = pose_ik_wiring(chain, design=design)
    forward_neurons, layer_neurons = neurons(chain, wiring, sizes, sheets)
    parts = {(p["part"], p["joint"]): p for p in wiring.parts()}
    n, turning, moving = chain.n, design.orientation, design.position
    on_sheets = moving and design.position_goal == "sheets"
    step, first = 3.9, 10.0                           # width of one joint's column, x of the first joint's share
    row_left, row_share, row_way, bus = 7.0, 5.5, 3.95, 3.02           # what is left; shares; what a joint does to the hand; the line of axes and levers
    y_gain, y_rings, robot_y = 1.95, 0.5, -1.25
    x_min, x_max, y_min, y_max = -0.1, first + step * (n - 1) + 5.0, -2.75, 9.55
    unit = 0.74
    fig, ax = plt.subplots(figsize=(unit * (x_max - x_min), unit * (y_max - y_min)))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    cx = 6.4                                          # the two comparisons

    # -- the orientation goal, top left ------------------------------------------
    if turning:
        for i, name in enumerate(("lift", "pitch", "yaw")):
            x = 0.7 + 0.95 * i
            ring(ax, x, 8.75, 0.27, PURPLE, 0.9 + 1.7 * i)
            label(ax, x, 9.25, name, INK, 8.5, fontweight="bold")
        label(ax, 3.25, 8.75, "the orientation of the goal:\n3 goal rings x %d" % sizes.ring, INK, 8.3, ha="left")
        goal = parts[("goal", -1)]
        box(ax, 1.9, 7.75, 3.5, 0.6, "cos, sin of each goal angle", PURPLE, "3 x %d neurons" % sizes.feature)
        arrow(ax, (1.65, 8.4), (1.65, 8.07))
        box(ax, 1.9, 6.75, 3.5, 0.8, "goal orientation  G", PURPLE, "%d products x %d, %d sums x %d" % (goal["products"], sizes.product, goal["sums"], sizes.entry))
        arrow(ax, (1.9, 7.43), (1.9, 7.17))
        turn = parts[("turn", -1)]
        box(ax, cx, row_left, 3.3, 1.06, "turn  e", ORANGE,
            "%d products x %d, %d sums x %d:\nabout which axis, and how far,\nthe hand still has to turn" % (turn["products"], sizes.product, turn["sums"], sizes.entry))
        arrow(ax, (3.67, 6.8), (cx - 1.67, row_left - 0.05))

    # -- the position goal: three numbers, and as packets on three goal sheets ----
    if moving:
        box(ax, 1.9, 5.45, 3.5, 0.72, "the position of the goal", PURPLE, "x, y, z as numbers: 3 gates x %d" % sizes.entry)
        if on_sheets:
            for i, (name, at) in enumerate((("xy", (0.2, 0.13)), ("xz", (0.2, -0.18)), ("yz", (0.13, -0.18)))):
                x = 0.75 + 1.15 * i
                sheet(ax, x, 4.2, 0.9, PURPLE, at)
                label(ax, x, 3.58, name, INK, 8.0)
                arrow(ax, (1.9, 5.07), (x, 4.68), PURPLE)
            label(ax, 3.0, 4.86, "anchor", PURPLE, ha="left")
            label(ax, 0.18, 3.08, "3 goal sheets, %d x %d neurons each:\nthe goal as a packet on each" % (sheets.size, sheets.size), INK, 8.0, ha="left")

    # -- the pose network, bottom left -------------------------------------------
    net_y = 0.3
    box(ax, 1.9, net_y, 3.5, 1.9, "pose network", GREEN,
        "unchanged, %s neurons:\njoint rings  >  cos, sin  >\norientation of every link  >  R;\nposition cells  >  3 forward sheets" % format(forward_neurons, ","))
    if on_sheets:
        for i, at in enumerate(((-0.12, -0.05), (-0.12, 0.16), (-0.05, 0.16))):
            sheet(ax, 4.3 + 0.68 * i, 2.15, 0.56, GREEN, at)
        label(ax, 4.98, 1.62, "forward sheets:\nthe hand as packets", GREEN, 7.6)
        arrow(ax, (3.67, 0.95), (4.2, 1.82), GREEN)
        label(ax, 0.18, 2.3, "every sheet has %d level cells, which\nhold its packet at one strength" % sizes.entry, GREY, 7.4, ha="left")

    # -- the comparison of positions: the ramp comparator -------------------------
    if moving:
        offset = parts[("offset", -1)]
        text = ("6 ramp readouts x %d, %d sums x %d:\ngoal ramp minus hand ramp,\nwhere the hand still has to go" % (RAMP_NEURONS, offset["sums"], sizes.entry)
                if on_sheets else "%d sums x %d:\ngoal number minus position cell,\nwhere the hand still has to go" % (offset["sums"], sizes.entry))
        box(ax, cx, 4.2, 3.55, 1.06, "offset  d", ORANGE, text)
        if on_sheets:
            arrow(ax, (3.5, 4.2), (cx - 1.8, 4.2), PURPLE)
            label(ax, 4.05, 4.4, "cold", PURPLE, 7.4)
            arrow(ax, (4.98, 2.47), (cx - 0.5, 3.65), GREEN)
            label(ax, 5.2, 3.1, "warm", GREEN, 7.4)
        else:
            arrow(ax, (3.67, 5.3), (cx - 1.5, 4.72), PURPLE)
            arrow(ax, (3.67, 0.8), (cx - 0.6, 3.65), GREEN)
            label(ax, 5.3, 2.1, "position cells", GREEN, 7.6, rotation=50)
    if turning:
        arrow(ax, (3.2, 1.27), (cx - 1.35, row_left - 0.55), GREEN, rad=-0.32)
        label(ax, 4.42, 5.75, "decoded R", GREEN, 7.6)

    # -- axes and levers: one line under the cascade -------------------------------
    last_share = first + step * (n - 1)
    ax.plot([3.67, 8.2, 8.2, last_share + (0.0 if moving else -0.5)], [-0.25, -0.25, bus, bus], color=GREEN, lw=1.3, solid_capstyle="round")
    levers = parts.get(("levers", -1), {"sums": 0})
    label(ax, 8.05, 1.0, "axis of every joint,  a_k:\none column of link k" +
          (";\nits lever to the end point:\n%d sums x %d" % (levers["sums"], sizes.entry) if moving else ""), GREEN, ha="right")

    # -- the cascade, the goal rings of the joints, the robot ---------------------
    previous, previous_half = cx, 1.66
    for k in range(n):
        x = first + step * k
        way, share, rest = parts[("direction", k)], parts[("share", k)], parts.get(("rest", k), {"products": 0, "sums": 0})
        moves = moving and way["sums"] > 0
        if moving:
            box(ax, x, row_way, 3.4, 1.1, "what joint %d does" % (k + 1), GREEN,
                (("turns the hand about a_%d, moves\nit along c%d = a%d x lever%d:\n" % (k + 1, k + 1, k + 1, k + 1) if turning
                  else "moves the end point along\nc%d = a%d x lever%d:\n" % (k + 1, k + 1, k + 1))
                 + "%s, %d sums x %d" % (counted(way["products"], sizes=sizes, none="no product"), way["sums"], sizes.entry)
                 if moves else ("turns the hand about a_%d;\n" % (k + 1) if turning else "") + "moves its end point nowhere"))
            arrow(ax, (x, bus), (x, row_way - 0.56), GREEN)
            arrow(ax, (x - 0.5, row_way + 0.56), (x - 0.5, row_share - 0.46), GREEN)
        else:
            arrow(ax, (x - 0.5, bus), (x - 0.5, row_share - 0.46), GREEN)
        box(ax, x, row_share, 2.5, 0.9, "share  t%d" % (k + 1), ORANGE, counted(share["products"], halves=share["halves"], sizes=sizes, none="axis is fixed: no product"))
        if turning:
            arrow(ax, (previous + 0.6, row_left - 0.5), (x - 0.75, row_share + 0.47))          # what is left so far, into the share
        if moving and k == 0:
            arrow(ax, (cx + 1.8, 4.45), (x - 1.27, row_share - 0.1))
        if design.cascade:
            rx = x + step / 2.0
            box(ax, rx, row_left, 2.3, 0.9, "left  " + ", ".join(v for v in ("e%d" % (k + 2) if turning else "", "d%d" % (k + 2) if moving else "") if v), ORANGE,
                counted(rest["products"], rest["sums"], sizes=sizes))
            if turning or k > 0:
                arrow(ax, (previous + previous_half, row_left), (rx - 1.17, row_left))        # what was left before, itself
            if moving and k == 0:
                arrow(ax, (cx + 1.2, 4.75), (rx - 1.0, row_left - 0.47), rad=-0.12)
            arrow(ax, (x + 0.75, row_share + 0.47), (rx - 0.4, row_left - 0.47))              # minus the share that was taken
            previous, previous_half = rx, 1.15

        # the joint's own rings
        box(ax, x + 0.95, y_gain, 2.3, 0.66, "gain pair %d" % (k + 1), PURPLE, "left and right: 2 x %d" % sizes.ring)
        arrow(ax, (x + 1.0, row_share - 0.46), (x + 1.0, y_gain + 0.35), PURPLE)
        label(ax, x + 1.12, 2.63, "its two halves move\nthe two sides apart", PURPLE, ha="left", size=7.3)
        state, goal_ring = (x + 0.15, y_rings), (x + 1.75, y_rings)
        bump = 0.9 + 1.1 * k
        ring(ax, state[0], state[1], 0.3, BLUE, bump)
        ring(ax, goal_ring[0], goal_ring[1], 0.3, PURPLE, bump + 0.55, ghost=bump)
        label(ax, state[0], y_rings - 0.6, "joint ring %d\n(where it is)" % (k + 1), INK, 7.8)
        label(ax, goal_ring[0], y_rings - 0.6, "goal ring %d\n(where next)" % (k + 1), INK, 7.8)
        arrow(ax, (state[0] + 0.05, y_rings + 0.37), (x + 0.45, y_gain - 0.35), BLUE)
        label(ax, x - 0.02, 1.28, "gates", BLUE, size=7.5, ha="right")
        arrow(ax, (x + 1.45, y_gain - 0.35), (goal_ring[0] - 0.05, y_rings + 0.39), PURPLE)
        label(ax, x + 1.93, 1.28, "writes", PURPLE, size=7.5, ha="left")
        arrow(ax, (goal_ring[0], y_rings - 0.94), (goal_ring[0], robot_y + 0.36), PURPLE)
        arrow(ax, (state[0], robot_y + 0.36), (state[0], y_rings - 0.94), BLUE)

    if design.cascade:
        label(ax, previous + 1.27, row_left, "what no\njoint took", ha="left")
    box(ax, first + step * (n - 1) / 2.0 + 0.95, robot_y, max(step * (n - 1) + 3.9, 10.4), 0.66, "robot: the light robot", INK,
        "every joint moves toward the angle its goal ring holds (purple);  its angle is written on its joint ring (blue)")
    arrow(ax, (first - 0.2, y_rings), (3.67, 0.35), BLUE, dash=(0, (4, 3)))
    label(ax, 5.6, 0.08, "the joint rings are the\npose network's own", BLUE, size=7.6)

    task = {"pose": "a pose goal (orientation and position)", "position": "a position goal", "orientation": "an orientation goal"}[design.task]
    ax.text(0.0, -2.35, "%s, %s%s%s: %s neurons in all, %s of them in this layer (%d products, %d sums, %d share halves; the goal's %s%s; per joint a gain pair and a goal ring).\n"
                        "Purple: goal.  Orange: the two comparisons and the cascade.  Blue: joint rings.  Green: the pose network and what is taken from it."
            % (chain.name, task, ", positions compared as numbers" if moving and not on_sheets else "", "" if design.cascade else ", every joint gets the whole error",
               format(forward_neurons + layer_neurons, ","), format(layer_neurons, ","), len(wiring.muls), len(wiring.relays), len(wiring.halves),
               "rings" if turning else "",
               (", sheets and ramp readouts" if turning else "sheets and ramp readouts") if on_sheets else ("" if turning else "gates")),
            fontsize=8.3, color=GREY, va="center", linespacing=1.4)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return forward_neurons, layer_neurons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chain", default="tiago4")
    parser.add_argument("--task", choices=TASKS, default="pose")
    parser.add_argument("--position-goal", choices=POSITION_GOALS, default=PoseIkDesign().position_goal)
    parser.add_argument("--all-at-once", action="store_true", help="draw the variant without the cascade")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "ik", "figures", "architecture.png"))
    args = parser.parse_args(argv)
    design = PoseIkDesign(task=args.task, position_goal=args.position_goal, cascade=not args.all_at_once)
    forward_neurons, layer_neurons = draw(get_chain(args.chain), design, args.out)
    print("written: %s (%s: pose network %d neurons, IK layer %d, together %d)" % (
        os.path.relpath(args.out), args.chain, forward_neurons, layer_neurons, forward_neurons + layer_neurons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
