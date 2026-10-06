#!/usr/bin/env python3
"""Draw the pose network (docs/figures/architecture.png), with the sizes counted from a chain's wiring.

    python3 -B scripts/draw_architecture.py                    # tiago7, the picture of the docs
    python3 -B scripts/draw_architecture.py --chain generic3 --out /out/generic3.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_so3.design import Sizes  # noqa: E402
from ring_se3.chains import get_chain  # noqa: E402
from ring_se3.nest.calibrate import BIAS_CELLS  # noqa: E402
from ring_se3.sheet import SheetDesign  # noqa: E402
from ring_se3.wiring import pose_wiring  # noqa: E402

INK, GREY, LIGHT, BLUE, GREEN, ORANGE, PURPLE = "#1c2330", "#8a94a3", "#eef1f5", "#2a78d6", "#1baf7a", "#eb6834", "#7a5cc9"


def box(ax, x, y, w, h, text, color, sub=None):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=LIGHT, ec=color, lw=1.6))
    lines = sub.count("\n") + 1 if sub else 0
    ax.text(x, y + (0.1 + 0.08 * lines if sub else 0.0), text, ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")
    if sub:
        ax.text(x, y - 0.14, sub, ha="center", va="center", fontsize=8, color=GREY, linespacing=1.25)


def arrow(ax, a, b, color=GREY, text=None, shift=(0.0, 0.12)):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, shrinkA=2, shrinkB=2))
    if text:
        ax.text((a[0] + b[0]) / 2 + shift[0], (a[1] + b[1]) / 2 + shift[1], text, ha="center", va="center", fontsize=7.8, color=GREY)


def sheet(ax, x, y, size, name, at):
    from matplotlib.patches import Circle, Rectangle

    ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, fc="white", ec=PURPLE, lw=1.6))
    n = 9
    for i in range(1, n):
        ax.plot([x - size / 2 + size * i / n] * 2, [y - size / 2, y + size / 2], color=LIGHT, lw=0.8)
        ax.plot([x - size / 2, x + size / 2], [y - size / 2 + size * i / n] * 2, color=LIGHT, lw=0.8)
    for radius, alpha in ((0.2, 0.25), (0.13, 0.55), (0.07, 1.0)):
        ax.add_patch(Circle((x + at[0] * size, y + at[1] * size), radius * size * 1.6, fc=PURPLE, ec="none", alpha=alpha))
    ax.text(x, y - size / 2 - 0.16, name, ha="center", va="center", fontsize=8.5, color=INK)


def draw(chain, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wiring, sizes, design = pose_wiring(chain), Sizes(), SheetDesign()
    so3 = wiring.so3.summary()
    levers = sum(1 for r in wiring.relays if r.node[0] == "lever")
    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    # left: inputs
    box(ax, 1.3, 5.0, 2.2, 0.9, "joint rings", BLUE, "%d rings x %d neurons\na bump at each joint angle" % (chain.n, sizes.ring))
    box(ax, 1.3, 1.6, 2.2, 0.9, "joint velocities", ORANGE, "%d populations x %d neurons\ngiven from outside" % (chain.n, sizes.entry))
    # SO(3)
    box(ax, 4.2, 5.0, 2.5, 1.1, "orientation of every link", GREEN,
        "the SO(3) network:\n%d products, %d link populations" % (so3["products"], so3["entries"]))
    arrow(ax, (2.42, 5.0), (2.93, 5.0))
    box(ax, 4.5, 3.35, 2.0, 0.9, "position cells x, y, z", GREEN, "3 x %d neurons: link entries\ntimes the link offsets" % sizes.entry)
    arrow(ax, (4.5, 4.43), (4.5, 3.82), text="sums only", shift=(0.5, 0.0))
    box(ax, 4.2, 1.5, 2.6, 1.3, "velocity populations", ORANGE,
        "axis x lever x joint velocity:\n%d products, %d lever populations,\n3 populations for vx, vy, vz" % (len(wiring.muls), levers))
    arrow(ax, (2.42, 1.6), (2.88, 1.6))
    arrow(ax, (3.2, 4.43), (3.2, 2.17), text="joint axes", shift=(-0.42, 0.0))
    arrow(ax, (4.9, 2.88), (4.9, 2.17), text="levers", shift=(0.42, 0.0))
    # gates and velocity cells
    box(ax, 7.3, 3.35, 2.1, 0.9, "gates and bias cells", PURPLE, "3 x %d + %d neurons\nswitch: anchor on / off" % (sizes.entry, BIAS_CELLS))
    arrow(ax, (5.52, 3.35), (6.23, 3.35))
    box(ax, 7.3, 1.6, 2.1, 0.9, "velocity cells", PURPLE, "6 groups x %d slow neurons\nswitch: path integration" % design.velocity_cells)
    arrow(ax, (5.52, 1.6), (6.23, 1.6))
    # sheets
    for i, (name, at) in enumerate((("sheet xy", (0.18, 0.12)), ("sheet xz", (0.18, -0.2)), ("sheet yz", (0.12, -0.2)))):
        y = 4.55 - 1.55 * i
        sheet(ax, 9.9, y, 1.1, name, at)
        arrow(ax, (8.37, 3.35), (9.3, y), color=PURPLE)
        arrow(ax, (8.37, 1.6), (9.3, y - 0.12), color=ORANGE)
    ax.text(9.9, 5.55, "three attractor sheets\n%d x %d neurons each" % (design.size, design.size), ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.text(7.3, 4.12, "anchor: holds each packet\nat the computed position", ha="center", va="center", fontsize=7.8, color=PURPLE)
    ax.text(7.3, 0.8, "each spike briefly silences the sheet\nneurons that point against the motion", ha="center", va="center", fontsize=7.8, color=ORANGE)
    # outputs
    box(ax, 12.0, 3.0, 1.7, 1.0, "position", PURPLE, "centre of the packets;\neach coordinate twice")
    for i in range(3):
        arrow(ax, (10.47, 4.55 - 1.55 * i), (11.13, 3.0 + 0.25 * (1 - i)))
    box(ax, 12.0, 5.0, 1.7, 0.9, "orientation", GREEN, "matrix of every link;\nlift, pitch, yaw rings")
    arrow(ax, (5.47, 5.3), (11.13, 5.3))
    ax.text(0.15, 0.25, "%s.  Blue: joint angles in.  Green: the SO(3) network and what is read from it without products.  Orange: velocity.  Purple: the sheets and what acts on them."
            % chain.name, fontsize=8.5, color=GREY, va="center")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chain", default="tiago7")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures", "architecture.png"))
    args = parser.parse_args(argv)
    draw(get_chain(args.chain), args.out)
    print("written:", os.path.relpath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
