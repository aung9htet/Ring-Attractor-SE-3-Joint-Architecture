"""One look for every data figure: journal sizes, 7 pt Liberation Sans (metric-compatible with Arial), thin lines,
one colour per method fixed across all figures, bold lower-case panel letters, vector output.

Colours: the orientation sources take the categorical slots of the validated reference palette in their fixed
order (M0 slot 1 … MQ slot 7); the observer method takes slot 8, for its spiking network and its exact twin (MB, drawn
dashed); the reference controllers without neurons are neutral inks told apart by line style, so a method keeps its
colour in every figure and no colour is reused for anything else.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from ..common import FIGURES, ensure_dirs
MM = 1.0 / 25.4
SINGLE, DOUBLE = 89.0 * MM, 183.0 * MM

INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#dcdad4"
#: the network (old method, spiking) is M0's colour
COLOURS: Dict[str, str] = {"M0": "#2a78d6", "M1": "#eb6834", "M2": "#1baf7a", "M3": "#eda100", "M4": "#e87ba4",
                           "M5": "#008300", "MQ": "#6250d6", "MB": "#e34948", "observer": "#e34948", "network": "#2a78d6"}
#: neutral references: (colour, line style)
REFERENCES: Dict[str, Tuple[str, Any]] = {"cascade": ("#1f1f1e", (0, (4, 2))), "dls": ("#6b6a66", (0, (1, 1.5))),
                                          "transpose": ("#a9a7a0", (0, (6, 2, 1, 2)))}
LABELS: Dict[str, str] = {"network": "spiking network (matrix)", "M0": "matrix (old)", "M1": "matrix + joint lead",
                          "M2": "readout rings", "M3": "observer, cue only", "M4": "observer + prediction",
                          "M5": "two-chart observer", "MQ": "SO(3) observer", "MB": "observer as built (exact)",
                          "observer": "observer method (spiking)", "cascade": "exact cascade",
                          "dls": "damped least squares", "transpose": "Jacobian transpose"}
ARMS: Dict[str, str] = {"tiago7": "TIAGo 7-DoF", "generic7": "generic 7-DoF", "tiago4": "TIAGo 4-DoF"}


def setup() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "Liberation Sans" if "Liberation Sans" in names else "DejaVu Sans", "font.size": 7.0, "axes.titlesize": 7.0, "axes.labelsize": 7.0,
        "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 6.0, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "lines.linewidth": 0.9, "lines.markersize": 3.0, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2, "pdf.fonttype": 42,
        "svg.fonttype": "none", "savefig.dpi": 600, "figure.dpi": 150, "axes.titlepad": 3.0, "axes.labelpad": 2.0,
    })


def colour(method: str) -> str:
    return COLOURS[method] if method in COLOURS else REFERENCES[method][0]


def line_style(method: str) -> Any:
    return REFERENCES[method][1] if method in REFERENCES else "solid"


def panel(ax: Any, letter: str, x: float = -0.18, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="bottom", ha="left")


def recessive_grid(ax: Any, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> Sequence[str]:
    ensure_dirs()
    paths = [str(FIGURES / (name + ext)) for ext in (".pdf", ".svg", ".png")]
    for path in paths:
        metadata = {".pdf": {"CreationDate": None}, ".svg": {"Date": None}}.get(path[-4:], {})
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=300, facecolor="white", metadata=metadata)
    plt.close(fig)
    return paths
