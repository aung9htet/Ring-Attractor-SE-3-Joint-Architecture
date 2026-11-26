"""How many neurons each part of a network has, counted from its wiring and the population sizes (no NEST).

:func:`neuron_budget` gives the counts by technique, as the model explanation's budget page groups them; the totals are
the ``neuron_count()`` of the networks NEST builds (checked in tests/test_budget.py and tests/test_network.py).
:func:`observer_budget` counts the observer method's orientation observer from its wiring.
"""

from __future__ import annotations

from typing import Dict

from reaching_benchmark.geometry.chains import Chain

from ..config import IkDesign, ObserverDesign, SheetDesign, Sizes
from ..neurons.tuning import RAMP_NEURONS
from ..spiking.sheet import BIAS_CELLS
from .forward_wiring import forward_wiring
from .ik_wiring import Half, Mul, Relay, ik_wiring
from .observer_wiring import observer_wiring

#: the techniques, in the order the budget page lists them
TECHNIQUES = ("value populations", "2-D attractor sheets", "gain pairs and written goal rings", "ring attractors",
              "warm/cold comparison", "anchors", "readout rings", "level cells", "motor rings", "orientation observer")


def neuron_budget(chain: Chain, design: IkDesign, method: str = "matrix", sizes: Sizes = Sizes(),
                  sheet: SheetDesign = SheetDesign(), ramp_neurons: int = RAMP_NEURONS) -> Dict[str, int]:
    """Neurons of the pose IK network by technique (level cells on every sheet, as the IK builds it)."""

    n = chain.n
    sheet_neurons = int(sheet.size) ** 2
    forward = forward_wiring(chain)
    wiring = ik_wiring(chain, design, sizes.ring, forward)
    on_sheets = design.position and design.position_goal == "sheets"
    ik_products = sum(1 for s in wiring.steps if isinstance(s, Mul))
    ik_sums = sum(1 for s in wiring.steps if isinstance(s, (Relay, Half)))
    offsets = len(wiring.offset)
    value = (n * sizes.feature + len(forward.products) * sizes.product + len(forward.entries) * sizes.entry + 3 * sizes.entry
             + (3 * sizes.feature if design.orientation else 0) + ik_products * sizes.product + (ik_sums - offsets) * sizes.entry)
    out = {
        "value populations": value,
        "2-D attractor sheets": (3 + (3 if on_sheets else 0)) * sheet_neurons,
        "gain pairs and written goal rings": n * 3 * sizes.ring,
        "ring attractors": (n + (3 if design.orientation else 0)) * sizes.ring,
        "warm/cold comparison": (2 * 3 * ramp_neurons if on_sheets else 0) + offsets * sizes.entry,
        "anchors": 3 * sizes.entry + (6 * sizes.entry if design.position else 0) + BIAS_CELLS * (2 if on_sheets else 1),
        "readout rings": 3 * sizes.output_ring + sizes.norm,
        "level cells": (3 + (3 if on_sheets else 0)) * sizes.entry,
        "motor rings": n * sizes.ring if method == "lead" else 0,
        "orientation observer": observer_budget(chain, sizes=sizes)["total"] if method == "observer" else 0,
    }
    out["total"] = sum(out[t] for t in TECHNIQUES)
    return out


def forward_budget(chain: Chain, sizes: Sizes = Sizes(), sheet: SheetDesign = SheetDesign()) -> int:
    """Neurons of the forward network alone (with level cells), as the forward measurements build it."""

    forward = forward_wiring(chain)
    return (chain.n * (sizes.ring + sizes.feature) + len(forward.products) * sizes.product + len(forward.entries) * sizes.entry
            + 3 * sizes.output_ring + sizes.norm + 6 * sizes.entry + 3 * int(sheet.size) ** 2 + BIAS_CELLS + 3 * sizes.entry)


def observer_budget(chain: Chain, design: ObserverDesign = ObserverDesign(), sizes: Sizes = Sizes()) -> Dict[str, int]:
    """Neurons of the observer method's orientation observer (network/observer_wiring.py), by part."""

    wiring = observer_wiring(chain, design)
    groups = {
        "observer rings (2 charts × 3)": 6 * sizes.ring,
        "shift cells": 6 * 2 * sizes.ring,
        "observer cos/sin, now and delayed": 12 * sizes.feature,
        "joint velocities": chain.n * sizes.entry,
        "products": len(wiring.muls) * sizes.product,
        "sums, selector, fallback and matrix gates": (len(wiring.relays) + len(wiring.halves)) * sizes.entry,
        "cue norms": len(wiring.norms) * sizes.norm,
    }
    groups["total"] = int(sum(groups.values()))
    return groups
