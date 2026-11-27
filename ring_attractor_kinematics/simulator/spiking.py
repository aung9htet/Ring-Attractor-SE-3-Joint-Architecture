"""The simulator's spiking back end: pose IK networks in NEST with the stored calibrations."""

from __future__ import annotations

from typing import Any, Callable

import nest

from reaching_benchmark.geometry.chains import Chain

from ..calibration.artifacts import Calibrations
from ..config import IkDesign
from ..network.pose_ik_network import PoseIkNetwork, ik_options
from ..neurons.tuning import Tunings


def network_factory(threads: int) -> Callable[[Chain, IkDesign], Any]:
    calibrations, tunings = Calibrations.load(), Tunings()
    return lambda chain, design: PoseIkNetwork(nest, chain, calibrations, tunings, ik_options(threads=threads), design)
