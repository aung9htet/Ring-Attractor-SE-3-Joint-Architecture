"""Networks with the stored calibrations: one call per kind."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from reaching_benchmark.geometry.chains import Chain, get_chain

from ..calibration.artifacts import ARTIFACTS, Calibrations
from ..config import GainPairSettings, IkDesign, NetworkOptions, ObserverDesign, default_task
from ..neurons.tuning import Tunings
from .forward import ForwardNetwork
from .pose_ik_network import PoseIkNetwork, ik_options


def build_forward(backend: Any, chain_name: str, directory: str = ARTIFACTS, **options: Any) -> Tuple[Chain, ForwardNetwork]:
    chain = get_chain(chain_name)
    return chain, ForwardNetwork(backend, chain, Calibrations.load(directory), Tunings(), NetworkOptions(**options))


def build_pose_ik(backend: Any, chain_name: str, ik_design: Optional[IkDesign] = None, gain_pairs: Optional[GainPairSettings] = None,
                  directory: str = ARTIFACTS, method: str = "matrix", observer_design: Optional[ObserverDesign] = None,
                  **options: Any) -> Tuple[Chain, PoseIkNetwork]:
    """The pose IK network for an arm; the task defaults to pose from six joints on, position below."""

    chain = get_chain(chain_name)
    design = ik_design or IkDesign(task=default_task(chain.n))
    return chain, PoseIkNetwork(backend, chain, Calibrations.load(directory), Tunings(), ik_options(**options), design, gain_pairs, method,
                                observer_design)
