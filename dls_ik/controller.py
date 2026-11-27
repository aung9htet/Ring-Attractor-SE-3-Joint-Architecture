"""DLS IK as a controller of the shared closed loop: no state, every tick a damped-least-squares step from the measured
joint angles and the true geometry, sent as the joints' targets."""

from __future__ import annotations

from typing import Any, Dict

from reaching_benchmark.closed_loop import RuleLoop
from reaching_benchmark.geometry.chains import Chain
from reaching_benchmark.task import IkDesign

from . import rule

#: what the controller reads, is given and learns (stored in every run file)
DESCRIPTION: Dict[str, Any] = {
    "name": "DLS IK",
    "reads": "the measured joint angles, every 50 ms tick",
    "given": "the true geometry: forward kinematics and the Jacobian",
    "learns": "nothing",
    "internal_model": "none: every step is recomputed from the measured state (reactive)",
    "rule": "damped least squares, lambda = %g; whole step scaled to the caps; joints at a limit left out" % rule.DAMPING,
    "gain": rule.GAIN,
}


def make_loop(chain: Chain, design: IkDesign, robot: Any) -> RuleLoop:
    return RuleLoop(chain, design, robot, rule.steps, rule.GAIN)
