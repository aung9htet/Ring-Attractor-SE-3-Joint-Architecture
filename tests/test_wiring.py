"""The step lists the network follows give the exact kinematics and the exact cascade."""

import unittest

import numpy as np

from ring_attractor_kinematics.config import IkDesign, default_task
from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import MODEL_NAMES, get_chain
from reaching_benchmark.geometry.rotations import lift_pitch_yaw
from ring_attractor_kinematics.network.forward_wiring import forward_wiring
from ring_attractor_kinematics.network.ik_wiring import ik_wiring
from ring_attractor_kinematics.rules.cascade_rule import TURN_BOUND, limit_room, offset, pose_cascade, turn

#: products, sums and share halves of the IK layer per arm (default task): fixed by the wiring rules
EXPECTED = {"tiago4": (38, 34, 8), "tiago7": (128, 80, 14), "generic7": (140, 89, 14)}


class ForwardWiringTest(unittest.TestCase):
    def test_exact_orientation_and_position(self):
        for name in MODEL_NAMES:
            chain = get_chain(name)
            wiring = forward_wiring(chain)
            for q in [chain.random_pose(np.random.default_rng(s)) for s in range(5)]:
                values = wiring.evaluate(q)
                self.assertTrue(np.allclose(wiring.tool_matrix(values), chain.rotation(q), atol=1e-12), name)
                self.assertTrue(np.allclose(wiring.position_m(values), kinematics.position(chain, q), atol=1e-12), name)


class IkWiringTest(unittest.TestCase):
    def test_shares_equal_the_exact_cascade(self):
        """With turn and offset inside what their populations hold, the wired shares are the rule's shares."""

        for name in ("tiago4", "tiago6", "tiago7", "generic7"):
            chain = get_chain(name)
            design = IkDesign(task=default_task(chain.n))
            wiring = ik_wiring(chain, design)
            rng = np.random.default_rng(7)
            for _ in range(10):
                q = chain.random_pose(rng)
                goal_q = chain.clip(q + rng.uniform(-0.05, 0.05, chain.n))
                goal_lpy, position = np.array(lift_pitch_yaw(chain.rotation(goal_q))), kinematics.position(chain, goal_q)
                room = limit_room(chain, q, design)
                values = wiring.evaluate(q, goal_lpy, position, room=room)
                e = np.clip(turn(chain.rotation(q), goal_lpy), -TURN_BOUND, TURN_BOUND) if design.orientation else np.zeros(3)
                d = np.clip(offset(chain, q, position), -0.25, 0.25) if design.position else np.zeros(3)
                shares, _ = pose_cascade(kinematics.joint_axes(chain, q), kinematics.moves(chain, q), e, d, design, wiring.lever,
                                         wiring.caps, room)
                self.assertTrue(np.allclose(wiring.share_values(values), shares, atol=1e-9), name)

    def test_population_counts(self):
        for name, (products, sums, halves) in EXPECTED.items():
            chain = get_chain(name)
            summary = ik_wiring(chain, IkDesign(task=default_task(chain.n))).summary()
            self.assertEqual((summary["products"], summary["sum_populations"], summary["share_populations"]), (products, sums, halves), name)


if __name__ == "__main__":
    unittest.main()
