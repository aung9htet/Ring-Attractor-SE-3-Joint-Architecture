"""Test 1: the exact maths.  Position as a weighted sum of link orientations, the velocity formula, the wiring."""

import unittest

import numpy as np

from ring_so3.chains import get_chain as so3_chain
from ring_se3 import kinematics
from ring_se3.chains import MODEL_NAMES, get_chain, lever_lengths, link_offsets, workspace
from ring_se3.wiring import QDOT_REF, pose_wiring


class PositionTests(unittest.TestCase):
    def test_position_is_the_last_origin_of_the_so3_frames(self):
        rng = np.random.default_rng(1)
        for name in MODEL_NAMES:
            chain = get_chain(name)
            for _ in range(20):
                q = chain.random_pose(rng)
                frames = chain.frames(q)
                np.testing.assert_allclose(kinematics.position(chain, q), frames[-1][0], atol=1e-12)
                np.testing.assert_allclose(kinematics.joint_origins(chain, q)[:-1], [f[0] for f in frames[1:-1]], atol=1e-12)
                np.testing.assert_allclose(kinematics.pose(chain, q)[:3, :3], chain.rotation(q), atol=1e-12)

    def test_shortened_tiago_arms_end_at_the_next_joint_and_keep_their_orientation(self):
        rng = np.random.default_rng(2)
        whole = get_chain("tiago7")
        for n in range(2, 7):
            chain = get_chain("tiago%d" % n)
            self.assertEqual(chain.tool_offset, whole.joints[n].offset)
            q = whole.random_pose(rng)
            np.testing.assert_allclose(kinematics.position(chain, q[:n]), kinematics.joint_origins(whole, q)[n], atol=1e-12)
            np.testing.assert_allclose(chain.rotation(q[:n]), so3_chain("tiago%d" % n).rotation(q[:n]), atol=1e-12)

    def test_every_position_lies_in_the_workspace_ball(self):
        rng = np.random.default_rng(3)
        for name in MODEL_NAMES:
            chain = get_chain(name)
            space = workspace(chain)
            self.assertGreater(space.radius, 0.05)
            scaled = np.array([space.scaled(kinematics.position(chain, chain.random_pose(rng))) for _ in range(500)])
            self.assertLessEqual(np.max(np.linalg.norm(scaled, axis=1)), 1.0 + 1e-9)
            np.testing.assert_allclose(space.metres(space.scaled([0.1, 0.2, 0.3])), [0.1, 0.2, 0.3], atol=1e-12)

    def test_levers_are_no_longer_than_their_bound(self):
        rng = np.random.default_rng(4)
        for name in MODEL_NAMES:
            chain = get_chain(name)
            bound = lever_lengths(chain)
            self.assertEqual(len(link_offsets(chain)), chain.n + 1)
            for _ in range(100):
                lengths = np.linalg.norm(kinematics.levers(chain, chain.random_pose(rng)), axis=1)
                self.assertTrue(np.all(lengths <= bound + 1e-9))


class VelocityTests(unittest.TestCase):
    def test_jacobian_against_finite_differences(self):
        rng = np.random.default_rng(5)
        step = 1e-6
        for name in MODEL_NAMES:
            chain = get_chain(name)
            for _ in range(10):
                q = chain.random_pose(rng)
                numeric = np.zeros((3, chain.n))
                for k in range(chain.n):
                    dq = np.zeros(chain.n)
                    dq[k] = step
                    numeric[:, k] = (kinematics.position(chain, q + dq) - kinematics.position(chain, q - dq)) / (2 * step)
                np.testing.assert_allclose(kinematics.jacobian(chain, q), numeric, atol=1e-6)


class WiringTests(unittest.TestCase):
    def test_wiring_gives_position_and_velocity_exactly(self):
        rng = np.random.default_rng(6)
        for name in MODEL_NAMES:
            chain = get_chain(name)
            for relay in (True, False):
                wiring = pose_wiring(chain, relay=relay)
                for _ in range(10):
                    q = chain.random_pose(rng)
                    qdot = rng.uniform(-QDOT_REF, QDOT_REF, chain.n)
                    values = wiring.evaluate(q, qdot)
                    np.testing.assert_allclose(wiring.position_m(values), kinematics.position(chain, q), atol=1e-9)
                    np.testing.assert_allclose(wiring.velocity_m_s(values), kinematics.velocity(chain, q, qdot), atol=1e-9)

    def test_every_value_a_population_holds_stays_in_its_range(self):
        rng = np.random.default_rng(7)
        for name in MODEL_NAMES:
            chain = get_chain(name)
            wiring = pose_wiring(chain)
            worst, clipped, total = 0.0, 0, 0
            for _ in range(300):
                q = chain.random_pose(rng)
                qdot = rng.choice([-1.0, 1.0], chain.n) * QDOT_REF          # every joint at the reference speed
                values = wiring.evaluate(q, qdot)
                for step in wiring.steps:
                    if step.node[0] in ("pos", "lever", "mul"):
                        worst = max(worst, abs(values[step.node]))
                    if step.node[0] == "vel":
                        total += 1
                        clipped += abs(values[step.node]) > 1.0
            self.assertLessEqual(worst, 1.0 + 1e-9, name)
            self.assertLess(clipped / total, 0.02, name)       # the velocity may leave its range only rarely, at full speed of all joints

    def test_size(self):
        wiring = pose_wiring(get_chain("tiago7"))
        self.assertLessEqual(len(wiring.muls), 9 * 7)
        self.assertEqual(len(wiring.position), 3)
        self.assertEqual(len(wiring.velocity_nodes), 3)
        self.assertEqual(len(pose_wiring(get_chain("tiago7"), velocity=False).muls), 0)


if __name__ == "__main__":
    unittest.main()
