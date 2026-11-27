"""Rotations, chains and the hand's kinematics."""

import unittest

import numpy as np

from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import GOAL_CHAIN, MODEL_NAMES, get_chain, lever_lengths, workspace
from reaching_benchmark.geometry.rotations import (from_lift_pitch_yaw, lift_pitch_yaw, nearest_rotation, rodrigues, rotation_angle,
                                         turn_between)


class RotationTest(unittest.TestCase):
    def test_lift_pitch_yaw_round_trip(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            lpy = rng.uniform([-3, -1.5, -3], [3, 1.5, 3])
            self.assertTrue(np.allclose(lift_pitch_yaw(from_lift_pitch_yaw(*lpy)), lpy, atol=1e-9))

    def test_goal_chain_is_the_lift_pitch_yaw_matrix(self):
        lift, pitch, yaw = 0.3, -0.7, 2.1
        self.assertTrue(np.allclose(GOAL_CHAIN.rotation([yaw, pitch, lift]), from_lift_pitch_yaw(lift, pitch, yaw)))

    def test_turn_is_sine_of_angle_about_axis(self):
        axis = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
        for angle in (0.1, 0.8, 1.5):
            r = from_lift_pitch_yaw(0.2, 0.4, -1.0)
            g = rodrigues(angle * axis) @ r
            self.assertTrue(np.allclose(turn_between(r, g), np.sin(angle) * axis, atol=1e-12))
            self.assertAlmostEqual(rotation_angle(r, g), angle, places=12)

    def test_nearest_rotation_is_proper(self):
        m = from_lift_pitch_yaw(0.1, 0.2, 0.3) + 0.05 * np.random.default_rng(0).normal(size=(3, 3))
        r = nearest_rotation(m)
        self.assertTrue(np.allclose(r.T @ r, np.eye(3)))
        self.assertAlmostEqual(np.linalg.det(r), 1.0)


class ChainTest(unittest.TestCase):
    def test_twelve_arms(self):
        self.assertEqual(len(MODEL_NAMES), 12)
        for name in MODEL_NAMES:
            chain = get_chain(name)
            self.assertGreater(workspace(chain).radius, 0.0)
            self.assertAlmostEqual(lever_lengths(chain)[0], workspace(chain).radius)

    def test_jacobian_matches_finite_differences(self):
        chain = get_chain("tiago7")
        q = chain.limits().mean(axis=1) + 0.1
        numeric = np.array([(kinematics.position(chain, q + 1e-6 * e) - kinematics.position(chain, q - 1e-6 * e)) / 2e-6
                            for e in np.eye(chain.n)]).T
        self.assertTrue(np.allclose(kinematics.jacobian(chain, q), numeric, atol=1e-7))

    def test_angular_velocity_turns_the_hand(self):
        chain = get_chain("generic7")
        q, qdot, dt = np.full(7, 0.4), np.linspace(-0.5, 0.5, 7), 1e-6
        omega = kinematics.angular_velocity(chain, q, qdot)
        self.assertTrue(np.allclose(rodrigues(omega * dt) @ chain.rotation(q), chain.rotation(q + qdot * dt), atol=1e-11))


if __name__ == "__main__":
    unittest.main()
