"""The observer method: its wiring computes what its exact twin computes, and the built network matches its budget."""

import importlib.util
import unittest

import numpy as np

from ring_attractor_kinematics.config import IkDesign
from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.geometry.rotations import from_lift_pitch_yaw, lift_pitch_yaw, nearest_rotation, rotation_angle
from ring_attractor_kinematics.network.budget import neuron_budget
from ring_attractor_kinematics.network.observer_wiring import observer_wiring
from ring_attractor_kinematics.rules.observer_rule import CHARTS, ObserverAsBuilt, chart_rates_with_secant

HAVE_NEST = importlib.util.find_spec("nest") is not None


class ObserverWiringTest(unittest.TestCase):
    def setUp(self):
        self.chain = get_chain("tiago7")
        self.wiring = observer_wiring(self.chain)
        self.rng = np.random.default_rng(3)

    def state(self):
        q = self.chain.random_pose(self.rng)
        rotation = self.chain.rotation(q)
        true = np.array([lift_pitch_yaw(u.T @ rotation) for u in CHARTS])
        return q, rotation, true + self.rng.normal(0, 0.05, (2, 3)), true + self.rng.normal(0, 0.08, (2, 3))

    def test_rates_and_innovations_match_the_twin(self):
        for _ in range(100):
            q, rotation, angles, past = self.state()
            qdot = self.rng.uniform(-0.8, 0.8, self.chain.n)
            values = self.wiring.evaluate(q, qdot, angles, past, 0, False)
            twin = ObserverAsBuilt(self.chain, 0.2)
            twin.reset(q)
            twin.history = [[past[c].copy()] for c in range(2)]
            omega = kinematics.joint_axes(self.chain, q).T @ qdot
            for c, u in enumerate(CHARTS):
                innovation = twin.innovation(c, rotation)
                np.testing.assert_allclose([values[("oinn", c, j)] for j in range(3)], innovation, atol=1e-9)
                rates = chart_rates_with_secant(angles[c], u.T @ omega, 1.0 / max(np.cos(angles[c][1]), 0.5))
                wanted = rates + innovation / twin.design.correction_s
                if np.all(np.abs(wanted) < 0.95 * twin.design.rate_full):
                    np.testing.assert_allclose(self.wiring.rate_values(values)[c], wanted, atol=1e-9)

    def test_matrix_gates_pass_the_chosen_route(self):
        q, rotation, angles, past = self.state()
        for selected in (0, 1):
            values = self.wiring.evaluate(q, np.zeros(self.chain.n), angles, past, selected, False)
            np.testing.assert_allclose(self.wiring.hand_matrix(values), CHARTS[selected] @ from_lift_pitch_yaw(*angles[selected]), atol=1e-9)
        values = self.wiring.evaluate(q, np.zeros(self.chain.n), angles, past, 0, True)
        np.testing.assert_allclose(self.wiring.hand_matrix(values), rotation, atol=1e-9)


@unittest.skipUnless(HAVE_NEST, "NEST is not installed")
class ObserverNetworkTest(unittest.TestCase):
    def test_built_network_matches_its_budget_and_holds_a_pose(self):
        nest = importlib.import_module("nest")
        factory = importlib.import_module("ring_attractor_kinematics.network.factory")
        design = IkDesign(task="orientation")
        chain, network = factory.build_pose_ik(nest, "generic3", design, method="observer", threads=4)
        self.assertEqual(network.neuron_count(), neuron_budget(chain, design, "observer")["total"])
        q = chain.limits().mean(axis=1)
        lpy, position = lift_pitch_yaw(chain.rotation(q)), kinematics.position(chain, q)
        for _ in range(30):
            tick = network.step_ik(q, lpy, position, velocity=np.zeros(chain.n), seed=True)
        for _ in range(20):
            tick = network.step_ik(q, lpy, position, velocity=np.zeros(chain.n))
        self.assertLess(np.degrees(rotation_angle(nearest_rotation(tick.observer.hand), chain.rotation(q))), 10.0)


if __name__ == "__main__":
    unittest.main()
