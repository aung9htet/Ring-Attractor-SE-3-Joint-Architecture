"""The copy without spikes of the position and velocity populations."""

import unittest

import numpy as np

from ring_se3 import kinematics
from ring_se3.chains import get_chain
from ring_se3.rate.network import PoseRateNetwork
from ring_se3.wiring import pose_wiring


class PoseRateTests(unittest.TestCase):
    def test_position_and_velocity_come_out_close_to_the_exact_ones(self):
        rng = np.random.default_rng(21)
        for name in ("generic3", "tiago4"):
            chain = get_chain(name)
            network = PoseRateNetwork(pose_wiring(chain))
            radius = network.wiring.space.radius
            position, velocity = [], []
            for _ in range(25):
                q, qdot = chain.random_pose(rng), rng.uniform(-0.5, 0.5, chain.n)
                out = network.decode(q, qdot)
                position.append(np.linalg.norm(out.position - kinematics.position(chain, q)) / radius)
                true = kinematics.velocity(chain, q, qdot)
                velocity.append(np.linalg.norm(out.velocity - true) / (np.linalg.norm(true) + 0.02 * radius))
            self.assertLess(np.median(position), 0.03, name)
            self.assertLess(np.median(velocity), 0.35, name)

    def test_without_the_velocity_path_only_the_position_is_there(self):
        chain = get_chain("generic2")
        out = PoseRateNetwork(pose_wiring(chain, velocity=False)).decode([0.3, -0.5])
        self.assertTrue(np.all(np.isnan(out.velocity)))
        self.assertLess(np.linalg.norm(out.position - kinematics.position(chain, [0.3, -0.5])), 0.03 * 0.4)


if __name__ == "__main__":
    unittest.main()
