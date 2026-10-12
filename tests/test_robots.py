"""The robot behind the pose IK: the light robot, and the Gazebo robot on a stand-in for ROS."""

import unittest

import numpy as np

from ring_se3 import kinematics
from ring_se3.chains import get_chain
from ring_se3.gazebo.link import arm_link, end_point, kinematics_check, quaternion_matrix
from ring_se3.ik.design import PoseIkDesign
from ring_se3.ik.session import PoseIkSession
from ring_se3.sim.ik_server import PoseIkStubNetwork
from ring_so3.kinematics import lift_pitch_yaw, rot
from ring_so3.robots import GazeboRobot, LightRobot


class FakeLink:
    """Seven joints that move toward what they were sent as a trajectory controller does, and the frames of their
    links as Gazebo would give them; no ROS."""

    def __init__(self, q):
        self.q = np.array(q, dtype=float)
        self.goal, self.arrive_s = self.q.copy(), 0.05
        self.t, self.overshoot_s, self.sent, self.base_resets = 0.0, 0.0, 0, 0
        self.arm = get_chain("tiago7")
        self.shift, self.known = np.zeros(3), True                  # for the tests: links that are somewhere else, or not there

    def joint_state(self):
        return list(self.q), [0.0] * 7, self.t

    def send(self, positions, time_s):
        self.goal, self.arrive_s, self.sent = np.array(positions, dtype=float), float(time_s), self.sent + 1

    def step(self, iterations):
        dt = iterations * 0.001
        self.q = self.q + (self.goal - self.q) * min(dt / self.arrive_s, 1.0)
        self.arrive_s = max(self.arrive_s - dt, 1e-6)
        self.t += dt
        return self.t

    def put_base_back(self):
        self.base_resets += 1
        return True

    def base_pose(self):
        return (0.0, 0.0, 0.0)

    def frames(self):
        return 0

    def link_pose(self, link, reference="torso_lift_link"):
        names = [arm_link(k) for k in range(1, 8)]
        if link not in names or not self.known:
            return None
        origin, rotation = self.arm.frames(self.q)[names.index(link) + 1]      # frames: the base, then link 1 .. 7, then the tool
        return origin + self.shift, rotation


class LightRobotTests(unittest.TestCase):
    def test_follow_is_a_proportional_step(self):
        chain = get_chain("generic3")
        robot = LightRobot(chain, [0.0, 0.0, 0.0])
        robot.follow([0.2, -0.1, np.nan], 0.8, 0.05)
        np.testing.assert_allclose(robot.q, [0.2 * 0.05 / 0.8, -0.1 * 0.05 / 0.8, 0.0])

    def test_limits_wrap_and_push(self):
        tiago = get_chain("tiago7")
        robot = LightRobot(tiago)
        robot.follow(tiago.limits()[:, 1] + 1.0, 0.05, 0.05)
        np.testing.assert_allclose(robot.q, tiago.limits()[:, 1])
        generic = LightRobot(get_chain("generic3"), [3.1, 0.0, 0.0])
        generic.follow([-3.1, 0.0, 0.0], 0.05, 0.05)                 # the short way round
        self.assertAlmostEqual(generic.q[0], -3.1, places=9)
        generic.displace(1, 0.3)
        self.assertAlmostEqual(generic.q[1], 0.3, places=9)


class GazeboRobotTests(unittest.TestCase):
    def setUp(self):
        self.chain = get_chain("tiago7")
        self.link = FakeLink(self.chain.limits().mean(axis=1))
        self.robot = GazeboRobot(self.chain, self.link)

    def test_one_follow_is_one_tick_of_physics(self):
        before = self.link.t
        target = self.robot.q + 0.1
        self.robot.follow(target, 0.8, 0.05)
        self.assertAlmostEqual(self.link.t - before, 0.05, places=9)
        self.assertEqual(self.link.sent, 1)
        np.testing.assert_allclose(self.robot.lag, target - self.robot.q)

    def test_targets_stay_inside_the_limits_and_reset_moves_there(self):
        self.robot.follow(self.chain.limits()[:, 1] + 1.0, 0.05, 0.05)
        np.testing.assert_allclose(self.link.goal, self.chain.limits()[:, 1])
        goal = self.chain.limits().mean(axis=1) + 0.3
        self.robot.reset(goal)
        np.testing.assert_allclose(self.robot.q, goal, atol=0.01)
        self.assertGreaterEqual(self.link.base_resets, 1)

    def test_the_pose_robot_reads_rest_from_the_angles(self):
        from ring_se3.gazebo.robot import PoseGazeboRobot

        self.link.joint_state = lambda: (list(self.link.q), [0.0] * 5 + [1.9, 0.0], self.link.t)      # a wrist joint that reports motion it does not make
        robot = PoseGazeboRobot(self.chain, self.link)
        goal = self.chain.limits().mean(axis=1) + 0.3
        before = self.link.sent
        robot.reset(goal)
        np.testing.assert_allclose(robot.q, goal, atol=0.01)
        self.assertLess(self.link.sent - before, 200)               # not the 2,000 ticks the velocity test would take

    def test_only_the_tiago_chains(self):
        with self.assertRaises(ValueError):
            GazeboRobot(get_chain("generic7"), self.link)

    def test_a_shorter_chain_holds_the_other_joints(self):
        chain = get_chain("tiago4")
        held = self.link.q[4:].copy()
        robot = GazeboRobot(chain, self.link)
        robot.reset(chain.limits().mean(axis=1) + 0.2)
        np.testing.assert_allclose(self.link.q[:4], chain.limits().mean(axis=1) + 0.2, atol=0.01)
        np.testing.assert_allclose(self.link.q[4:], held, atol=1e-12)

    def test_the_pose_loop_runs_on_it(self):
        design = PoseIkDesign(task="pose")
        session = PoseIkSession(PoseIkStubNetwork(self.chain, design, noise_deg=0.1, noise_position=0.002), robot=self.robot)
        session.warm_up(5)
        there = self.chain.limits().mean(axis=1) + 0.25
        trial = session.reach(lift_pitch_yaw(self.chain.rotation(there)), kinematics.position(self.chain, there))
        self.assertTrue(trial.reached(design))
        self.assertIsNotNone(trial.lag)
        self.assertEqual(trial.lag.shape, trial.q.shape)

    def test_a_position_goal_on_four_joints(self):
        chain = get_chain("tiago4")
        design = PoseIkDesign(task="position")
        robot = GazeboRobot(chain, self.link)
        session = PoseIkSession(PoseIkStubNetwork(chain, design, noise_deg=0.1, noise_position=0.002), robot=robot)
        session.warm_up(5)
        there = chain.limits().mean(axis=1) + 0.25
        trial = session.reach(lift_pitch_yaw(chain.rotation(there)), kinematics.position(chain, there))
        self.assertTrue(trial.reached(design))


class GazeboKinematicsTests(unittest.TestCase):
    def test_quaternion(self):
        np.testing.assert_allclose(quaternion_matrix(0.0, 0.0, 0.0, 1.0), np.eye(3), atol=1e-12)
        angle = 0.7
        np.testing.assert_allclose(quaternion_matrix(0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)), rot("z", angle), atol=1e-12)
        np.testing.assert_allclose(quaternion_matrix(2 * np.sin(angle / 2), 0.0, 0.0, 2 * np.cos(angle / 2)), rot("x", angle), atol=1e-12)   # not of unit length

    def test_end_point_of_every_tiago_chain_is_where_the_links_are(self):
        rng = np.random.default_rng(3)
        whole = get_chain("tiago7")
        for _ in range(5):
            link = FakeLink(whole.random_pose(rng))
            for n in range(2, 8):
                chain = get_chain("tiago%d" % n)
                position, rotation = end_point(link, chain)
                np.testing.assert_allclose(position, kinematics.position(chain, link.q[:n]), atol=1e-12)
                np.testing.assert_allclose(rotation, chain.rotation(link.q[:n]), atol=1e-12)

    def test_check_reports_no_difference_on_a_true_arm_and_the_difference_on_a_shifted_one(self):
        chain = get_chain("tiago7")
        link = FakeLink(chain.limits().mean(axis=1))
        poses = [chain.limits().mean(axis=1) + 0.2, chain.limits().mean(axis=1) - 0.2]
        document = kinematics_check(link, GazeboRobot(chain, link), chain, poses, log=None)
        self.assertEqual(len(document["rows"]), 2)
        self.assertLess(document["largest_position_error_m"], 1e-9)
        self.assertLess(document["largest_orientation_error_deg"], 1e-6)
        link.shift = np.array([0.003, 0.0, 0.0])
        shifted = kinematics_check(link, GazeboRobot(chain, link), chain, poses, log=None)
        self.assertAlmostEqual(shifted["largest_position_error_m"], 0.003, places=9)
        link.known = False                                          # Gazebo does not have the link
        self.assertIsNone(end_point(link, chain))
        self.assertIsNone(kinematics_check(link, GazeboRobot(chain, link), chain, poses[:1], log=None)["largest_position_error_m"])


if __name__ == "__main__":
    unittest.main()
