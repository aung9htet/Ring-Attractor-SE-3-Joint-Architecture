"""IK test 1, second half: the wiring of the pose IK layer computes what the exact rule computes."""

import unittest

import numpy as np

from ring_se3 import kinematics
from ring_se3.chains import MODEL_NAMES, get_chain
from ring_se3.ik import rules
from ring_se3.ik.design import PoseIkDesign
from ring_se3.ik.wiring import pose_ik_wiring, ramp_nodes
from ring_se3.wiring import pose_wiring
from ring_so3.ik import rules as so3
from ring_so3.ik.wiring import Half, Mul, Relay
from ring_so3.kinematics import lift_pitch_yaw


def near_goal(chain, q, rng, size=0.05):
    other = q + rng.normal(0.0, size, chain.n)
    return np.array(lift_pitch_yaw(chain.rotation(other))), kinematics.position(chain, other)


class WiringTests(unittest.TestCase):
    def test_wiring_equals_the_rule(self):
        rng = np.random.default_rng(11)
        for name in MODEL_NAMES:
            chain = get_chain(name)
            for task in ("pose", "position", "orientation"):
                for position_goal in ("sheets", "cells"):
                    for cascade in (True, False):
                        design = PoseIkDesign(task=task, position_goal=position_goal, cascade=cascade)
                        wiring = pose_ik_wiring(chain, design=design)
                        q = chain.random_pose(rng)
                        goal_lpy, goal_position = near_goal(chain, q, rng)
                        values = wiring.evaluate(q, goal_lpy, goal_position)
                        e = so3.turn(chain.rotation(q), goal_lpy, "matrix") if design.orientation else np.zeros(3)
                        d = rules.offset(chain, q, goal_position) if design.position else np.zeros(3)
                        shares, rests = rules.pose_cascade(so3.joint_axes(chain, q), rules.moves(chain, q), e, d, design, wiring.lever, wiring.caps)
                        np.testing.assert_allclose(wiring.turn_vector(values), e, atol=1e-9, err_msg=name)
                        np.testing.assert_allclose(wiring.offset_vector(values), d, atol=1e-9, err_msg=name)
                        np.testing.assert_allclose(wiring.share_values(values), shares, atol=1e-9, err_msg=name)
                        np.testing.assert_allclose(wiring.left_vector(values), rests[-1], atol=1e-9, err_msg=name)

    def test_every_value_stays_in_its_range(self):
        rng = np.random.default_rng(12)
        for name in ("generic7", "tiago7", "tiago4"):
            chain = get_chain(name)
            wiring = pose_ik_wiring(chain, design=PoseIkDesign())
            for _ in range(20):
                q, goal = chain.random_pose(rng), chain.random_pose(rng)
                values = wiring.evaluate(q, lift_pitch_yaw(chain.rotation(goal)), kinematics.position(chain, goal))
                for step in wiring.steps:
                    low = 0.0 if isinstance(step, Half) else -1.0
                    self.assertTrue(low - 1e-9 <= values[step.node] <= 1.0 + 1e-9, (name, step.node, values[step.node]))

    def test_shares_are_held_down_at_a_joint_limit(self):
        chain = get_chain("tiago7")
        design = PoseIkDesign()
        wiring = pose_ik_wiring(chain, design=design)
        q = chain.limits().mean(axis=1)
        q[3] = chain.joints[3].upper
        beyond = q.copy()
        beyond[3] += 0.03                                             # a goal that asks joint 4 to go on past its limit
        goal_lpy, goal_position = lift_pitch_yaw(chain.rotation(beyond)), kinematics.position(chain, beyond)
        free = wiring.share_values(wiring.evaluate(q, goal_lpy, goal_position))
        self.assertGreater(free[3], 1e-3)
        room = so3.limit_room(chain, q, design)
        values = wiring.evaluate(q, goal_lpy, goal_position, room=room)
        self.assertLessEqual(wiring.share_values(values)[3], 1e-12)
        e, d = so3.turn(chain.rotation(q), goal_lpy, "matrix"), rules.offset(chain, q, goal_position)
        shares, rests = rules.pose_cascade(so3.joint_axes(chain, q), rules.moves(chain, q), e, d, design, wiring.lever, wiring.caps, room)
        np.testing.assert_allclose(wiring.share_values(values), shares, atol=1e-9)
        np.testing.assert_allclose(wiring.left_vector(values), rests[-1], atol=1e-9)      # and what it does not take is handed on

    def test_parts_of_the_seven_joint_arm(self):
        wiring = pose_ik_wiring(get_chain("tiago7"), design=PoseIkDesign())
        summary = wiring.summary()
        self.assertEqual(summary["share_populations"], 14)
        self.assertEqual(summary["sheet_readouts"], 6)
        self.assertEqual(wiring.ramps, ramp_nodes("goal") + ramp_nodes("hand"))
        self.assertEqual(len(wiring.turn), 3)
        self.assertEqual(len(wiring.offset), 3)
        self.assertEqual(len(wiring.rests), 7)
        self.assertEqual(len(wiring.rests_position), 7)
        # the first joint turns about a fixed axis: what it asks for of the turn, and what it leaves of it, need no product
        first_share = next(step for step in wiring.steps if isinstance(step, Half))
        products = [step for step in wiring.steps if isinstance(step, Mul) and step.name is None]
        before = [step for step in wiring.steps[: wiring.steps.index(first_share)] if isinstance(step, Mul) and step.name is None]
        self.assertEqual(len(before), 18 + 2)                        # the comparison of orientations, and c_1 · d (c_1 has no z part)
        self.assertEqual(len(products), summary["products"] - 14)    # 14 products build the goal matrix
        # the last joint's lever lies along its own axis: it moves nothing and gets no direction populations
        self.assertFalse(any(isinstance(step, Relay) and step.node[:2] == ("jac", 6) for step in wiring.steps))
        self.assertEqual(wiring.rests_position[-1], wiring.rests_position[-2])

    def test_position_goal_on_cells_needs_no_sheet_readout(self):
        wiring = pose_ik_wiring(get_chain("tiago4"), design=PoseIkDesign(task="position", position_goal="cells"))
        self.assertEqual(wiring.ramps, [])
        self.assertEqual(wiring.turn, [])
        self.assertEqual(wiring.goal, None)

    def test_levers_are_taken_from_the_pose_wiring_when_it_has_them(self):
        chain = get_chain("generic4")
        with_velocity = pose_ik_wiring(chain, pose_wiring(chain, velocity=True), PoseIkDesign(task="position"))
        without = pose_ik_wiring(chain, pose_wiring(chain, velocity=False), PoseIkDesign(task="position"))
        own = [step for step in without.steps if isinstance(step, Relay) and step.node[0] == "lever"]
        self.assertEqual(len([step for step in with_velocity.steps if isinstance(step, Relay) and step.node[0] == "lever"]), 0)
        self.assertGreater(len(own), 0)
        self.assertEqual(len(with_velocity.muls), len(without.muls))

    def test_the_exact_normaliser_is_refused(self):
        with self.assertRaises(ValueError):
            pose_ik_wiring(get_chain("tiago4"), design=PoseIkDesign(normaliser="exact"))


if __name__ == "__main__":
    unittest.main()
