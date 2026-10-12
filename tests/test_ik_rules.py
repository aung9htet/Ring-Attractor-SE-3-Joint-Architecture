"""IK test 1: the exact maths of the pose comparison, the pose cascade and the closed loop."""

import unittest

import numpy as np

from ring_se3 import kinematics
from ring_se3.chains import MODEL_NAMES, get_chain, workspace
from ring_se3.ik import rules
from ring_se3.ik.design import PoseIkDesign
from ring_se3.ik.pairs import DEVELOPMENT_SEED, STUDY_SEED, TEST_SEED, pairs
from ring_se3.standard import ik as standard
from ring_so3.ik import rules as so3
from ring_so3.ik.design import IkDesign
from ring_so3.kinematics import lift_pitch_yaw


class ComparisonTests(unittest.TestCase):
    def test_moves_are_how_far_the_end_point_goes_per_radian(self):
        rng = np.random.default_rng(1)
        for name in MODEL_NAMES:
            chain = get_chain(name)
            q = chain.random_pose(rng)
            move, radius, h = rules.moves(chain, q), workspace(chain).radius, 1e-6
            for k in range(chain.n):
                step = np.zeros(chain.n)
                step[k] = h
                numeric = (kinematics.position(chain, q + step) - kinematics.position(chain, q - step)) / (2 * h * radius)
                np.testing.assert_allclose(move[k], numeric, atol=1e-6)

    def test_offset_is_nothing_at_the_goal_and_in_reaches(self):
        chain = get_chain("tiago7")
        q = chain.limits().mean(axis=1)
        here = kinematics.position(chain, q)
        np.testing.assert_allclose(rules.offset(chain, q, here), 0.0, atol=1e-12)
        np.testing.assert_allclose(rules.offset(chain, q, here + [workspace(chain).radius, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)

    def test_lever_constants(self):
        for name in MODEL_NAMES:
            chain = get_chain(name)
            lever = rules.levers(chain)
            self.assertAlmostEqual(lever.ratios[0], 1.0)
            self.assertTrue(np.all(np.diff(lever.ratios) <= 1e-12))                      # a later joint has a shorter lever
            self.assertTrue(np.all(lever.mean_square <= lever.ratios ** 2 + 1e-12))       # |m_k| is never more than the lever's ratio
            self.assertIs(rules.levers(chain), lever)                                    # worked out once per chain

    def test_normalisers(self):
        chain = get_chain("generic7")
        lever = rules.levers(chain)
        move = rules.moves(chain, chain.random_pose(np.random.default_rng(2)))
        exact = rules.normalisers(PoseIkDesign(normaliser="exact"), lever, move)
        bound = rules.normalisers(PoseIkDesign(normaliser="bound"), lever)
        mean = rules.normalisers(PoseIkDesign(normaliser="mean"), lever)
        self.assertTrue(np.all(exact <= bound + 1e-12) and np.all(mean <= bound + 1e-12) and np.all(exact >= 1.0))
        np.testing.assert_allclose(rules.normalisers(PoseIkDesign(normaliser="none"), lever), 1.0)
        np.testing.assert_allclose(rules.normalisers(PoseIkDesign(task="orientation"), lever), 1.0)   # no position: nothing to divide by


class CascadeTests(unittest.TestCase):
    def test_shares_add_up_to_turn_and_offset_less_what_is_left(self):
        rng = np.random.default_rng(3)
        for name in MODEL_NAMES:
            chain = get_chain(name)
            lever = rules.levers(chain)
            for normaliser in ("mean", "bound", "exact", "none"):
                design = PoseIkDesign(normaliser=normaliser)
                q = chain.random_pose(rng)
                axes, move = so3.joint_axes(chain, q), rules.moves(chain, q)
                e, d = rng.normal(0.0, 0.3, 3), rng.normal(0.0, 0.2, 3)
                caps = rng.uniform(0.05, 0.4, chain.n)
                shares, rests = rules.pose_cascade(axes, move, e, d, design, lever, caps)
                np.testing.assert_allclose(axes.T @ shares, e - rests[-1, :3], atol=1e-9)
                np.testing.assert_allclose(move.T @ shares, d - rests[-1, 3:], atol=1e-9)
                np.testing.assert_allclose(rests[0], np.concatenate([e, d]), atol=0)
                self.assertTrue(np.all(np.abs(shares) <= caps + 1e-12))

    def test_the_exact_share_leaves_the_least(self):
        rng = np.random.default_rng(4)
        chain = get_chain("tiago7")
        design = PoseIkDesign(normaliser="exact")
        lever = rules.levers(chain)
        for _ in range(20):
            q = chain.random_pose(rng)
            axes, move = so3.joint_axes(chain, q), rules.moves(chain, q)
            e, d = rng.normal(0.0, 0.1, 3), rng.normal(0.0, 0.1, 3)
            shares, rests = rules.pose_cascade(axes, move, e, d, design, lever)
            first = rules.left_size(rests[1], design)
            for other in (0.9 * shares[0], 1.1 * shares[0]):                              # any other share of the first joint leaves more
                rest = np.concatenate([e - axes[0] * other, d - move[0] * other])
                self.assertLess(first, rules.left_size(rest, design) + 1e-12)
            sizes = [rules.left_size(rest, design) for rest in rests]
            self.assertTrue(np.all(np.diff(sizes) <= 1e-12))                              # what is left never grows down the arm

    def test_orientation_alone_is_the_cascade_of_the_so3_repo(self):
        rng = np.random.default_rng(5)
        chain = get_chain("generic7")
        q = chain.random_pose(rng)
        axes, e, caps = so3.joint_axes(chain, q), rng.normal(0.0, 0.3, 3), np.full(chain.n, 0.25)
        shares, rests = rules.pose_cascade(axes, rules.moves(chain, q), e, rng.normal(size=3), PoseIkDesign(task="orientation"),
                                           rules.levers(chain), caps)
        reference, reference_rests = so3.cascade(axes, e, caps)
        np.testing.assert_allclose(shares, reference, atol=1e-12)
        np.testing.assert_allclose(rests[:, :3], reference_rests, atol=1e-12)
        np.testing.assert_allclose(rests[:, 3:], 0.0, atol=0)

    def test_position_alone_leaves_the_turn_out(self):
        rng = np.random.default_rng(6)
        chain = get_chain("tiago4")
        q = chain.random_pose(rng)
        design = PoseIkDesign(task="position")
        with_turn, _ = rules.pose_cascade(so3.joint_axes(chain, q), rules.moves(chain, q), [0.4, -0.2, 0.3], [0.05, 0.02, -0.03], design, rules.levers(chain))
        without, rests = rules.pose_cascade(so3.joint_axes(chain, q), rules.moves(chain, q), [0.0, 0.0, 0.0], [0.05, 0.02, -0.03], design, rules.levers(chain))
        np.testing.assert_allclose(with_turn, without, atol=1e-12)
        np.testing.assert_allclose(rests[:, :3], 0.0, atol=0)

    def test_without_the_cascade_every_joint_gets_the_whole(self):
        rng = np.random.default_rng(7)
        chain = get_chain("generic4")
        design = PoseIkDesign(cascade=False)
        q = chain.random_pose(rng)
        axes, move, lever = so3.joint_axes(chain, q), rules.moves(chain, q), rules.levers(chain)
        e, d = rng.normal(0.0, 0.1, 3), rng.normal(0.0, 0.05, 3)
        shares, rests = rules.pose_cascade(axes, move, e, d, design, lever)
        expected = (axes @ e + design.move_weight * (move @ d)) / rules.normalisers(design, lever)
        np.testing.assert_allclose(shares, expected, atol=1e-12)
        np.testing.assert_allclose(rests[-1], np.concatenate([e, d]), atol=0)

    def test_a_share_fades_at_a_joint_limit(self):
        chain = get_chain("tiago7")
        design = PoseIkDesign()
        q = chain.limits().mean(axis=1)
        q[1] = chain.joints[1].upper                                                  # joint 2 at its upper limit
        room = so3.limit_room(chain, q, design)
        axes, move = so3.joint_axes(chain, q), rules.moves(chain, q)
        up = move[1] / np.linalg.norm(move[1])
        shares, _ = rules.pose_cascade(axes, move, np.zeros(3), 0.1 * up, design, rules.levers(chain), so3.share_caps(chain, design), room)
        self.assertLessEqual(shares[1], 1e-12)                                         # nothing beyond the limit
        back, _ = rules.pose_cascade(axes, move, np.zeros(3), -0.1 * up, design, rules.levers(chain), so3.share_caps(chain, design), room)
        self.assertLess(back[1], -1e-3)                                                # and all of it away from the limit


class LoopTests(unittest.TestCase):
    def test_goal_travels_on_a_straight_line(self):
        point = rules.travel_position(np.zeros(3), [3.0, 4.0, 0.0], 1.0)
        np.testing.assert_allclose(point, [0.6, 0.8, 0.0], atol=1e-12)
        np.testing.assert_allclose(rules.travel_position(point, [0.6, 0.9, 0.0], 1.0), [0.6, 0.9, 0.0], atol=1e-12)

    def test_first_time_within(self):
        t = np.arange(1, 7) * 0.5
        self.assertEqual(rules.first_time_within(t, [False, True, False, True, True, True]), 2.0)
        self.assertTrue(np.isnan(rules.first_time_within(t, [True, True, True, True, True, False])))
        self.assertEqual(rules.first_time_within(t, [True] * 6), 0.5)

    def test_a_near_pose_goal_is_reached(self):
        for name, task in (("generic7", "pose"), ("tiago7", "pose"), ("tiago4", "position"), ("generic4", "position"), ("generic7", "orientation")):
            chain = get_chain(name)
            design = PoseIkDesign(task=task)
            reached = 0
            for pair in pairs(chain, 6, DEVELOPMENT_SEED, spread=0.4):
                trace = rules.simulate(chain, pair.start, pair.goal, pair.goal_position, design, delay_s=0.25)
                reached += trace.reached(design)
            self.assertGreaterEqual(reached, 5, name)

    def test_the_loop_ends_on_the_goal_whatever_the_motor_does(self):
        chain = get_chain("tiago4")
        design = PoseIkDesign(task="position")
        pair = pairs(chain, 1, DEVELOPMENT_SEED, spread=0.4)[0]
        for motor_scale in (0.7, 1.3):
            trace = rules.simulate(chain, pair.start, pair.goal, pair.goal_position, design, motor_scale=motor_scale, stop_when_settled=False)
            self.assertLess(trace.final_error_reach, 0.01)

    def test_a_readout_that_is_off_shows_as_that_much_error(self):
        chain = get_chain("tiago4")
        design = PoseIkDesign(task="position")
        pair = pairs(chain, 1, DEVELOPMENT_SEED, spread=0.4)[0]
        trace = rules.simulate(chain, pair.start, pair.goal, pair.goal_position, design, position_bias=[0.02, 0.0, 0.0], stop_when_settled=False)
        self.assertAlmostEqual(trace.final_error_reach, 0.02, delta=0.004)

    def test_standard_methods_run_in_the_same_loop(self):
        chain = get_chain("generic7")
        pair = pairs(chain, 1, DEVELOPMENT_SEED, spread=0.4)[0]
        for method in standard.METHODS:
            design = PoseIkDesign(kappa=standard.GAINS[method])
            trace = rules.simulate(chain, pair.start, pair.goal, pair.goal_position, design, rule=standard.RULES[method])
            self.assertLess(trace.final_error_reach, trace.start_error_reach)
            self.assertLess(trace.final_error_deg, trace.start_error_deg)
        dls = rules.simulate(chain, pair.start, pair.goal, pair.goal_position, PoseIkDesign(), rule=standard.dls)
        self.assertTrue(dls.reached(PoseIkDesign()))


class PairTests(unittest.TestCase):
    def test_goals_are_poses_of_the_arm_and_the_seeds_differ(self):
        chain = get_chain("tiago7")
        test = pairs(chain, 5, TEST_SEED, spread=0.8)
        for pair in test:
            np.testing.assert_allclose(kinematics.position(chain, pair.goal_pose), pair.goal_position, atol=1e-12)
            np.testing.assert_allclose(lift_pitch_yaw(chain.rotation(pair.goal_pose)), pair.goal, atol=1e-12)
            self.assertTrue(np.all(np.abs(pair.goal_pose - pair.start) <= 0.8 + 1e-12))
            limits = chain.limits()
            self.assertTrue(np.all(pair.goal_pose >= limits[:, 0] - 1e-12) and np.all(pair.goal_pose <= limits[:, 1] + 1e-12))
        for seed in (DEVELOPMENT_SEED, STUDY_SEED):
            self.assertFalse(np.allclose(pairs(chain, 1, seed, spread=0.8)[0].start, test[0].start))
        self.assertFalse(np.allclose(pairs(chain, 1, TEST_SEED)[0].start, test[0].start))            # anywhere: another stream

    def test_the_pose_design_serves_the_so3_rules(self):
        chain = get_chain("tiago7")
        np.testing.assert_allclose(so3.share_caps(chain, PoseIkDesign()), so3.share_caps(chain, IkDesign()), atol=0)
        q = chain.limits()[:, 1]
        np.testing.assert_allclose(so3.limit_room(chain, q, PoseIkDesign()), so3.limit_room(chain, q, IkDesign()), atol=0)


if __name__ == "__main__":
    unittest.main()
