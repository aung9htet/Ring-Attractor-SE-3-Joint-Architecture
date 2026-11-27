"""The shared benchmark and the non-learned baselines, without NEST, ROS or a GPU: the shared limits are the RAK model's,
DLS IK is the rule it was, CCD IK takes the best turn per joint and reaches, the shared runner writes, resumes and
summarises, and the comparison pairs every model with the RAK model."""

import os
import tempfile
import unittest

import numpy as np

from ccd_ik import rule as ccd_rule
from ccd_ik.controller import make_loop as make_ccd_loop
from dls_ik import rule as dls_rule
from dls_ik.controller import make_loop
from jacobian_transpose_ik import rule as transpose_rule
from reaching_benchmark import compare, results, summary
from reaching_benchmark.benchmark import run_reach, test_pairs
from reaching_benchmark.geometry import kinematics
from reaching_benchmark.geometry.chains import get_chain, workspace
from reaching_benchmark.limits import faded, limit_room, step_caps
from reaching_benchmark.plants.light_robot import LightRobot
from reaching_benchmark.task import IkDesign, default_task
from ring_attractor_kinematics.neurons.ring_layout import RingMap
from ring_attractor_kinematics.rules.cascade_rule import share_caps


def old_dls(axes, move, turn_, offset_, design, lever, caps, room=None, damping=0.1):
    """DLS as it was before the repository was split into one package per model (kept here to prove the move changed
    nothing)."""

    ra, rb = np.sqrt(design.turn_weight), np.sqrt(design.move_weight)
    jacobian = np.concatenate([ra * np.asarray(axes, dtype=float).T, rb * np.asarray(move, dtype=float).T])
    error = np.concatenate([ra * np.asarray(turn_, dtype=float), rb * np.asarray(offset_, dtype=float)])
    caps = np.asarray(caps, dtype=float)
    toward = jacobian.T @ error
    weights = np.ones(caps.size) if room is None else np.array([room[k, 1] if toward[k] > 0 else room[k, 0] for k in range(caps.size)])
    weighted = jacobian * weights[None, :]
    shares = weighted.T @ np.linalg.solve(weighted @ weighted.T + damping ** 2 * np.eye(6), error)
    scale = float(np.max(np.abs(shares) / caps)) if shares.size else 0.0
    shares = shares / scale if scale > 1.0 else shares
    if room is None:
        return shares
    return np.array([faded(shares[k], caps[k], room[k, 0], room[k, 1], "scale") for k in range(shares.size)])


class SharedLimitsTest(unittest.TestCase):
    def test_caps_and_room_are_the_ring_maps(self):
        for name in ("tiago7", "generic7", "tiago4"):
            chain = get_chain(name)
            design = IkDesign(task=default_task(chain.n))
            maps = [RingMap.for_joint(j, 200) for j in chain.joints]
            np.testing.assert_array_equal(step_caps(chain, design), [design.lead_indices * m.radians_per_index for m in maps])
            np.testing.assert_array_equal(share_caps(chain, design, 200), step_caps(chain, design))
            rng = np.random.default_rng(3)
            for _ in range(20):
                q = chain.random_pose(rng)
                expected = np.ones((chain.n, 2))
                for k, (joint, m) in enumerate(zip(chain.joints, maps)):
                    if not joint.circular:
                        taper = max(design.limit_taper_indices * m.radians_per_index, 1e-9)
                        expected[k] = [np.clip((q[k] - joint.lower) / taper, 0, 1), np.clip((joint.upper - q[k]) / taper, 0, 1)]
                np.testing.assert_array_equal(limit_room(chain, q, design), expected)


class RuleTest(unittest.TestCase):
    def test_dls_is_the_rule_it_was(self):
        chain = get_chain("tiago7")
        design = IkDesign()
        rng = np.random.default_rng(5)
        caps = step_caps(chain, design)
        for _ in range(50):
            q = chain.random_pose(rng)
            axes, move = kinematics.joint_axes(chain, q), kinematics.moves(chain, q)
            turn_, offset_ = rng.normal(0, 0.2, 3), rng.normal(0, 0.1, 3)
            room = limit_room(chain, q, design)
            np.testing.assert_array_equal(dls_rule.dls(axes, move, turn_, offset_, design, None, caps, room),
                                          old_dls(axes, move, turn_, offset_, design, None, caps, room))
            self.assertTrue(np.all(np.abs(transpose_rule.steps(axes, move, turn_, offset_, design, caps, room)) <= caps + 1e-12))

    def test_dls_reaches_on_the_kinematic_plant(self):
        chain = get_chain("tiago7")
        design = IkDesign()
        loop = make_loop(chain, design, LightRobot(chain))
        for pair in test_pairs(chain, 3):
            loop.place(pair.start)
            trial = loop.reach(pair.goal, pair.goal_position)
            self.assertTrue(trial.reached(design), "pair %d: %.2f deg, %.4f reach" % (pair.index, trial.final_error_deg, trial.final_error_reach))


class RunnerTest(unittest.TestCase):
    def test_run_resume_summary_and_comparison(self):
        chain = get_chain("tiago7")
        design = IkDesign()
        saved = results.ROOT
        with tempfile.TemporaryDirectory() as root:
            results.ROOT = root
            try:
                pairs = test_pairs(chain, 2)
                run = {"arm": "tiago7", "pairs": 2, "kappa": 1.0, "plant": "light", "kind": "near"}
                path = run_reach("dls_ik", "reach_kinematic", "tiago7", run, make_loop(chain, design, LightRobot(chain)), pairs,
                                 {"name": "DLS IK"})
                document = results.read_run("dls_ik", "reach_kinematic", "tiago7")
                self.assertEqual([r["index"] for r in document["rows"]], [0, 1])
                written = os.path.getmtime(path)
                run_reach("dls_ik", "reach_kinematic", "tiago7", run, make_loop(chain, design, LightRobot(chain)), pairs, {"name": "DLS IK"})
                self.assertEqual(os.path.getmtime(path), written)                  # resumed: nothing left to run
                dls = summary.summarise("dls_ik", "reach_kinematic")
                self.assertEqual(dls["runs"]["tiago7"]["success"]["n"], 2)
                results.write_summary("dls_ik", "reach_kinematic", dls)
                # a RAK summary in the shape reach_summary writes, with one trial missed
                network = {k: dls["runs"]["tiago7"][k] for k in ("success", "time_median_s", "final_error_median_deg",
                                                                  "final_error_median_reach", "path_ratio_median", "per_trial")}
                network = dict(network, per_trial=dict(network["per_trial"], reached=[True, False]))
                results.write_summary("ring_attractor_kinematics", "reach_kinematic", {"groups": {"tiago7_pose_k1": {
                    "arm": "tiago7", "task": "pose", "kappa": 1.0, "condition": None, "network": network}}})
                table = compare.compare("reach_kinematic")
            finally:
                results.ROOT = saved
        labels = [c["label"] for c in table["controllers"]]
        self.assertEqual(labels, ["RAK", "DLS IK"])
        self.assertEqual(table["controllers"][1]["against_rak"]["reached"], [1, 2])


class CcdTest(unittest.TestCase):
    """CCD IK: each joint's closed-form angle is the best one, and it reaches on the kinematic plant."""

    def test_the_closed_form_angle_is_the_best_turn(self):
        for name in ("tiago7", "generic7"):
            chain = get_chain(name)
            design = IkDesign()
            radius = workspace(chain).radius
            rng = np.random.default_rng(11)
            grid = np.linspace(-np.pi, np.pi, 20001)
            for _ in range(10):
                q, goal = chain.random_pose(rng), chain.random_pose(rng)
                goal_r, goal_p = chain.rotation(goal), kinematics.position(chain, goal)
                k = int(rng.integers(chain.n))
                origin = kinematics.joint_origins(chain, q)[k]

                def objective(phi):
                    moved = q.copy()
                    moved[k] += phi
                    return (design.turn_weight * np.trace(goal_r.T @ chain.rotation(moved))
                            + 2 * design.move_weight / radius ** 2 * (goal_p - origin) @ (kinematics.position(chain, moved) - origin))
                best = ccd_rule.best_angle(kinematics.joint_axes(chain, q)[k], origin, chain.rotation(q), kinematics.position(chain, q),
                                           goal_r, goal_p, design, radius)
                searched = grid[int(np.argmax([objective(phi) for phi in grid]))]
                self.assertAlmostEqual(float(np.angle(np.exp(1j * (best - searched)))), 0.0, places=3)

    def test_ccd_reaches_on_the_kinematic_plant(self):
        chain = get_chain("tiago7")
        design = IkDesign()
        loop = make_ccd_loop(chain, design, LightRobot(chain))
        for pair in test_pairs(chain, 3):
            loop.place(pair.start)
            trial = loop.reach(pair.goal, pair.goal_position)
            self.assertTrue(trial.reached(design), "pair %d: %.2f deg, %.4f reach" % (pair.index, trial.final_error_deg, trial.final_error_reach))

if __name__ == "__main__":
    unittest.main()
