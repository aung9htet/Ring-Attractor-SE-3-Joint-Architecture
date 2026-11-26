"""The exact cascade, the standard methods and the orientation sources of the new method."""

import unittest

import numpy as np

from ring_attractor_kinematics.config import IkDesign
from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.geometry.rotations import from_lift_pitch_yaw, lift_pitch_yaw
from ring_attractor_kinematics.rules import observer_rule
from ring_attractor_kinematics.rules.references import RULES
from ring_attractor_kinematics.rules.cascade_rule import limit_room, share_caps, simulate
from reaching_benchmark.targets import STUDY_SEED, pose_pairs


class CascadeTest(unittest.TestCase):
    def test_reaches_near_goals(self):
        chain = get_chain("tiago7")
        design = IkDesign(task="pose")
        reached = [simulate(chain, p.start, p.goal, p.goal_position, design).reached(design) for p in pose_pairs(chain, 8, STUDY_SEED)]
        self.assertGreaterEqual(sum(reached), 7)

    def test_no_share_beyond_a_limit(self):
        chain = get_chain("tiago4")
        design = IkDesign(task="position")
        q = chain.limits()[:, 1].copy()
        self.assertTrue(np.all(limit_room(chain, q, design)[:, 1] == 0.0))

    def test_standard_methods_reduce_the_error(self):
        chain = get_chain("generic7")
        design = IkDesign(task="pose")
        pair = pose_pairs(chain, 1, STUDY_SEED)[0]
        for name, rule in RULES.items():
            trace = simulate(chain, pair.start, pair.goal, pair.goal_position, design, rule=rule)
            self.assertLess(trace.final_error_deg, 0.5 * trace.start_error_deg, name)

    def test_caps_are_eight_ring_indices(self):
        chain = get_chain("generic3")
        self.assertTrue(np.allclose(share_caps(chain, IkDesign()), 8 * 2 * np.pi / 200))


class ObserverTest(unittest.TestCase):
    def test_chart_rates_invert_the_euler_rate_map(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            lpy = rng.uniform([-3, -1.0, -3], [3, 1.0, 3])
            rates = rng.normal(size=3)
            omega = observer_rule.euler_rate_map(lpy) @ rates
            self.assertTrue(np.allclose(observer_rule.chart_rates(lpy, omega, bound=1e9), rates, atol=1e-9))

    def test_two_charts_have_disjoint_poles(self):
        pole = from_lift_pitch_yaw(0.3, np.pi / 2, 1.1)
        pitches = [lift_pitch_yaw(u.T @ pole)[1] for u in observer_rule.CHARTS]
        self.assertAlmostEqual(abs(pitches[0]), np.pi / 2, places=6)
        self.assertLess(abs(pitches[1]), 1e-6)

    def test_predicting_observers_follow_a_move_without_lag_or_noise(self):
        chain = get_chain("tiago7")
        rng = np.random.default_rng(5)
        path = observer_rule.smooth_path(chain, chain.random_pose(rng), [chain.random_pose(rng) for _ in range(2)], 0.5)
        settings = observer_rule.SourceSettings(cue_noise_deg=0.0)
        for code in ("M5", "MQ"):
            errors = observer_rule.open_loop(observer_rule.make_source(code, chain, 0.0, settings), chain, path)["error_deg"]
            self.assertLess(np.median(errors), 0.2, code)

    def test_lagging_matrix_is_late_by_its_delay(self):
        chain = get_chain("tiago7")
        rng = np.random.default_rng(6)
        path = observer_rule.smooth_path(chain, chain.random_pose(rng), [chain.random_pose(rng)], 1.0)
        source = observer_rule.make_source("M0", chain, 0.3, observer_rule.SourceSettings(cue_noise_deg=0.0))
        result = observer_rule.open_loop(source, chain, path)
        late = [np.degrees(np.arccos(np.clip((np.trace(result["estimates"][t].T @ chain.rotation(path[t - 6])) - 1) / 2, -1, 1)))
                for t in range(6, len(path))]
        self.assertLess(max(late), 1e-4)

    def test_every_source_runs_in_the_closed_loop(self):
        chain = get_chain("generic7")
        design = IkDesign(task="pose")
        pair = pose_pairs(chain, 1, STUDY_SEED)[0]
        for code in observer_rule.CODES:
            trace = simulate(chain, pair.start, pair.goal, pair.goal_position, design, delay_s=0.3,
                             orientation=observer_rule.make_source(code, chain, 0.3))
            self.assertTrue(np.all(np.isfinite(trace.estimate_error_deg)), code)


if __name__ == "__main__":
    unittest.main()
