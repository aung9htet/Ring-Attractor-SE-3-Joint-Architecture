"""Paired statistics and the closed-loop summaries."""

import unittest

import numpy as np

from reaching_benchmark.statistics import censored, mcnemar, paired, success_summary, wilson


class StatisticsTest(unittest.TestCase):
    def test_wilson_contains_the_rate(self):
        low, high = wilson(28, 30)
        self.assertLess(low, 28 / 30)
        self.assertGreater(high, 28 / 30)

    def test_mcnemar_counts_discordant_pairs(self):
        result = mcnemar([True] * 10 + [False] * 2, [True] * 4 + [False] * 8)
        self.assertEqual((result["only_first"], result["only_second"]), (6, 0))
        self.assertLess(result["p"], 0.05)

    def test_paired_median_difference(self):
        a = np.arange(20, dtype=float) + 1.0
        result = paired(a, a - 1.0)
        self.assertAlmostEqual(result["median_difference"], 1.0)
        self.assertLess(result["p"], 0.001)

    def test_failures_count_as_the_trial_length(self):
        self.assertTrue(np.array_equal(censored([1.0, None, float("nan")], 25.0), [1.0, 25.0, 25.0]))

    def test_success_summary(self):
        self.assertEqual(success_summary([True, False, True, True])["reached"], 3)


if __name__ == "__main__":
    unittest.main()
