"""The report tool: statistics, the delay estimate, the gap summary, the tables, and the number and link checks of the documents."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import check_doc_numbers  # noqa: E402
import report  # noqa: E402


def tracking_document(delay_ticks, radius=0.5):
    """An end point on a circle, 4 s moving then 1 s at rest, over and over; the readout is the truth, late."""

    true, moving, phase = [], [], 0.0
    for _ in range(4):
        for i in range(100):
            is_moving = i < 80
            phase += 0.05 if is_moving else 0.0
            true.append([0.3 * np.cos(phase), 0.3 * np.sin(phase), 0.1 * np.sin(2 * phase)])
            moving.append(is_moving)
    ticks = []
    for i, (p, m) in enumerate(zip(true, moving)):
        late = true[max(i - delay_ticks, 0)]
        ticks.append({"t": 0.05 * i, "moving": m, "true_position": p, "position": late, "position_tick": late, "cells_position": p,
                      "true_velocity": [0.3, 0.0, 0.0] if m else [0.0, 0.0, 0.0], "velocity": [0.33, 0.0, 0.0], "error_deg": 1.0})
    return {"radius_m": radius, "ticks": ticks}


def gap_document(drift_per_tick, recover_ticks, radius=0.5):
    """One gap of 2 s with the arm at rest: the readout drifts along x, and is back some ticks after the gap."""

    ticks = []

    def add(phase, error_x):
        ticks.append({"phase": phase, "gap": 0, "gap_s": 2.0, "true_position": [0.1, 0.2, 0.3], "position_tick": [0.1 + error_x, 0.2, 0.3],
                      "tick_error_m": abs(error_x), "sheet_spikes": [20, 20, 20]})

    for _ in range(40):
        add("before", 0.004)
    for i in range(40):
        add("gap", 0.004 + drift_per_tick * (i + 1))
    for i in range(40):
        add("after", 0.2 if i < recover_ticks else 0.004)
    return {"radius_m": radius, "ticks": ticks}


class StatisticsTests(unittest.TestCase):
    def test_stats_ignore_missing_values(self):
        self.assertEqual(report.stats([1.0, None, 3.0, float("nan"), 2.0]), (2.0, 2.8, 3.0))
        self.assertTrue(report.similar([1.0] * 9 + [3.9]))
        self.assertFalse(report.similar([2.1] * 10))
        self.assertFalse(report.similar([1.0] * 8 + [5.0, 5.0]))

    def test_delay_of_a_readout_that_is_late_by_a_known_time(self):
        for ticks in (0, 3, 6):
            summary = report.tracking_summary(tracking_document(ticks))
            self.assertAlmostEqual(summary["delay_s"], 0.05 * ticks, delta=0.011)
            self.assertLess(summary["lagged_moving"], 0.05)
        late = report.tracking_summary(tracking_document(4))
        self.assertGreater(late["error_moving"][0], 5.0)             # 4 ticks x 0.3 m/s over a reach of 0.5 m: about 12%
        self.assertLess(late["after_stop"][0], 1e-9)
        self.assertAlmostEqual(report.velocity_error(tracking_document(0))[0], 10.0, delta=0.5)

    def test_gap_summary(self):
        gap, = report.dropout_summary(gap_document(drift_per_tick=0.0005, recover_ticks=3))
        self.assertTrue(gap["alive"])
        self.assertAlmostEqual(gap["packet_moved"], 100.0 * (0.0005 * 38) / 0.5, delta=0.05)       # mean of the last five ticks
        self.assertAlmostEqual(gap["true_moved"], 0.0)
        self.assertAlmostEqual(gap["recover_s"], 0.15)
        self.assertAlmostEqual(gap["level"], 0.8)


class TableTests(unittest.TestCase):
    def _static(self, name, errors_mm):
        rows = [{"kind": "random", "index": i, "position_error_m": e / 1000.0, "cells_error_m": e / 2000.0, "held_position_error_m": e / 4000.0,
                 "radius_m": 0.5, "disagreement_m": [0.001, -0.002, 0.0005], "sheet_error_m": [0.004, 0.005, 0.006], "sheet_spikes": [200, 210, 220],
                 "error_deg": 1.0 + 0.1 * i} for i, e in enumerate(errors_mm)]
        return {"chain": name, "joints": 2, "radius_m": 0.5, "neurons": 11276, "so3_neurons": 1860, "pose_products": 8, "pose_sums": 9,
                "synapses": 11000000, "sheet": {"size": 48}, "rows": rows}

    def test_tables_are_written_from_the_data_and_the_number_check_reads_them(self):
        with tempfile.TemporaryDirectory() as root:
            data = os.path.join(root, "docs", "results", "data")
            os.makedirs(os.path.join(data, "static"))
            with open(os.path.join(data, "static", "generic2.json"), "w", encoding="utf-8") as stream:
                json.dump(self._static("generic2", [5.0, 10.0, 15.0, 20.0]), stream)
            path, written = report.build(report.load(data), os.path.join(root, "docs", "results"), figures=False)
            with open(path, encoding="utf-8") as stream:
                text = stream.read()
            self.assertIn("| generic2 | 4 | 2.50 / 3.70 / 4.00 | 12.5 / 18.5 / 20.0 |", text)          # percent of a reach of 0.5 m, and mm
            self.assertIn("| generic2 | 2 | 0.500 | 1,860 | 8 | 9 | 11,276 | 11,000,000 |", text)
            self.assertIn("| generic2 | 1.25 / 1.85 | 2.50 / 3.70 | 0.62 / 0.93 |", text)
            self.assertIn("| 4. Held poses: position |", text)
            self.assertIn("0 of 1 models", text)                                                          # the median is above 2%
            self.assertEqual(written, [])
            with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as stream:
                stream.write("# x\n\n## 1. Result\n\nThe median is 2.50% of the reach, p90 3.70%; `9.99` in code is ignored.\n\n## 2. Setup\n\n7.77 is outside.\n")
            self.assertEqual(check_doc_numbers.check(root), (2, []))
            with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as stream:
                stream.write("## 1. Result\n\nThe median is 2.51% of the reach, in 55% of the poses.\n\n## 2. Setup\n")
            self.assertEqual(check_doc_numbers.check(root), (2, [("README.md", "2.51%"), ("README.md", "55%")]))

    def test_documents_of_this_repo_quote_only_numbers_from_the_tables(self):
        if not os.path.exists(os.path.join(ROOT, check_doc_numbers.TABLES)):
            self.skipTest("docs/results/tables.md is not generated yet")
        checked, missing = check_doc_numbers.check(ROOT)
        self.assertEqual(missing, [], "run scripts/report.py, then correct the documents")


class DocumentTests(unittest.TestCase):
    def test_links_and_named_files_in_the_documents_exist(self):
        import glob
        import re

        documents = [os.path.join(ROOT, "README.md")] + sorted(glob.glob(os.path.join(ROOT, "docs", "**", "*.md"), recursive=True))
        problems = []
        for path in documents:
            with open(path, encoding="utf-8") as stream:
                text = stream.read()
            folder = os.path.dirname(path)
            for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
                if target.startswith(("http://", "https://")):
                    continue
                if not os.path.exists(os.path.normpath(os.path.join(folder, target))):
                    problems.append("%s -> %s" % (os.path.relpath(path, ROOT), target))
            for name in re.findall(r"`((?:ring_so3|ring_se3|scripts|tests|docker|docs|artifacts)/[\w./-]+\.(?:py|md|bash|env|json|png|mp4))`", text):
                if "<" in name or "*" in name:
                    continue
                if not os.path.exists(os.path.join(ROOT, name)):
                    problems.append("%s names %s" % (os.path.relpath(path, ROOT), name))
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
