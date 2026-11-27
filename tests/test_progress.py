"""The progress bar and ``python -m reaching_benchmark status`` (standard library only)."""

import io
import json
import os
import tempfile
import unittest

from reaching_benchmark.progress import Progress, bar, clock, trial_detail
from reaching_benchmark.status import collect, render


class ProgressTest(unittest.TestCase):
    def test_clock_and_bar(self):
        self.assertEqual(clock(3725), "1:02:05")
        self.assertEqual(clock(None), "--:--:--")
        self.assertEqual(bar(0.5, 10), "[#####-----]")
        self.assertIn("3.1 deg", trial_detail(7.3, 25, 3.14, 21.0))

    def test_lines_mode_prints_one_line_per_trial(self):
        stream = io.StringIO()
        progress = Progress("joint_loss lock", 2, stream=stream, mode="lines")
        progress.start(0)
        progress.update("t 1.0/25 s")
        progress.finish(True, "pair 0: reached")
        progress.close()
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertIn("pair 0: reached", lines[0])
        self.assertIn("1/2", lines[0])
        self.assertIn("reached 1", lines[0])

    def test_bar_mode_redraws_in_place(self):
        stream = io.StringIO()
        progress = Progress("demo", 3, stream=stream, mode="bar")
        progress.start(0)
        progress.finish(False)
        progress.close()
        text = stream.getvalue()
        self.assertIn("\r\033[K", text)
        self.assertTrue(text.endswith("\n"))

    def test_a_resumed_run_estimates_from_its_earlier_pace(self):
        progress = Progress("resumed", 30, done=10, wall_before=1000.0, stream=io.StringIO(), mode="off")
        self.assertAlmostEqual(progress.eta(), 2000.0)


class StatusTest(unittest.TestCase):
    def test_a_run_in_progress(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "ring_attractor_kinematics", "joint_loss"))
            document = {"run": {"pairs": 30}, "shard": [0, 1], "planned": 30, "wall_s": 3600.0,
                        "rows": [{"index": i, "reached": i % 2 == 0} for i in range(6)], "skipped": []}
            with open(os.path.join(root, "ring_attractor_kinematics", "joint_loss", "tiago7_seed1_lock.json"), "w") as stream:
                json.dump(document, stream)
            with open(os.path.join(root, "ring_attractor_kinematics", "joint_loss", "summary.json"), "w") as stream:
                json.dump({"groups": {}}, stream)                       # a summary is not a run
            entries = collect(root, [])
            self.assertEqual(len(entries), 1)
            self.assertEqual((entries[0]["done"], entries[0]["planned"], entries[0]["reached"]), (6, 30, 3))
            text = render(entries)
            self.assertIn("6/30", text)
            self.assertIn("left 4:00:00", text)

    def test_older_files_work_out_their_plan_from_the_shard(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "ring_attractor_kinematics", "reach_kinematic"))
            with open(os.path.join(root, "ring_attractor_kinematics", "reach_kinematic", "a_shard1of3.json"), "w") as stream:
                json.dump({"run": {"pairs": 30}, "shard": [1, 3], "wall_s": 10.0, "rows": [{"index": 1, "reached": True}]}, stream)
            self.assertEqual(collect(root, ["reach_kinematic"])[0]["planned"], 10)
            self.assertEqual(collect(root, ["ring_attractor_kinematics/reach_kinematic"])[0]["planned"], 10)


if __name__ == "__main__":
    unittest.main()
