"""Re-scoring a stored trial at other tolerances (reaching_benchmark.rescore)."""

import unittest

from reaching_benchmark.rescore import rescore_document, rescore_row
from reaching_benchmark.task import IkDesign


def row(final_reach):
    return {"t": [0.0, 0.2, 0.4, 0.6], "error_deg": [20.0, 4.0, 6.0, 3.0], "error_reach": [0.2, 0.06, 0.045, 0.045],
            "duration_s": 0.65, "final_error_deg": 2.0, "final_error_reach": final_reach}


class Rescore(unittest.TestCase):
    def test_reached_and_entry(self):
        design = {"task": "pose", "reach_deg": 5.0, "reach_reach": 0.05}
        r = row(0.045)
        rescore_row(r, design)
        self.assertTrue(r["reached"])
        self.assertAlmostEqual(r["time_to_reach_s"], 0.6)    # 0.4 is outside in orientation
        r = row(0.045)
        rescore_row(r, dict(design, reach_reach=0.04))
        self.assertFalse(r["reached"])

    def test_document_takes_current_tolerance(self):
        document = {"design": dict(IkDesign().describe(), reach_reach=0.04), "rows": [row(0.045)]}
        self.assertTrue(rescore_document(document))
        self.assertEqual(document["design"]["reach_reach"], IkDesign().reach_reach)
        self.assertEqual(document["scoring"]["run_with"]["reach_reach"], 0.04)
        self.assertTrue(document["rows"][0]["reached"])
        self.assertFalse(rescore_document(document))          # a second time changes nothing


if __name__ == "__main__":
    unittest.main()
