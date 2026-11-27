"""The analysis package: every figure is listed, and Fig. 3 redrawn from results/ shows the reported counts."""

import csv
import unittest

try:
    import matplotlib  # noqa: F401
    HAVE_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAVE_MATPLOTLIB = False


@unittest.skipUnless(HAVE_MATPLOTLIB, "needs Matplotlib (the model image)")
class AnalysisTest(unittest.TestCase):
    def test_every_figure_is_listed(self):
        from analysis.__main__ import figure_table

        names = [name for name, _, _ in figure_table()]
        self.assertEqual(len(names), len(set(names)))
        for name in ("fig1", "fig2", "fig3", "fig4", "fig5", "supp_screening"):
            self.assertIn(name, names)

    def test_fig3_counts(self):
        from analysis.__main__ import run_figures
        from analysis.common import EXPORT

        run_figures(["fig3"])
        with open(EXPORT / "fig3_gazebo_success.csv", encoding="utf-8") as stream:
            reached = {row["controller"]: int(row["reached"]) for row in csv.DictReader(stream)}
        self.assertEqual(reached, {"rak": 28, "rak_pe": 27, "dls_ik": 30, "ccd_ik": 30, "mpc_nn_dynamics": 29, "pets": 30})


if __name__ == "__main__":
    unittest.main()
