"""Neuron budgets counted from the wiring equal the networks NEST builds (the totals of the golden master)."""

import unittest

from ring_attractor_kinematics.config import IkDesign, default_task
from reaching_benchmark.geometry.chains import get_chain
from ring_attractor_kinematics.network.budget import TECHNIQUES, forward_budget, neuron_budget, observer_budget

#: neuron_count() of the built pose IK networks (old method) and of the forward networks alone
TOTALS = {"tiago4": 35916, "tiago7": 69036, "generic7": 71976}
FORWARD = {"tiago4": 14968, "tiago7": 24208}


class BudgetTest(unittest.TestCase):
    def budget(self, arm, method="matrix"):
        chain = get_chain(arm)
        return chain, neuron_budget(chain, IkDesign(task=default_task(chain.n)), method)

    def test_totals_equal_the_built_networks(self):
        for arm, total in TOTALS.items():
            _, budget = self.budget(arm)
            self.assertEqual(budget["total"], total, arm)
            self.assertEqual(sum(budget[t] for t in TECHNIQUES), total, arm)

    def test_the_new_method_adds_one_motor_ring_per_joint(self):
        for arm in TOTALS:
            chain, old = self.budget(arm)
            _, new = self.budget(arm, "lead")
            self.assertEqual(new["total"] - old["total"], chain.n * 200, arm)
            self.assertEqual(new["motor rings"], chain.n * 200, arm)

    def test_forward_network_alone(self):
        for arm, total in FORWARD.items():
            self.assertEqual(forward_budget(get_chain(arm)), total, arm)

    def test_the_observer_method_adds_its_observer(self):
        for arm in ("tiago7", "generic7"):
            chain, old = self.budget(arm)
            _, observer = self.budget(arm, "observer")
            parts = observer_budget(chain)
            self.assertEqual(parts["total"], sum(v for k, v in parts.items() if k != "total"))
            self.assertEqual(observer["total"] - old["total"], parts["total"], arm)
            self.assertEqual(observer["orientation observer"], parts["total"], arm)


if __name__ == "__main__":
    unittest.main()
