"""The attractor sheet without NEST: where its neurons sit, its weights, the anchor current, the readout."""

import unittest

import numpy as np

from ring_se3.nest.calibrate import drive_for_speed
from ring_se3.sheet import DIRECTIONS, PLANES, SheetDesign, SheetMap, against, combine


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.map = SheetMap(SheetDesign(size=16))

    def test_places_cover_the_workspace_with_a_margin(self):
        places = self.map.places
        self.assertEqual(places.shape, (256, 2))
        self.assertAlmostEqual(places.min(), -self.map.design.extent + self.map.cell_size / 2)
        self.assertAlmostEqual(places.max(), self.map.design.extent - self.map.cell_size / 2)
        self.assertGreater(self.map.design.extent, 1.0)
        np.testing.assert_allclose(self.map.to_scaled(self.map.to_cells([0.3, -0.7])), [0.3, -0.7], atol=1e-12)

    def test_four_labels_in_equal_numbers(self):
        for label in range(4):
            self.assertEqual(self.map.group(label).size, 64)
        np.testing.assert_array_equal(self.map.directions, DIRECTIONS[self.map.labels])
        self.assertEqual(against(0, +1), 1)        # a motion along +a is opposed by the group labelled -a
        self.assertEqual(against(1, -1), 2)

    def test_weights_excite_near_and_inhibit_around_and_are_shifted_along_the_label(self):
        weights = self.map.weight_matrix()
        pre = int(np.flatnonzero((self.map.cells == [8, 8]).all(axis=1))[0])       # label 0: +a
        self.assertEqual(self.map.labels[pre], 0)
        row = weights[pre].reshape(16, 16)
        peak = np.unravel_index(np.argmax(row), row.shape)
        self.assertEqual(peak, (8 + int(self.map.design.shift), 8))
        self.assertGreater(row.max(), 0)
        self.assertLess(row.min(), 0)

    def test_shifts_of_the_four_groups_cancel(self):
        centres = self.map.cells + self.map.design.shift * self.map.directions
        np.testing.assert_allclose((centres - self.map.cells).sum(axis=0), 0.0, atol=1e-9)


class AnchorTests(unittest.TestCase):
    def test_anchor_current_is_largest_at_the_anchored_place_and_splits_as_documented(self):
        sheet = SheetMap(SheetDesign(size=24))
        rng = np.random.default_rng(0)
        gain = sheet.anchor_gain_per_unit()
        for _ in range(20):
            x = rng.uniform(-1, 1, 2)
            current = sheet.anchor_current(x)
            nearest = np.argmin(np.linalg.norm(sheet.places - x, axis=1))
            self.assertEqual(int(np.argmax(current)), int(nearest))
            self.assertLessEqual(current.max(), sheet.design.anchor_offset + 1e-9)
            # the same current as a weighted sum of x and x² per coordinate plus the fixed bias of every neuron
            moving = gain * (sheet.places @ x - 0.5 * np.sum(x ** 2))
            np.testing.assert_allclose(moving + sheet.anchor_bias(), current, atol=1e-9)


class ReadoutTests(unittest.TestCase):
    def test_centre_of_a_packet(self):
        sheet = SheetMap(SheetDesign(size=24))
        target = np.array([0.31, -0.52])
        counts = np.exp(-np.sum((sheet.places - target) ** 2, axis=1) / (2 * 0.12 ** 2))
        np.testing.assert_allclose(sheet.centre(counts), target, atol=2e-3)
        self.assertTrue(np.all(np.isnan(sheet.centre(np.zeros(sheet.count)))))

    def test_three_sheets_give_each_coordinate_twice(self):
        position = np.array([0.2, -0.4, 0.6])
        estimates = {name: position[[a, b]] for name, a, b in PLANES}
        combined, difference = combine(estimates)
        np.testing.assert_allclose(combined, position)
        np.testing.assert_allclose(difference, 0.0, atol=1e-12)
        estimates["xy"] = estimates["xy"] + [0.02, 0.0]
        combined, difference = combine(estimates)
        np.testing.assert_allclose(combined, position + [0.01, 0, 0])
        np.testing.assert_allclose(difference, [0.02, 0, 0], atol=1e-12)
        estimates["xz"] = np.array([np.nan, np.nan])                      # a silent sheet: the other copy is used
        combined, _ = combine(estimates)
        np.testing.assert_allclose(combined, position + [0.02, 0, 0])


class SpeedCurveTests(unittest.TestCase):
    def test_inverse_of_the_speed_curve(self):
        drive = np.linspace(0, 0.3, 7)
        speed = np.array([0.0, 1.0, 2.0, 1.9, 4.0, 6.0, 8.0])           # one dip: made non-decreasing before inverting
        np.testing.assert_allclose(drive_for_speed(drive, speed, np.array([0.0, 1.0, 6.0])), [0.0, 0.05, 0.25], atol=1e-6)
        self.assertAlmostEqual(float(drive_for_speed(drive, speed, np.array([50.0]))[0]), 0.3)       # beyond the curve: full drive


if __name__ == "__main__":
    unittest.main()
