"""A fitted gain repairs amplitude but cannot establish positional fidelity."""
import unittest

import numpy as np

from distill.tile40_amplitude_probe import fit_gain


class GainTests(unittest.TestCase):
    def test_gain_recovers_scale_without_fitting_a_height_offset(self):
        reference = np.arange(256).reshape(16, 16).astype(float)
        actual = reference*.5+300.
        self.assertAlmostEqual(fit_gain([(actual, reference)]), 2.)

    def test_equal_scene_weighting_does_not_use_only_the_largest_mountain(self):
        reference = np.arange(256).reshape(16, 16).astype(float)
        self.assertAlmostEqual(fit_gain([(reference*.5, reference), (reference*50., reference*100.)]), 2.)

    def test_flat_reversed_or_nonfinite_prediction_is_not_a_gain_fix(self):
        reference = np.arange(256).reshape(16, 16).astype(float)
        for actual in (np.zeros_like(reference), -reference, reference*np.nan):
            with self.subTest(actual=actual[0, 0]), self.assertRaises(ValueError):
                fit_gain([(actual, reference)])


if __name__ == '__main__':
    unittest.main()
