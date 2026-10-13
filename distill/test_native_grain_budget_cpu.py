import unittest

import numpy as np

from distill.native_grain_budget_cpu import projection, noise_regression


class GrainBudgetTests(unittest.TestCase):
    def test_projection_retains_correlated_part_with_nonzero_crop_mean(self):
        target = np.array([1., 2., 3., 4.])
        orthogonal = np.array([1., -1., -1., 1.])
        actual = .8*target+orthogonal
        result = projection(actual, target)
        self.assertAlmostEqual(result['correlated_slope'], .8)
        self.assertAlmostEqual(result['orthogonal_rms_units'], 1.)
        self.assertAlmostEqual(result['orthogonal_energy_ratio'], 1/7.5)
        self.assertNotAlmostEqual(result['centered_correlation']*result['rms_ratio'], .8)

    def test_noise_regression_controls_teacher_gain_and_intercept(self):
        teacher = np.array([-2., -1., 0., 1., 2.])
        noise = np.array([1., -1., 0., -1., 1.])
        error = .3*teacher+.7*noise+2.
        result = noise_regression(error, teacher, noise)
        self.assertAlmostEqual(result['partial_r2_noise'], 1.)
        self.assertAlmostEqual(result['noise_coefficient'], .7)
        self.assertAlmostEqual(result['teacher_coefficient'], .3)

    def test_zero_energy_reference_is_unavailable(self):
        self.assertFalse(projection(np.ones(8), np.zeros(8))['available'])


if __name__ == '__main__':
    unittest.main()
