"""Sketch units, climate counterfactuals and full40km data split invariants."""
import unittest

import numpy as np

from distill.tile40_conditioning import CLIMATES, SketchConditioning, cases, sample_sketch, sketch


class ConditioningTests(unittest.TestCase):
    def test_physical_altitude_is_preserved_without_per_tile_normalization(self):
        values = np.array([[100., 200.], [300., 400.]], np.float32)
        actual = sample_sketch(values, np.array([0., 20000., 40000.]), np.array([0., 40000.]))
        np.testing.assert_array_equal(actual, [[100., 150., 200.], [300., 350., 400.]])
        doubled = sample_sketch(values*2, np.array([20000.]), np.array([20000.]))
        self.assertEqual(float(doubled[0, 0]), 500.)

    def test_changing_climate_preserves_sketch_and_changes_physical_conditioning(self):
        values = sketch(10, 'mountains')
        fields = [SketchConditioning(values, climate).sample(np.array([0., 20000.]), np.array([20000.]))
                  for climate in CLIMATES]
        for value in fields[1:]:
            np.testing.assert_array_equal(value[0], fields[0][0])
            self.assertFalse(np.array_equal(value[1:], fields[0][1:]))

    def test_coarse_teacher_input_uses_signed_sqrt_once_and_preserves_climate_units(self):
        conditioning = SketchConditioning(np.full((2, 2), 1600., np.float32), 'arid')
        raw = conditioning.sample_raw(0, 0, 2, 2)
        model = conditioning(0, 0, 2, 2).numpy()
        np.testing.assert_array_equal(model[0], np.full((2, 2), 40.))
        np.testing.assert_array_equal(model[1:], raw[1:])
        self.assertEqual(float(raw[2, 0, 0]), 700.)

    def test_splits_are_by_geometry_seed_and_keep_all_five_climates_together(self):
        rows = cases()
        seeds = {split:{row['seed'] for row in rows if row['split'] == split} for split in ('train', 'val', 'test')}
        self.assertFalse(seeds['train'] & seeds['val'])
        self.assertFalse(seeds['train'] & seeds['test'])
        self.assertFalse(seeds['val'] & seeds['test'])
        self.assertEqual(sum(row['split'] == 'test' for row in rows), 30)
        for seed in set.union(*seeds.values()):
            self.assertEqual({row['climate'] for row in rows if row['seed'] == seed}, set(CLIMATES))

    def test_relief_contrast_is_present_before_neural_refinement(self):
        spans = [np.ptp(sketch(10, relief)) for relief in ('plain', 'hills', 'mountains')]
        self.assertGreater(spans[1], spans[0]*10)
        self.assertGreater(spans[2], spans[1]*4)

    def test_macro_interpolation_has_no_slope_jump_or_height_overshoot(self):
        values = np.tile([100., 400., 200., 300.], (4, 1)).astype(np.float32)
        x = np.linspace(0., 40000., 1001)
        sampled = sample_sketch(values, x, np.array([20000.]))[0]
        self.assertGreaterEqual(float(sampled.min()), 100.)
        self.assertLessEqual(float(sampled.max()), 400.)
        knot = 40000./3
        near = sample_sketch(values, np.array([knot-1., knot, knot+1.]), np.array([20000.]))[0]
        self.assertLess(abs(float((near[2]-near[1])-(near[1]-near[0]))), .0001)


if __name__ == '__main__':
    unittest.main()
