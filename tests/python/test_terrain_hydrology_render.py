"""Small fixtures for the CPU physical water appearance contract."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import unittest

import numpy as np

import terrain_hydrology_render as hr


def tile(level=0., coverage=0., kind=hr.DRY, valid=1., shape=(6, 8)):
    water = np.zeros(shape+(4,), np.float32)
    water[...] = level, coverage, kind, valid
    return water


class HydrologyRenderTests(unittest.TestCase):
    def setUp(self):
        self.elevation = np.linspace(-40, 60, 48, dtype=np.float32).reshape(6, 8)
        self.rgb = np.random.default_rng(3).integers(0, 256, (2, 4, 3), dtype=np.uint8)

    def test_legacy_appearance_without_authority_or_for_ocean(self):
        for water in (None, tile(5., 1., hr.LAKE, valid=0.), tile(0., 1., hr.OCEAN)):
            np.testing.assert_array_equal(hr.ground_for_shading(self.elevation, water), self.elevation)
            out = hr.composite_water(self.rgb, self.elevation, water, crop=2)
            self.assertEqual(out.dtype, np.uint8)
            np.testing.assert_array_equal(out, self.rgb)

    def test_dry_endorheic_land_below_sea_shades_as_ground(self):
        elevation = np.array([[-120., -.2, .3, 80.]], np.float32)
        ground = hr.ground_for_shading(elevation, tile(shape=(1, 4)))
        np.testing.assert_array_equal(ground, [[.5, .5, .5, 80.]])
        # Without authority a negative sample remains legacy sea.
        np.testing.assert_array_equal(hr.ground_for_shading(elevation, tile(valid=0., shape=(1, 4))), elevation)
        rgb = np.full((1, 4, 3), 200, np.uint8)
        np.testing.assert_array_equal(hr.composite_water(rgb, elevation, tile(shape=(1, 4))), rgb)

    def test_lake_depth_coverage_and_crop_alignment(self):
        elevation = np.zeros((4, 5), np.float32)
        water = tile(30., 1., hr.LAKE, shape=(4, 5))           # 30 m deep: deep colour
        water[1, 1] = .15, 1., hr.RIVER, 1.                     # at the shallow limit
        water[1, 2] = 30., .5, hr.LAKE, 1.                      # half wet
        water[1, 3] = 30., 7., hr.LAKE, 1.                      # coverage clamps to 1
        water[2, 1:4, 1] = 0., -1., 0.                          # zero or negative coverage
        rgb = np.ones((2, 3, 3), np.float32)
        out = hr.composite_water(rgb, elevation, water, crop=1)
        np.testing.assert_allclose(out[0, 0], hr.SHALLOW, atol=1e-6)
        np.testing.assert_allclose(out[0, 1], (1+hr.DEEP)/2, atol=1e-6)
        np.testing.assert_allclose(out[0, 2], hr.DEEP, atol=1e-6)
        np.testing.assert_array_equal(out[1], 1.)
        np.testing.assert_allclose(hr.water_colour([6.075])[0], (hr.SHALLOW+hr.DEEP)/2, atol=1e-6)
        with self.assertRaises(ValueError):  # crop is explicit, never inferred
            hr.composite_water(rgb, elevation, water, crop=0)

    def test_non_finite_inputs_stay_finite_and_in_range(self):
        water = tile(12., 1., hr.RIVER)
        water[0, :, 0] = np.nan
        water[1, :, 1] = np.inf
        water[2, :, 3] = np.nan
        elevation = self.elevation.copy()
        elevation[3, 2:6] = np.nan
        rgb = np.random.default_rng(4).random((6, 8, 3)).astype(np.float32)
        out = hr.composite_water(rgb, elevation, water)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(((out >= 0) & (out <= 1)).all())
        np.testing.assert_array_equal(out[:3], rgb[:3])          # no authority from bad channels
        np.testing.assert_allclose(out[3, 2], hr.SHALLOW, atol=1e-6)  # unknown bed: zero depth
        ground = hr.ground_for_shading(self.elevation, tile(np.nan, 0., hr.DRY))
        np.testing.assert_array_equal(ground, self.elevation)


if __name__ == '__main__':
    unittest.main()
