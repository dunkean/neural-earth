
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import json
import unittest

import numpy as np

from terrain_styles import STYLES, apply_contours, parse_contours, render_style


class TerrainStylesTests(unittest.TestCase):
    def test_all_city_styles_and_rust_elevation_view(self):
        self.assertEqual(set(STYLES), {'parchment', 'atlas', 'watabou', 'engraving',
            'cadastre', 'blueprint', 'illuminated', 'topographic', 'night', 'copernicus'})
        elevation = np.broadcast_to(np.linspace(-1000, 5000, 100), (50, 100)).copy()
        results = []
        for mode in STYLES:
            rgb = render_style(elevation, 30, mode)
            self.assertEqual(rgb.shape, (50, 100, 3))
            self.assertTrue(np.isfinite(rgb).all())
            self.assertGreaterEqual(rgb.min(), 0)
            self.assertLessEqual(rgb.max(), 1)
            results.append(rgb[25, 40].round(3).tobytes())
        self.assertEqual(len(set(results)), 10)

    def test_contours_are_fixed_physical_levels_and_main_levels_heavier(self):
        elevation = np.broadcast_to(np.arange(304)*2, (304, 304)).astype(float)
        rgb = np.ones((304, 304, 3))
        settings = dict(enabled=True, interval=100)
        result = apply_contours(rgb, elevation, settings, 'topographic')
        np.testing.assert_array_equal(result[:, 25], rgb[:, 25])  # 50 m, between lines
        self.assertTrue((result[:, 50]<1).all())  # 100 m
        self.assertTrue((result[:, 250]<result[:, 50]).all())  # main line at 500 m
        np.testing.assert_array_equal(result[:, 0], rgb[:, 0])  # zero level is shore
        np.testing.assert_array_equal(apply_contours(rgb, elevation, dict(enabled=False, interval=100)), rgb)

    def test_flat_fields_and_ocean_are_not_filled_with_contour_ink(self):
        rgb = np.ones((32, 32, 3))
        for height in (100, 500, 0, -100):
            np.testing.assert_array_equal(apply_contours(rgb, np.full((32, 32), height),
                dict(enabled=True, interval=100)), rgb)

    def test_steep_slope_has_contours_instead_of_being_hidden(self):
        elevation = np.broadcast_to(np.arange(32)*80+20, (32, 32)).astype(float)
        rgb = np.ones((32, 32, 3))
        result = apply_contours(rgb, elevation, dict(enabled=True, interval=100), 'topographic')
        self.assertLess(result[16, 6, 0], .95)  # 500 m, despite 80 m per pixel

    def test_contours_and_palettes_do_not_restart_at_tile_boundaries(self):
        elevation = np.broadcast_to(np.arange(608)*1.3+100, (304, 608)).copy()
        settings = dict(enabled=True, interval=100)
        for mode in STYLES:
            whole = apply_contours(render_style(elevation, 30, mode), elevation, settings, mode)
            # Adjacent physical tiles contain the same 24-pixel halo.
            for x in (24, 280):
                field = elevation[:, x-24:x+280]
                tile = apply_contours(render_style(field, 30, mode, origin=((x-24)*30, 0)), field, settings, mode)
                np.testing.assert_allclose(tile[24:-24,24:-24], whole[24:-24,x:x+256], atol=1e-5)

    def test_paper_grain_does_not_turn_into_blocks_at_fine_lods(self):
        elevation = np.full((64, 64), 1000.)
        for lod in (0, -1, -2, -3):
            resolution = 30*2**lod
            for mode in ('parchment', 'illuminated', 'blueprint', 'cadastre', 'engraving'):
                with self.subTest(lod=lod, mode=mode):
                    rgb = render_style(elevation, resolution, mode, origin=(-20e6, -10e6))
                    different = np.any(np.abs(np.diff(rgb, axis=1)) > 1e-6, axis=2)
                    self.assertGreater(different.mean(), .95, 'grain must vary between terrain pixels')

    def test_fine_style_grain_matches_at_tile_boundaries(self):
        elevation = np.full((64, 128), 1000.)
        resolution = 30*2**-3
        origin = (-20e6, -10e6)
        for mode in ('parchment', 'illuminated', 'engraving'):
            whole = render_style(elevation, resolution, mode, origin=origin)
            for x in (0, 64):
                tile = render_style(elevation[:, x:x+64], resolution, mode,
                                    origin=(origin[0]+x*resolution, origin[1]))
                np.testing.assert_allclose(tile, whole[:, x:x+64], atol=1e-10)

    def test_invalid_settings_rejected(self):
        for value in ({'interval':0}, {'interval':True}, {'interval':10001}, {'enabled':1}, {'unexpected':1},
                      {'width':0}, {'width':4.1}, {'width':True}, {'width':float('nan')},
                      {'density':3}, {'density':201}, {'density':False}, {'density':float('inf')}):
            with self.assertRaises(ValueError):
                parse_contours(json.dumps(value))
        self.assertEqual(parse_contours('{"enabled":true,"interval":20}'), dict(enabled=True, interval=20, width=.9))
        self.assertEqual(parse_contours('{"enabled":true,"interval":40,"width":2,"density":50,"automatic":true}'),
                         dict(enabled=True, interval=40, width=2))
        for density in (3.125, 12.5, 25, 200):
            self.assertEqual(parse_contours(json.dumps(dict(enabled=True, interval=400, density=density))),
                             dict(enabled=True, interval=400, width=.9))

    def test_width_changes_strokes_without_moving_levels(self):
        elevation = np.broadcast_to(np.arange(304)*2, (304, 304)).astype(float)
        rgb = np.ones((304, 304, 3))
        thin = apply_contours(rgb, elevation, dict(enabled=True, interval=100, width=.3), 'topographic')
        wide = apply_contours(rgb, elevation, dict(enabled=True, interval=100, width=4), 'topographic')
        self.assertGreater(np.sum(1-wide), np.sum(1-thin)*2)
        self.assertLess(wide[150, 52, 0], thin[150, 52, 0])
        np.testing.assert_array_equal(wide[:, 25], rgb[:, 25])


if __name__ == '__main__':
    unittest.main()
