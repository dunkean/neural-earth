
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import unittest
import numpy as np
from terrain_priority import land_mask, tile_relief_stats


class LandMaskTests(unittest.TestCase):
    def test_islands_coasts_and_unknowns_are_kept(self):
        heights = np.full((8, 12), -2000.)
        heights[0, 0] = 1  # Even one island sample protects the whole cell.
        heights[5, 5] = 0
        heights[6, 9] = np.nan
        mask = land_mask(heights, [-12, -8, 12, 8])
        self.assertEqual(mask['rows'], ['100', '011'])
        self.assertEqual((mask['width'], mask['height']), (3, 2))
        self.assertEqual(mask['bounds'], [-12, -8, 12, 8])

    def test_threshold_and_coastal_mask(self):
        heights = np.full((4, 8), -10.)
        heights[1, 5] = 2
        mask = land_mask(heights, [0, 0, 8, 4])
        self.assertEqual(mask['rows'], ['01'])
        self.assertEqual(mask['sea_rows'], ['11'])

    def test_stats_ignore_halo_and_keep_shallow_relief(self):
        heights = np.full((6, 6), 100.)
        heights[1:-1, 1:-1] = -10
        self.assertEqual(tile_relief_stats(heights, 1),
                         dict(elevation_min=-10., elevation_max=-10.))
        heights[2, 2] = -9.99
        self.assertGreater(tile_relief_stats(heights, 1)['elevation_max'], -10)
        heights[2, 2] = np.nan
        self.assertEqual(tile_relief_stats(heights, 1), {})


if __name__ == '__main__':
    unittest.main()
