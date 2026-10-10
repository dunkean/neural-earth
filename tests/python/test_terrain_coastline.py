"""Native coast cleanup: only pixel-scale near-zero land/sea specks change."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from terrain_coastline import EPSILON, HALO, MAX_PIXELS, remove_coast_specks, read_native


def coast(size=64):
    """Sea on the left, a gentle beach crossing zero, land on the right."""
    x = np.linspace(-3, 3, size, dtype=np.float32)
    return np.repeat(x[None], size, axis=0)


class CoastlineTests(unittest.TestCase):
    def test_isolated_near_zero_flips_take_their_neighbourhood_sign(self):
        heights = coast()
        heights[10, 5] = .02     # a dry pixel in shallow sea
        heights[40, 60] = -.03   # a wet pixel on land
        cleaned = remove_coast_specks(torch.from_numpy(heights)).numpy()
        self.assertLess(cleaned[10, 5], 0)
        self.assertGreater(cleaned[40, 60], 0)
        # Magnitude is kept, so the change stays below 2 * EPSILON.
        self.assertAlmostEqual(abs(cleaned[10, 5]), .02, places=6)
        self.assertLess(np.abs(cleaned-heights).max(), 2*EPSILON)

    def test_real_features_and_heights_beyond_epsilon_are_unchanged(self):
        heights = coast()
        heights[20:26, 8:14] = 2.0  # a 6x6 islet in the sea: real relief
        heights[30:33, 3:6] = .3    # a 3x3 low sandbar: larger than a speck
        heights[50, 2] = 3.         # a one-pixel rock with real relief
        cleaned = remove_coast_specks(torch.from_numpy(heights)).numpy()
        far = np.abs(heights) >= EPSILON
        np.testing.assert_array_equal(cleaned[far], heights[far])
        self.assertTrue((cleaned[30:33, 3:6] > 0).all())
        self.assertTrue((cleaned[20:26, 8:14] == 2).all())
        self.assertEqual(cleaned[50, 2], 3.)

    def test_noisy_beach_has_no_single_pixel_components(self):
        rng = np.random.default_rng(3)
        heights = coast(96) * .3 + rng.normal(0, .15, (96, 96)).astype(np.float32)
        before = np.mean(np.sign(heights[:, 1:]) != np.sign(heights[:, :-1]))
        cleaned = remove_coast_specks(torch.from_numpy(heights)).numpy()
        after = np.mean(np.sign(cleaned[:, 1:]) != np.sign(cleaned[:, :-1]))
        self.assertLess(after, before / 3)

    def test_overlapping_reads_agree_exactly(self):
        rng = np.random.default_rng(7)
        field = rng.normal(0, .3, (200, 200)).astype(np.float32)
        origin = 100
        world = SimpleNamespace(get=lambda i1, j1, i2, j2, with_climate: {
            'elev': torch.from_numpy(field[i1+origin:i2+origin, j1+origin:j2+origin].copy())})
        whole = read_native(world, -40, -40, 40, 40).numpy()
        left = read_native(world, -40, -40, 40, 5).numpy()
        right = read_native(world, -40, 0, 40, 40).numpy()
        np.testing.assert_array_equal(left, whole[:, :45])
        np.testing.assert_array_equal(right, whole[:, 40:])
        self.assertEqual(whole.shape, (80, 80))
        self.assertGreaterEqual(HALO, MAX_PIXELS)


if __name__ == '__main__':
    unittest.main()
