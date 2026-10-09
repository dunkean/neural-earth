"""Coastal noise, parent means and tile seams in the sub-native cascade."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(_REPO_ROOT / 'terrain-diffusion'))
from terrain_refinement import sample_refined


class DetailField:
    def __init__(self, device, *, smooth=False):
        self.device = device
        self.smooth = smooth

    def __getitem__(self, key):
        _, rows, cols = key
        yy = torch.arange(rows.start, rows.stop, device=self.device)[:, None]
        xx = torch.arange(cols.start, cols.stop, device=self.device)[None, :]
        # Strong, globally anchored decoder residuals stress both signs.
        detail = torch.where((xx + yy) % 2 == 0, 3., -3.)
        if self.smooth:
            detail = .6*detail + torch.sin(.71*xx + .37*yy) + .8*torch.cos(.29*xx - .53*yy)
        return torch.stack((detail, torch.ones_like(detail)))


class CoastalRefinementTests(unittest.TestCase):
    def devices(self):
        return ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']

    def world(self, device, heights):
        def get(i1, j1, i2, j2, *, with_climate):
            self.assertFalse(with_climate)
            yy = torch.arange(i1, i2, device=device)[:, None]
            xx = torch.arange(j1, j2, device=device)[None, :]
            return {'elev': torch.broadcast_to(heights(yy, xx), (i2-i1, j2-j1)).float()}
        return SimpleNamespace(device=device, get=get, kwargs={'residual_std': 1.1678})

    def test_low_land_and_shallow_sea_do_not_fragment_at_any_level(self):
        for device in self.devices():
            with patch('terrain_refinement._detail_field', return_value=DetailField(device)):
                for height in (-1., -.1, -.001, 0., .001, .1, 1.):
                    world = self.world(device, lambda y, x: torch.tensor(height, device=device))
                    for level in (1, 2, 3):
                        with self.subTest(device=device, height=height, lod=-level):
                            actual = sample_refined(world, level, -16, -16, 16, 16)
                            self.assertTrue(torch.isfinite(actual).all())
                            if height == 0:
                                self.assertEqual(torch.count_nonzero(actual).item(), 0)
                            else:
                                self.assertTrue((actual * height > 0).all())
                                self.assertGreater(actual.std().item(), 0, 'Keep fine detail above/below sea level')

    def test_each_level_preserves_parent_means_away_from_the_native_coast(self):
        for device in self.devices():
            world = self.world(device, lambda y, x: (x + .3) * .05 + y * .003)
            with patch('terrain_refinement._detail_field', return_value=DetailField(device, smooth=True)):
                for level in (1, 2, 3):
                    with self.subTest(device=device, lod=-level):
                        extent = 8*2**(level-1)
                        parent = sample_refined(world, level-1, -extent, -extent, extent, extent)
                        child = sample_refined(world, level, -2*extent, -2*extent, 2*extent, 2*extent)
                        means = F.avg_pool2d(child[None, None], 2)[0, 0]
                        # The native coastal band deliberately relaxes mean
                        # offsets; compare conservative blocks further inland.
                        columns = torch.arange(-extent, extent, device=device)
                        outside = (columns.div(2**(level-1), rounding_mode='floor').abs() >= 3)
                        self.assertTrue(outside.any())
                        torch.testing.assert_close(means[:, outside], parent[:, outside], rtol=2e-6, atol=1e-7)
                        self.assertTrue((child < 0).any())
                        self.assertTrue((child > 0).any())

    def test_adjacent_unaligned_reads_match_a_single_read(self):
        for device in self.devices():
            world = self.world(device, lambda y, x: (x + .3) * .05 + y * .003)
            with patch('terrain_refinement._detail_field', return_value=DetailField(device, smooth=True)):
                for level in (1, 2, 3):
                    full = sample_refined(world, level, -15, -17, 17, 15)
                    left = sample_refined(world, level, -15, -17, 17, -1)
                    right = sample_refined(world, level, -15, -1, 17, 15)
                    top = sample_refined(world, level, -15, -17, 1, 15)
                    bottom = sample_refined(world, level, 1, -17, 17, 15)
                    with self.subTest(device=device, lod=-level):
                        torch.testing.assert_close(torch.cat((left, right), 1), full, rtol=0, atol=0)
                        torch.testing.assert_close(torch.cat((top, bottom), 0), full, rtol=0, atol=0)

    def test_low_cells_beside_higher_ground_stay_on_the_same_side_of_zero(self):
        for device in self.devices():
            with patch('terrain_refinement._detail_field', return_value=DetailField(device, smooth=True)):
                for sign in (-1, 1):
                    world = self.world(device, lambda y, x: sign * torch.where((x//2 + y//3) % 2 == 0, .001, 12.))
                    for level in (1, 2, 3):
                        with self.subTest(device=device, sign=sign, lod=-level):
                            actual = sample_refined(world, level, -17, -15, 15, 17)
                            self.assertTrue((actual * sign > 0).all())

    def test_high_ground_keeps_decoder_detail_and_native_is_unchanged(self):
        for device in self.devices():
            height = 4000.
            world = self.world(device, lambda y, x: torch.tensor(height, device=device))
            with patch('terrain_refinement._detail_field', return_value=DetailField(device)) as detail:
                native = sample_refined(world, 0, -16, -16, 16, 16)
                detail.assert_not_called()
                torch.testing.assert_close(native, torch.full_like(native, height), rtol=0, atol=0)
                child = sample_refined(world, 1, -16, -16, 16, 16)
                old_amplitude = 3 * height**.5 * world.kwargs['residual_std']
                actual_amplitude = (child-height).abs().max().item()
                self.assertGreater(actual_amplitude / old_amplitude, .99)
                self.assertLessEqual(actual_amplitude, old_amplitude)

    def test_mixed_neighbours_still_refine_the_coast_inside_a_parent_cell(self):
        for device in self.devices():
            world = self.world(device, lambda y, x: (x + .1) * .05)
            with patch('terrain_refinement._detail_field', return_value=DetailField(device)):
                # Parent column zero is positive; its west neighbour is sea.
                parent = sample_refined(world, 0, 0, 0, 1, 1)
                child = sample_refined(world, 1, 0, 0, 2, 2)
                self.assertGreater(parent.item(), 0)
                self.assertTrue((child[:, 0] < 0).all())
                self.assertTrue((child[:, 1] > 0).all())
                torch.testing.assert_close(child.mean(), parent.squeeze())

    def test_irregular_coast_follows_native_surface_without_new_pixel_islands(self):
        for device in self.devices():
            # A connected diagonal coastline, with strongly varying heights.
            # Conservative 2x2 offsets previously detached pixels along it.
            heights = lambda y, x: (x+y+.2)*torch.where((x//2+y//3)%2 == 0, .01, 12.)
            world = self.world(device, heights)
            native = world.get(-10, -10, 10, 10, with_climate=False)['elev']
            with patch('terrain_refinement._detail_field', return_value=DetailField(device, smooth=True)):
                for level in (1, 2, 3):
                    scale = 2**level
                    actual = sample_refined(world, level, -8*scale, -8*scale, 8*scale, 8*scale)
                    reference = F.interpolate(native[None, None], scale_factor=scale,
                                              mode='bilinear', align_corners=False)[0, 0]
                    reference = reference[2*scale:-2*scale, 2*scale:-2*scale]
                    with self.subTest(device=device, lod=-level):
                        self.assertTrue(torch.equal(actual >= 0, reference >= 0))

    def test_native_coast_is_stable_at_far_and_negative_coordinates(self):
        for device in self.devices():
            for origin in (-100000003, 100000003):
                world = self.world(device, lambda y, x: (x-origin+.2)*.05)
                with patch('terrain_refinement._detail_field', return_value=DetailField(device)):
                    for level in (1, 2, 3):
                        scale = 2**level
                        start = origin*scale
                        actual = sample_refined(world, level, -2*scale, start-2*scale,
                                                2*scale, start+2*scale)
                        columns = torch.arange(-2*scale, 2*scale, device=device)
                        reference = ((columns.float()+.5)/scale-.5+.2)*.05
                        with self.subTest(device=device, origin=origin, lod=-level):
                            self.assertTrue(torch.equal(actual >= 0, (reference >= 0)[None].expand_as(actual)))


if __name__ == '__main__':
    unittest.main()
