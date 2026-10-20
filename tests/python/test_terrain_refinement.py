"""Coastal noise, parent consistency, grid lattices, slope-guided detail and tile seams in sub-native refinement."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(_REPO_ROOT / 'terrain-diffusion'))
from terrain_refinement import (HALF_BAND_RADIUS, MIN_LOD, NOTCH_RADIUS, _HALF_BAND, _NOTCH, _kernel,
                                _procedural_detail, _slope_cap, sample_refined)


class DetailField:
    def __init__(self, device, *, smooth=False, zero=False, stripes=False):
        self.device = device
        self.smooth = smooth
        self.zero = zero
        self.stripes = stripes

    def __getitem__(self, key):
        _, rows, cols = key
        yy = torch.arange(rows.start, rows.stop, device=self.device)[:, None]
        xx = torch.arange(cols.start, cols.stop, device=self.device)[None, :]
        # Strong, globally anchored decoder residuals stress both signs: a
        # diagonal wave at a quarter of the sample rate (the Nyquist notch
        # removes a checkerboard entirely).
        detail = torch.where((xx + yy) % 4 < 2, 3., -3.)
        if self.stripes:
            detail = torch.where(yy % 2 == 0, 2., -2.) + torch.where(xx % 2 == 0, 1., -1.)
        if self.smooth:
            detail = .6*detail + torch.sin(.71*xx + .37*yy) + .8*torch.cos(.29*xx - .53*yy)
        if self.zero:
            detail = torch.zeros_like(detail)
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
                    for level in range(1, -MIN_LOD+1):
                        with self.subTest(device=device, height=height, lod=-level):
                            actual = sample_refined(world, level, -16, -16, 16, 16)
                            self.assertTrue(torch.isfinite(actual).all())
                            if height == 0:
                                self.assertEqual(torch.count_nonzero(actual).item(), 0)
                            else:
                                self.assertTrue((actual * height > 0).all())
                                self.assertGreater(actual.std().item(), 0, 'Keep fine detail above/below sea level')

    def test_each_level_keeps_its_parent_at_coarser_scales(self):
        smooth = lambda y, x: 1000 + 50*torch.sin(.37*x + .11*y) + 30*torch.cos(.23*y - .41*x)
        ramp = lambda y, x: (x + .3) * .05 + y * .003
        for device in self.devices():
            for name, heights in (('coastal ramp', ramp), ('smooth relief', smooth)):
                world = self.world(device, heights)
                with patch('terrain_refinement._detail_field', return_value=DetailField(device, smooth=True)):
                    for level in range(1, -MIN_LOD+1):
                        with self.subTest(device=device, terrain=name, lod=-level):
                            parent = sample_refined(world, level-1, -32, -32, 32, 32).double()
                            child = sample_refined(world, level, -64, -64, 64, 64).double()
                            added = child - parent.repeat_interleave(2, 0).repeat_interleave(2, 1)
                            # Each level adds its own octave; 8x8-parent-cell
                            # means must not move by more than a fraction of it.
                            # Near sea level the sign-keeping soft limit is not
                            # additive across octaves, so the coast allows more.
                            coarse = F.avg_pool2d(parent[None, None], 8) - F.avg_pool2d(child[None, None], 16)
                            limit = .4 if name == 'coastal ramp' else .1
                            self.assertLess(coarse.abs().max().item(), limit * added.std().item())
                            if name == 'coastal ramp':
                                self.assertTrue((child < 0).any())
                                self.assertTrue((child > 0).any())
            # Inland and without decoder detail, the spline reproduces a linear surface.
            world = self.world(device, lambda y, x: 100 + (x + .3) * .05 + y * .003)
            with patch('terrain_refinement._detail_field', return_value=DetailField(device, zero=True)):
                for level in range(1, -MIN_LOD+1):
                    child = sample_refined(world, level, -32, -32, 32, 32)
                    centres = (torch.arange(-32, 32, device=device).double() + .5) / 2**level - .5
                    expected = 100 + (centres[None, :] + .3) * .05 + centres[:, None] * .003
                    torch.testing.assert_close(child.double(), expected, rtol=0, atol=2e-5)

    def test_smooth_native_terrain_shows_no_source_grid_lattice(self):
        # Bilinear and block-mean cascades concentrated curvature on the 30 m
        # grid (phase ratio ~2 at LOD -3), a lattice that hillshading reveals.
        smooth = lambda y, x: 1000 + 50*torch.sin(.37*x + .11*y) + 30*torch.cos(.23*y - .41*x)
        for device in self.devices():
            world = self.world(device, smooth)
            with patch('terrain_refinement._detail_field', return_value=DetailField(device, zero=True)):
                z = sample_refined(world, 3, -64, -64, 64, 64).double()
            curvature = (z[2:, 1:-1] + z[:-2, 1:-1] + z[1:-1, 2:] + z[1:-1, :-2] - 4*z[1:-1, 1:-1]).abs()
            phase = (torch.arange(-63, 63, device=device) % 8)
            for axis in (0, 1):
                profile = torch.stack([(curvature[phase == k] if axis == 0 else curvature[:, phase == k]).mean()
                                       for k in range(8)])
                with self.subTest(device=device, axis=axis):
                    self.assertLess((profile.max() / profile.min()).item(), 1.1)

    def test_procedural_detail_does_not_streak_slopes_and_is_isotropic_on_flats(self):
        # Below LOD -1, only rare 7.5 m kernels elongate down steep slopes:
        # more combed them into streaks. Flats show no direction at all.
        for name, heights in (('slope', lambda y, x: 1000 + 15*x + 0*y),
                              ('flat', lambda y, x: 1000 + 0*x + 0*y)):
            world = self.world('cpu', heights)
            with patch('terrain_refinement._detail_field', return_value=DetailField('cpu', smooth=True)):
                ratios = {}
                for level in (2, 6):
                    detail = _procedural_detail(world, level, -128, -128, 128, 128)
                    ratios[level] = (detail.diff(dim=0).std() / detail.diff(dim=1).std()).item()
            with self.subTest(terrain=name):
                for ratio in ratios.values():
                    self.assertAlmostEqual(ratio, 1, delta=.15 if name == 'slope' else .1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA only')
    def test_cuda_kernels_match_the_pytorch_fallback_exactly(self):
        if _kernel('octave', 'cuda') is None:
            self.skipTest('CuPy kernels unavailable')
        world = self.world('cuda', lambda y, x: 1000 + 15*torch.sin(.3*x) + 9*torch.cos(.2*y))
        with patch('terrain_refinement._detail_field', return_value=DetailField('cuda', smooth=True)):
            for level in range(1, -MIN_LOD+1):
                fused = sample_refined(world, level, -37, 21, 19, 90)
                with patch('terrain_refinement._kernel', return_value=None):
                    loop = sample_refined(world, level, -37, 21, 19, 90)
                with self.subTest(lod=-level):
                    torch.testing.assert_close(fused, loop, rtol=0, atol=0)

    def test_adjacent_unaligned_reads_match_a_single_read(self):
        for device in self.devices():
            world = self.world(device, lambda y, x: (x + .3) * .05 + y * .003)
            with patch('terrain_refinement._detail_field', return_value=DetailField(device, smooth=True)):
                for level in range(1, -MIN_LOD+1):
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
                    for level in range(1, -MIN_LOD+1):
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
                # The coast limit leaves high ground alone; only the slope cap applies.
                # The band keeps 1 - H^2 of the diagonal wave, H ~ 1/2 the
                # half band's response at a quarter of the sample rate; the
                # Nyquist notch passes it with a ripple R per axis.
                keep = 1 - _HALF_BAND[HALF_BAND_RADIUS]**2
                ripple = math.fsum(tap * math.cos(math.pi*n/2) for n, tap in
                                   zip(range(-NOTCH_RADIUS, NOTCH_RADIUS+1), _NOTCH))
                raw = 3 * keep * height**.5 * world.kwargs['residual_std']
                old_amplitude = raw * ripple**2 / (1 + (raw/_slope_cap(1))**2)**.5
                actual_amplitude = (child-height).abs().max().item()
                # Signed-sqrt composition adds the second-order (ds)^2 term.
                self.assertGreater(actual_amplitude / old_amplitude, .99)
                self.assertLessEqual(actual_amplitude, old_amplitude * (1 + 1e-3))

    def test_decoder_and_native_nyquist_stripes_do_not_become_a_grid(self):
        # Decoders leave alternating rows and columns; interpolating splines
        # kept them and magnified hillshading showed a 60 m and 30 m grid.
        stripes = lambda y, x: 1000 + 4*(1 - 2*(x % 2)) + 3*(1 - 2*(y % 2))
        for device in self.devices():
            world = self.world(device, stripes)
            with patch('terrain_refinement._detail_field', return_value=DetailField(device, stripes=True)):
                child = sample_refined(world, 1, -64, -64, 64, 64).double()
            with self.subTest(device=device):
                self.assertLess((child - 1000).abs().max().item(), .05)

    def test_a_land_parent_cell_holds_both_land_and_sea_at_fine_levels(self):
        for device in self.devices():
            world = self.world(device, lambda y, x: (x + .1) * .05)
            with patch('terrain_refinement._detail_field', return_value=DetailField(device)):
                # Native cell (0, 0) is land; its west neighbour is sea.
                self.assertGreater(sample_refined(world, 0, 0, 0, 1, 1).item(), 0)
                block = sample_refined(world, 3, 0, 0, 8, 8)
                self.assertTrue((block < 0).any())
                self.assertTrue((block > 0).any())

    def coast_checks(self, actual, signed_distance, tolerance):
        """Sign follows the reference away from its coast; no islands or lakes appear."""
        from scipy import ndimage
        land = (actual >= 0).cpu().numpy()
        distance = signed_distance.cpu().numpy()
        far = abs(distance) > tolerance
        self.assertTrue(((distance >= 0) == land)[far].all())
        for mask in (land, ~land):
            labels, count = ndimage.label(mask)
            sizes = np.bincount(labels.ravel())[1:]
            self.assertEqual(count, 1)

    def centres(self, level, start, stop, device):
        return (torch.arange(start, stop, device=device).double() + .5) / 2**level - .5

    def test_irregular_coast_stays_near_the_native_coast_without_new_islands_or_lakes(self):
        for device in self.devices():
            # A diagonal coastline with strongly varying heights on both sides.
            heights = lambda y, x: (x+y+.2)*torch.where((x//2+y//3)%2 == 0, .01, 12.)
            world = self.world(device, heights)
            with patch('terrain_refinement._detail_field', return_value=DetailField(device, smooth=True)):
                for level in range(1, -MIN_LOD+1):
                    scale = 2**level
                    actual = sample_refined(world, level, -8*scale, -8*scale, 8*scale, 8*scale)
                    c = self.centres(level, -8*scale, 8*scale, device)
                    with self.subTest(device=device, lod=-level):
                        self.coast_checks(actual, (c[None, :] + c[:, None] + .2) / 2**.5, 1.5)

    def test_raster_staircase_coast_becomes_straight(self):
        # Land 3 m beside sea at -1 cm along a diagonal of whole cells: the 30 m
        # raster previously showed as one-cell steps at every fine level.
        for device in self.devices():
            world = self.world(device, lambda y, x: torch.where(x + 2*y >= 0, 3., -.01))
            with patch('terrain_refinement._detail_field', return_value=DetailField(device, zero=True)),                     patch('terrain_refinement.COAST_ANGLE', 0.):
                for level in range(1, -MIN_LOD+1):
                    scale = 2**level
                    land = sample_refined(world, level, -8*scale, -8*scale, 8*scale, 8*scale) >= 0
                    # First land column of each row, against a straight-line fit.
                    edge = land.double().argmax(dim=1).double() / scale
                    rows = torch.arange(len(edge), device=device).double() / scale
                    inside = (edge > 2) & (edge < 14)
                    a = torch.stack((rows[inside], torch.ones_like(rows[inside])), 1)
                    fit = torch.linalg.lstsq(a.cpu(), edge[inside].cpu()[:, None]).solution
                    residual = edge[inside].cpu() - (a.cpu() @ fit)[:, 0]
                    with self.subTest(device=device, lod=-level):
                        self.assertLess(residual.std().item(), .06)

    def test_detail_never_moves_the_coastline(self):
        # The coast is the zero line of the base: strong decoder and
        # procedural detail leave every pixel on the same side.
        for device in self.devices():
            heights = lambda y, x: (x+y+.2)*torch.where((x//2+y//3)%2 == 0, .01, 12.)
            world = self.world(device, heights)
            for level in range(1, -MIN_LOD+1):
                scale = 2**level
                read = lambda: sample_refined(world, level, -8*scale, -8*scale, 8*scale, 8*scale)
                with patch('terrain_refinement._detail_field', return_value=DetailField(device, smooth=True)):
                    detailed = read()
                with patch('terrain_refinement._detail_field', return_value=DetailField(device, zero=True)):
                    bare = read()
                with self.subTest(device=device, lod=-level):
                    self.assertTrue(torch.equal(detailed >= 0, bare >= 0))

    def test_native_coast_is_stable_at_far_and_negative_coordinates(self):
        for device in self.devices():
            for origin in (-100000003, 100000003):
                world = self.world(device, lambda y, x: (x-origin+.2)*.05)
                with patch('terrain_refinement._detail_field', return_value=DetailField(device)):
                    for level in range(1, -MIN_LOD+1):
                        scale = 2**level
                        start = origin*scale
                        actual = sample_refined(world, level, -4*scale, start-4*scale,
                                                4*scale, start+4*scale)
                        c = self.centres(level, -4*scale, 4*scale, device)
                        with self.subTest(device=device, origin=origin, lod=-level):
                            self.coast_checks(actual, (c[None, :] + .2).expand_as(actual), 1.)


if __name__ == '__main__':
    unittest.main()
