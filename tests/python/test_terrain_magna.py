"""Pinned upstream agreement, native coast safety and tile independence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'terrain-diffusion'))

import json
import math
import os
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import torch
import terrain_magna as magna
import terrain_refinement as refinement
from test_terrain_amplification import fake_world, analytic_detail


def climate(world, xs, ys, **kwargs):
    out = np.zeros((5, len(ys), len(xs)), np.float32)
    out[0], out[2], out[4] = 14, 900, -.0065
    return out


class MagnaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def devices(self):
        return ['cpu', 'cuda'] if torch.cuda.is_available() and os.getenv('TERRAIN_TEST_CUDA') == '1' else ['cpu']

    def test_matches_pinned_magna_noise_and_ground(self):
        reference = json.loads(subprocess.check_output(['node', 'tools/verification/magna_surface_reference.cjs'], text=True))
        for device in self.devices():
            x, y = [torch.tensor([row[k] for row in reference], dtype=torch.float64, device=device) for k in ('x', 'y')]
            def base(x, y):
                return 1000+100*torch.sin(x/200)+60*torch.cos(y/140)
            heights, slope = magna.noise_height(base, x, y, 17331)
            np.testing.assert_allclose(heights.cpu(), [row['noise'] for row in reference], rtol=0, atol=2e-9)
            local = 14-.0065*(2000+.6*x+.4*y).clamp_min(0)
            math_slope = math.hypot(.6, .4)
            snow = (1-magna._smooth(-5, 1, local))*(1-magna._smooth(.55, 1.3, torch.full_like(x, math_slope)))
            exposed = 1-snow
            rock = magna._smooth(.25, .9, torch.full_like(x, math_slope))*exposed
            grass = magna._scalar_smooth(250, 1400, 900)*magna._smooth(-8, 4, local)*(1-magna._smooth(28, 42, local))*(exposed-rock)
            gain = magna._scalar_smooth(.025, .18, math_slope)*magna._smooth(0, 2, 2000+.6*x+.4*y)*(rock+.16*(exposed-rock-grass)+.045*grass+.018*snow)
            delta = magna._ground_height(x, y, 17331)*gain
            np.testing.assert_allclose(delta.cpu(), [row['ground'] for row in reference], rtol=0, atol=3e-9)

    def test_identity_validation_cancellation(self):
        world = fake_world()
        with analytic_detail('cpu'):
            for level, strength in ((0, 1), (2, 0), (6, 0)):
                args = level, -5, -7, 11, 13
                h, s = magna.sample_magna(world, *args, intensity=strength)
                self.assertTrue(torch.equal(h, refinement.sample_refined(world, *args)))
                self.assertEqual(s.count_nonzero().item(), 0)
        for args in ((7,0,0,1,1), (2,.5,0,1,1), (2,0,0,0,1), (2,0,0,1025,1025)):
            with self.assertRaises(ValueError):
                magna.sample_magna(world, *args)
        for strength in (-1, 2.1, float('nan'), float('inf'), None):
            with self.assertRaises(ValueError):
                magna.sample_magna(world, 2, 0, 0, 2, 2, intensity=strength)
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            magna.sample_magna(world, 2, 0, 0, 2, 2, check=lambda: (_ for _ in ()).throw(RuntimeError('cancelled')))

    @patch('terrain_climate.sample_coarse_climate', climate)
    def test_overlap_seed_and_bounded_materials(self):
        for device in self.devices():
            for origin in (-103, 0, 100003):
                world = fake_world(device, 'hills')
                args = 4, origin, origin-13, origin+34, origin+39
                a, s = magna.sample_magna(world, *args)
                b, t = magna.sample_magna(world, 4, origin+7, origin-2, origin+31, origin+30)
                self.assertTrue(torch.equal(a[7:31, 11:43], b))
                self.assertTrue(torch.equal(s[:, 7:31, 11:43], t))
                self.assertTrue(torch.isfinite(a).all())
                self.assertTrue(((s >= 0) & (s <= 1)).all())
                self.assertEqual(s[:2].count_nonzero().item(), 0)
            first = magna.sample_magna(fake_world(device, 'hills', 17), 6, -16, -16, 16, 16)[0]
            other = magna.sample_magna(fake_world(device, 'hills', 18), 6, -16, -16, 16, 16)[0]
            self.assertFalse(torch.equal(first, other))

    @patch('terrain_climate.sample_coarse_climate', climate)
    def test_ocean_coast_and_flat_ground(self):
        for device in self.devices():
            world = fake_world(device, 'coast')
            bounds = -32, -64, 32, 64
            surface = magna._NativeSurface(world, 4, bounds, None)
            y = (torch.arange(-32,32,device=device,dtype=torch.float64)[:,None]+.5)*30/16
            x = (torch.arange(-64,64,device=device,dtype=torch.float64)[None,:]+.5)*30/16
            base = surface(x,y).float()
            height, _ = magna.sample_magna(world, 4, *bounds, intensity=2)
            ocean = base <= 0
            self.assertTrue(ocean.any())
            self.assertTrue(torch.equal(height[ocean], base[ocean]))
            self.assertTrue(torch.equal(height > 0, base > 0))
            flat, _ = magna.sample_magna(fake_world(device, 'plateau'), 6, -16,-16,16,16)
            self.assertTrue(torch.equal(flat, torch.full_like(flat, 500)))



if __name__ == '__main__':
    unittest.main()
