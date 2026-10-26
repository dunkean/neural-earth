"""Phasor equations, analytic cascade gradients and globally anchored sampling."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'terrain-diffusion'))

from types import SimpleNamespace
import math
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import terrain_phasor as phasor
import terrain_refinement as refinement
from test_terrain_amplification import fake_world, analytic_detail


class ConstantControls:
    def sample(self, y, x):
        zero = torch.zeros_like(y+x)
        return zero+12, zero, zero, zero+.3, zero+.4


def rms(values):
    return values.double().square().mean().sqrt().item()


def drainage_world(device='cpu', seed=17, flat=False):
    """Open native valley; accumulation runs toward +y along its thalweg."""
    world = fake_world(device, seed=seed)
    def get(i1, j1, i2, j2, *, with_climate):
        y = torch.arange(i1, i2, device=device, dtype=torch.float64)[:, None]
        x = torch.arange(j1, j2, device=device, dtype=torch.float64)[None, :]
        height = torch.zeros_like(y+x)+900 if flat else 900+.4*x*x-6*y
        return {'elev': height.float()}
    world.get = get
    return world


def channel_area(world, i1, j1, i2, j2):
    y = torch.arange(i1, i2, device=world.device, dtype=torch.float64)[:, None]
    x = torch.arange(j1, j2, device=world.device, dtype=torch.float64)[None, :]
    return (900+240000*torch.exp(-x*x/(2*.9**2))*torch.sigmoid((y+35)/7)).float()


class PhasorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def devices(self):
        return ['cpu', 'cuda'] if torch.cuda.is_available() and os.getenv('TERRAIN_TEST_CUDA') == '1' else ['cpu']

    def test_identity_and_validation(self):
        for device in self.devices():
            world = fake_world(device)
            with analytic_detail(device):
                for level, intensity in ((0, 1), (2, 0), (4, 0), (6, 0)):
                    base = refinement.sample_refined(world, level, -12, -9, 11, 16)
                    height, surface = phasor.sample_phasor(world, level, -12, -9, 11, 16, intensity=intensity)
                    self.assertTrue(torch.equal(base, height))
                    self.assertEqual(surface.count_nonzero().item(), 0)
                for args in ((7, 0, 0, 1, 1), (2, .5, 0, 1, 1), (2, 0, 0, 0, 1), (2, 0, 0, 1025, 1025)):
                    with self.assertRaises(ValueError):
                        phasor.sample_phasor(world, *args)
                for strength in (-1, float('nan'), float('inf'), None):
                    with self.assertRaises(ValueError):
                        phasor.sample_phasor(world, 2, 0, 0, 2, 2, intensity=strength)

    def test_determinism_seed_and_surfaces(self):
        for device in self.devices():
            with analytic_detail(device):
                a = phasor.sample_phasor(fake_world(device), 4, -40, -36, 24, 28)
                b = phasor.sample_phasor(fake_world(device), 4, -40, -36, 24, 28)
                c = phasor.sample_phasor(fake_world(device, seed=18), 4, -40, -36, 24, 28)
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
                self.assertFalse(torch.equal(a[0], c[0]))
                self.assertEqual(a[0].dtype, torch.float32)
                self.assertEqual(a[1].dtype, torch.float32)
                self.assertEqual(a[1].shape, (4, 64, 64))
                self.assertTrue(torch.isfinite(a[0]).all())
                self.assertTrue(((a[1] >= 0) & (a[1] <= 1)).all())
                self.assertGreater(a[1].max().item(), .05)

    def test_exact_negative_mosaic_and_far_origin(self):
        for device in self.devices():
            with analytic_detail(device):
                for level in range(1, 7):
                    for origin in (-43, 2**27):
                        world = fake_world(device, 'hills')
                        a, b = origin, origin-13
                        whole = phasor.sample_phasor(world, level, a, b, a+48, b+52)
                        rows = []
                        for ya, yb in ((a, a+21), (a+21, a+48)):
                            tiles = [phasor.sample_phasor(world, level, ya, xa, yb, xb)
                                     for xa, xb in ((b, b+29), (b+29, b+52))]
                            rows.append(tuple(torch.cat([t[k] for t in tiles], dim=-1) for k in range(2)))
                        mosaic = tuple(torch.cat([r[k] for r in rows], dim=-2) for k in range(2))
                        with self.subTest(device=device, level=level, origin=origin):
                            self.assertTrue(all(torch.equal(x, y) for x, y in zip(whole, mosaic)))

    def test_sign_preservation_flats_and_smooth_ice_proxy(self):
        for device in self.devices():
            with analytic_detail(device, zero=True):
                for name in ('coast', 'plateau', 'alpine'):
                    world = fake_world(device, name)
                    for level in (2, 4, 6):
                        base = phasor._base(world, level, -32, -32, 32, 32, None)
                        h, s = phasor.sample_phasor(world, level, -32, -32, 32, 32, intensity=100)
                        self.assertTrue(torch.equal(h.sign(), base.sign()))
                        if name == 'coast':
                            refined = refinement.sample_refined(world, level, -32, -32, 32, 32)
                            self.assertTrue(torch.equal(h.sign(), refined.sign()))
                        if name == 'plateau':
                            self.assertTrue(torch.equal(h, base))
                            self.assertEqual(s.count_nonzero().item(), 0)
                        if name == 'coast':
                            band = base <= 4
                            self.assertTrue(torch.equal(h[band], base[band]))
                            self.assertEqual(s[:, band].count_nonzero().item(), 0)
                world = fake_world(device)
                def plane(i1, j1, i2, j2, *, with_climate):
                    y = torch.arange(i1, i2, device=device).double()[:, None]
                    x = torch.arange(j1, j2, device=device).double()[None, :]
                    return {'elev': (1000+3*x+2*y).float()}
                world.get = plane
                base = phasor._base(world, 4, -32, -32, 32, 32, None)
                h, s = phasor.sample_phasor(world, 4, -32, -32, 32, 32)
                self.assertTrue(torch.equal(base, h))
                self.assertEqual(s.count_nonzero().item(), 0)

    def test_profile_radial_limit_and_lod_fades(self):
        p, _ = phasor._profile(torch.tensor([0., math.pi], dtype=torch.float64))
        torch.testing.assert_close(p, torch.tensor([0., 1.], dtype=torch.float64))
        h, hr, hi, _, _ = phasor._transfer(torch.tensor(0., dtype=torch.float64), torch.tensor(0., dtype=torch.float64))
        self.assertAlmostEqual(h.item(), phasor._PROFILE_MEAN)
        self.assertTrue(torch.isfinite(hr+hi))
        angles = torch.linspace(-math.pi, math.pi, 100, dtype=torch.float64)
        tiny_h = phasor._transfer(1e-9*angles.cos(), 1e-9*angles.sin())[0]
        self.assertLess((tiny_h-phasor._PROFILE_MEAN).abs().max().item(), 1e-8)
        for wave in phasor.WAVELENGTHS:
            self.assertEqual(phasor._lod_weight(0, wave), 0)
            self.assertEqual(phasor._lod_weight(6, wave), 1)
            for level in range(7):
                if phasor._lod_weight(level, wave):
                    self.assertGreaterEqual(wave/(30/2**level), 4)

    def test_cascade_amplitudes_and_compact_support_jacobian(self):
        for device in self.devices():
            patterns = phasor._Patterns(SimpleNamespace(seed=31), ConstantControls())
            y = torch.linspace(-50, 50, 5, device=device, dtype=torch.float64)[:, None]
            x = torch.linspace(-60, 60, 7, device=device, dtype=torch.float64)[None, :]
            parent = patterns.sample(0, y, x)
            for index in (1, 2):
                child = patterns.sample(index, y, x)
                self.assertTrue((child[5] <= .2*parent[0]+1e-12).all())
                self.assertTrue((child[0] >= 0).all())
                parent = child
            # Stress the tapered Gaussian annulus, including either side of
            # a hash-cell boundary; no finite-difference derivative is used
            # by the implementation itself.
            eps = 1e-5
            y = torch.tensor([[179.999, 180.001, -180.001]], device=device, dtype=torch.float64)
            x = torch.tensor([[13., 13., 13.]], device=device, dtype=torch.float64)
            actual = patterns.sample(0, y, x)[1]
            finite = (patterns.sample(0, y+eps, x)[0]-patterns.sample(0, y-eps, x)[0])/(2*eps)
            torch.testing.assert_close(actual, finite, atol=1e-7, rtol=2e-6)

    def test_r1_analytic_jacobian(self):
        for device in self.devices():
            # Include varying native amplitude so this tests the product rule,
            # not only a constant-amplitude Gaussian field.
            world = fake_world(device)
            controls = phasor._Controls(world, (-1600, -1600, 1600, 1600), None)
            patterns = phasor._Patterns(world, controls)
            y = torch.linspace(-110, 130, 9, device=device, dtype=torch.float64)[:, None]
            x = torch.linspace(-140, 90, 11, device=device, dtype=torch.float64)[None, :]
            r = patterns.sample(0, y, x)
            eps = .0001
            for axis in (1, 2):
                offset_y, offset_x = (eps, 0) if axis == 1 else (0, eps)
                finite = (patterns.sample(0, y+offset_y, x+offset_x)[0] -
                          patterns.sample(0, y-offset_y, x-offset_x)[0])/(2*eps)
                torch.testing.assert_close(r[axis], finite, rtol=3e-5, atol=3e-6)

    def test_inclined_plane_spectral_orientation(self):
        # Fourier wavevector is perpendicular to ravine lines: compare the
        # inferred line direction with the plane's downslope, modulo pi.
        for device in self.devices():
            patterns = phasor._Patterns(SimpleNamespace(seed=31), ConstantControls())
            y = torch.arange(-128, 128, device=device, dtype=torch.float64)[:, None]*3
            x = torch.arange(-128, 128, device=device, dtype=torch.float64)[None, :]*3
            r = patterns.sample(0, y, x)[0].cpu().numpy()
            spectrum = np.abs(np.fft.fft2((r-r.mean())*np.outer(np.hanning(256), np.hanning(256))))**2
            freq = np.fft.fftfreq(256)
            ky, kx = np.meshgrid(freq, freq, indexing='ij')
            band = (kx*kx+ky*ky > (.02)**2) & (kx*kx+ky*ky < .15**2)
            tensor = np.array([[np.sum(spectrum*band*kx*kx), np.sum(spectrum*band*kx*ky)],
                               [np.sum(spectrum*band*kx*ky), np.sum(spectrum*band*ky*ky)]])
            _, vectors = np.linalg.eigh(tensor)
            line = vectors[:, 0]
            angle = math.degrees(math.acos(np.clip(abs(np.dot(line, [.8, .6])), 0, 1)))
            self.assertLess(angle, 15)

    def test_cancellation(self):
        def cancel():
            raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            phasor.sample_phasor(fake_world(), 4, -8, -8, 8, 8, check=cancel)

    def test_impulse_frequency_direction_and_weight_variation(self):
        for device in self.devices():
            patterns = phasor._Patterns(SimpleNamespace(seed=31), ConstantControls())
            y = torch.tensor([[-1800.], [1800.]], device=device, dtype=torch.float64)
            x = y.T
            params, _ = patterns._parameters(0, y, x)
            frequency = (params[3]**2+params[4]**2).sqrt()/(2*math.pi)
            self.assertGreater(frequency.std().item(), .08)
            self.assertGreaterEqual(frequency.min().item(), .8-1e-12)
            self.assertLessEqual(frequency.max().item(), 1.25+1e-12)
            direction = torch.atan2(params[3], params[4])
            nominal = math.atan2(.4, -.3)
            difference = torch.atan2((direction-nominal).sin(), (direction-nominal).cos())
            self.assertGreater(difference.std().item(), .1)
            self.assertLessEqual(difference.abs().max().item(), math.radians(18)+1e-12)
            weights = params[2]
            kurtosis = weights.pow(4).mean()/weights.square().mean().square()
            self.assertGreater(kurtosis.item(), 2.2)

    def test_patch_clusters_head_taper_and_patch_jacobian(self):
        for device in self.devices():
            y = torch.linspace(-700, 700, 65, device=device, dtype=torch.float64)[:, None]
            x = torch.linspace(-600, 600, 71, device=device, dtype=torch.float64)[None, :]
            patch, py, px = phasor._patch(17, 1, y, x)
            self.assertGreater((patch < .05).double().mean().item(), .1)
            self.assertGreater((patch > .9).double().mean().item(), .02)
            eps = .0001
            for axis, analytic in ((0, py), (1, px)):
                dy, dx = (eps, 0) if axis == 0 else (0, eps)
                finite = (phasor._patch(17, 1, y+dy, x+dx)[0]-
                          phasor._patch(17, 1, y-dy, x-dx)[0])/(2*eps)
                torch.testing.assert_close(analytic, finite, atol=2e-8, rtol=3e-5)
            q = torch.tensor([0., .2, .4, .7, 1.], device=device, dtype=torch.float64)
            zero, one = torch.zeros_like(q), torch.ones_like(q)
            gate, _, _ = phasor._head_gate((q, one, zero, zero, zero, one, zero, zero))
            self.assertEqual(gate[0].item(), 0)
            self.assertEqual(gate[-1].item(), 0)
            self.assertEqual(gate[2].item(), 1)

    def test_sublinear_intensity_and_surface_stability(self):
        self.assertEqual(phasor._intensity_gain(.5), .5)
        self.assertEqual(phasor._intensity_gain(1), 1)
        self.assertAlmostEqual(phasor._intensity_gain(2), 4/3)
        self.assertLess(phasor._intensity_gain(100), 1.5)
        for device in self.devices():
            with analytic_detail(device):
                world = fake_world(device)
                args = (4, -32, -32, 32, 32)
                a, sa = phasor.sample_phasor(world, *args)
                b, sb = phasor.sample_phasor(world, *args, intensity=2)
                base = phasor._base(world, *args, None)
                self.assertTrue(torch.equal(sa, sb))
                self.assertLessEqual(rms(b-base), 1.335*rms(a-base))

    def test_drainage_channel_distance_is_capped_and_seamless(self):
        for device in self.devices():
            world = fake_world(device)
            def area(world, y0, x0, y1, x1):
                y = torch.arange(y0, y1, device=device)[:, None]
                x = torch.arange(x0, x1, device=device)[None, :]
                return torch.where((y % 19 == 0) | (x % 23 == 0), 1e6, 900.)
            with patch('terrain_amplification.sample_drainage', side_effect=area):
                whole = phasor._drainage_distance(world, -31, -37, 39, 47)
                part = phasor._drainage_distance(world, -15, -17, 25, 31)
                self.assertTrue(torch.equal(part, whole[16:56, 20:68]))
                self.assertEqual(whole[31, 37].item(), 0)
                with patch('terrain_phasor.USE_NATIVE_DRAINAGE', True), analytic_detail(device):
                    whole_h, whole_s = phasor.sample_phasor(world, 4, -16, -20, 24, 30)
                    part_h, part_s = phasor.sample_phasor(world, 4, -9, -11, 19, 23)
                    self.assertTrue(torch.equal(part_h, whole_h[7:35, 9:43]))
                    self.assertTrue(torch.equal(part_s, whole_s[:, 7:35, 9:43]))
            def empty(world, y0, x0, y1, x1):
                return torch.zeros((y1-y0, x1-x0), device=device)
            with patch('terrain_amplification.sample_drainage', side_effect=empty):
                result = phasor._drainage_distance(world, -3, -7, 9, 11)
                self.assertTrue((result == 240).all())

    def test_explicit_drainage_channel_strength_direction_and_heads(self):
        for device in self.devices():
            world = drainage_world(device)
            with patch('terrain_amplification.sample_drainage', side_effect=channel_area):
                gain, fy, fx, oy, ox, near = phasor._drainage_controls(world, -80, -20, 40, 21)
                self.assertGreater(gain[80:, 20].mean().item(), .5)
                self.assertLess(gain[:20, 20].max().item(), .01, 'No incision above the contributing head')
                self.assertEqual(gain[:, 0].count_nonzero().item(), 0, 'Remote interfluve stays quiet')
                self.assertGreater(fy[80:, 20].mean().item(), .9)
                self.assertLess(fx[80:, 20].abs().max().item(), .02)
                a = phasor._Controls(world, (-700,-700,700,700), None, drainage=True)
                b = phasor._Controls(world, (-700,-700,700,700), None)
                y = torch.tensor([[180.]], device=device, dtype=torch.float64)
                x = torch.tensor([[120.]], device=device, dtype=torch.float64)
                ca, cb = a.sample(y,x), b.sample(y,x)
                self.assertLess((ca[4]/ca[3]).abs().item(), (cb[4]/cb[3]).abs().item())

    def test_explicit_drainage_phase_locked_analytic_jacobian(self):
        for device in self.devices():
            world = drainage_world(device)
            with patch('terrain_amplification.sample_drainage',side_effect=channel_area):
                controls = phasor._Controls(world,(-1600,-1600,1600,1600),None,drainage=True)
                patterns = phasor._Patterns(world,controls,drainage=True)
                y = torch.linspace(-500,100,5,device=device,dtype=torch.float64)[:,None]
                x = torch.linspace(-80,110,7,device=device,dtype=torch.float64)[None,:]
                result = patterns.sample(0,y,x)
                self.assertGreater(result[5].max().item(),.01)
                eps = 1e-4
                for axis in (0,1):
                    dy,dx = (eps,0) if axis==0 else (0,eps)
                    finite = (patterns.sample(0,y+dy,x+dx)[0]-
                              patterns.sample(0,y-dy,x-dx)[0])/(2*eps)
                    torch.testing.assert_close(result[axis+1],finite,atol=1e-7,rtol=2e-5)

    def test_explicit_drainage_determinism_seams_and_surface(self):
        for device in self.devices():
            with patch('terrain_amplification.sample_drainage', side_effect=channel_area), analytic_detail(device):
                world = drainage_world(device)
                for level in (2, 4, 6):
                    full = phasor.sample_phasor(world, level, -42,-39,50,51, drainage=True)
                    again = phasor.sample_phasor(drainage_world(device), level, -42,-39,50,51, drainage=True)
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(full,again)))
                    for y0,y1 in ((-42,3),(3,50)):
                        for x0,x1 in ((-39,5),(5,51)):
                            tile = phasor.sample_phasor(world,level,y0,x0,y1,x1,drainage=True)
                            for k in (0,1):
                                self.assertTrue(torch.equal(tile[k],full[k][...,y0+42:y1+42,x0+39:x1+39]))
                    self.assertTrue(((full[1]>=0)&(full[1]<=1)).all())
                    self.assertTrue(torch.isfinite(full[0]).all())
                seed = phasor.sample_phasor(drainage_world(device, seed=18),4,-42,-39,50,51,drainage=True)
                reference = phasor.sample_phasor(world,4,-42,-39,50,51,drainage=True)
                self.assertFalse(torch.equal(seed[0],reference[0]))

    def test_explicit_drainage_mean_native_consistency_and_zoom(self):
        for device in self.devices():
            with patch('terrain_amplification.sample_drainage', side_effect=channel_area), analytic_detail(device):
                world = drainage_world(device)
                for level in (2,4):
                    args = (level,-256,-256,256,256)
                    height,_ = phasor.sample_phasor(world,*args,drainage=True)
                    delta = height-phasor._base(world,*args,None)
                    self.assertGreater(rms(delta), .01)
                    self.assertLess(abs(delta.double().mean().item()), .02*rms(delta))
                    if level == 2:
                        self.assertLess(rms(F.avg_pool2d(delta[None,None],4)), .15*rms(delta))
                previous = None
                for level in range(1,7):
                    half = 4*2**level
                    args = (level,-half,-half,half,half)
                    height,_ = phasor.sample_phasor(world,*args,drainage=True)
                    base = phasor._base(world,*args,None)
                    delta = height-base
                    if previous is not None:
                        down = lambda v: F.avg_pool2d(v[None,None],2)[0,0]
                        bound = {2:.25,3:.25,4:.1,5:.05,6:.02}[level]
                        self.assertLess(rms(down(delta)-previous[1]), bound*rms(delta)+1e-4)
                        self.assertLess(rms(down(height)-previous[0]), .5)
                    previous = height,delta

    def test_explicit_drainage_identity_flat_and_fused_fallback(self):
        for device in self.devices():
            with patch('terrain_amplification.sample_drainage', side_effect=channel_area), analytic_detail(device,zero=True):
                world = drainage_world(device,flat=True)
                for level in (0,2,4,6):
                    args = (level,-16,-16,16,16)
                    height,surface = phasor.sample_phasor(world,*args,drainage=True)
                    base = refinement.sample_refined(world,*args)
                    self.assertTrue(torch.equal(height,base))
                    self.assertEqual(surface.count_nonzero().item(),0)
                world = drainage_world(device)
                args = (4,-24,-24,24,24)
                base = refinement.sample_refined(world,*args)
                height,surface = phasor.sample_phasor(world,*args,intensity=0,drainage=True)
                self.assertTrue(torch.equal(height,base))
                self.assertEqual(surface.count_nonzero().item(),0)
                if device == 'cuda':
                    fast = phasor.sample_phasor(world,*args,drainage=True)
                    with patch('terrain_phasor._kernel',return_value=None):
                        slow = phasor.sample_phasor(world,*args,drainage=True)
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(fast,slow)))

    def test_large_area_centered_displacement(self):
        # 3.84 km at -2 and .96 km at -4: finite-area bias <2% of detail RMS.
        for device in self.devices():
            with analytic_detail(device):
                world = fake_world(device)
                for level in (2, 4):
                    h, _ = phasor.sample_phasor(world, level, -256, -256, 256, 256)
                    base = phasor._base(world, level, -256, -256, 256, 256, None)
                    delta = h-base
                    self.assertGreater(rms(delta), .1)
                    self.assertLess(abs(delta.double().mean().item()), .02*rms(delta))

    def test_native_cell_mean_consistency(self):
        # Compare averaged output with the same averaged C2/decoder base,
        # isolating added detail from existing spline quadrature differences.
        # Required bound: native-cell mean residual <15% of detail RMS.
        for device in self.devices():
            with analytic_detail(device):
                world = fake_world(device)
                args = (2, -128, -128, 128, 128)
                h, _ = phasor.sample_phasor(world, *args)
                base = phasor._base(world, *args, None)
                delta = h-base
                means = F.avg_pool2d(delta[None, None], 4)[0, 0]
                self.assertLess(rms(means), .15*rms(delta))
                # Guard the usefulness of the test: unprojected 60 m waves
                # do move native-cell means substantially.
                with patch('terrain_phasor.PROJECT_NATIVE_MEAN', False):
                    raw, _ = phasor.sample_phasor(world, *args)
                self.assertGreater(rms(F.avg_pool2d((raw-base)[None, None], 4)), .2*rms(raw-base))

    def test_zoom_continuity_same_physical_footprint(self):
        # Adjacent LODs on the same 240 m square. Bounds are relative to fine
        # delta RMS: <=25% at the first two transitions, <=10/5/2% thereafter.
        # Also bound the actual full-height transition to .5 m RMS; report
        # baseline quadrature separately rather than hiding it in detail.
        for device in self.devices():
            with analytic_detail(device):
                for relief in ('alpine', 'montserrat', 'new_zealand'):
                    world = fake_world(device, relief)
                    previous = None
                    for level in range(1, 7):
                        half = 4*2**level
                        args = (level, -half, -half, half, half)
                        h, _ = phasor.sample_phasor(world, *args)
                        base = phasor._base(world, *args, None)
                        delta = h-base
                        if previous is not None:
                            coarse = F.avg_pool2d(h[None, None], 2)[0, 0]
                            coarse_delta = F.avg_pool2d(delta[None, None], 2)[0, 0]
                            bound = {2: .25, 3: .25, 4: .1, 5: .05, 6: .02}[level]
                            with self.subTest(device=device, relief=relief, level=level):
                                self.assertLess(rms(coarse_delta-previous[1]), bound*rms(delta)+1e-4)
                                self.assertLess(rms(coarse-previous[0]), .5)
                        previous = h, delta

    def test_centered_projected_analytic_jacobian(self):
        for device in self.devices():
            world = fake_world(device)
            controls = phasor._Controls(world, (-1600, -1600, 1600, 1600), None)
            patterns = phasor._Patterns(world, controls, bounds=(-200, -200, 200, 200))
            projection = phasor._project_first(patterns, patterns.bounds, 2, device)
            y = torch.linspace(-110, 130, 9, device=device, dtype=torch.float64)[:, None]
            x = torch.linspace(-140, 90, 11, device=device, dtype=torch.float64)[None, :]
            raw, c = patterns.sample(0, y, x), projection.sample(y, x)
            eps = .0001
            def displacement(y, x):
                r = patterns.sample(0, y, x)
                return r[0]-r[8]*r[5]-projection.sample(y, x)[0]
            for axis in (1, 2):
                dy, dx = (eps, 0) if axis == 1 else (0, eps)
                finite = (displacement(y+dy, x+dx)-displacement(y-dy, x-dx))/(2*eps)
                analytic = raw[axis]-raw[8]*raw[axis+5]-c[axis]
                torch.testing.assert_close(analytic, finite, rtol=3e-5, atol=3e-6)

    @unittest.skipUnless(torch.cuda.is_available() and os.getenv('TERRAIN_TEST_CUDA') == '1', 'CUDA gate')
    def test_fused_fallback_equivalence(self):
        if phasor._kernel('cuda') is None:
            self.skipTest('CuPy/NVRTC unavailable; CUDA PyTorch fallback covered by other tests')
        world = fake_world('cuda')
        with analytic_detail('cuda'):
            for level, bounds in ((2, (-24, -28, 29, 31)), (4, (-24, -28, 29, 31)),
                                  (6, (-152, -152, 152, 152))):
                fast = phasor.sample_phasor(world, level, *bounds)
                with patch('terrain_phasor._kernel', return_value=None):
                    slow = phasor.sample_phasor(world, level, *bounds)
                # Transcendentals may differ by an FP64 ulp, but final FP32
                # products must agree bit for bit on the supported CUDA path.
                with self.subTest(level=level):
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(fast, slow)))


if __name__ == '__main__':
    unittest.main()
