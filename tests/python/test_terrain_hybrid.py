"""Hybrid height/material composition, drainage seams and conservative zooms."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'terrain-diffusion'))

import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import terrain_amplification as amplification
import terrain_hybrid as hybrid
import terrain_phasor as phasor
import terrain_refinement as refinement
from test_terrain_phasor import drainage_world, channel_area, analytic_detail, rms


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def devices(self):
        return ['cpu','cuda'] if torch.cuda.is_available() and os.getenv('TERRAIN_TEST_CUDA') == '1' else ['cpu']

    def test_identity_validation_and_cancellation(self):
        for device in self.devices():
            world = drainage_world(device)
            with analytic_detail(device), patch('terrain_amplification.sample_drainage',side_effect=AssertionError('off path reads drainage')):
                for level,strength in ((0,1),(2,0),(6,0)):
                    args = level,-13,-17,19,23
                    h,s = hybrid.sample_hybrid(world,*args,intensity=strength)
                    self.assertTrue(torch.equal(h,refinement.sample_refined(world,*args)))
                    self.assertEqual(s.count_nonzero().item(),0)
            for args in ((7,0,0,1,1),(2,.5,0,1,1),(2,0,0,0,1),(2,0,0,1025,1025)):
                with self.assertRaises(ValueError):
                    hybrid.sample_hybrid(world,*args)
            for strength in (-1,float('nan'),float('inf'),None):
                with self.assertRaises(ValueError):
                    hybrid.sample_hybrid(world,2,0,0,2,2,intensity=strength)
            def cancel():
                raise RuntimeError('cancelled')
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                hybrid.sample_hybrid(world,4,-4,-4,4,4,check=cancel)

    def test_surface_halo_is_evaluated_once_and_cropped(self):
        for device in self.devices():
            world = drainage_world(device)
            def surface(world,level,i1,j1,i2,j2,height_fn,*,check=None):
                h = height_fn(i1-7,j1-7,i2+7,j2+7)
                # Querying the centre again must hit the provider's first halo.
                self.assertTrue(torch.equal(height_fn(i1,j1,i2,j2),h[7:-7,7:-7]))
                return torch.ones((4,i2-i1,j2-j1),device=h.device)*.25
            with patch('terrain_amplification.sample_drainage',side_effect=channel_area), analytic_detail(device):
                expected = phasor.sample_phasor(world,4,-19,-21,23,25,drainage=True)[0]
                with patch.object(amplification,'sample_surface',surface,create=True), \
                     patch('terrain_phasor.sample_phasor',wraps=phasor.sample_phasor) as calls:
                    h,s = hybrid.sample_hybrid(world,4,-19,-21,23,25)
                    self.assertEqual(calls.call_count,1)
                    self.assertEqual(calls.call_args.args[2:],(-26,-28,30,32))
                self.assertTrue(torch.equal(h,expected))
                self.assertTrue((s==.25).all())

    def test_fallback_is_explicit_phasor_surface(self):
        for device in self.devices():
            world = drainage_world(device)
            with patch('terrain_amplification.sample_drainage',side_effect=channel_area), analytic_detail(device), \
                 patch.object(amplification,'sample_surface',None,create=True):
                expected = phasor.sample_phasor(world,4,-20,-24,24,28,drainage=True)
                actual = hybrid.sample_hybrid(world,4,-20,-24,24,28)
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,actual)))

    def test_provider_reuses_split_surface_halo(self):
        world = drainage_world()
        def read(world,level,a,b,c,d,**kwargs):
            self.assertTrue(kwargs['drainage'])
            y = torch.arange(a,c,dtype=torch.float32)[:,None]
            x = torch.arange(b,d,dtype=torch.float32)[None,:]
            h = y*.2+x*.01
            return h,torch.ones((4,c-a,d-b))*.3
        with patch('terrain_phasor.sample_phasor',side_effect=read) as calls:
            provider = hybrid._HeightProvider(world,4,1,None)
            for ya,yb in ((-6,0),(0,6)):
                for xa,xb in ((-6,0),(0,6)):
                    provider(ya,xa,yb,xb)
            h = provider(-4,-4,4,4)
            self.assertEqual(calls.call_count,4,'Visible heights must reuse the four halo pieces')
            y = torch.arange(-4,4,dtype=torch.float32)[:,None]
            x = torch.arange(-4,4,dtype=torch.float32)[None,:]
            self.assertTrue(torch.equal(h,y*.2+x*.01))

    def test_rectangular_native_decoder_cannot_move_hybrid_heights(self):
        # Reproduce the upstream short-side resize failure without models:
        # arbitrary rectangular native reads acquire a datum error. The new
        # path must use fixed square native reads for heights AND diagnostics.
        for device in self.devices():
            clean,broken = drainage_world(device),drainage_world(device)
            original = broken.get
            def get(i1,j1,i2,j2,*,with_climate):
                data = original(i1,j1,i2,j2,with_climate=with_climate)
                if i2-i1 != j2-j1:
                    data['elev'] = data['elev']+17
                return data
            broken.get = get
            with patch('terrain_amplification.sample_drainage',side_effect=channel_area), analytic_detail(device):
                for level in (4,6):
                    args = level,-56,-62,84,78
                    expected = hybrid.sample_hybrid(clean,*args)
                    whole = hybrid.sample_hybrid(broken,*args)
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,whole)))
                    part = hybrid.sample_hybrid(broken,level,-40,-45,65,53)
                    self.assertTrue(torch.equal(part[0],whole[0][16:121,17:115]))
                    self.assertTrue(torch.equal(part[1],whole[1][:,16:121,17:115]))

    @unittest.skipUnless(callable(getattr(amplification,'sample_surface',None)),
                         'Public terrain_amplification.sample_surface has not landed; hybrid uses phasor surface fallback')
    def test_public_surface_and_deterministic_negative_mosaic(self):
        for device in self.devices():
            with patch('terrain_amplification.sample_drainage',side_effect=channel_area), analytic_detail(device):
                world = drainage_world(device)
                for level in (2,4,6):
                    args = level,-40,-36,32,40
                    whole = hybrid.sample_hybrid(world,*args)
                    again = hybrid.sample_hybrid(drainage_world(device),*args)
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(whole,again)))
                    rows = []
                    for y0,y1 in ((-40,-3),(-3,32)):
                        tiles = [hybrid.sample_hybrid(world,level,y0,x0,y1,x1) for x0,x1 in ((-36,5),(5,40))]
                        rows.append(tuple(torch.cat([t[k] for t in tiles],dim=-1) for k in (0,1)))
                    mosaic = tuple(torch.cat([r[k] for r in rows],dim=-2) for k in (0,1))
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(whole,mosaic)))
                    h,s = whole
                    self.assertEqual(h.dtype,torch.float32)
                    self.assertEqual(s.dtype,torch.float32)
                    self.assertEqual(s.shape,(4,72,76))
                    self.assertTrue(h.is_contiguous() and s.is_contiguous())
                    self.assertTrue(torch.isfinite(s).all())
                    self.assertTrue(((s>=0)&(s<=1)).all())
                    self.assertGreater(s.max().item(),.05)
                seed = hybrid.sample_hybrid(drainage_world(device,seed=18),4,-40,-36,32,40)
                reference = hybrid.sample_hybrid(world,4,-40,-36,32,40)
                self.assertFalse(torch.equal(seed[0],reference[0]))

    def test_mean_native_consistency_continuity_and_flat(self):
        for device in self.devices():
            with patch('terrain_amplification.sample_drainage',side_effect=channel_area), analytic_detail(device):
                world = drainage_world(device)
                h,_ = hybrid.sample_hybrid(world,2,-192,-192,192,192)
                delta = h-phasor._base(world,2,-192,-192,192,192,None)
                self.assertGreater(rms(delta),.01)
                self.assertLess(abs(delta.double().mean().item()),.02*rms(delta))
                self.assertLess(rms(F.avg_pool2d(delta[None,None],4)),.15*rms(delta))
                previous = None
                for level in range(1,7):
                    half = 4*2**level
                    args = level,-half,-half,half,half
                    h,_ = hybrid.sample_hybrid(world,*args)
                    base = phasor._base(world,*args,None)
                    d = h-base
                    if previous is not None:
                        down = lambda x: F.avg_pool2d(x[None,None],2)[0,0]
                        bound = {2:.25,3:.25,4:.1,5:.05,6:.02}[level]
                        self.assertLess(rms(down(d)-previous[1]),bound*rms(d)+1e-4)
                        self.assertLess(rms(down(h)-previous[0]),.5)
                    previous = h,d
            with patch('terrain_amplification.sample_drainage',side_effect=channel_area), analytic_detail(device,zero=True):
                flat = drainage_world(device,flat=True)
                h,_ = hybrid.sample_hybrid(flat,6,-32,-32,32,32)
                self.assertTrue(torch.equal(h,refinement.sample_refined(flat,6,-32,-32,32,32)))

    def test_composite_identity(self):
        self.assertIn(phasor.PHASOR_DRAINAGE_VERSION,hybrid.VERSION)
        self.assertIn(hybrid.SURFACE_VERSION,hybrid.VERSION)
        self.assertEqual(hybrid.SURFACE_CHANNELS,('flow','deposit','scree','rock'))

    @unittest.skipUnless(torch.cuda.is_available() and os.getenv('TERRAIN_TEST_CUDA')=='1','CUDA gate')
    def test_canonical_native_cache_can_cross_cuda_streams(self):
        world = drainage_world('cuda')
        first,second = torch.cuda.Stream(),torch.cuda.Stream()
        with patch('terrain_amplification.sample_drainage',side_effect=channel_area), analytic_detail('cuda'):
            with torch.cuda.stream(first):
                a = phasor.sample_phasor(world,4,-32,-32,32,32,drainage=True)
            with torch.cuda.stream(second):
                b = phasor.sample_phasor(world,4,-32,-32,32,32,drainage=True)
            torch.cuda.synchronize()
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))


if __name__ == '__main__':
    unittest.main()
