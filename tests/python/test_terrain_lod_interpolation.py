"""Geometry, continuity and boundedness of the globally anchored display spline."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import terrain_lod_interpolation as interp


def terrain(y0, x0, y1, x1):
    y, x = np.mgrid[y0:y1, x0:x1]
    h = 20*np.sin(x*.15)+12*np.cos(y*.21)+x*.2-y*.3
    return h, np.stack([.5+.3*np.sin(x*.1+y*.2)]*4)


class InterpolationTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ, TERRAIN_RELIEF_FLOORS='');self.env.start();self.addCleanup(self.env.stop)

    def test_exact_knots_and_plane(self):
        v,_=terrain(-10,-12,14,16)
        ys=np.arange(-7,10);xs=np.arange(-9,12)
        out=interp.evaluate(v[None],-10,-12,ys,xs)[0]
        np.testing.assert_array_equal(out,v[3:20,3:24])
        y,x=np.mgrid[-10:14,-12:16];plane=4+2*x-3*y
        ys=np.linspace(-6,8,61);xs=np.linspace(-8,10,87)
        np.testing.assert_allclose(interp.evaluate(plane[None].astype(float),-10,-12,ys,xs)[0],4+2*xs[None]-3*ys[:,None],atol=2e-13)

    def test_overlaps_negative_far_coordinates_determinism_and_cache(self):
        for origin in (-100003,0,100003):
            world=SimpleNamespace();read=Mock(side_effect=terrain)
            a,_=interp.sample_interpolated(world,-12,origin,origin,origin+60,origin+80,reader=read)
            count=read.call_count
            b,s=interp.sample_interpolated(world,-12,origin+13,origin+17,origin+53,origin+73,reader=read)
            np.testing.assert_array_equal(a[13:53,17:73],b);self.assertEqual(read.call_count,count)
            self.assertTrue(np.all((s>=0)&(s<=1)))
            c,_=interp.sample_interpolated(SimpleNamespace(),-12,origin,origin,origin+60,origin+80,reader=terrain)
            np.testing.assert_array_equal(a,c)

    def test_c2_second_derivatives_cross_lattice_lines(self):
        h,_=terrain(-10,-10,14,14)
        for axis in (0,1):
            errors=[]
            for eps in (.0004,.0001):
                step=eps/4;values=[]
                for centre in (-eps,eps):
                    positions=np.array([centre-step,centre,centre+step])
                    ys,xs=(positions,np.array([.37])) if axis==0 else (np.array([.37]),positions)
                    line=interp.evaluate(h[None],-10,-10,ys,xs).ravel()
                    values.append((line[0]-2*line[1]+line[2])/step**2)
                errors.append(abs(values[0]-values[1]))
            self.assertLess(errors[1],errors[0]*.4+1e-5)
            self.assertLess(errors[1],.02)

    def test_cliff_has_no_overshoot_and_surfaces_are_bounded(self):
        y,x=np.mgrid[-6:8,-6:8];h=np.where(x>=0,100.,0.)
        out=interp.evaluate(h[None],-6,-6,np.linspace(-2,3,97),np.linspace(-2,3,193))[0]
        self.assertGreaterEqual(out.min(),-.02000001);self.assertLessEqual(out.max(),100.02000001)

    def test_policy_override_and_identity(self):
        self.assertEqual(interp.MIN_LOD,-12)
        with patch.dict(os.environ,TERRAIN_RELIEF_FLOORS='{"classic":-3,"hybrid":-4}'):
            self.assertEqual(interp.relief_floor(),-3)
            self.assertEqual(interp.interpolation_suffix(-12),'/'+'interp-from-3-v1')
            self.assertEqual(interp.interpolation_suffix(-4,'hybrid'),'')
            self.assertEqual(interp.interpolation_suffix(-5,'hybrid'),'/interp-from-4-v1')
        for raw in ('[]','{"mse":-7}','{"unknown":-3}','{"classic":true}'):
            with patch.dict(os.environ,TERRAIN_RELIEF_FLOORS=raw),self.assertRaises(ValueError):interp.relief_floor()

    def test_floor_tiles_seed_blocks_without_resampling(self):
        world=SimpleNamespace();h,s=terrain(-24,-24,280,280)
        interp.remember_floor(world,'classic','fixture',-24,-24,h)
        read=Mock(side_effect=AssertionError('cached floor block should be reused'))
        interp.sample_interpolated(world,-8,80,80,140,140,reader=read,identity='fixture')
        read.assert_not_called()

    @unittest.skipUnless(os.environ.get('TERRAIN_TEST_CUDA')=='1','optional CUDA parity')
    def test_cuda_limiter_is_exact_and_window_independent(self):
        import torch
        values=np.random.default_rng(4).normal(size=(5,80,80))
        cpu=interp._fields(values);gpu=interp._fields(values,0)
        self.assertIsNone(interp._limiter_error)
        for key in cpu:np.testing.assert_array_equal(cpu[key],gpu[key])
        world=SimpleNamespace(device=torch.device('cuda:0'))
        a,_=interp.sample_interpolated(world,-7,-101,-103,203,201,reader=terrain)
        b,_=interp.sample_interpolated(world,-7,-81,-83,183,181,reader=terrain)
        np.testing.assert_array_equal(a[20:-20,20:-20],b)


if __name__=='__main__':unittest.main()
