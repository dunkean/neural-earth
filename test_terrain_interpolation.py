"""Coarse preview monotonicity, grid slope continuity, halos and real read paths."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.ndimage import map_coordinates
from terrain_interpolation import monotone_grid


class InterpolationTests(unittest.TestCase):
    def test_knots_terminal_knot_constants_and_finite_extremes(self):
        field=np.array([[0,2,-3,5],[1,4,-2,6],[2,5,1,7]],np.float32)
        actual=monotone_grid(field,np.arange(4),np.arange(3))
        self.assertEqual(actual.tobytes(),field.tobytes())
        constant=np.full((5,5),3,np.float32)
        self.assertTrue(np.all(monotone_grid(constant,[.1,1.3,4.],[.8,3.2])==3))
        extreme=field/7*np.finfo(np.float32).max
        self.assertTrue(np.isfinite(monotone_grid(extreme,[.25,1.5,2.75],[.5,1.5])).all())

    def test_bounds_no_overshoot_or_new_land_in_ocean_cell(self):
        rng=np.random.default_rng(31)
        field=rng.uniform(-100,100,(8,9)).astype(np.float32)
        x=np.linspace(.01,7.99,115);y=np.linspace(.01,6.99,91)
        values=monotone_grid(field,x,y)
        ix=np.floor(x).astype(int);iy=np.floor(y).astype(int)
        corners=np.stack([field[np.ix_(iy+dy,ix+dx)] for dy,dx in ((0,0),(0,1),(1,0),(1,1))])
        self.assertTrue(np.all(values>=corners.min(0)))
        self.assertTrue(np.all(values<=corners.max(0)))
        ocean=np.full((6,6),-10,np.float32);ocean[0,:]=100;ocean[-1,:]=100
        self.assertTrue(np.all(monotone_grid(ocean,np.linspace(1,4,61),np.linspace(2,3,21))<0))

    def test_c1_slope_at_grid_lines(self):
        row=np.array([0,1,5,6,3,0],np.float32)
        field=np.repeat(row[None,:],6,0)
        step=.001
        for knot in (1,2,3,4):
            values=monotone_grid(field,[knot-step,knot,knot+step],[2.25])[0].astype(np.float64)
            slopes=np.diff(values)/step
            self.assertLess(abs(slopes[0]-slopes[1]),.025)

    def test_tile_halos_and_grouped_sampling_same_global_values_and_reads(self):
        calls=[]
        def block(world,source,x0,y0,x1,y1):
            calls.append((source,x0,y0,x1,y1))
            yy,xx=np.meshgrid(np.arange(y0,y1),np.arange(x0,x1),indexing='ij')
            return (np.sin(xx*.03)*40+np.cos(yy*.07)*30-15).astype(np.float32)
        tree=ast.parse(Path(__file__).with_name('terrain_server.py').read_text(encoding='utf-8'))
        function=next(node for node in tree.body if getattr(node,'name',None)=='sample_field')
        namespace=dict(np=np,math=math,map_coordinates=map_coordinates,_field_block=block)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<actual server sampler>','exec'),namespace)
        sample=namespace['sample_field']
        ys=(np.arange(-24,280)+.5)*16
        xs=(np.arange(-24,280)+.5)*16
        left=sample(None,xs,ys,'coarse',smooth_coarse=True)
        right=sample(None,xs+4096,ys,'coarse',smooth_coarse=True)
        self.assertEqual(left[:,-48:].tobytes(),right[:,:48].tobytes())
        calls.clear();sample(None,xs,ys,'coarse');reference_calls=calls.copy()
        calls.clear();sample(None,xs,ys,'coarse',smooth_coarse=True)
        self.assertEqual(calls,reference_calls)
        # Sparse blocks include negative coordinates and exact integer endpoints.
        x=np.array([-1000.2,-500.,.25,600.,1000.8]);y=np.array([-1000.,-250.2,0.,500.5,1000.])
        calls.clear();sparse=sample(None,(x+.5)*256,(y+.5)*256,'coarse',smooth_coarse=True)
        read_calls=calls.copy()
        whole=block(None,'coarse',-1002,-1001,1003,1003)
        expected=monotone_grid(whole,x+1002,y+1001)
        expected=np.sign(expected)*expected**2
        self.assertEqual(sparse.tobytes(),expected.tobytes())
        calls.clear();sample(None,(x+.5)*256,(y+.5)*256,'coarse')
        self.assertEqual(calls,read_calls)

    def test_reject_invalid_coordinates_and_nonfinite(self):
        with self.assertRaises(ValueError):monotone_grid(np.ones((4,4)),[-.1],[1.])
        with self.assertRaises(ValueError):monotone_grid(np.full((4,4),np.nan),[1.],[1.])


if __name__=='__main__':unittest.main()
