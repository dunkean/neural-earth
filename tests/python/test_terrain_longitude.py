"""Finite neural longitude strips join before lighting, on nonintegral worlds."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[2]/'backend')]
import terrain_longitude as L


class LongitudeTests(unittest.TestCase):
    bounds = (-2e6*np.pi, -1e6*np.pi, 2e6*np.pi, 1e6*np.pi)

    def test_every_lod_has_identical_two_sided_limits(self):
        for lod in (10, 7, 4, 2, 0, -2):
            step = 30.*2.**lod
            # A deliberately nonperiodic learned realization, with local detail.
            def height(x):
                return 500 + x/2e4 + 30*np.sin(x/15000) + 10*np.sin(x/1300)
            def read(col):
                x = (col*256+np.arange(-24,280)+.5)*step
                return np.broadcast_to(height(x), (3,304)).astype(np.float32)
            eps = min(.01, step/1000)
            x = np.array([self.bounds[2]-eps, self.bounds[0]+eps])
            tx = int(np.floor((x[0]/step-.5)/256))
            # Two independent windows approaching the exact physical seam.
            # Supply an actual source tile for the cache address, not the probe.
            v = L.stitch(np.broadcast_to(height(x),(3,2)), x, self.bounds, lod, tx+100000, read)
            np.testing.assert_allclose(v[:,0], v[:,1], atol=2e-4, err_msg=f'LOD {lod}')

    def test_interior_is_unchanged_and_reads_are_local(self):
        a = np.arange(304*3,dtype=np.float32).reshape(3,304)
        result = L.stitch(a,np.arange(304)*30,self.bounds,0,0,
                          lambda _: self.fail('An interior tile must not fetch other tiles'))
        self.assertIs(result,a)

    def test_both_end_tiles_use_the_same_canonical_strip(self):
        step = 120.
        def read(col):
            x=(col*256+np.arange(-24,280)+.5)*step
            return np.broadcast_to(300+x/3e4+np.sin(x/300)*30,(3,304)).astype(np.float32)
        for edge in (self.bounds[0],self.bounds[2]):
            col=int(np.floor(edge/(256*step)))
            x=(col*256+np.arange(-24,280)+.5)*step
            a=L.stitch(read(col),x,self.bounds,2,col,read)
            b=L.stitch(read(col),x+(self.bounds[2]-self.bounds[0]),self.bounds,2,col,read)
            np.testing.assert_allclose(a,b,atol=2e-4)


if __name__ == '__main__':
    unittest.main()
