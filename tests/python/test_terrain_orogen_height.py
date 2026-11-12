"""V3 physical heights: linear coasts, uncapped imports and coordinate frames."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools._bootstrap import activate
activate()

import json
import subprocess
import unittest
import numpy as np
from terrain_orogen import _metres, _elevation, _atlas_to_elevation


class OrogenPhysicalHeightTests(unittest.TestCase):
    def test_coasts_are_linear_and_mountains_are_not_capped_at_6km(self):
        heights = np.array([-11000, -1, -.001, 0, .001, 1, 10, 100, 6000, 8550, 12000], np.float64)
        np.testing.assert_allclose(_metres(_elevation(heights)), heights, rtol=1e-7, atol=1e-6)
        np.testing.assert_array_equal(_metres(np.array([-.01, 0, .01])), [-10, 0, 10])
        # The old quartic plateau is gone: 1 cm in physical input stays 1 cm.
        self.assertAlmostEqual(float(_metres(.00001)), .01, places=8)

    def test_city_graph_resampling_keeps_ocean_and_high_mountains(self):
        points = np.array([[1., 0, 0], [0, 0, 1], [0, 0, -1]], np.float32)
        source = np.full((2, 4), 8500, np.float32)
        out = _atlas_to_elevation(source, points, np.array([1., -3., 2.], np.float32))
        np.testing.assert_array_equal(_metres(out), [8500, -3000, 8500])

    def test_shared_grid_sampling_has_the_same_longitude_and_pixel_centres(self):
        module = (ROOT/'native/orogen/climate.mjs').as_uri()
        script = f"""import {{sample}} from {json.dumps(module)};
const xyz=new Float32Array([Math.SQRT1_2,Math.SQRT1_2,0,Math.SQRT1_2,-Math.SQRT1_2,0]);
console.log(JSON.stringify([0,1].map(c=>sample({{w:4,h:2}},[0,1,2,3,0,1,2,3],xyz,c))));"""
        result = subprocess.run(['node','--input-type=module','-e',script], capture_output=True, text=True, check=True)
        np.testing.assert_allclose(json.loads(result.stdout), [2, 1], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
