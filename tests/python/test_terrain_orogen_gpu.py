"""V3 keeps CUDA spatial projection; the V1 source-rewriting engine is retired."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools._bootstrap import activate
activate()

import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial import cKDTree
import terrain_orogen as core
import terrain_orogen_gpu as gpu
from terrain_generation import OROGEN_GPU_PARAMETERS


class GPUOptions(unittest.TestCase):
    def test_only_the_ported_projection_is_offered(self):
        self.assertEqual(set(OROGEN_GPU_PARAMETERS), {'orogen_gpu_raster'})
        with patch.object(gpu, 'implementation_identity', side_effect=AssertionError('GPU initialized')):
            config = core.style_config('earthlike')
        self.assertFalse(config['orogen_gpu_raster'])
        with self.assertRaises(ValueError):
            core.style_config('earthlike', {'orogen_gpu_climate': True})

    def test_unavailable_projection_uses_cpu_tree(self):
        identity = dict(available=False, reason='missing NVRTC', fallback='CPU cKDTree')
        with patch.object(gpu, 'implementation_identity', return_value=identity):
            config = core.style_config('earthlike', {'orogen_gpu_raster': True})
        points = np.eye(3, dtype=np.float32)
        points = np.concatenate((points, -points))
        query = core.sphere_raster(16, 8)
        np.testing.assert_array_equal(core._nearest_regions(points, query, config), cKDTree(points).query(query, k=4)[1])


@unittest.skipUnless(gpu.implementation_identity()['available'], 'optional CUDA projection unavailable')
class GPUProjection(unittest.TestCase):
    def test_search_matches_cpu_at_seams_and_poles(self):
        rng = np.random.default_rng(42)
        points = rng.normal(size=(4000, 3)).astype(np.float32)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        query = np.concatenate((core.sphere_raster(64,32), np.eye(3, dtype=np.float32), -np.eye(3, dtype=np.float32)))
        np.testing.assert_array_equal(gpu.nearest_regions(points, query), cKDTree(points).query(query, k=4)[1])


if __name__ == '__main__':
    unittest.main()
