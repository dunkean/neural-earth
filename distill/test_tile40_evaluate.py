import unittest
from pathlib import Path
import tempfile
import numpy as np
from distill.tile40_evaluate import directional_coherence,compare_height


class DirectionalCoherenceTests(unittest.TestCase):
    def test_parallel_ridges_have_more_direction_than_isotropic_texture(self):
        axis=np.arange(256)
        ridge=np.broadcast_to(10*np.sin(axis[None,:]*.3),(256,256)).copy()
        noise=np.random.default_rng(817).standard_normal((256,256))*10
        self.assertGreater(directional_coherence(ridge),.99)
        self.assertLess(directional_coherence(noise),.5)
        self.assertAlmostEqual(directional_coherence(ridge),directional_coherence(ridge+1000),places=8)
        self.assertEqual(directional_coherence(np.ones((256,256))),0.)

    def test_reference_cache_preserves_every_metric_and_rebuilds_corrupt_data(self):
        rng=np.random.default_rng(871)
        axis=np.arange(1334,dtype=np.float64)
        target=100.+axis[:,None]*.1+8*np.sin(axis[None,:]*.04)+rng.normal(size=(1334,1334))
        prediction=target+3*np.sin(axis[:,None]*.07)+rng.normal(size=target.shape)*.3
        expected=compare_height(prediction,target)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertEqual(compare_height(prediction,target,reference_cache=root),expected)
            self.assertEqual(compare_height(prediction,target,reference_cache=root),expected)
            path=next(root.glob('*.npz'))
            with np.load(path,allow_pickle=False) as saved:
                payload={name:saved[name].copy() for name in saved.files}
            payload['band0']+=100.
            np.savez(path,**payload)
            self.assertEqual(compare_height(prediction,target,reference_cache=root),expected)
            changed=compare_height(prediction,target+7.,reference_cache=root)
            self.assertEqual(len(list(root.glob('*.npz'))),2)
            self.assertNotEqual(changed['reference_stats'],expected['reference_stats'])


if __name__=='__main__':unittest.main()
