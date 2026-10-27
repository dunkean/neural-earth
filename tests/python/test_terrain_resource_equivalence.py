"""Full-world byte equality against e2046f9 (before performance edits).

Retained Orogen inputs are outside Git. Recreate with benchmark_resources_compile
--prepare on the pinned revision, or set TERRAIN_RESOURCE_REFERENCE_INPUTS.
All arrays (floats included), units and complete deposit dictionaries are checked.
"""
from pathlib import Path
import json
import os
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools._bootstrap import activate
activate()
import numpy as np
import terrain_resources as R
from tools.benchmarks.benchmark_resources_compile import fingerprint, load_inputs

INPUTS = Path(os.environ.get('TERRAIN_RESOURCE_REFERENCE_INPUTS',
                             'E:/TerrainDiffusionRuntime/resources/perf-inputs'))


class FullWorldEquivalenceTests(unittest.TestCase):
    def test_reference_worlds_are_bit_identical_twice(self):
        for seed in (42, 20261027):
            path = INPUTS / f'{seed}.npz'
            if not path.exists():
                self.skipTest(f'Retained Orogen atlas missing: {path}; see module docstring')
            expected = json.loads((ROOT / f'tests/fixtures/resources-e2046f9-{seed}.json').read_text())
            self.assertEqual(expected['revision'], 'e2046f93bfe1b844bd8a8b838d6742b79e04a7ed')
            layers, height, _, bounds = load_inputs(path)
            self.assertEqual(fingerprint(dict(layers,height_m=height),[]), expected['inputs'])
            for repeat in range(2):
                with self.subTest(seed=seed, repeat=repeat), patch.dict(os.environ, TERRAIN_RESOURCES_FAST=str(1-repeat)):
                    grid, fields, deposits = R.compile_world(layers, height, seed, bounds=bounds)
                    self.assertEqual(grid.shape, (206, 410))
                    self.assertEqual(fingerprint(fields, deposits), expected['fingerprint'])

    def test_first_productivity_is_final_authority(self):
        from test_terrain_resources import atlas
        layers,height = atlas()
        snapshots = []
        _, fields, _ = R.compile_world(layers,height,42,lod=12,balance=False,
            bounds=(-np.pi*5e5,-np.pi*2.5e5,np.pi*5e5,np.pi*2.5e5),
            progress=lambda stage,f:snapshots.append((stage,dict(f))))
        self.assertEqual(snapshots[0][0],'climate')
        self.assertNotIn('rock_bias',snapshots[0][1])
        np.testing.assert_array_equal(snapshots[0][1]['npp_g_m2'], fields['npp_g_m2'])

    def test_background_job_and_lossless_disk_round_trip(self):
        import tempfile
        import threading
        from types import SimpleNamespace
        import terrain_resource_layers as L
        from test_terrain_resources import atlas
        layers,height = atlas()
        source = SimpleNamespace(layers=layers,height_m=height,seed=42,
                                 bounds=(-np.pi*5e5,-np.pi*2.5e5,np.pi*5e5,np.pi*2.5e5))
        original = R.compile_world
        entered, release = threading.Event(), threading.Event()
        def delayed(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError('test did not release compiler')
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,TERRAIN_RESOURCES_CACHE=directory):
            with patch.object(R,'compile_world',side_effect=delayed):
                job=L._Compile(L._identity(source),source)
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertIsNone(job.result)
                    self.assertIsNone(job.fields)
                    self.assertEqual(job.status()['state'],'compiling')
                finally:
                    release.set()
                job._thread.join(30)
            self.assertFalse(job._thread.is_alive())
            self.assertIsNone(job.error)
            self.assertIsNotNone(job.result)
            loaded=L._from_disk(job.key,source.bounds)
            self.assertIsNotNone(loaded)
            self.assertEqual(fingerprint(loaded[1],loaded[2]),fingerprint(job.result[1],job.result[2]))
            self.assertFalse(list(L._cache_root().glob('*.tmp.*')))

    def test_months_equal_independent_render_v2_reference(self):
        from terrain_resource_environment import months
        from terrain_render_v2 import evaluate
        rng = np.random.default_rng(913)
        point = rng.uniform(-6e6, 6e6, (256, 3)).astype(np.float32)
        gradient = rng.uniform(-.8, .8, (256, 2)).astype(np.float32)
        tpi = rng.uniform(-.3, .3, 256).astype(np.float32)
        ts = rng.uniform(-20, 40, 256).astype(np.float32)
        tw = rng.uniform(-40, 30, 256).astype(np.float32)
        ps, pw = [rng.uniform(0, 2500, 256).astype(np.float32) for _ in range(2)]
        for footprint in (30., 480., 30720.):
            got = months(point, gradient, tpi, ts, tw, ps, pw, 42, footprint, point)
            for month in range(12):
                reference = evaluate(point, np.zeros(256, np.float32), gradient, tpi, ts, tw, ps, pw,
                                     np.full(256, .4, np.float32), 0., (month+.5)/12, 42, footprint, point)
                for name, value in got[month].items():
                    np.testing.assert_array_equal(value, np.asarray(reference[name], np.float32),
                                                  err_msg=f'{footprint} month {month}: {name}')


if __name__ == '__main__':
    unittest.main()
