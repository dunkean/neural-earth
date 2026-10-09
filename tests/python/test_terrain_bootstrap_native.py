"""CPU integration checks of the real pinned atlas exporter; no neural model."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import hashlib
import json
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

import terrain_bootstrap as bootstrap


class NativeHeightmapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='bootstrap-tests-')
        cls.cache_patch = patch.object(bootstrap, 'CACHE_ROOT', Path(cls.temporary.name))
        cls.cache_patch.start()
        cls.worlds = {style: bootstrap.WorldHeightmap(0, style) for style in bootstrap.STYLES}

    @classmethod
    def tearDownClass(cls):
        cls.cache_patch.stop()
        cls.temporary.cleanup()

    def test_real_source_provenance_and_signed_metres(self):
        identity = bootstrap.implementation_identity()
        self.assertEqual(identity['upstream_commit'], bootstrap.UPSTREAM_COMMIT)
        self.assertIn('Cargo.lock', identity['bridge_files'])
        self.assertIn('crates/world-core/src/geometry.rs', identity['upstream_files'])
        self.assertEqual(identity['binary_sha256'], bootstrap._sha_file(bootstrap.BINARY))
        for style, world in self.worlds.items():
            with self.subTest(style=style):
                self.assertEqual(world.height_m.shape, (512, 1024))
                self.assertEqual(world.height_m.dtype, np.float32)
                self.assertTrue(np.isfinite(world.height_m).all())
                self.assertLess(world.height_m.min(), -3000)
                self.assertGreater(world.height_m.max(), 2000)
                self.assertEqual(world.metadata['generator_version'], bootstrap.UPSTREAM_VERSION)
                self.assertEqual(world.metadata['native_config']['gravity_m_s2'], 9.81)
                self.assertEqual(world.metadata['native_config']['erosion_steps'], 0)
                self.assertTrue(np.array_equal(world.raw_height_m < 0, world.height_m < 0))
                self.assertTrue(world.metadata['attempts'][-1]['accepted'])
                self.assertEqual(world.metadata['selected_seed_u64'], world.metadata['attempts'][-1]['seed_u64'])
                self.assertFalse(world.height_m.flags.writeable)

    def test_styles_have_measured_topology(self):
        for style, world in self.worlds.items():
            topo = world.metadata['canonical_height_topology']
            with self.subTest(style=style):
                self.assertTrue(.05 < topo['land_fraction'] < .5)
                largest = topo['largest_land_fraction']
                if style == 'gondwana':
                    self.assertGreaterEqual(largest, .85)
                elif style == 'continents':
                    self.assertLessEqual(largest, .65)
                    self.assertGreaterEqual(topo['significant_components'], 3)
                elif style == 'earthlike':
                    self.assertLessEqual(largest, .75)
                    self.assertGreaterEqual(topo['significant_components'], 2)
                else:
                    self.assertLessEqual(largest, .15)
                    self.assertGreaterEqual(topo['components'], 100)

    def test_sampling_global_centres_wrapping_and_polar_clamp(self):
        world = self.worlds['earthlike']
        x0, y0, x1, y1 = world.bounds
        xs = x0 + (np.arange(world.width) + .5) / world.width * (x1-x0)
        ys = y0 + (np.arange(world.height) + .5) / world.height * (y1-y0)
        self.assertTrue(np.array_equal(world.sample_height_m(xs, ys), world.height_m))
        self.assertTrue(np.array_equal(world.sample_height_m(xs + (x1-x0), ys), world.height_m))
        self.assertTrue(np.array_equal(world.sample_height_m(xs, [y0-1e6, y1+1e6]), world.height_m[[0, -1]]))
        seam = world.sample_height_m([x0, x1], [-1234, 5678])
        self.assertTrue(np.array_equal(seam[:, 0], seam[:, 1]))

    def test_random_access_order_and_partition_invariance(self):
        world = self.worlds['continents']
        rng = np.random.default_rng(724)
        xs = rng.uniform(-21e6, 21e6, 73)
        ys = rng.uniform(-11e6, 11e6, 41)
        whole = world.sample_height_m(xs, ys)
        self.assertTrue(np.array_equal(whole, np.concatenate([
            world.sample_height_m(xs[:19], ys), world.sample_height_m(xs[19:], ys)], axis=1)))
        permutation = rng.permutation(xs.size)
        self.assertTrue(np.array_equal(whole[:, permutation], world.sample_height_m(xs[permutation], ys)))

    def test_hypsometry_preserves_rank_zero_and_spherical_distribution(self):
        probabilities, land, depth, _ = bootstrap._source_hypsometry()
        world = self.worlds['archipelago']
        weights = np.broadcast_to(np.cos((np.arange(world.height)+.5)/world.height*np.pi-np.pi/2)[:, None], world.height_m.shape)
        for sign, target in [(1, land), (-1, depth)]:
            mask = sign * world.raw_height_m > 0
            order = np.argsort(sign * world.raw_height_m[mask])
            mapped = sign * world.height_m[mask]
            self.assertGreaterEqual(np.diff(mapped[order]).min(), 0)
            actual = bootstrap._weighted_quantile(mapped.astype(np.float64), weights[mask], probabilities)
            self.assertLess(np.max(np.abs(actual-target)), 80.)
        raw = world.raw_height_m.copy()
        raw[0, 0] = 0
        final, _ = bootstrap._remap_hypsometry(raw)
        self.assertEqual(final[0, 0], 0)

    def test_cache_reload_checks_bytes_and_receipt(self):
        world = self.worlds['earthlike']
        loaded = bootstrap.WorldHeightmap(0, 'earthlike')
        self.assertEqual(loaded.cache_key, world.cache_key)
        self.assertTrue(np.array_equal(loaded.height_m, world.height_m))
        namespace = world.metadata['namespace']
        original = world.cache_path.read_bytes()
        try:
            with np.load(world.cache_path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            arrays['height_m'] = arrays['height_m'].copy()
            arrays['height_m'][10, 20] += 50
            np.savez_compressed(world.cache_path, **arrays)
            self.assertIsNone(world._load(namespace))
            world.cache_path.write_bytes(b'PK\x03\x04broken archive')
            self.assertIsNone(world._load(namespace))
        finally:
            world.cache_path.write_bytes(original)

    def test_production_cpu_export_is_identical_across_thread_counts_and_u64_seed(self):
        request = {'seed': 2**64-1, 'style': 'earthlike',
                   'resolution': bootstrap.NATIVE_RESOLUTION,
                   'width': bootstrap.RASTER_WIDTH, 'height': bootstrap.RASTER_HEIGHT}
        self.assertEqual((request['resolution'], request['width'], request['height']), (256, 1024, 512))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'request.json').write_text(json.dumps(request))
            receipts = []
            outputs = []
            for threads in (1, 4):
                output = root/str(threads)
                subprocess.run([str(bootstrap.BINARY), str(root/'request.json'), str(output)],
                    env={**os.environ, 'RAYON_NUM_THREADS': str(threads)}, check=True, capture_output=True)
                outputs.append([(output/name).read_bytes() for name in ('height.f32', 'ocean.u8')])
                receipts.append(json.loads((output/'metadata.json').read_text()))
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(len(outputs[0][0]), 1024 * 512 * 4)
            self.assertEqual(len(outputs[0][1]), 1024 * 512)
            self.assertEqual(receipts[0]['attempts'], receipts[1]['attempts'])
            self.assertEqual(receipts[0]['native_world_id'], receipts[1]['native_world_id'])
            self.assertEqual(receipts[0]['native_config'], receipts[1]['native_config'])
            self.assertEqual(receipts[0]['requested_seed_u64'], str(2**64-1))

    def test_invalid_requests_fail_explicitly(self):
        for seed in (-1, 2**64, True, 1.5, '-1', 'abc'):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                bootstrap.WorldHeightmap(seed)
        with self.assertRaises(ValueError):
            bootstrap.WorldHeightmap(0, 'unknown')
        for xs, ys in [([np.nan], [0]), ([[0]], [0]), ([0], [np.inf])]:
            with self.assertRaises(ValueError):
                self.worlds['earthlike'].sample_height_m(xs, ys)

    def test_active_process_rejects_changed_source_or_executable(self):
        identity = deepcopy(bootstrap.implementation_identity())
        self.assertTrue(bootstrap.verify_implementation_identity(identity))
        self.assertTrue(bootstrap.verify_implementation_identity())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(bootstrap, '_native_source_identity', return_value=({}, 'changed')):
                # The identity cache is warm; the verifier still reads current bytes.
                with self.assertRaisesRegex(RuntimeError, 'sources changed'):
                    bootstrap.verify_implementation_identity()
                with self.assertRaisesRegex(RuntimeError, 'sources changed'):
                    bootstrap._generation_binary(identity, directory)
            executable = directory/'replacement.exe'
            executable.write_bytes(b'changed executable')
            with patch.object(bootstrap, 'BINARY', executable):
                with tempfile.TemporaryDirectory() as destination:
                    with self.assertRaisesRegex(RuntimeError, 'exporter changed'):
                        bootstrap._generation_binary(identity, Path(destination))


if __name__ == '__main__':
    unittest.main()
