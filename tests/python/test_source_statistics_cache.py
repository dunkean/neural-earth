"""CPU-only durable source-statistics cache contracts using tiny real rasters."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

import terrain_world as world


class SourceStatisticsCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        for index, name in enumerate(world.SOURCE_FILES):
            values = np.arange(36 * 8, dtype=np.float32).reshape(36, 8) + index * 100
            if index == 0:
                values -= 150
            with rasterio.open(self.data / name, 'w', driver='GTiff', width=8,
                               height=36, count=1, dtype='float32',
                               transform=from_origin(-180, 90, 45, 5)) as dataset:
                dataset.write(values, 1)
        self.stats_path = self.data / 'synthetic_map_stats.json'
        self.stats_path.write_text(json.dumps({'noise_quantile_tables': [[0, .5, 1]] * 4}))
        self.patches = [patch.object(world, 'DATA_ROOT', self.data),
                        patch.object(world, 'SOURCE_STATS_CACHE_ROOT', self.root / 'cache'),
                        patch.dict('os.environ', {'TERRAIN_SOURCE_STATS_CACHE_DISABLE': '0'})]
        for item in self.patches:
            item.start()
        world.source_distributions.cache_clear()

    def tearDown(self):
        world.source_distributions.cache_clear()
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def assertExact(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            if isinstance(expected[key], np.ndarray):
                self.assertEqual(expected[key].dtype, actual[key].dtype)
                self.assertEqual(expected[key].shape, actual[key].shape)
                self.assertEqual(expected[key].tobytes(), actual[key].tobytes())
            else:
                self.assertEqual(expected[key], actual[key])

    def test_warm_cache_is_exact_and_skips_raster_reads_and_quantiles(self):
        baseline = world.source_distributions()
        world.source_distributions.cache_clear()
        with patch.object(world.rasterio, 'open', side_effect=AssertionError('raster read')), \
             patch.object(world.np, 'quantile', side_effect=AssertionError('sort')):
            self.assertExact(baseline, world.source_distributions())

    def test_corrupt_archive_recomputes_and_repairs(self):
        baseline = world.source_distributions()
        next((self.root / 'cache').glob('*.npz')).write_bytes(b'corrupted')
        world.source_distributions.cache_clear()
        with patch.object(world.rasterio, 'open', wraps=world.rasterio.open) as compute:
            self.assertExact(baseline, world.source_distributions())
            self.assertEqual(compute.call_count, 5)
        world.source_distributions.cache_clear()
        with patch.object(world.rasterio, 'open', side_effect=AssertionError('raster read')):
            self.assertExact(baseline, world.source_distributions())

    def test_metadata_array_hash_and_dtype_are_checked(self):
        baseline = world.source_distributions()
        path = next((self.root / 'cache').glob('*.npz'))
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays['climate'] = arrays['climate'].astype(np.float64)
        np.savez(path, **arrays)
        world.source_distributions.cache_clear()
        with patch.object(world.rasterio, 'open', wraps=world.rasterio.open) as compute:
            self.assertExact(baseline, world.source_distributions())
            self.assertEqual(compute.call_count, 5)

    def test_source_content_and_stats_dependency_changes_invalidate(self):
        baseline = world.source_distributions()
        with rasterio.open(self.data / world.SOURCE_FILES[0], 'r+') as dataset:
            values = dataset.read(1)
            values[0, 0] += 10
            dataset.write(values, 1)
        world.source_distributions.cache_clear()
        changed = world.source_distributions()
        self.assertNotEqual(baseline['source_digest'], changed['source_digest'])
        self.stats_path.write_text(json.dumps({'noise_quantile_tables': [[0, .6, 1]] * 4}))
        world.source_distributions.cache_clear()
        changed_stats = world.source_distributions()
        self.assertNotEqual(changed['source_digest'], changed_stats['source_digest'])
        self.assertEqual(float(changed_stats['noise_quantiles'][0, 1]), float(np.float32(.6)))
        self.assertEqual(len(list((self.root / 'cache').glob('*.npz'))), 3)

    def test_disable_recomputes_without_persistent_write(self):
        with patch.dict('os.environ', {'TERRAIN_SOURCE_STATS_CACHE_DISABLE': '1'}):
            baseline = world.source_distributions()
        self.assertFalse((self.root / 'cache').exists())
        world.source_distributions.cache_clear()
        self.assertExact(baseline, world.source_distributions())

    def test_source_order_invalidates_without_computing_wrong_channel_mapping(self):
        world.source_distributions()
        identity, key = world._source_stats_identity()
        path = self.root / 'cache' / f'{key}.npz'
        self.assertIsNotNone(world._read_source_stats_cache(path, identity))
        with patch.object(world, 'SOURCE_FILES', tuple(reversed(world.SOURCE_FILES))):
            reordered, reordered_key = world._source_stats_identity()
            self.assertEqual(identity['sources'], reordered['sources'])
            self.assertNotEqual(key, reordered_key)
            self.assertIsNone(world._read_source_stats_cache(path, reordered))
            self.assertFalse((path.parent / f'{reordered_key}.npz').exists())

    def test_gdal_version_invalidates(self):
        identity, key = world._source_stats_identity()
        self.assertEqual(identity['gdal'], rasterio.__gdal_version__)
        with patch.object(world.rasterio, '__gdal_version__', 'fixture-next-gdal'):
            changed, changed_key = world._source_stats_identity()
        self.assertNotEqual(key, changed_key)
        self.assertEqual(changed['gdal'], 'fixture-next-gdal')

    def test_computation_source_change_invalidates_and_recomputes(self):
        baseline = world.source_distributions()
        _, key = world._source_stats_identity()
        original_getsource = world.inspect.getsource

        def changed_getsource(function):
            source = original_getsource(function)
            if function is world._compute_source_distributions:
                source += '\n# fixture implementation revision\n'
            return source

        world.source_distributions.cache_clear()
        with patch.object(world.inspect, 'getsource', side_effect=changed_getsource), \
             patch.object(world.rasterio, 'open', wraps=world.rasterio.open) as reads:
            _, changed_key = world._source_stats_identity()
            self.assertNotEqual(key, changed_key)
            self.assertExact(baseline, world.source_distributions())
            self.assertEqual(reads.call_count, 5)
        self.assertTrue((self.root / 'cache' / f'{changed_key}.npz').is_file())
        self.assertEqual(len(list((self.root / 'cache').glob('*.npz'))), 2)

    def test_unwritable_cache_falls_back_and_cleans_temporary_files(self):
        # Mock permissions rather than chmod: Windows ACLs and privileged test
        # users do not reliably enforce POSIX read-only directory modes.
        with patch.dict('os.environ', {'TERRAIN_SOURCE_STATS_CACHE_DISABLE': '1'}):
            baseline = world.source_distributions()
        targets = ('pathlib.Path.mkdir', 'terrain_world.tempfile.NamedTemporaryFile',
                   'terrain_world.os.fsync', 'terrain_world.os.replace')
        for target in targets:
            with self.subTest(target=target):
                world.source_distributions.cache_clear()
                with patch(target, side_effect=PermissionError('fixture read-only cache')) as failure:
                    self.assertExact(baseline, world.source_distributions())
                    failure.assert_called_once()
                self.assertFalse(list((self.root / 'cache').glob('*.npz')))
                self.assertFalse(list((self.root / 'cache').glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
