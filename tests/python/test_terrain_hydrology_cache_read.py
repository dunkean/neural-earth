"""Cache-only Base delivery: no model creation, generation or eviction race."""
import ast
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock

from tools._bootstrap import activate
activate()
import numpy as np
import torch


class CachedBaseTests(unittest.TestCase):
    def setUp(self):
        # Load the server helper alone: these checks never initialize a GPU/server.
        source = Path('backend/terrain_server.py').read_text(encoding='utf-8')
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                        and n.name == 'read_hydrology_cached_base')
        levels = torch.tensor([[-5., 10.], [0., 30.]])
        data = torch.zeros((6, 2, 2))
        data[-1] = 2.
        data[4] = (levels+31.4)/38.6*2.
        self.data = data
        self.pins = 0
        def begin(_):
            self.pins += 1
        def end(_):
            self.pins -= 1
        def read(_, region):
            self.assertGreater(self.pins, 0)
            self.assertEqual(tuple((s.stop-s.start-1)//s.step+1 for s in region), (6, 2, 2))
            return data
        self.store = SimpleNamespace(begin_access=begin, end_access=end,
            is_window_processed=Mock(return_value=True), read_pixels=Mock(side_effect=read))
        window = SimpleNamespace(intersecting_windows=Mock(side_effect=lambda *a, **k: iter([(0, 0, 0)])))
        tensor = SimpleNamespace(shape=(6, None, None), uuid='base', _store=self.store, output_window=window)
        self.runtime = SimpleNamespace(lock=threading.Lock(), worlds={('fixture', 42): SimpleNamespace(latents=tensor)})
        self.settings = {'world_topology': 'plane'}
        namespace = dict(np=np, torch=torch, OrderedDict=OrderedDict,
            runtimes_lock=threading.Lock(), runtimes={0: self.runtime},
            _hydrology_base_lock=threading.Lock(), _hydrology_base_revision=0,
            resolve_generation=lambda _: SimpleNamespace(settings=self.settings),
            profile_bounds=lambda _: (-20e6, -10e6, 20e6, 10e6))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<cached Base reader>', 'exec'), namespace)
        self.read = namespace['read_hydrology_cached_base']
        self.namespace = namespace
        self.bounds = (0., 0., 480., 480.)

    def test_reads_complete_fused_base_under_eviction_pin(self):
        from infinite_tensor import TensorWindow
        from infinite_tensor.tilestore import MemoryTileStore
        # Exercise the installed store's slice and fusion contract, rather
        # than allowing a mock to accept invalid unnormalised slices.
        tensor = self.runtime.worlds[('fixture', 42)].latents
        tensor.output_window = TensorWindow((6, 2, 2))
        tensor.dtype, tensor.device = torch.float32, torch.device('cpu')
        tensor._store = MemoryTileStore()
        tensor._store.register_tensor(tensor)
        tensor._store.notify_window_processed(tensor.uuid, (0, 0, 0), self.data)
        result = self.read(42, 'fixture', self.bounds)
        np.testing.assert_allclose(result['base_height'], [[-25., 100.], [0., 900.]], atol=.001)
        self.assertEqual(result['base_bounds'], self.bounds)
        self.assertEqual(self.pins, 0)
        self.assertFalse(self.runtime.lock.locked())
        self.assertEqual(self.namespace['_hydrology_base_revision'], 1)
        tensor._store.clear_cache(tensor.uuid)
        self.runtime.lock.acquire()
        try:
            # The NN can evict its centre windows while generating the edges.
            # Captured CPU Base must survive and remain readable without it.
            again = self.read(42, 'fixture', self.bounds)
            np.testing.assert_array_equal(again['base_height'], result['base_height'])
            self.assertFalse(again['base_height'].flags.writeable)
            self.assertEqual(self.namespace['_hydrology_base_revision'], 1)
        finally:
            self.runtime.lock.release()

    def test_missing_busy_unloaded_or_blended_terrain_never_reads(self):
        self.store.is_window_processed.return_value = False
        self.assertIsNone(self.read(42, 'fixture', self.bounds))
        self.store.is_window_processed.return_value = True
        self.runtime.lock.acquire()
        try:
            self.assertIsNone(self.read(42, 'fixture', self.bounds))
        finally:
            self.runtime.lock.release()
        self.assertIsNone(self.read(43, 'fixture', self.bounds))
        self.settings['world_topology'] = 'sphere'
        self.assertIsNone(self.read(42, 'fixture', (0., 6.1e6, 480., 6.1e6+480.)))
        self.store.read_pixels.assert_not_called()
        self.assertEqual(self.pins, 0)
        self.assertFalse(self.runtime.lock.locked())

    def test_fast_height_delivery_keeps_useful_tiles_across_subscription_epochs(self):
        source = Path('backend/terrain_server.py').read_text(encoding='utf-8')
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                        and n.name == 'hydrology_display')
        service = SimpleNamespace(ensure=Mock(return_value={'status': 'ready', 'key': 'fixture'}),
            physical=Mock(side_effect=AssertionError('fast height delivery rasterized rivers')))
        namespace = dict(request=SimpleNamespace(endpoint='height_tile', args={
            'hydro': '1', 'session': 'old-view-session', 'epoch': '0'}),
            hydrology_options=lambda: dict(detail=False, cosmetics=()), hydrology_service=lambda: service,
            world_manifest=lambda *a: {}, world_identity=lambda _: 'world', jobs=object(),
            _hydrology_base_revision=0)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<fast height delivery>', 'exec'), namespace)
        raw = np.ones((4, 4), np.float32)
        height, water, report = namespace['hydrology_display'](42, 'fixture', 2, 0, 0, raw, {})
        self.assertIs(height, raw)
        self.assertIsNone(water)
        self.assertTrue(report['hydro_vector_preview'])
        service.physical.assert_not_called()


if __name__ == '__main__':
    unittest.main()
