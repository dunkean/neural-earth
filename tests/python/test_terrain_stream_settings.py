"""Settings API and grouped persistence contracts, without server startup."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path

import ast
from collections import OrderedDict
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from flask import Flask, jsonify, request
from test_terrain_windows import _world, _manifest
from terrain_coarse import CoarsePreparation
from terrain_background import CoarseBackground
from terrain_jobs import TerrainJobs


class StreamSettingsTests(unittest.TestCase):
    def test_api_validates_and_applies_without_clearing_world_stores(self):
        app=Flask(__name__)
        owner=SimpleNamespace()
        old_pool=Mock()
        owner._terrain_stream_pool=('old',old_pool)
        profile=SimpleNamespace(cuda_graphs=True,coarse_solver_graphs=True,coarse_streams=1)
        world=SimpleNamespace(coarse_model=owner,device='cpu',_terrain_profile=profile,
                              coarse=SimpleNamespace(_batch_size=1),tile_store=object())
        original_store=world.tile_store
        setting={'coarse':4}
        namespace=dict(app=app,jsonify=jsonify,request=request,gpu_lock=threading.RLock(),
            coarse_stream_setting=setting,shared_pipeline=world,background_world=world,
            worlds=OrderedDict(a=world),preview_worlds={},polar_worlds={})
        source=(source_path('terrain_server.py', root=_REPO_ROOT))
        tree=ast.parse(source.read_text(encoding='utf-8'))
        node=next(n for n in tree.body if getattr(n,'name',None)=='inference_stream_settings')
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        client=app.test_client()
        self.assertEqual(client.get('/api/inference/streams').json['coarse'],4)
        for value in (None,True,0,3,32,'4',4.0):
            self.assertEqual(client.post('/api/inference/streams',json={'coarse':value}).status_code,400)
        self.assertEqual(setting['coarse'],4)
        response=client.post('/api/inference/streams',json={'coarse':16})
        self.assertEqual(response.status_code,200)
        self.assertEqual(world._terrain_coarse_streams,16)
        self.assertEqual(world.coarse._batch_size,1)  # CPU keeps scalar path.
        self.assertIs(world.tile_store,original_store)
        old_pool.close.assert_called_once()
        self.assertEqual(client.get('/api/inference/streams').json['coarse'],16)

    def test_grouped_persistence_keeps_window_budget_and_replays_saved_data(self):
        with tempfile.TemporaryDirectory() as directory:
            world=_world(cache_bytes=100000)
            world._terrain_profile=SimpleNamespace(coarse_batch=1,coarse_streams=4)
            world.coarse._batch_size=4
            preparation=CoarsePreparation(directory,_manifest(701),bounds=(0,0,16*7680,8*7680),
                                          async_persistence=False).install(world)
            try:
                groups=[]
                original=preparation._model_f
                # The wrapper retains this callback, so observe the real tensor
                # output function rather than replacing the repair callback.
                installed=world.coarse._f
                def observe(indices,*args):
                    groups.append(list(indices))
                    return installed(indices,*args)
                world.coarse._f=observe
                result=preparation.step(world,budget_windows=6)
                self.assertEqual([len(g) for g in groups],[4,2])
                self.assertEqual(result['quantum_windows'],6)
                self.assertEqual(world.calls['coarse'],list(preparation._indices[:6]))
                values=[preparation._load_window(i) for i in preparation._indices[:6]]
                self.assertTrue(all(value is not None for value in values))
                before=len(world.calls['coarse'])
                world.coarse._f(list(preparation._indices[:6]))
                self.assertEqual(len(world.calls['coarse']),before)
            finally:
                preparation.close()

    def test_background_counts_windows_and_bounds_last_quantum(self):
        jobs=TerrainJobs()
        counts=[]
        def compute(seed,profile,*,budget_windows):
            counts.append(budget_windows)
            return dict(complete_windows=sum(counts),total_windows=100,quantum_windows=budget_windows)
        background=CoarseBackground(jobs,compute,None,window_quantum=lambda:4)
        try:
            background.start(42,'natural',max_windows=6)
            import time
            deadline=time.monotonic()+3
            while background.status()['state']=='running':
                if time.monotonic()>deadline:self.fail('Background budget did not complete')
                time.sleep(.01)
            self.assertEqual(counts,[4,2])
            self.assertEqual(background.status()['computed_windows'],6)
            self.assertEqual(background.status()['quanta'],2)
        finally:
            background.close();jobs.close()


if __name__=='__main__':unittest.main()
