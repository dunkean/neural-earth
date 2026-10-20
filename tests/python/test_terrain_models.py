"""Model API validation, persisted changes and finite-tile asset isolation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import hashlib
import json
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from flask import Flask
from backend.terrain_models import register_model_routes
from backend.terrain_tile40 import Tile40Assets,register_tile40_routes
from backend.terrain_tile40_python import Tile40Python
from distill.live_runtime import LiveModels,DEFAULT_SELECTION


class ModelRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'config.json'
        choices={stage:{'teacher':dict(label='Original',path=None),choice:dict(label=choice,path=None)}
            for stage,choice in DEFAULT_SELECTION.items()}
        self.models=LiveModels(self.path,choices)
        self.app=Flask(__name__);self.restart=Mock()
        self.app.extensions['terrain_model_restart']=self.restart
        self.server=SimpleNamespace(app=self.app,live_models=lambda:self.models,
            generation_coordinator=SimpleNamespace(lock=threading.RLock(),suspend=Mock()),
            RESTART_COMMAND=None,WEB_ROOT=Path('.'))
        register_model_routes(self.server)
        self.client=self.app.test_client()

    def test_get_exposes_original_and_selected_models(self):
        response=self.client.get('/api/models')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json['selection'],DEFAULT_SELECTION)
        self.assertEqual(response.json['tile40_url'],'/tile40/')

    def test_retired_choice_rejected_without_restart_or_persisted_changes(self):
        before=self.path.read_bytes()
        response=self.client.post('/api/models',json=dict(selection=dict(DEFAULT_SELECTION,coarse='solver4')))
        self.assertEqual(response.status_code,400)
        self.assertEqual(self.path.read_bytes(),before)
        self.restart.assert_not_called()

    def test_valid_selection_saves_then_requests_restart(self):
        desired=dict.fromkeys(DEFAULT_SELECTION,'teacher')
        response=self.client.post('/api/models',json=dict(selection=desired))
        self.assertEqual(response.status_code,202)
        self.assertEqual(json.loads(self.path.read_text())['selection'],desired)
        self.restart.assert_called_once()
        self.assertTrue(self.client.get('/api/models').json['switching'])
        self.assertEqual(self.client.post('/api/models',json=dict(selection=desired)).status_code,409)

    def test_restart_unavailable_does_not_modify_saved_selection(self):
        del self.app.extensions['terrain_model_restart']
        before=self.path.read_bytes()
        self.assertEqual(self.client.post('/api/models',json=dict(selection=DEFAULT_SELECTION)).status_code,503)
        self.assertEqual(self.path.read_bytes(),before)


class Tile40RouteTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'catalog.json').write_text(json.dumps(dict(status='ready',decoders=[])))
        directory=self.root/'decoder16';directory.mkdir()
        self.model=directory/'decoder.onnx';self.model.write_bytes(b'model')
        item=dict(file='decoder.onnx',bytes=5,sha256=hashlib.sha256(b'model').hexdigest())
        (directory/'manifest.json').write_text(json.dumps(dict(models={'decoder':item})))
        (directory/'expected-height.bin').write_bytes(b'private fixture')
        self.app=Flask(__name__)
        self.assets=Tile40Assets(self.root)
        register_tile40_routes(self.app,self.assets)
        self.client=self.app.test_client()

    def test_scoped_assets_are_isolated_and_private_fixtures_are_not_served(self):
        response=self.client.get('/tile40/model/decoder16/decoder.onnx')
        self.assertEqual(response.data,b'model')
        self.assertEqual(response.headers['Cross-Origin-Opener-Policy'],'same-origin')
        response.close()
        self.assertEqual(self.client.get('/tile40/model/decoder16/expected-height.bin').status_code,404)
        self.assertEqual(self.client.get('/tile40/model/retired/decoder.onnx').status_code,404)
        self.assertEqual(self.client.get('/tile40/model/decoder16/../manifest.json').status_code,404)

    def test_tampered_model_is_not_served(self):
        self.model.write_bytes(b'other')
        self.assertEqual(self.client.get('/tile40/model/decoder16/decoder.onnx').status_code,503)

    def test_worker_and_runtime_have_working_scoped_paths(self):
        for name in ['worker.js','demo.mjs','runtime/tile40_runtime.mjs','runtime/noise.mjs','ort/ort.webgpu.min.js']:
            response=self.client.get('/tile40/'+name)
            self.assertEqual(response.status_code,200,name)
            self.assertEqual(response.headers['Cross-Origin-Embedder-Policy'],'require-corp')
            response.close()

    def test_python_catalog_is_available_without_browser_exports(self):
        (self.root/'catalog.json').unlink()
        response=self.client.get('/tile40/python/catalog.json?decoder=decoder64')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json['selected_models']['decoder'],'decoder64')
        self.assertEqual(response.json['models']['encoder']['training_step'],8000)
        self.assertEqual(self.client.get('/tile40/python/catalog.json?decoder=retired').status_code,400)

    def test_python_generation_returns_binary_and_matching_provenance(self):
        runtime=Mock()
        runtime.generate.return_value=(b'\0'*16,dict(pixels=2,provider='python-cpu'))
        app=Flask('python-response');register_tile40_routes(app,self.assets,runtime)
        response=app.test_client().post('/tile40/python/generate',json={'decoder':'decoder64'})
        self.assertEqual(response.data,b'\0'*16)
        self.assertEqual(json.loads(response.headers['X-Terrain-Metadata'])['provider'],'python-cpu')
        self.assertEqual(response.headers['Cache-Control'],'no-store')
        runtime.generate.side_effect=BlockingIOError('busy')
        self.assertEqual(app.test_client().post('/tile40/python/generate',json={}).status_code,409)

    def test_python_rejects_invalid_inputs_before_loading_models(self):
        import numpy as np
        runtime=Tile40Python()
        payload=dict(decoder='decoder16',provider='cpu',input=dict(sketch_m=np.zeros((16,16)).tolist(),climate='arid',seed=1))
        with patch.object(runtime,'_load') as load:
            for field,value in [('seed',True),('climate','unknown'),('sketch_m',[[float('nan')]*16]*16),('sketch_m',[[9000]*16]*16)]:
                invalid=dict(payload,input=dict(payload['input'],**{field:value}))
                with self.assertRaises(ValueError):runtime.generate(invalid)
            with self.assertRaises(ValueError):runtime.generate(None)
            load.assert_not_called()
        self.assertEqual(self.client.post('/tile40/python/generate',data='invalid',content_type='application/json').status_code,400)

    def test_python_keeps_models_resident_and_reloads_after_decoder_change(self):
        import numpy as np
        runtime=Tile40Python()
        generated=dict(height_m=np.zeros((1334,1334),np.float32),side_m=40000,spacing_m=40000/1334,
                       timing=dict(native_total_seconds=.1))
        generator=Mock();generator.generate.return_value=generated
        payload=dict(decoder='decoder16',provider='cpu',input=dict(sketch_m=np.zeros((16,16)).tolist(),climate='arid',seed=1))
        with patch.object(runtime,'_load',return_value=generator) as load:
            for decoder in ['decoder16','decoder16','decoder64']:
                body,metadata=runtime.generate(dict(payload,decoder=decoder))
                self.assertEqual(len(body),1334*1334*4)
                self.assertEqual(metadata['manifest']['selected_models']['decoder'],decoder)
            self.assertEqual(load.call_count,2)


if __name__=='__main__':unittest.main()
