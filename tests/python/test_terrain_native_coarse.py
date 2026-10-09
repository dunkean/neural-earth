"""Source alignment and transport tests without loading neural checkpoints."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import ast
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from flask import Flask, Response, jsonify, request

import terrain_native_coarse as native


class NativeCoarseTests(unittest.TestCase):
    bounds = (-20e6, -10e6, 20e6, 10e6)

    def test_neighbour_halos_are_identical_at_negative_coordinates(self):
        x, y = native.sample_axes(-2, -1, self.bounds)
        neighbour, _ = native.sample_axes(-1, -1, self.bounds)
        np.testing.assert_array_equal(x[-2*native.HALO:], neighbour[:2*native.HALO])
        self.assertEqual(len(x), native.WIDTH)
        np.testing.assert_array_equal(x/256-.5, np.rint(x/256-.5))
        x, _ = native.climate_axes(-2, -1, self.bounds)
        neighbour, _ = native.climate_axes(-1, -1, self.bounds)
        np.testing.assert_array_equal(x[-9:], neighbour[:9])

    def test_world_edges_extend_actual_source_cell_centres(self):
        x, _ = native.sample_axes(-21, 0, self.bounds)
        self.assertEqual(x[0]/256, np.floor(self.bounds[0]/7680)+.5)
        x, _ = native.sample_axes(20, 0, self.bounds)
        self.assertEqual(x[-1]/256, np.ceil(self.bounds[2]/7680)-.5)

    def test_fused_native_cells_are_not_interpolated_or_squared(self):
        x, y = native.sample_axes(-1, 0, self.bounds)
        def read(world, source, y0, x0, y1, x1, check):
            values=torch.arange(x0,x1,dtype=torch.float32)[None,:].expand(y1-y0,-1)
            return torch.stack((values*3,torch.full_like(values,3)))
        roots=native.read_native(None,x,y,read,lambda:None)
        np.testing.assert_array_equal(roots[0],np.rint(x/256-.5))
        self.assertEqual(roots.dtype,np.float32)
        self.assertTrue(roots.flags.c_contiguous)

    def test_encoding_preserves_physical_height_and_sea(self):
        h=np.array([-10000,-.2,0,.01,500,6000],np.float32)
        roots=native.encode_height(h)
        np.testing.assert_allclose(np.sign(roots)*roots**2,h,rtol=1e-6)

    def test_endpoint_reuses_source_and_promotes_preview_without_new_lod(self):
        app=Flask(__name__)
        state={'ready':False,'reads':0}
        def submit(key,compute,finalize,**kwargs):
            return SimpleNamespace(wait=lambda:finalize(compute()))
        def preview(seed,wp,x,y):
            state['reads']+=1
            return {'elev':np.full((len(y),len(x)),625,np.float32),
                    'climate':np.zeros((5,len(y),len(x)),np.float32)}
        ns=dict(app=app,np=np,torch=torch,Response=Response,jsonify=jsonify,request=request,
            native_coarse=native,VERSION='natural-v1',PROFILE='test',NATIVE=30,CLIMATE_SIZE=33,
            WORLD_BOUNDS=self.bounds,generation_profile=lambda:'natural',
            world_manifest=lambda *args:{},world_identity=lambda m:'identity',
            _tile_coordinates=lambda seed,lod,tx,ty:(seed,lod,int(tx),int(ty)),
            native_coarse_ready=lambda *args:state['ready'],
            native_coarse_cache=OrderedDict(),native_coarse_cache_lock=threading.Lock(),
            nullcontext=nullcontext,measured_lock=lambda *args:nullcontext(),gpu_lock=None,
            time=time,jobs=SimpleNamespace(submit=submit,check_current_interest=lambda:None),
            conditioning_preview=preview,get_world=lambda *args:None,
            sample_coarse_climate=lambda w,x,y,check:np.zeros((5,len(y),len(x)),np.float32),
            tile_relief_stats=lambda *args:{},_session=lambda x:x,
            _tile_headers=lambda response,report,hit:(response.headers.update({'Stage':report['stage'],'Cache':str(hit)}) or response),
            JobCancelled=type('JobCancelled',(Exception,),{}),QueueFull=type('QueueFull',(Exception,),{}))
        source=ast.parse(Path('terrain_server.py').read_text(encoding='utf-8'))
        selected=[node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in ('native_coarse_key','native_coarse_tile')]
        exec(compile(ast.Module(body=selected,type_ignores=[]),'terrain_server.py','exec'),ns)
        client=app.test_client();url='/coarse/natural-v1/42/-1/0.bin?profile=test'
        first=client.get(url);self.assertEqual(first.status_code,200)
        self.assertEqual(first.headers['X-Terrain-Encoding'],'signed-sqrt')
        payload=np.frombuffer(first.data,dtype='<f4')
        self.assertEqual(payload.size,native.WIDTH**2+5*native.CLIMATE_SIZE**2)
        self.assertTrue(np.all(payload[:native.WIDTH**2]==25))
        self.assertEqual(client.get(url).headers['Cache'],'True')
        self.assertEqual(state['reads'],2)
        from unittest.mock import patch
        with patch.object(native,'read_native',return_value=np.full((native.WIDTH,native.WIDTH),-10,np.float32)):
            promoted=client.get(url+'&learned=1')
        self.assertEqual(promoted.headers['Stage'],'coarse')
        self.assertEqual(client.get(url).headers['Stage'],'coarse')
        self.assertEqual(state['reads'],2)


if __name__=='__main__':
    unittest.main()
