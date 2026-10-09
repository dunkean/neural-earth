"""Queue and Flask protocol tests; no model load or CUDA execution."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path

import importlib
import json
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from terrain_jobs import TerrainJobs, JobCancelled
from terrain_disk_cache import TerrainDiskCache


class DiskCacheTests(unittest.TestCase):
    def test_restart_indexes_parameterized_worlds_as_independent_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            names=['natural--g'+'a'*24,'natural--g'+'b'*24]
            for name in names:
                p=source_path(name, root=root)/'physical-v1'/'42'/'0'/'0_0.npy'
                p.parent.mkdir(parents=True)
                p.write_bytes(b'payload')
            cache=TerrainDiskCache(root,'v',budget_bytes=100,grace_seconds=60)
            try:
                deadline=time.monotonic()+2
                while not cache.status()['indexed'] and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertEqual(set(cache.groups),{f'v/{name}/42/0/0/0' for name in names})
                self.assertEqual(cache.status()['bytes'],14)
            finally:
                cache.close()

    def test_restart_indexes_macro_profile_groups_for_quota(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'active-profile'
            protected='v/terrestrial-gondwana/17/0/0/0'
            files={}
            for profile in ('terrestrial-gondwana','terrestrial-continents','terrestrial-archipelago'):
                base=root/profile/'physical-v1'/'17'/'0'/'0_0'
                image=root/profile/'17'/'0'/'0_0'
                paths=[base.with_suffix('.npy'),base.with_suffix('.json'),
                       image.with_suffix('.png'),image.with_suffix('.json')]
                for path in paths:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(b'x'*40)
                files[profile]=paths
            # All files precede construction, as after a server restart.
            cache=TerrainDiskCache(root,'v',budget_bytes=180,
                protected_keys=lambda:{protected},grace_seconds=60)
            try:
                deadline=time.monotonic()+2
                while not cache.status()['indexed'] and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertTrue(cache.status()['indexed'])
                self.assertEqual(cache.status()['bytes'],480)
                self.assertEqual(set(cache.groups),{
                    f'v/{profile}/17/0/0/0' for profile in files})
                cache.grace_seconds=0
                cache.trim()
                self.assertEqual(cache.status()['bytes'],160)
                self.assertTrue(all(path.exists() for path in files['terrestrial-gondwana']))
                self.assertTrue(all(not path.exists() for profile in ('terrestrial-continents','terrestrial-archipelago')
                                    for path in files[profile]))
            finally:
                cache.close()

    def test_group_quota_pins_and_root_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'active-profile'
            root.mkdir()
            untouched=Path(temporary)/'initial-export.png'
            untouched.write_bytes(b'keep export')
            other=Path(temporary)/'other-profile'
            other.mkdir();(other/'cached.png').write_bytes(b'keep other profile')
            protected={'v/1/0/1/0'}
            cache=TerrainDiskCache(root,'v',budget_bytes=180,
                                   protected_keys=lambda:protected,grace_seconds=60)
            try:
                lease=cache.acquire('v/1/0/0/0')
                groups=[]
                for tx in range(3):
                    raw=root/'physical-v1'/'1'/'0'/f'{tx}_0'
                    image=root/'1'/'0'/f'{tx}_0'
                    paths=[raw.with_suffix('.npy'),raw.with_suffix('.json'),
                           image.with_suffix('.png'),image.with_suffix('.json')]
                    for path in paths:
                        path.parent.mkdir(parents=True,exist_ok=True)
                        path.write_bytes(b'x'*40)
                    cache.record(f'v/1/0/{tx}/0',paths)
                    groups.append(paths)
                deadline=time.monotonic()+2
                while not cache.status()['indexed'] and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertTrue(cache.status()['indexed'])
                cache.grace_seconds=0;cache.trim()
                self.assertTrue(all(path.exists() for path in groups[0]))
                self.assertTrue(all(path.exists() for path in groups[1]))
                self.assertFalse(any(path.exists() for path in groups[2]))
                self.assertTrue(cache.status()['over_budget'])
                lease.release();cache.trim()
                self.assertFalse(any(path.exists() for path in groups[0]))
                self.assertEqual(cache.status()['bytes'],160)
                self.assertEqual(cache.status()['evictions'],2)
                self.assertEqual(untouched.read_bytes(),b'keep export')
                self.assertTrue((other/'cached.png').exists())
                with self.assertRaises(ValueError):cache.record('escape',[untouched])
            finally:
                cache.close()

class QueueTests(unittest.TestCase):
    def setUp(self):
        self.queue = TerrainJobs()
    def tearDown(self):
        self.queue.close()
    def test_cancel_deduplicate_prioritize(self):
        entered, release = threading.Event(), threading.Event()
        order=[]
        def hold():
            entered.set();release.wait(3);return 'running'
        self.queue.update_view('view1',1,{'running':2000,'old':2001,'shared':2002})
        first=self.queue.submit('running',hold,session='view1',epoch=1)
        self.assertTrue(entered.wait(2))
        old=self.queue.submit('old',lambda: order.append('old'),session='view1',epoch=1)
        self.queue.update_view('view2',1,{'shared':0})
        shared=self.queue.submit('shared',lambda: order.append('shared'),session='view1',epoch=1)
        duplicate=self.queue.submit('shared',lambda: self.fail('duplicate compute'),session='view2',epoch=1)
        self.assertIs(shared,duplicate)
        self.queue.update_view('view1',2,{'new':2000})
        self.assertFalse(self.queue.update_view('view1',1,{'old':0}))
        new=self.queue.submit('new',lambda: order.append('new'),session='view1',epoch=2)
        with self.assertRaises(JobCancelled):old.wait(1)
        release.set();self.assertEqual(first.wait(3),'running');shared.wait(3);new.wait(3)
        self.assertEqual(order,['shared','new'])
        metrics=self.queue.status()
        self.assertEqual(metrics['deduplicated'],1)
        self.assertEqual(metrics['cancelled_queued'],1)
        self.assertEqual(metrics['obsolete_completed'],1)
    def test_encoding_does_not_block_compute(self):
        encoding, release, computed=threading.Event(),threading.Event(),threading.Event()
        def encode(value):encoding.set();release.wait(3);return value
        first=self.queue.submit('first',lambda:1,encode)
        self.assertTrue(encoding.wait(2))
        second=self.queue.submit('second',lambda:computed.set())
        self.assertTrue(computed.wait(2),'CPU encoding should not own compute lane')
        release.set();first.wait(3);second.wait(3)
    def test_running_macro_stops_at_safe_boundary(self):
        entered, release = threading.Event(), threading.Event()
        def compute():
            entered.set();release.wait(3)
            self.queue.check_current_interest()
            self.fail('obsolete macro must not launch its next block')
        self.queue.update_view('macroview',1,{'macro':0})
        job=self.queue.submit('macro',compute,session='macroview',epoch=1)
        self.assertTrue(entered.wait(2))
        self.queue.update_view('macroview',2,{})
        release.set()
        with self.assertRaises(JobCancelled):job.wait(3)
        self.assertEqual(self.queue.status()['cancelled_computing'],1)
    def test_expiration_cancels_queued(self):
        self.queue.session_ttl=.03
        entered,release=threading.Event(),threading.Event()
        first=self.queue.submit('hold',lambda:(entered.set(),release.wait(3)))
        self.assertTrue(entered.wait(2))
        self.queue.update_view('expired',1,{'queued':0})
        queued=self.queue.submit('queued',lambda:self.fail('expired work'),session='expired',epoch=1)
        time.sleep(.05)
        self.queue.submit('legacy',lambda:None)
        with self.assertRaises(JobCancelled):queued.wait(1)
        release.set();first.wait(3)

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.output=Path(cls.temporary.name)
        sys.path.insert(0,str(_REPO_ROOT/'terrain-diffusion'))
        terrain_app=ModuleType('terrain_app')
        terrain_app.ROOT=_REPO_ROOT
        terrain_app.OUTPUT=cls.output
        terrain_app.MODEL='test'
        terrain_app.gpu_calls={'coarse':0,'base':0,'decoder':0}
        terrain_app.load_pipeline=lambda seed:None
        inference=ModuleType('terrain_inference')
        inference.VERSION='test-v1'
        inference.choose_profile=lambda:SimpleNamespace(name='test',gpu_windows=True,latent_batch=16,coarse_streams=1)
        inference.inference_status=lambda world:None
        # Restore only the two stubs; removing newly imported Torch modules would
        # make a subsequent import reinitialize native operators in the process.
        replacements={'terrain_app':terrain_app,'terrain_inference':inference}
        previous={name:sys.modules.get(name) for name in replacements}
        sys.modules.update(replacements)
        try:
            cls.server=importlib.import_module('terrain_server')
        finally:
            for name,module in previous.items():
                if module is None:sys.modules.pop(name,None)
                else:sys.modules[name]=module
        cls.server.app.testing=True
    @classmethod
    def tearDownClass(cls):
        cls.server.jobs.close();cls.server.physical_delivery.close();cls.server.disk_cache.close();cls.temporary.cleanup()
    def setUp(self):
        self.server.physical_delivery.flush()
        self.client=self.server.app.test_client()

    def test_busy_gpu_is_unknown_not_a_fresh_cpu_preview_admission(self):
        entered,release=threading.Event(),threading.Event()
        def hold():
            with self.server.gpu_lock:
                entered.set();release.wait(2)
        thread=threading.Thread(target=hold)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertIsNone(self.server.learned_tile_state(42,'natural',11,0,0))
        finally:
            release.set();thread.join(2)
    def test_seed_manifest_matches_actual_export(self):
        (self.output/'terrain.json').write_text(json.dumps(dict(seed='17',resolution=30,x=-2,y=3,width=4,height=5)))
        (self.output/'terrain.png').write_bytes(b'placeholder')
        with patch.object(self.server.torch.cuda,'get_device_name',return_value='mock GPU'):
            self.assertIsNone(self.client.get('/api/world?seed=42').json['initial_image'])
            matching=self.client.get('/api/world?seed=17&world_profile=natural').json
        self.assertEqual(matching['initial_bounds'],[-60,90,60,240])
        self.assertEqual(matching['initial_image'],'/generated/terrain.png')

    def test_generation_settings_api_replay_and_invalid_inputs(self):
        settings={'cond_snr':[.2,.5,.5,.5,.5],'frequency_mult':[1.7,1.,1.,1.,1.]}
        with patch.object(self.server.torch.cuda,'get_device_name',return_value='mock GPU'):
            original=self.client.get('/api/world?seed=17&world_profile=natural').json
            response=self.client.get('/api/world',query_string={
                'seed':'17','world_profile':'natural','generation':json.dumps(settings)})
            self.assertEqual(response.status_code,200)
            info=response.json
            replay=self.client.get('/api/world',query_string={
                'seed':'17','world_profile':info['generation_profile']}).json
        self.assertEqual(info['world_profile'],'natural')
        self.assertRegex(info['generation_profile'],r'^natural--g[0-9a-f]{24}$')
        self.assertEqual(info['generation_settings']['cond_snr'],settings['cond_snr'])
        self.assertEqual(info['world_identity'],replay['world_identity'])
        self.assertNotEqual(info['world_identity'],original['world_identity'])
        self.assertIsNone(info['initial_image'])
        self.assertIn(info['generation_profile'],info['overview'])
        self.assertEqual(self.client.get('/api/generation/schema').status_code,200)
        for value in ('not JSON','null','[]','{"unknown":1}','{"cond_snr":[0.5]}',
                      '{"frequency_mult":[1,1,1,1,Infinity]}'):
            with self.subTest(value=value):
                invalid=self.client.get('/api/world',query_string={'generation':value})
                self.assertEqual(invalid.status_code,400)
                self.assertIn('error',invalid.json)

    def test_parameterized_tiles_views_and_background_keep_same_namespace(self):
        from terrain_generation import register_generation
        a=register_generation('natural',{'cond_snr':[.2]*5})
        b=register_generation('natural',{'cond_snr':[.3]*5})
        computed=[]
        def sample(world,seed,profile,*args):
            computed.append(profile)
            return (np.full((304,304),1. if profile==a else 2.,np.float32),
                    np.zeros((5,33,33),np.float32),'decoder')
        with patch.object(self.server,'get_world',return_value=object()),\
             patch.object(self.server,'sample_physical',side_effect=sample):
            for profile in (a,b,a):
                view=self.client.post('/api/view',json=dict(session='configured_view',epoch=1,
                    seed='9797',world_profile=profile,tiles=[dict(lod=0,tx=0,ty=0,priority=2000)]))
                self.assertEqual(view.status_code,200)
                response=self.client.get('/height/natural-v1/9797/0/0/0.bin',
                    query_string={'world_profile':profile})
                self.assertEqual(response.status_code,200)
                self.assertTrue(np.all(np.frombuffer(response.data,dtype='<f4')==(1. if profile==a else 2.)))
            self.assertEqual(response.headers['X-Terrain-Cache'],'hit')
        self.assertEqual(computed,[a,b])
        self.assertNotEqual(self.server._physical_paths(9797,0,0,0,a),
                            self.server._physical_paths(9797,0,0,0,b))
        with patch.object(self.server.coarse_background,'start',return_value={'state':'running'}) as start:
            prep=self.client.post('/api/coarse/prepare',json=dict(seed='9797',world_profile=a,max_windows=1))
            self.assertEqual(prep.status_code,202)
            start.assert_called_once_with(9797,a,1)
        stale=self.client.get('/height/natural-v1/9797/0/0/0.bin',query_string={
            'world_profile':b,'world_identity':self.server.world_manifest(9797,a)['world_hash']})
        self.assertEqual(stale.status_code,400)

    def test_lod4_interpolation_separates_interest_physical_and_png_caches(self):
        from flask import has_request_context
        server=self.server
        seed=9798
        monotone_key=server.request_tile_key(seed,4,0,0,'natural',coarse_interpolation='monotone')
        bilinear_key=server.request_tile_key(seed,4,0,0,'natural',coarse_interpolation='bilinear')
        self.assertNotEqual(monotone_key,bilinear_key)
        self.assertEqual(server.request_tile_key(seed,3,0,0,'natural',coarse_interpolation='monotone'),
                         server.request_tile_key(seed,3,0,0,'natural',coarse_interpolation='bilinear'))
        physical_paths={option:server._physical_paths(seed,4,0,0,'natural',option)
                        for option in ('monotone','bilinear')}
        self.assertNotEqual(physical_paths['monotone'],physical_paths['bilinear'])
        for option,key in (('monotone',monotone_key),('bilinear',bilinear_key)):
            self.assertEqual(server.disk_cache._key(physical_paths[option][0]),key)

        calls=[]
        def sample(world,actual_seed,profile,lod,tx,ty,*,coarse_interpolation):
            self.assertFalse(has_request_context())
            self.assertEqual((actual_seed,lod,tx,ty),(seed,4,0,0))
            calls.append(coarse_interpolation)
            value=1. if coarse_interpolation=='monotone' else 2.
            return (np.full((304,304),value,np.float32),
                    np.zeros((5,33,33),np.float32),'coarse')
        def render(elevation,*args,**kwargs):
            return np.full((2,2,3),int(elevation[0,0]),np.uint8)
        with patch.object(server,'get_world',return_value=object()),\
             patch.object(server,'sample_physical',side_effect=sample),\
             patch.object(server,'render_elevation',side_effect=render):
            for epoch,option in enumerate(('monotone','bilinear'),start=1):
                view=self.client.post('/api/view',json=dict(session='interpolation_view',epoch=epoch,
                    seed=str(seed),world_profile='natural',coarse_interpolation=option,
                    tiles=[dict(lod=4,tx=0,ty=0,priority=2000)]))
                self.assertEqual(view.status_code,200)
                self.assertIn(server.request_tile_key(seed,4,0,0,'natural',coarse_interpolation=option),
                              server.jobs.views['interpolation_view']['wants'])
                height=self.client.get(f'/height/natural-v1/{seed}/4/0/0.bin',
                    query_string={'coarse_interpolation':option})
                self.assertEqual(height.status_code,200)
                self.assertEqual(height.headers['X-Terrain-Coarse-Interpolation'],option)
                self.assertTrue(np.all(np.frombuffer(height.data,dtype='<f4')==float(epoch)))
                png=self.client.get(f'/tiles/natural-v1/{seed}/4/0/0.png',
                    query_string={'coarse_interpolation':option})
                self.assertEqual(png.status_code,200)
                self.assertEqual(png.headers['X-Terrain-Coarse-Interpolation'],option)
                if option=='monotone':
                    monotone_png=png.data
                else:
                    self.assertNotEqual(png.data,monotone_png)
                png.close()
            for option,value in (('monotone',1.),('bilinear',2.)):
                repeat=self.client.get(f'/height/natural-v1/{seed}/4/0/0.bin',
                    query_string={'coarse_interpolation':option})
                self.assertEqual(repeat.status_code,200)
                self.assertEqual(repeat.headers['X-Terrain-Cache'],'hit')
                self.assertTrue(np.all(np.frombuffer(repeat.data,dtype='<f4')==value))
        self.assertEqual(calls,['monotone','bilinear'])
        invalid=self.client.get(f'/height/natural-v1/{seed}/4/0/0.bin',
            query_string={'coarse_interpolation':'bicubic'})
        self.assertEqual(invalid.status_code,400)
        invalid_view=self.client.post('/api/view',json=dict(session='interpolation_view',epoch=3,
            seed=str(seed),coarse_interpolation='bicubic',tiles=[]))
        self.assertEqual(invalid_view.status_code,400)

    def test_lod4_interpolation_selects_only_coarse_sampling_filter(self):
        def field(world,xs,ys,source,*,smooth_coarse):
            self.assertEqual(source,'coarse')
            return np.full((len(ys),len(xs)),2. if smooth_coarse else 1.,np.float32)
        with patch.object(self.server,'sample_field',side_effect=field):
            monotone,stage=self.server.sample_elevation(None,4,0,0,
                                                          coarse_interpolation='monotone')
            bilinear,_=self.server.sample_elevation(None,4,0,0,
                                                     coarse_interpolation='bilinear')
        self.assertEqual(stage,'coarse')
        self.assertTrue(np.all(monotone==2.))
        self.assertTrue(np.all(bilinear==1.))

    def test_world_manifest_keeps_startup_file_snapshot_for_new_seed(self):
        import terrain_manifest
        with patch.object(terrain_manifest,'_files',side_effect=AssertionError('rehash after startup')):
            manifest=self.server.world_manifest(987654321)
        self.assertEqual(manifest['files'],self.server._reference_manifest['files'])
        self.assertEqual(self.server.world_identity(manifest),manifest['world_hash'])

    def test_bootstrap_initialization_failure_is_json(self):
        with patch.object(self.server,'metadata',side_effect=RuntimeError('Native generation failed')):
            response=self.client.get('/api/world?seed=17&world_profile=terrestrial-earthlike')
        self.assertEqual(response.status_code,503)
        self.assertEqual(response.json['error'],'Native generation failed')

    def test_lazy_native_snapshot_refuses_unloaded_python_bytes(self):
        import terrain_bootstrap
        self.server.terrestrial_file_snapshot.cache_clear()
        try:
            with patch.object(terrain_bootstrap,'implementation_identity',return_value={'python_source_sha256':'imported'}),\
                 patch.object(self.server,'_digest',return_value={'sha256':'edited','size':1}):
                with self.assertRaisesRegex(RuntimeError,'changed since startup'):
                    self.server.terrestrial_file_snapshot()
        finally:
            self.server.terrestrial_file_snapshot.cache_clear()

    def test_native_only_identity_change_invalidates_height_payload(self):
        current={'identity':'native-a','height':1.}
        def sample(*args):
            return (np.full((304,304),current['height'],np.float32),
                    np.zeros((5,33,33),np.float32),'decoder')
        with patch.object(self.server,'world_identity',side_effect=lambda _:current['identity']),\
             patch.object(self.server,'get_world',return_value=object()),\
             patch.object(self.server,'sample_physical',side_effect=sample) as compute:
            url='/height/natural-v1/9292/0/0/0.bin'
            first=self.client.get(url)
            self.assertEqual(first.status_code,200)
            self.assertEqual(self.client.get(url).headers['X-Terrain-Cache'],'hit')
            current.update(identity='native-b',height=2.)
            fresh=self.client.get(url)
            self.assertEqual(fresh.headers['X-Terrain-Cache'],'miss')
            self.assertTrue(np.all(np.frombuffer(fresh.data,dtype='<f4')==2.))
            self.assertEqual(compute.call_count,2)

    def test_material_revision_invalidates_images_but_reuses_physical_overview(self):
        from PIL import Image
        import io
        self.server.physical_overview.cache_clear()
        self.addCleanup(self.server.physical_overview.cache_clear)
        values={'elev':np.ones((2,2),np.float32),'climate':np.zeros((5,2,2),np.float32)}
        def albedo(*args,**kwargs):
            value=.25 if self.server.APPEARANCE_IDENTITY=='review-a' else .75
            return np.full((2,2,3),value,np.float32)
        with patch.object(self.server,'metadata',return_value={'overview_bounds':[-20e6,-10e6,20e6,10e6],'world_identity':'same-neural-world'}),\
             patch.object(self.server,'conditioning_preview',return_value=values) as physical,\
             patch.object(self.server,'biome_atlas',return_value=(object(),{})),\
             patch.object(self.server,'colorize_surface',side_effect=albedo) as material,\
             patch.object(self.server,'APPEARANCE_IDENTITY','review-a'):
            def image():
                response=self.client.get('/api/overview/natural-v1/9394.png?mode=soil&world_profile=orogen')
                self.assertEqual(response.status_code,200)
                with Image.open(io.BytesIO(response.data)) as png:result=np.asarray(png).copy()
                response.close();return result
            before=image();np.testing.assert_array_equal(image(),before)
            with patch.object(self.server,'APPEARANCE_IDENTITY','review-b'):
                after=image()
            self.assertGreater(after.mean(),before.mean())
            self.assertEqual(physical.call_count,1)
            self.assertEqual(material.call_count,2)

    def test_overview_identity_receipts_are_per_display_mode(self):
        self.server.physical_overview.cache_clear()
        self.addCleanup(self.server.physical_overview.cache_clear)
        current={'identity':'native-a'}
        def metadata(*args):
            return {'overview_bounds':[-20e6,-10e6,20e6,10e6],
                    'world_identity':current['identity']}
        values={'elev':np.ones((2,2),np.float32),'climate':np.zeros((5,2,2),np.float32)}
        with patch.object(self.server,'metadata',side_effect=metadata),\
             patch.object(self.server,'conditioning_preview',return_value=values) as compute,\
             patch.object(self.server,'get_relief_map',return_value=np.zeros((2,2,3),np.float32)),\
             patch.object(self.server,'colorize',return_value=np.zeros((2,2,3),np.float32)):
            for identity in ('native-a','native-b'):
                current['identity']=identity
                for mode in ('relief','biomes'):
                    response=self.client.get(f'/api/overview/natural-v1/9393.png?mode={mode}')
                    self.assertEqual(response.status_code,200)
                    response.close()
            self.assertEqual(compute.call_count,2,'Display modes share the physical overview for each identity')
            cached=self.client.get('/api/overview/natural-v1/9393.png?mode=biomes')
            cached.close()
            self.assertEqual(compute.call_count,2)
    def test_latent_preview_has_distinct_subscription_cache_and_sampling_grid(self):
        h=np.full((176,176),7.,np.float32);c=np.zeros((5,33,33),np.float32)
        preview=object();native=object()
        with patch.object(self.server,'get_preview_world',return_value=preview) as get_preview, \
             patch.object(self.server,'get_world',return_value=native) as get_native, \
             patch.object(self.server,'sample_latent_preview',return_value=(h,c,'latent')) as sample_preview, \
             patch.object(self.server,'sample_physical',return_value=(np.ones((304,304),np.float32),c,'decoder')) as sample_native:
            view=dict(session='patch_camera',epoch=1,seed='88112',tiles=[dict(lod=2,tx=-1,ty=-1,source_lod=3,priority=2000)])
            self.assertEqual(self.client.post('/api/view',json=view).status_code,200)
            url='/height/natural-v1/88112/2/-1/-1.bin?source_lod=3&session=patch_camera&epoch=1&climate=1'
            first=self.client.get(url)
            self.assertEqual(first.status_code,200,first.data[:200])
            self.assertEqual(first.headers['X-Terrain-Width'],'176')
            self.assertEqual(first.headers['X-Terrain-Resolution'],'240')
            self.assertEqual(first.headers['X-Terrain-Stage'],'latent')
            self.assertEqual(len(first.data),(176**2+5*33**2)*4)
            self.assertEqual(self.client.get(url).headers['X-Terrain-Cache-Source'],'ram')
            self.server.physical_delivery.flush()
            key='natural-v1/natural/88112/2/-1/-1/source3'
            paths=list(self.server.CACHE.rglob('*-source3/*.npy'))
            self.assertTrue(paths)
            self.assertEqual(self.server.disk_cache._key(paths[0]),key)
            # The same geometric address without source3 is a separate job.
            rejected=self.client.get('/height/natural-v1/88112/2/-1/-1.bin?session=patch_camera&epoch=1')
            self.assertEqual(rejected.status_code,409)
            final=self.client.get('/height/natural-v1/88112/2/-1/-1.bin')
            self.assertEqual(final.status_code,200)
            self.assertEqual(final.headers['X-Terrain-Stage'],'decoder')
            self.assertEqual(sample_preview.call_count,1);self.assertEqual(sample_native.call_count,1)
            self.assertEqual(get_preview.call_count,1);self.assertEqual(get_native.call_count,1)
        for lod,source in ((0,3),(2,2),(3,3),(1,'bogus')):
            self.assertEqual(self.client.get(f'/height/natural-v1/88112/{lod}/0/0.bin?source_lod={source}').status_code,400)

    def test_latent_preview_png_reuses_disk_physical_data_with_source_shading(self):
        height=np.ones((176,176),np.float32);climate=np.zeros((5,33,33),np.float32)
        with patch.object(self.server,'get_preview_world',return_value=object()), \
             patch.object(self.server,'sample_latent_preview',return_value=(height,climate,'latent')) as sample, \
             patch.object(self.server,'render_elevation',return_value=np.zeros((128,128,3),np.uint8)) as render:
            url='/height/natural-v1/88113/2/-1/-1.bin?source_lod=3'
            self.assertEqual(self.client.get(url).status_code,200)
            self.server.physical_delivery.flush()
            # Simulate a restarted RAM delivery cache while retaining disk.
            with self.server.physical_delivery.lock:
                self.server.physical_delivery.entries.clear();self.server.physical_delivery.bytes=0
            result=self.client.get('/tiles/natural-v1/88113/2/-1/-1.png?source_lod=3')
            self.assertEqual(result.status_code,200,result.data[:200]);result.close()
            self.assertEqual(result.headers['X-Terrain-Source-LOD'],'3')
            self.assertEqual(result.headers['X-Terrain-Cache-Source'],'disk')
            self.assertEqual(render.call_args.args[1],3,'CPU shader must use source sample spacing, not geometric LOD2')
            self.assertEqual(sample.call_count,1)

    def test_preview_view_read_batches_only_adjacent_interested_source3_tiles(self):
        import terrain_window_scheduler
        session='batch_camera';prefix='natural-v1/natural/42/2/'
        wants={prefix+address:2000 for address in ('-4/2/source3','-3/2/source3','200/100/source3','-4/3')}
        self.server.jobs.update_view(session,1,wants)
        with patch.object(terrain_window_scheduler,'ensure_rect') as ensure:
            self.assertEqual(self.server.prepare_preview_view(object(),42,'natural',2,-4,2,session),2)
            _,stage,y0,x0,y1,x1=ensure.call_args.args
            self.assertEqual(stage,'latent')
            self.assertEqual((x0,x1),(-537,-231));self.assertEqual((y0,y1),(231,409))
            self.assertLess((x1-x0)*(y1-y0),384**2)
            self.assertEqual(ensure.call_args.kwargs['check'],self.server.jobs.check_current_interest)
        self.server.jobs.update_view(session,2,{prefix+'-4/2/source3':2000})
        with patch.object(terrain_window_scheduler,'ensure_rect') as ensure:
            self.assertEqual(self.server.prepare_preview_view(object(),42,'natural',2,-4,2,session),1)
            ensure.assert_not_called()

    def test_latent_preview_grid_matches_parent_crop_including_negative_coordinates(self):
        world=SimpleNamespace(_terrain_generation_settings={})
        def field(world,xs,ys,source):
            self.assertEqual(source,'latent')
            return (xs[None,:]*.01+ys[:,None]*.02).astype(np.float32)
        with patch.object(self.server,'sample_field',side_effect=field), \
             patch.object(self.server,'sample_coarse_climate',return_value=np.zeros((5,33,33),np.float32)):
            full=self.server.sample_elevation(world,3,-1,-1)[0]
            full=self.server.gaussian_filter(full,.65,mode='reflect').astype(np.float32)
            for lod in (1,2):
                count=2**(3-lod);inner=256//count
                for x,y in ((0,0),(count-1,count-1)):
                    patch_height,_,_=self.server.sample_latent_preview(world,lod,-count+x,-count+y)
                    np.testing.assert_array_equal(patch_height[24:-24,24:-24],full[24+y*inner:24+(y+1)*inner,24+x*inner:24+(x+1)*inner])

    def test_optional_base_prewarm_failure_is_reported_without_discarding_model(self):
        from terrain_cuda_graphs import CudaGraphModel
        world=SimpleNamespace(base_model=object(),_terrain_profile=SimpleNamespace(latent_batch=16))
        with patch('terrain_cuda_graphs.prewarm_base_forms',side_effect=RuntimeError('injected allocation failure')):
            result=self.server.warm_base_forms(world)
        self.assertEqual(result['state'],'failed')
        self.assertFalse(result['fully_warmed'])
        self.assertIn('injected allocation failure',result['error'])
        self.assertEqual(result['warmup_cuda_forward_calls'],dict.fromkeys(self.server.gpu_calls,0))

    def test_all_graph_families_prepared_by_default_and_partial_failure_reported(self):
        world=SimpleNamespace(base_model=object(),decoder_model=object(),decoder_tile_size=512)
        ready=dict(enabled=True,fully_warmed=True)
        with patch('terrain_inference.set_coarse_streams') as streams, \
             patch('terrain_inference.prewarm_coarse_graphs',return_value=ready) as coarse, \
             patch.object(self.server,'warm_base_forms',return_value=ready) as base, \
             patch('terrain_cuda_graphs.prewarm_decoder_form',return_value=ready) as decoder, \
             patch.dict('os.environ',{'TERRAIN_PREWARM_BASE':'1'}):
            result=self.server.warm_graphs(world)
            self.assertTrue(result['fully_warmed'])
            self.assertEqual(set(result['models']),{'coarse','base','decoder'})
            streams.assert_called_once_with(world,self.server.coarse_stream_setting['coarse'])
            coarse.assert_called_once_with(world);base.assert_called_once_with(world)
            decoder.assert_called_once_with(world.decoder_model,512)
            coarse.side_effect=RuntimeError('capture refused')
            result=self.server.warm_graphs(world)
            self.assertFalse(result['fully_warmed'])
            self.assertEqual(result['models']['coarse']['state'],'failed')
            self.assertTrue(result['models']['decoder']['fully_warmed'])
            self.assertNotIn('warming_model',self.server.preload_state)

    def test_preview_worlds_have_isolated_bounded_stores_and_close_before_eviction(self):
        made=[]
        class World:
            def __init__(self):
                self.tile_store=object();self.closed=False
                self._terrain_coarse_preparation=SimpleNamespace(close=lambda:made.append('drained'))
            def empty_cache(self):pass
            def close(self):self.closed=True;made.append('closed')
        def create(seed,profile,limit):
            self.assertEqual(limit,128*1024*1024);world=World();made.append(world);return world
        with patch.object(self.server,'preview_worlds',OrderedDict()),patch.object(self.server,'_create_world',side_effect=create):
            a=self.server.get_preview_world(1,'natural');b=self.server.get_preview_world(2,'natural')
            self.assertIsNot(a.tile_store,b.tile_store)
            self.assertIs(self.server.get_preview_world(1,'natural'),a)
            self.server.get_preview_world(3,'natural')
            self.assertTrue(b.closed);self.assertFalse(a.closed)
            self.assertEqual(made[-2:],['drained','closed'])

    def test_height_png_share_compute_and_cache_profile(self):
        height=np.arange(304*304,dtype=np.float32).reshape(304,304)
        climate=np.zeros((5,33,33),dtype=np.float32)
        with patch.object(self.server,'get_world',return_value=object()),patch.object(self.server,'sample_physical',return_value=(height,climate,'decoder')) as sample,patch.object(self.server,'render_elevation',return_value=np.zeros((256,256,3),dtype=np.uint8)):
            view=self.client.post('/api/view',json=dict(session='test_session',epoch=1,seed='91',tiles=[dict(lod=0,tx=-2,ty=3,priority=2000)]))
            self.assertEqual(view.status_code,200)
            raw=self.client.get('/height/natural-v1/91/0/-2/3.bin?session=test_session&epoch=1')
            self.assertEqual(raw.status_code,200)
            self.assertEqual(raw.headers['X-Terrain-Halo'],'24')
            self.assertEqual(raw.headers['X-Terrain-Width'],'304')
            np.testing.assert_array_equal(np.frombuffer(raw.data,dtype='<f4').reshape(304,304),height)
            combined=self.client.get('/height/natural-v1/91/0/-2/3.bin?climate=1')
            self.assertEqual(combined.headers['X-Terrain-Climate-Width'],'33')
            self.assertEqual(len(combined.data),(304*304+5*33*33)*4)
            png=self.client.get('/tiles/natural-v1/91/0/-2/3.png?session=test_session&epoch=1')
            self.assertEqual(png.status_code,200)
            self.assertEqual(self.server.disk_cache.status()['pinned_groups'],1)
            png.close()
            self.assertEqual(self.server.disk_cache.status()['pinned_groups'],0)
            self.assertEqual(sample.call_count,1)
            self.assertEqual(self.client.get('/height/natural-v1/91/0/-2/3.bin?profile=obsolete').status_code,400)
            self.assertEqual(self.client.get('/tiles/natural-v1/91/0/-2/3.png?profile=obsolete').status_code,400)
    def test_terrestrial_manifest_and_bootstrap_do_not_load_nn(self):
        with patch.object(self.server.torch.cuda,'get_device_name',return_value='mock GPU'):
            info=self.client.get('/api/world?seed=42&world_profile=terrestrial-earthlike').json
        self.assertEqual(info['generation_profile'],'terrestrial-earthlike')
        self.assertEqual(info['initial_bounds'],[-20e6,-10e6,20e6,10e6])
        self.assertIsNone(info['initial_image'])
        self.assertEqual(info['bootstrap_raster_spacing_m'],[39062.5,39062.5])
        self.assertEqual(info['conditioning_sample_spacing_m'],7680)
        with patch.object(self.server,'get_world',side_effect=AssertionError('Aperçu sans réseau')):
            raw=self.client.get('/height/natural-v1/123/11/0/0.bin?climate=1&world_profile=terrestrial-earthlike')
            self.assertEqual(raw.status_code,200)
            self.assertEqual(raw.headers['X-Terrain-Stage'],'conditioning-preview')
            self.assertEqual(float(raw.headers['X-Terrain-Source-Resolution']),39062.5)
        self.assertEqual(self.client.get('/height/natural-v1/123/12/0/0.bin').status_code,400)
        for rejected in ('earth','macro-a1','macro-a2','macro-a3','macro-a4'):
            self.assertEqual(self.client.get('/api/world?seed=0&world_profile='+rejected).status_code,400)
        self.assertEqual(self.client.get('/height/natural-v1/123/11/0/0.bin?world_profile=invalid').status_code,400)
    def test_natural_is_default_worldwide_and_high_lod_is_explicit_preview(self):
        with patch.object(self.server.torch.cuda,'get_device_name',return_value='mock GPU'):
            info=self.client.get('/api/world?seed=0').json
        self.assertEqual(info['world_profile'],'natural')
        self.assertEqual(info['seed'],'0')
        self.assertEqual(info['world_bounds'],[-20e6,-10e6,20e6,10e6])
        self.assertEqual(len(info['world_identity']),64)
        with patch.object(self.server,'get_world',side_effect=AssertionError('preview must not run NN')):
            raw=self.client.get('/height/natural-v1/0/11/0/0.bin?climate=1')
        self.assertEqual(raw.status_code,200)
        self.assertEqual(raw.headers['X-Terrain-Stage'],'conditioning-preview')
        self.assertEqual(raw.headers['X-Terrain-Exact-Final-Mip'],'false')
        self.assertIn('max-age=2',raw.headers['Cache-Control'])
    def test_preview_temperature_lapse_is_applied_once(self):
        fields=np.zeros((5,2,2),np.float32)
        climate=np.zeros((5,2,2),np.float32)
        fields[0]=1000
        climate[0]=10
        climate[4]=-.0065
        with patch.object(self.server,'sample_conditioning_preview',return_value={'elev':fields[0],'climate':climate}):
            preview=self.server.conditioning_preview(0,'natural',np.arange(2),np.arange(2))
        np.testing.assert_allclose(preview['climate'][0]+preview['climate'][4]*preview['elev'],10,atol=1e-6)
    def test_old_unwanted_camera_rejected(self):
        self.client.post('/api/view',json=dict(session='old_camera',epoch=3,seed='92',tiles=[]))
        r=self.client.get('/height/natural-v1/92/0/0/0.bin?session=old_camera&epoch=1')
        self.assertEqual(r.status_code,409)
    def test_invalid_priority_and_seed(self):
        self.assertEqual(self.client.post('/api/view',json=dict(session='test_session',epoch=1,seed='-1',tiles=[])).status_code,400)
        self.assertEqual(self.client.post('/api/view',json=dict(session='test_session',epoch=1,seed='1',tiles=[dict(lod=0,tx=0,ty=0,priority=5000)])).status_code,400)
    def test_sparse_field_uses_bounded_blocks_and_same_samples(self):
        import torch
        queries=[]
        class LazyField:
            def __getitem__(self,index):
                _,ys,xs=index
                queries.append((xs.stop-xs.start,ys.stop-ys.start))
                yy,xx=np.meshgrid(np.arange(ys.start,ys.stop),np.arange(xs.start,xs.stop),indexing='ij')
                data=np.zeros((7,ys.stop-ys.start,xs.stop-xs.start),dtype=np.float32)
                data[0]=1+xx*.001+yy*.002
                data[-1]=1
                return torch.from_numpy(data)
        world=SimpleNamespace(coarse=LazyField())
        xs=np.linspace(-501.125,601.25,23)*256
        ys=np.linspace(-602.75,701.375,19)*256
        actual=self.server.sample_field(world,xs,ys,'coarse')
        yy,xx=np.meshgrid(ys/256-.5,xs/256-.5,indexing='ij')
        expected=1+xx*.001+yy*.002
        expected=np.sign(expected)*expected**2
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
        self.assertGreater(len(queries),1)
        self.assertLessEqual(max(w*h for w,h in queries),51*51)

    def test_background_world_does_not_touch_foreground_lru_or_seed(self):
        worlds=OrderedDict([(('natural',7),object()),(('natural',8),object())])
        created=[]
        class FakeWorld:
            def __init__(self):
                self.closed=False
                self._terrain_coarse_preparation=SimpleNamespace(
                    prioritize=lambda *args:None,
                    step=lambda world,**kw:{'complete_windows':1,'total_windows':2})
            def empty_cache(self):pass
            def close(self):self.closed=True
        def make(seed,profile,limit):
            created.append((seed,profile,limit,FakeWorld()))
            return created[-1][-1]
        with patch.object(self.server,'worlds',worlds),\
             patch.object(self.server,'active_seed',7),\
             patch.object(self.server,'background_world',None),\
             patch.object(self.server,'background_world_key',None),\
             patch.object(self.server,'_create_world',side_effect=make):
            self.server.prepare_coarse_quantum(99,'natural')
            self.server.prepare_coarse_quantum(99,'natural')
            self.assertEqual(len(created),1)
            self.server.prepare_coarse_quantum(100,'natural')
            self.assertTrue(created[0][-1].closed)
            self.assertIs(self.server._available_world(100,'natural'),created[-1][-1])
            self.assertEqual(self.server.active_seed,7)
            self.assertEqual(list(self.server.worlds),list(worlds))
            self.assertEqual([x[2] for x in created],[64*1024*1024]*2)

    def test_clipped_ready_coordinates_are_used_for_elevation(self):
        preparation=SimpleNamespace(ready_for_samples=lambda *args,**kw:True)
        world=SimpleNamespace(_terrain_coarse_preparation=preparation,_terrain_world_profile='natural')
        seen=[]
        def field(world,xs,ys,source):
            seen.append((xs.copy(),ys.copy()))
            return np.zeros((len(ys),len(xs)),np.float32)
        with patch.object(self.server,'existing_final_mip',return_value=None),\
             patch.object(self.server,'sample_field',side_effect=field),\
             patch.object(self.server,'sample_coarse_climate',return_value=np.zeros((5,33,33),np.float32)):
            _,_,stage=self.server.sample_physical(world,1,'natural',7,-21,-21)
        self.assertEqual(stage,'coarse')
        self.assertEqual(len(seen),1)
        bounds=self.server.profile_bounds('natural')
        self.assertGreaterEqual(seen[0][0].min(),bounds[0]/self.server.NATIVE)
        self.assertGreaterEqual(seen[0][1].min(),bounds[1]/self.server.NATIVE)

    def test_global_area_mean_is_physical_and_bounded_at_world_edge(self):
        import terrain_window_scheduler
        grid=SimpleNamespace(i1=0,j1=0,i2=520,j2=520)
        prep=SimpleNamespace(grid=grid,complete=lambda *args:True)
        world=SimpleNamespace(_terrain_coarse_preparation=prep)
        rectangles=[]
        def read(world,source,i1,j1,i2,j2,**kw):
            self.assertEqual(source,'coarse')
            rectangles.append((i1,j1,i2,j2))
            self.assertLessEqual((i2-i1)*(j2-j1)*7,4_000_000)
            yy,xx=np.meshgrid(np.arange(i1,i2),np.arange(j1,j2),indexing='ij')
            value=1+yy/200+xx/100
            data=np.zeros((7,i2-i1,j2-j1),np.float32)
            data[0]=value
            data[-1]=1
            return torch.from_numpy(data)
        with patch.object(terrain_window_scheduler,'read_rect',side_effect=read):
            actual=self.server.sample_coarse_area(world,9,1,1,halo=0)
        # Pixel (row 0, col 0) averages four *physical* heights; after the
        # world edge every footprint repeats the last coarse cell.
        values=np.array([1+y/200+x/100 for y in (512,513) for x in (512,513)])
        self.assertAlmostEqual(float(actual[0,0]),float(np.mean(values**2)),places=5)
        self.assertAlmostEqual(float(actual[10,10]),float((1+519/200+519/100)**2),places=5)
        self.assertEqual(actual.shape,(256,256))
        self.assertGreater(len(rectangles),1)
        self.assertTrue(all(0<=a<b<=520 and 0<=c<d<=520 for a,c,b,d in rectangles))

    def test_area_readiness_preflights_every_exact_source_chunk(self):
        grid=SimpleNamespace(i1=0,j1=0,i2=520,j2=520)
        queries=[]
        preparation=SimpleNamespace(grid=grid,
            ready_for_samples=lambda *args,**kw:True,
            _required_indices=lambda *rect:queries.append(rect) or [(0,0,0)],
            _valid_metadata=lambda index:False)
        xs=np.arange(304,dtype=np.float64)
        self.assertFalse(self.server._learned_ready(preparation,xs,xs,9,1,1))
        self.assertEqual(len(queries),1)
        self.assertEqual(queries[0],(464,464,520,520))

    def test_area_readiness_verifies_shared_window_only_once(self):
        grid=SimpleNamespace(i1=0,j1=0,i2=520,j2=520)
        checked=[]
        preparation=SimpleNamespace(grid=grid,
            _required_indices=lambda *rect:[(0,0,0)],
            _valid_metadata=lambda index:checked.append(index) or True)
        self.assertTrue(self.server._coarse_area_ready(preparation,9,1,1))
        self.assertEqual(checked,[(0,0,0)])

    def test_lod12_area_chunks_fit_scheduler_budget(self):
        grid=SimpleNamespace(i1=-3000,j1=-3000,i2=3000,j2=3000)
        preparation=SimpleNamespace(grid=grid)
        chunks=list(self.server._coarse_area_chunks(preparation,12,0,0))
        self.assertEqual(len(chunks),100)
        self.assertLessEqual(max(7*(int(yi[-1]-yi[0])+1)*(int(xi[-1]-xi[0])+1)
                                 for _,_,yi,xi in chunks),4_000_000)

    def test_lod1_mip_promotes_only_when_real_decoder_children_are_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            plan=self.server.plan_mip(1,0,0)
            for x,y in plan.children:
                base=root/'natural'/'physical-v1'/'17'/'0'/f'{x}_{y}'
                base.parent.mkdir(parents=True,exist_ok=True)
                np.save(base.with_suffix('.npy'),np.ones((304,304),np.float32))
                base.with_suffix('.json').write_text(json.dumps({
                    'stage':'decoder','world_identity':'identity'}))
            with patch.object(self.server,'CACHE',root),\
                 patch.object(self.server,'world_identity',return_value='identity'),\
                 patch.object(self.server.disk_cache,'acquire',return_value=nullcontext()),\
                 patch.object(self.server,'sample_elevation',side_effect=AssertionError('unexpected NN')) ,\
                 patch.object(self.server,'sample_coarse_climate',return_value=np.zeros((5,33,33),np.float32)):
                self.assertFalse(self.server.valid_physical_report(
                    {'stage':'decoder','world_identity':'identity'},17,'natural',1,0,0))
                elevation,_,stage=self.server.sample_physical(object(),17,'natural',1,0,0)
                self.assertEqual(stage,'final-dem-mip')
                self.assertEqual(float(elevation.mean()),1)
                x,y=plan.children[0]
                (root/'natural'/'physical-v1'/'17'/'0'/f'{x}_{y}.npy').write_bytes(b'')
                self.assertTrue(self.server.valid_physical_report(
                    {'stage':'decoder','world_identity':'identity'},17,'natural',1,0,0))

    def test_cached_preview_does_not_wait_for_busy_gpu(self):
        seed=1234567
        path,report_path=self.server._physical_paths(seed,11,0,0,'natural')
        np.save(path,np.zeros((304,304),np.float32))
        np.save(path.with_suffix('.climate.npy'),np.zeros((5,33,33),np.float32))
        report_path.write_text(json.dumps({'stage':'conditioning-preview',
            'world_identity':self.server.world_identity(self.server.world_manifest(seed)),
            'resolution':61440,'seconds':0.01,'width':304,'halo':24}))
        entered,release=threading.Event(),threading.Event()
        def hold_gpu():
            with self.server.gpu_lock:
                entered.set()
                release.wait(1.5)
        thread=threading.Thread(target=hold_gpu)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            with patch.object(self.server,'_available_world',side_effect=AssertionError('probe loaded world')):
                started=time.monotonic()
                response=self.client.get(f'/height/natural-v1/{seed}/11/0/0.bin')
                elapsed=time.monotonic()-started
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.headers['X-Terrain-Stage'],'conditioning-preview')
            self.assertEqual(response.headers['X-Terrain-Provisional'],'true')
            self.assertEqual(response.headers['X-Terrain-Cache-Source'],'disk')
            self.assertEqual(response.cache_control.max_age,0)
            self.assertLess(elapsed,.25)
        finally:
            release.set()
            thread.join(2)

    def test_disk_reader_cannot_mix_height_climate_and_receipt_during_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);height=root/'height.npy';climate=root/'climate.npy';receipt=root/'receipt.json'
            np.save(height,np.ones((2,2),np.float32));np.save(climate,np.full((5,2,2),10,np.float32))
            receipt.write_text(json.dumps({'stage':'old'}))
            key='coherent-replacement';entered,release,read=threading.Event(),threading.Event(),threading.Event()
            result=[]
            def writer():
                with self.server.physical_locks[hash(key)%len(self.server.physical_locks)]:
                    np.save(height,np.full((2,2),2,np.float32))
                    entered.set();release.wait(2)
                    np.save(climate,np.full((5,2,2),20,np.float32))
                    receipt.write_text(json.dumps({'stage':'new'}))
            def reader():
                result.append(self.server.read_physical_disk(key,height,receipt,climate));read.set()
            writing=threading.Thread(target=writer);reading=threading.Thread(target=reader)
            writing.start()
            try:
                self.assertTrue(entered.wait(1));reading.start()
                self.assertFalse(read.wait(.05))
                release.set();self.assertTrue(read.wait(1))
                self.assertEqual(result[0][0]['stage'],'new')
                self.assertTrue(np.all(result[0][1][0]==2));self.assertTrue(np.all(result[0][1][1]==20))
            finally:
                release.set();writing.join(2);reading.join(2)

    def test_empty_existing_final_mip_is_a_cache_miss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            base=root/'natural'/'physical-v1'/'17'/'0'/'0_0'
            base.parent.mkdir(parents=True)
            base.with_suffix('.npy').write_bytes(b'')
            base.with_suffix('.json').write_text(json.dumps({'stage':'decoder','world_identity':'identity'}))
            with patch.object(self.server,'CACHE',root),\
                 patch.object(self.server,'native_mip_paths',return_value=(object(),[(base.with_suffix('.npy'),base.with_suffix('.json'))])),\
                 patch.object(self.server,'world_identity',return_value='identity'),\
                 patch.object(self.server,'read_mip',side_effect=lambda plan,child:child(0,0)),\
                 patch.object(self.server.disk_cache,'acquire',return_value=nullcontext()):
                self.assertIsNone(self.server.existing_final_mip(17,'natural',1,0,0))

if __name__=='__main__':unittest.main(verbosity=2)
