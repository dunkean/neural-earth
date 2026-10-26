"""Server delivery/material integration with an isolated amplification contract stub."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()

import ast
from contextlib import nullcontext
import io
import math
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from flask import Flask, request, has_request_context
from terrain_delivery import PhysicalDelivery
import test_terrain_tile_rendering as tile_fixture
import test_terrain_render as material_fixture
import terrain_render
from terrain_lod_interpolation import relief_floor, interpolation_suffix

SOURCE=Path('backend/terrain_server.py')

def load(namespace,names):
    namespace.update(relief_floor=relief_floor,interpolation_suffix=interpolation_suffix)
    namespace.setdefault('REFINEMENT_VERSION','fixture')
    names=set(names)|{'amplification_mode','selected_amplification_mode','selected_amplification','request_amplification'}
    nodes=[n for n in ast.parse(SOURCE.read_text(encoding='utf-8')).body if getattr(n,'name',None) in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),namespace)


def stub(mode='mse'):
    module=ModuleType({'mse':'terrain_amplification','phasor':'terrain_phasor','hybrid':'terrain_hybrid'}[mode]);module.VERSION=mode+'-test'
    def parse(raw):return round(min(2.,max(0.,float(raw))),2)
    def sample(world,level,i0,j0,i1,j1,*,intensity=1.,check=None):
        h=np.full((i1-i0,j1-j0),{'mse':100,'phasor':200,'hybrid':300}[mode]+intensity*level,np.float32)
        surface=np.zeros((4,*h.shape),np.float32);surface[0]=.25;surface[1]=.5;surface[2]=.75;surface[3]=1
        return h,surface
    if mode=='mse':module.parse_intensity=Mock(side_effect=parse)
    setattr(module,{'mse':'sample_amplified','phasor':'sample_phasor','hybrid':'sample_hybrid'}[mode],Mock(side_effect=sample))
    return module


class AmplificationDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.module=stub();self.phasor=stub('phasor');self.hybrid=stub('hybrid');self.patch=patch.dict(sys.modules,terrain_amplification=self.module,terrain_phasor=self.phasor,terrain_hybrid=self.hybrid);self.patch.start();self.addCleanup(self.patch.stop)
        self.fixture=tile_fixture.BiomeTileRenderingTests('runTest');self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.ns=self.fixture.namespace;self.app=self.fixture.app
        self.ns.update(math=math,hydrology_display=lambda seed,wp,lod,tx,ty,h,r:(h,None,r))
        load(self.ns,{'selected_amplification','amplification_capability','tile_key','request_tile_key'})
        self.ns.update(VERSION='natural-v1',generation_profile=lambda value=None:'orogen',latent_preview_source=lambda lod,value=None:None)
        self.raw=self.fixture.physical.side_effect
        def physical(seed,lod,tx,ty,**options):
            path,report,cached,arrays=self.raw(seed,lod,tx,ty,**options)
            selected=self.ns['request_amplification']();amount=selected['amplify'];mode=selected['amplify_mode']
            if amount and lod<0:
                spec=self.ns['amplification_mode'](mode)
                h,s=spec['sampler'](None,-lod,0,0,304,304,intensity=amount)
                report.update(amplification=spec['amplification'],amplification_mode=mode,amplification_version=spec['version'],
                    amplification_intensity=amount,height_filter=spec['amplification'],source_kind=spec['source_kind'])
                if 'stages' in spec:report['amplification_stages']=spec['stages']
                arrays=(h,arrays[1],s)
            return path,report,cached,arrays
        self.fixture.physical.side_effect=physical

    def test_keys_off_identity_and_canonical_intensity(self):
        for query in ('','?amplify=0','?amplify=off','?amplify=-1'):
            with self.app.test_request_context('/'+query):
                self.assertEqual(self.ns['request_tile_key'](42,-2,0,0,**self.ns['request_amplification']()),'natural-v1/orogen/42/-2/0/0')
        for query in ('?amplify=1.234','?amplify=1.23'):
            with self.app.test_request_context('/'+query):
                self.assertTrue(self.ns['request_tile_key'](42,-2,0,0,**self.ns['request_amplification']()).endswith('/amp-hybrid-hybrid-test-1.23'))
                self.assertEqual(self.ns['tile_key'](42,0,0,0),'natural-v1/orogen/42/0/0/0')
        self.assertEqual(self.ns['amplification_capability']()['modes']['mse']['version'],'mse-test')

    def test_bad_parameter_returns_400_before_work(self):
        for endpoint in ('height','tiles'):
            for value in ('garbage','nan','inf'):
                response=self.app.test_client().get(f'/{endpoint}/natural-v1/42/-1/0/0.'+('bin' if endpoint=='height' else 'png')+f'?amplify={value}')
                self.assertEqual(response.status_code,400)
        self.fixture.physical.assert_not_called()

    def test_binary_legacy_bytes_and_optional_surface_order_headers(self):
        client=self.app.test_client();url='/height/natural-v1/42/-1/0/0.bin?mode=relief&amplify_mode=mse'
        old=client.get(url);off=client.get(url+'&amplify=0')
        self.assertEqual(old.data,off.data);self.assertNotIn('X-Terrain-Surface-Format',old.headers)
        after=client.get(url+'&amplify=1')
        self.assertEqual(after.status_code,200);self.assertEqual(after.headers['X-Terrain-Amplification-Version'],'mse-test')
        self.assertEqual(after.headers['X-Terrain-Amplification-Intensity'],'1.0')
        self.assertEqual(json.loads(after.headers['X-Terrain-Amplification-Stages']),[7.5,1.875,.47])
        self.assertEqual(after.headers['X-Terrain-Source-Kind'],'experimental-erosion-amplification')
        self.assertIn('no-store',after.headers['Cache-Control'])
        self.assertEqual(after.headers['X-Terrain-Surface-Format'],'rgba8unorm')
        self.assertEqual(len(after.data),304*304*8)
        surface=np.frombuffer(after.data,np.uint8,offset=304*304*4).reshape(304,304,4)
        np.testing.assert_array_equal(surface[100,100],[64,128,191,255])
        with patch.dict(self.ns,transport_climate=lambda seed,wp,xs,ys,c:c):
            climate=client.get(url+'&amplify=1&climate=1')
        self.assertEqual(len(climate.data),len(after.data)+5*33*33*4)
        self.assertEqual(climate.data[-304*304*4:],after.data[-304*304*4:])

    def test_png_none_surface_is_identical_and_surface_changes_material(self):
        render=self.ns['render_elevation'];h=self.fixture.elevations[-1]
        options=dict(mode='render',seed=42,world_profile='orogen')
        with patch.dict('os.environ',TERRAIN_RENDER_DEVICE='cpu'):
            base=render(h,-1,**options)
            np.testing.assert_array_equal(base,render(h,-1,surface=None,**options))
            surface=np.zeros((4,304,304),np.float32);surface[0]=1;surface[2]=1
            changed=render(h,-1,surface=surface,**options)
        self.assertFalse(np.array_equal(base,changed))
        np.testing.assert_array_equal(base[:, :60],changed[:, :60]) # ocean palette remains unchanged
        self.assertEqual(self.app.test_client().get('/tiles/natural-v1/42/-1/0/0.png?mode=render&amplify=1').status_code,200)

    def test_subscription_keys_match_tile_requests_and_allow_explicit_baseline(self):
        jobs=SimpleNamespace(update_view=Mock(return_value=True))
        self.ns.update(jobs=jobs,_session=lambda s:s,COARSE_INTERPOLATIONS=('monotone','bilinear'))
        load(self.ns,{'update_view'})
        data=dict(session='test',epoch=1,seed=42,amplify=1.25,tiles=[dict(lod=-1,tx=0,ty=0),dict(lod=-1,tx=1,ty=0,amplify=0)])
        response=self.app.test_client().post('/api/view',json=data)
        self.assertEqual(response.status_code,200)
        wants=jobs.update_view.call_args.args[2]
        self.assertIn('natural-v1/orogen/42/-1/0/0/amp-hybrid-hybrid-test-1.25',wants)
        self.assertIn('natural-v1/orogen/42/-1/1/0',wants)
        data['amplify_mode']='phasor'
        self.assertEqual(self.app.test_client().post('/api/view',json=data).status_code,200)
        self.assertIn('natural-v1/orogen/42/-1/0/0/amp-phasor-phasor-test-1.25',jobs.update_view.call_args.args[2])
        for mode in ('bogus',None):
            data['amplify_mode']=mode
            self.assertEqual(self.app.test_client().post('/api/view',json=data).status_code,400)
        data['amplify_mode']='mse'
        with patch.dict(sys.modules,terrain_phasor=None):
            data['amplify_mode']='phasor'
            self.assertEqual(self.app.test_client().post('/api/view',json=data).status_code,400)
        data['amplify_mode']='mse'
        data['amplify']='garbage'
        self.assertEqual(self.app.test_client().post('/api/view',json=data).status_code,400)

    def test_height_layers_and_water_receive_selected_heights(self):
        client=self.app.test_client()
        shade=Mock(return_value=np.zeros((256,256,3),np.uint8))
        for mode in ('relief','biomes','temperature','precipitation','orogen-biomes','orogen-koppen','render','soil'):
            with self.subTest(mode=mode),patch.dict(self.ns,render_elevation=shade):
                response=client.get('/tiles/natural-v1/42/-1/0/0.png?amplify=1&amplify_mode=mse&mode='+mode+'&contours={"enabled":true,"interval":10}')
                self.assertEqual(response.status_code,200)
                np.testing.assert_array_equal(shade.call_args.args[0],np.full((304,304),101,np.float32))
                self.assertEqual(shade.call_args.kwargs['surface'].shape,(4,304,304))
        water=np.full((4,304,304),.125,np.float32)
        def hydro(seed,wp,lod,tx,ty,h,r):
            np.testing.assert_array_equal(h,np.full((304,304),101,np.float32))
            return h,water,r
        with patch.dict(self.ns,hydrology_display=hydro):
            response=client.get('/height/natural-v1/42/-1/0/0.bin?amplify=1&amplify_mode=mse&hydro=1')
        count=304*304
        self.assertEqual(response.headers['X-Terrain-Hydro-Layers'],'4')
        np.testing.assert_array_equal(np.frombuffer(response.data,'<f4',count=count*4,offset=count*4).reshape(4,304,304),water)
        self.assertEqual(len(response.data),count*(4+16+4))

    def test_modes_capability_validation_and_optional_wavelengths(self):
        capability=self.ns['amplification_capability']()
        self.assertTrue(capability['available'])
        self.assertEqual(capability['modes']['phasor'],dict(available=True,version='phasor-test'))
        client=self.app.test_client();url='/height/natural-v1/42/-1/0/0.bin?amplify=1'
        mse=client.get(url+'&amplify_mode=mse');phasor=client.get(url+'&amplify_mode=phasor')
        self.assertEqual(mse.headers['X-Terrain-Amplification-Mode'],'mse')
        self.assertEqual(phasor.headers['X-Terrain-Amplification'],'phasor-noise')
        self.assertEqual(phasor.headers['X-Terrain-Amplification-Version'],'phasor-test')
        self.assertEqual(phasor.headers['X-Terrain-Source-Kind'],'experimental-phasor-amplification')
        self.assertNotIn('X-Terrain-Amplification-Stages',phasor.headers)
        self.assertNotEqual(mse.data,phasor.data)
        self.phasor.WAVELENGTHS=(90.,30.,10.)
        response=client.get(url+'&amplify_mode=phasor')
        self.assertEqual(json.loads(response.headers['X-Terrain-Amplification-Stages']),[90,30,10])
        with self.app.test_request_context('/?amplify=9&amplify_mode=phasor'):
            self.assertEqual(self.ns['selected_amplification'](),2)
        with self.app.test_request_context('/?amplify=1.234&amplify_mode=phasor'):
            self.assertEqual(self.ns['selected_amplification'](),1.23)
        for endpoint in ('height','tiles'):
            extension='bin' if endpoint=='height' else 'png'
            self.assertEqual(client.get(f'/{endpoint}/natural-v1/42/-1/0/0.{extension}?amplify_mode=bogus').status_code,400)
            with patch.dict(sys.modules,terrain_phasor=None):
                self.assertEqual(client.get(f'/{endpoint}/natural-v1/42/-1/0/0.{extension}?amplify_mode=phasor&amplify=1').status_code,400)
                capability=self.ns['amplification_capability']()
                self.assertTrue(capability['modes']['mse']['available'])
                self.assertFalse(capability['modes']['phasor']['available'])

    def test_hybrid_default_delivery_composite_identity_and_unavailable_reason(self):
        self.hybrid.VERSION='hybrid-test+phasor-test+mse-test'
        self.hybrid.RAVINE_WAVELENGTHS=(60.,20.,20/3)
        capability=self.ns['amplification_capability']()
        self.assertEqual(list(capability['modes']),['hybrid','phasor','mse'])
        self.assertEqual(capability['modes']['hybrid']['version'],self.hybrid.VERSION)
        client=self.app.test_client();url='/height/natural-v1/42/-1/0/0.bin?amplify=1'
        default=client.get(url);explicit=client.get(url+'&amplify_mode=hybrid')
        self.assertEqual(default.status_code,200);self.assertEqual(default.data,explicit.data)
        for key,value in {'Mode':'hybrid','Version':self.hybrid.VERSION,'Intensity':'1.0'}.items():
            self.assertEqual(default.headers['X-Terrain-Amplification-'+key],value)
        self.assertEqual(default.headers['X-Terrain-Amplification'],'hybrid-drainage-erosion')
        self.assertEqual(default.headers['X-Terrain-Source-Kind'],'experimental-hybrid-amplification')
        self.assertEqual(default.headers['X-Terrain-Surface-Channels'],'flow,deposit,scree,rock')
        self.assertEqual(json.loads(default.headers['X-Terrain-Amplification-Stages']),list(self.hybrid.RAVINE_WAVELENGTHS))
        self.assertEqual(len(default.data),304*304*8)
        keys=[self.ns['tile_key'](42,-4,0,0,amplify=1,amplify_mode=mode) for mode in ('hybrid','phasor','mse')]
        self.assertEqual(len(set(keys)),3);self.assertTrue(keys[0].endswith('/amp-hybrid-'+self.hybrid.VERSION+'-1.00'))
        for endpoint,extension in (('height','bin'),('tiles','png')):
            with patch.dict(sys.modules,terrain_hybrid=None):
                self.assertEqual(client.get(f'/{endpoint}/natural-v1/42/-1/0/0.{extension}?amplify=1').status_code,400)
                self.assertEqual(client.get(f'/{endpoint}/natural-v1/42/-1/0/0.{extension}?amplify=0').status_code,200)
        with patch('importlib.import_module',side_effect=RuntimeError('hybrid dependency missing')):
            capability=self.ns['amplification_capability']()
            self.assertFalse(capability['modes']['hybrid']['available'])
            self.assertIn('RuntimeError: hybrid dependency missing',capability['modes']['hybrid']['reason'])
            with self.assertRaisesRegex(ValueError,'hybrid dependency missing'):self.ns['amplification_mode']('hybrid')

    def test_pure_keys_do_not_inherit_request_and_match_worker_identity(self):
        key=self.ns['request_tile_key']
        with self.app.test_request_context('/?amplify=2&amplify_mode=phasor'):
            baseline=key(42,-2,0,0)
            self.assertEqual(key(42,-2,0,0,amplify=None),baseline)
            with self.assertRaises(ValueError):key(42,-2,0,0,amplify=1,amplify_mode=None)
            explicit=key(42,-2,0,0,amplify=1.25,amplify_mode='mse')
            request_key=key(42,-2,0,0,**self.ns['request_amplification']())
        self.assertEqual(baseline,'natural-v1/orogen/42/-2/0/0')
        self.assertNotEqual(explicit,request_key)
        self.assertTrue(request_key.endswith('/amp-phasor-phasor-test-2.00'))
        with ThreadPoolExecutor(max_workers=1) as worker:
            self.assertEqual(worker.submit(key,42,-2,0,0,amplify=1.25,amplify_mode='mse').result(),explicit)
            self.assertEqual(worker.submit(key,42,-2,0,0).result(),baseline)

    def test_negative_lod_capture_only_reads_raw_hydrology_base(self):
        raw={'base_height':np.full((2,2),240,np.float32)}
        reader=Mock(return_value=raw)
        self.ns.update(_hydrology_service=object(),read_hydrology_cached_base=reader)
        load(self.ns,{'capture_hydrology_cached_base'})
        captures=[]
        for mode in ('hybrid','phasor','mse'):
            with self.app.test_request_context('/?amplify=2&amplify_mode='+mode):
                self.ns['capture_hydrology_cached_base'](42,'orogen',-2,0,0)
            captures.append(reader.call_args_list[:]);reader.reset_mock()
        self.assertTrue(captures[0]);self.assertEqual(captures[0],captures[1]);self.assertEqual(captures[0],captures[2])
        self.fixture.physical.assert_not_called()
        self.module.sample_amplified.assert_not_called();self.phasor.sample_phasor.assert_not_called();self.hybrid.sample_hybrid.assert_not_called()
        np.testing.assert_array_equal(raw['base_height'],np.full((2,2),240,np.float32))

    def test_missing_module_keeps_disabled_paths_available(self):
        with patch.dict(sys.modules,terrain_amplification=None,terrain_phasor=None,terrain_hybrid=None):
            with self.app.test_request_context('/?amplify=0'):
                self.assertEqual(self.ns['selected_amplification'](),0)
                self.assertFalse(self.ns['amplification_capability']()['available'])
            with self.app.test_request_context('/?amplify=1'):
                with self.assertRaisesRegex(ValueError,'unavailable'):self.ns['selected_amplification']()

    def test_fine_request_range_keys_and_optional_stable_gradients(self):
        self.ns['MIN_LOD']=-12
        def raw(seed,lod,tx,ty,**kw):
            y,x=np.mgrid[:304,:304]
            return Path('unused.npy'),dict(stage='decoder-refinement',resolution=30*2**lod,seconds=0,width=304,halo=24,relief_floor_lod=-6,relief_interpolation='bounded-c2-biquintic-v1',height_filter='bounded-c2'),False,(1000+x*.0001+y*.0002,self.fixture.climate)
        self.raw=raw;client=self.app.test_client()
        for lod in range(-7,-13,-1):
            url=f'/height/natural-v1/42/{lod}/0/0.bin'
            old=client.get(url);new=client.get(url+'?gradients=1')
            self.assertEqual(new.status_code,200);self.assertEqual(new.headers['X-Terrain-Gradient-Format'],'rg32float')
            self.assertEqual(new.headers['X-Terrain-Relief-Floor-LOD'],'-6')
            self.assertEqual(new.data[:len(old.data)],old.data);self.assertEqual(len(new.data),len(old.data)+304*304*8)
            gradient=np.frombuffer(new.data,'<f4',offset=len(old.data)).reshape(304,304,2)
            np.testing.assert_allclose(gradient[100,100],[.0001/(30*2**lod),.0002/(30*2**lod)],rtol=1e-6)
            for mode in ('hybrid','phasor','mse'):
                self.assertTrue(self.ns['tile_key'](42,lod,0,0,amplify=1,amplify_mode=mode).endswith('/interp-from-6-v1'))
        self.assertEqual(client.get('/height/natural-v1/42/-13/0/0.bin').status_code,400)


class PhysicalAmplificationTests(unittest.TestCase):
    def test_interpolated_modes_only_dispatch_at_the_selected_floor(self):
        modules={'mse':stub(),'phasor':stub('phasor'),'hybrid':stub('hybrid')}
        classic=Mock(side_effect=lambda world,level,a,b,c,d,**kw:np.full((c-a,d-b),17.,np.float32))
        ns=dict(np=np,torch=torch,math=math,has_request_context=has_request_context,request=request,
            TILE=256,HALO=24,NATIVE=30,jobs=SimpleNamespace(check_current_interest=lambda:None),sample_refined=classic)
        load(ns,{'sample_elevation'})
        with patch.dict(sys.modules,terrain_amplification=modules['mse'],terrain_phasor=modules['phasor'],terrain_hybrid=modules['hybrid']),patch.dict('os.environ',TERRAIN_RELIEF_FLOORS='{"classic":-3,"hybrid":-3,"phasor":-3,"mse":-3}'):
            for mode in ('classic','hybrid','phasor','mse'):
                world=SimpleNamespace()
                for lod in range(-7,-13,-1):
                    surfaces=[];height,stage=ns['sample_elevation'](world,lod,-1,-1,amplify=0 if mode=='classic' else 1,amplify_mode='hybrid' if mode=='classic' else mode,surface_out=surfaces)
                    self.assertEqual(height.shape,(304,304));self.assertEqual(stage,'decoder-refinement')
                    self.assertEqual(len(surfaces),0 if mode=='classic' else 1)
            for sampler in [classic,modules['mse'].sample_amplified,modules['phasor'].sample_phasor,modules['hybrid'].sample_hybrid]:
                self.assertTrue(sampler.called);self.assertTrue(all(c.args[1]==3 for c in sampler.call_args_list))

    def test_sample_tile_and_polar_physical_thread_the_same_mode_and_surface(self):
        mse,phasor,hybrid=stub(),stub('phasor'),stub('hybrid');jobs=SimpleNamespace(check_current_interest=lambda:None)
        ns=dict(np=np,torch=torch,math=math,has_request_context=has_request_context,request=request,
            TILE=256,HALO=24,NATIVE=30,CLIMATE_SIZE=33,jobs=jobs,
            sample_refined=Mock(side_effect=AssertionError('selected sampler was bypassed')))
        load(ns,{'sample_elevation','sample_tile','sample_physical'})
        render=Mock(return_value=np.zeros((256,256,3),np.uint8));ns['render_elevation']=render
        ns.update(existing_final_mip=lambda *a:None,resolve_generation=lambda wp:SimpleNamespace(settings={}),
            sample_coarse_climate=lambda *a,**kw:np.zeros((5,33,33),np.float32))
        world=SimpleNamespace()
        with patch.dict(sys.modules,terrain_amplification=mse,terrain_phasor=phasor,terrain_hybrid=hybrid):
            for mode in ('hybrid','phasor','mse'):
                ns['sample_tile'](world,42,-2,0,0,amplify=1,amplify_mode=mode)
                expected_height=render.call_args.args[0];expected_surface=render.call_args.kwargs['surface']
                for polar in (False,True):
                    surfaces=[]
                    height,climate,stage=ns['sample_physical'](world,42,'fixture',-2,0,0,polar=polar,
                        amplify=1,amplify_mode=mode,surface_out=surfaces)
                    np.testing.assert_array_equal(height,expected_height)
                    np.testing.assert_array_equal(surfaces[0],expected_surface)
            for sample in (mse.sample_amplified,phasor.sample_phasor,hybrid.sample_hybrid):self.assertEqual(sample.call_count,3)

    def test_sampling_delivery_cache_report_and_cpu_torch_material(self):
        app=Flask(__name__);module=stub();phasor=stub('phasor');hybrid=stub('hybrid');delivery=PhysicalDelivery();self.addCleanup(delivery.close)
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        jobs=SimpleNamespace(check_current_interest=lambda:None,current=SimpleNamespace(job=None))
        jobs.submit=lambda key,compute,finalize,**kw:SimpleNamespace(sequence=1,key=key,wait=lambda:finalize(compute()))
        procedural=Mock(return_value=torch.full((304,304),100.))
        ns=dict(np=np,torch=torch,request=request,has_request_context=has_request_context,math=math,
            TILE=256,HALO=24,NATIVE=30,CLIMATE_SIZE=33,VERSION='natural-v1',PROFILE='fixture',REFINEMENT_VERSION='fixture',
            generation_profile=lambda value=None:'orogen',selected_neural_chart=lambda:None,
            selected_coarse_interpolation=lambda lod:None,latent_preview_source=lambda lod,value=None:None,
            _coarse_cache_level=lambda lod,value:lod,physical_delivery=delivery,metrics={'cache_hits':0,'generated_tiles':0},
            _physical_paths=lambda *a:(Path(temporary.name)/'height.npy',Path(temporary.name)/'height.json'),
            valid_physical_report=lambda *a:True,response_report=lambda r,*a:dict(r),
            world_manifest=lambda *a:{},world_identity=lambda _: 'world',
            current_runtime=lambda:SimpleNamespace(index=0,lock=None),nullcontext=nullcontext,
            measured_lock=lambda *a:nullcontext(),span=lambda *a,**kw:nullcontext(),trace=lambda *a:nullcontext(),
            time=__import__('time'),gpu_calls={},jobs=jobs,device_provenance=lambda _: {},
            get_world=lambda *a:SimpleNamespace(),capture_hydrology_cached_base=lambda *a:None,
            tile_relief_stats=lambda *a:{},sample_refined=procedural,_session=lambda s:s)
        load(ns,{'selected_amplification','tile_key','request_tile_key','sample_elevation','physical_tile'})
        def sample(world,seed,wp,lod,tx,ty,**kw):
            h,stage=ns['sample_elevation'](world,lod,tx,ty,**kw)
            return h,np.zeros((5,33,33),np.float32),stage
        ns['sample_physical']=sample
        with patch.dict(sys.modules,terrain_amplification=module,terrain_phasor=phasor,terrain_hybrid=hybrid):
            results=[]
            for intensity in (None,'0','1','1.00','2'):
                with app.test_request_context('/?amplify_mode=mse'+('' if intensity is None else '&amplify='+intensity)):
                    results.append(ns['physical_tile'](42,-2,0,0,return_arrays=True))
            phasor.KEEP_PROCEDURAL_BASE=False
            with app.test_request_context('/?amplify=1&amplify_mode=phasor'):
                phased=ns['physical_tile'](42,-2,0,0,return_arrays=True)
            self.assertEqual(phased[1]['amplification_mode'],'phasor')
            self.assertEqual(phased[1]['source_kind'],'experimental-phasor-amplification')
            self.assertNotIn('amplification_stages',phased[1])
            self.assertNotIn('procedural-sparse-convolution-octaves',phased[1]['height_filter'])
            self.assertIn('phasor-ravine-detail',phased[1]['height_filter'])
            self.assertEqual(phased[1]['procedural_octaves'],0)
            self.assertEqual(phasor.sample_phasor.call_count,1)
            with app.test_request_context('/?amplify=1'):
                combined=ns['physical_tile'](42,-2,0,0,return_arrays=True)
            self.assertEqual(combined[1]['amplification_mode'],'hybrid')
            self.assertEqual(combined[1]['amplification'],'hybrid-drainage-erosion')
            self.assertEqual(combined[1]['source_kind'],'experimental-hybrid-amplification')
            self.assertEqual(combined[1]['amplification_version'],'hybrid-test')
            self.assertIn('native-drainage-driven-phasor-ravine-detail+mse-surface-processes',combined[1]['height_filter'])
            self.assertNotIn('procedural-sparse-convolution-octaves',combined[1]['height_filter'])
            self.assertEqual(combined[1]['procedural_octaves'],0)
            self.assertEqual(hybrid.sample_hybrid.call_count,1)
            self.assertEqual(len(delivery.entries),5)
            np.testing.assert_array_equal(results[0][3][0],results[1][3][0])
            self.assertEqual(procedural.call_count,1);self.assertEqual(module.sample_amplified.call_count,2)
            self.assertEqual(results[2][1]['amplification_version'],'mse-test')
            self.assertEqual(len(results[2][3]),3);self.assertEqual(results[2][1]['persistence'],'memory-only')
            self.assertFalse(list(Path(temporary.name).iterdir()))
        # Exercise the real CUDA mirror when the device is available.
        import terrain_render_torch as twin
        atlas,xs,ys,h,_=material_fixture.material_case(size=20,resolution=7.5)
        gradient,tpi=terrain_render.terrain_derivatives(h,7.5)
        point=np.zeros((20,20,3),np.float32);north=np.zeros((20,20,2),np.float32)
        soil=np.broadcast_to(material_fixture.SOIL[:,None,None],(9,20,20))
        seasons=np.broadcast_to(np.array(material_fixture.TEMPERATE,np.float32)[:,None,None],(6,20,20))
        args=(h,gradient,tpi,point,north,7.5,soil,seasons,1234,None,False,None,None,None)
        surface=np.full((4,20,20),.6,np.float32)
        a=terrain_render.surface_material(*args,surface)
        if twin.available():
            b=twin.surface_material(*args,surface)
            np.testing.assert_allclose(a,b,atol=2e-6)


if __name__=='__main__':unittest.main()
