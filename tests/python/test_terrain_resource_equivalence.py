"""Full-world statistical contract, deterministic arrays, and CPU/CUDA parity.

Retained Orogen inputs are outside Git. Recreate with benchmark_resources_compile
--prepare on the pinned revision, or set TERRAIN_RESOURCE_REFERENCE_INPUTS.
The two retained Orogen worlds are compared with round-one reference statistics.
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
    def test_batched_deposits_independent_of_family_and_query_order(self):
        from test_terrain_resources import atlas
        layers,height=atlas()
        bounds=(-np.pi*5e5,-np.pi*2.5e5,np.pi*5e5,np.pi*2.5e5)
        grid,fields,first=R.compile_world(layers,height,42,bounds=bounds,balance=False)
        with patch.object(R,'FAMILIES',dict(reversed(list(R.FAMILIES.items())))):
            second=R.world_deposits(grid,fields,42)
        self.assertEqual(first.fingerprint(),second.fingerprint())
        order=list(range(len(first)-1,-1,-1))
        self.assertEqual([(first[i]['id'],first[i]['position_cell']) for i in order],
                         [(second[i]['id'],second[i]['position_cell']) for i in order])

    def test_columnar_gpu_packages_match_record_view(self):
        import terrain_resource_layers as L
        from test_terrain_resources import atlas
        layers,height=atlas()
        grid,fields,sites=R.compile_world(layers,height,42,balance=False,
            bounds=(-np.pi*5e5,-np.pi*2.5e5,np.pi*5e5,np.pi*2.5e5))
        columnar=L._World(grid,fields,sites,42,'columnar-package')
        records=L._World(grid,fields,sites.records(),42,'record-package')
        for mode in ('resources-bio-npp','resources-dep-copper','resources-dep-points'):
            a=L._gpu_package(columnar,mode);b=L._gpu_package(records,mode)
            self.assertEqual(a[0],b[0])
            self.assertEqual(a[2],b[2])
    def test_worker_arrays_survive_process_exit_and_match_in_process(self):
        from terrain_resource_service import Worker
        import terrain_resource_layers as L
        from terrain_resource_acceleration import prepare
        from test_terrain_resources import atlas
        layers,height=atlas()
        bounds=(-np.pi*5e5,-np.pi*2.5e5,np.pi*5e5,np.pi*2.5e5)
        worker=Worker()
        try:
            _,fields,deposits=worker.compile(layers,height,42,bounds)
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'atlas.npz'
                np.savez(path,height_m=height,**{'layer_'+k:v for k,v in layers.items()})
                _,mapped,sites=worker.compile(layers,height,42,bounds,atlas_path=str(path))
                self.assertEqual(fingerprint(fields,deposits),fingerprint(mapped,sites))
        finally:
            worker.close()
        prepare()
        _,reference,sites=R.compile_world(layers,height,42,bounds=bounds)
        self.assertEqual(fingerprint(fields,deposits),fingerprint(L._slim(reference),sites))
        self.assertGreater(float(fields['npp_g_m2'].sum()),0)

    def test_fused_noise_and_sparse_gaussian_equal_reference(self):
        import terrain_resource_noise as N
        N.prepare()
        rng=np.random.default_rng(192)
        x=np.linspace(-2e6,2e6,173)[None,:]
        y=np.linspace(-1e6,1e6,91)[:,None]
        with patch.object(N,'ready',False):reference=R.value_noise(42,'parity',13,x,y)
        np.testing.assert_array_equal(R.value_noise(42,'parity',13,x,y),reference)
        height=rng.uniform(-100,4000,(137,273)).astype(np.float32)
        rows=np.arange(0,137,3)
        for power in (1,2):
            expected=R.gaussian_filter(height.astype(np.float64)**power,1.5,mode=('nearest','wrap'))[rows]
            np.testing.assert_array_equal(R._height_moment(height,rows,power),expected)

    def test_analytic_water_matches_shared_solver_tolerance(self):
        from terrain_resource_water import prepare,evaluate
        from terrain_hydrology_balance import monthly_water_balance
        prepare()
        rng=np.random.default_rng(761)
        ts=rng.uniform(-15,35,256);tw=rng.uniform(-30,25,256)
        ps=rng.uniform(100,2000,256);pw=rng.uniform(100,2000,256)
        lat=rng.uniform(-85,85,256)
        t,p=R._months(ts[None],tw[None],ps[None],pw[None],lat[None])
        t,p=t[:,0].astype(np.float32),p[:,0].astype(np.float32)
        got=evaluate(t,p,lat)
        import terrain_resource_water as W
        with patch.object(W,'ready',False):fallback=evaluate(t,p,lat)
        for compiled,python in zip(got,fallback):np.testing.assert_array_equal(compiled,python)
        ref=monthly_water_balance(None,None,latitude_deg=lat,temp_summer_c=t[6],temp_winter_c=t[0],
                                  precip_summer_mm=p[6]*6,precip_winter_mm=p[0]*6)
        expected=(ref['potential_evaporation_mm'],ref['evapotranspiration_mm'],
                  ref['runoff_mm']+ref['baseflow_mm'],ref['snow_mm'])
        for actual,old in zip(got,expected):
            np.testing.assert_allclose(actual,old,rtol=1e-5,atol=.05)
    def test_reference_statistics_determinism_and_cpu_gpu_agreement(self):
        from terrain_resource_acceleration import prepare
        from tools.verification.resource_statistics import world_statistics,compare_statistics
        prepare()
        references=json.loads((ROOT/'tests/fixtures/resources-statistics-round1.json').read_text())['worlds']
        for seed in (42, 20261027):
            path = INPUTS / f'{seed}.npz'
            if not path.exists():
                self.skipTest(f'Retained Orogen atlas missing: {path}; see module docstring')
            expected = references[str(seed)]
            layers, height, _, bounds = load_inputs(path)
            self.assertEqual(fingerprint(dict(layers,height_m=height),[]), expected['inputs'])
            original=None
            for repeat,device in enumerate(('auto','auto','cpu')):
                with self.subTest(seed=seed, repeat=repeat,device=device), patch.dict(os.environ,TERRAIN_RESOURCES_DEVICE=device):
                    grid, fields, deposits = R.compile_world(layers, height, seed, bounds=bounds)
                    self.assertEqual(grid.shape, (206, 410))
                    compare_statistics(expected['statistics'],world_statistics(fields,deposits))
                    current=fingerprint(fields,deposits)
                    if original is not None:
                        self.assertEqual(current,original)
                    original=current

    def test_cuda_scores_tolerance_and_reproducibility(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        import terrain_resource_cuda as C
        import terrain_resource_gpu as G
        rng=np.random.default_rng(782)
        data=rng.uniform(-3,1,(len(R.ROCKS)+2,12,24)).astype(np.float32)
        pkg=dict(seed32=619273,layer=R.ROCK_LAYER,classes=len(R.ROCKS),aligned=R.ROCK_ALIGNED,
            groups=R.ROCK_GROUPS,group_aligned=R.ROCK_GROUP_ALIGNED,
            field=G.Field(data,-12,-6,(-368640.,-184320.,368640.,184320.)))
        for lod in (3,9,10):
            gi=rng.integers(-12*(1 << (10-lod)),12*(1 << (10-lod)),2048)
            gj=rng.integers(-6*(1 << (10-lod)),6*(1 << (10-lod)),2048)
            cpu=G.realized_class(pkg,gi,gj,lod,return_scores=True)
            with torch.inference_mode():
                gpu=C.scores(pkg,gi,gj,lod).cpu().numpy()
                again=C.scores(pkg,gi,gj,lod).cpu().numpy()
            np.testing.assert_array_equal(gpu,again)
            np.testing.assert_allclose(gpu,cpu,rtol=0,atol=2e-4)
            np.testing.assert_array_equal(C.realized_class(pkg,gi,gj,lod),cpu.argmax(axis=0))

    def test_actual_viewer_world_statistics_and_cpu_gpu_agreement(self):
        from terrain_resource_acceleration import prepare
        from tools.verification.resource_statistics import world_statistics,compare_statistics
        path=Path(os.environ.get('TERRAIN_RESOURCE_VIEWER_INPUTS',
                                 'E:/TerrainDiffusionRuntime/resources/perf-viewer-inputs'))/'42.npz'
        if not path.exists():self.skipTest('Retained actual viewer atlas unavailable')
        prepare()
        reference=json.loads((ROOT/'tests/fixtures/resources-statistics-round1.json').read_text())['worlds']['viewer-42']
        layers,height,seed,bounds=load_inputs(path)
        self.assertEqual(fingerprint(dict(layers,height_m=height),[]),reference['inputs'])
        previous=None
        for device in ('auto','cpu'):
            with patch.dict(os.environ,TERRAIN_RESOURCES_DEVICE=device):
                _,fields,deposits=R.compile_world(layers,height,seed,bounds=bounds)
            compare_statistics(reference['statistics'],world_statistics(fields,deposits))
            current=fingerprint(fields,deposits)
            if previous is not None:self.assertEqual(previous,current)
            previous=current

    def test_distance_kernel_matches_scipy(self):
        from terrain_resource_distance import prepare
        prepare()
        rng=np.random.default_rng(908)
        for shape in ((7,13),(37,63)):
            for rate in (.01,.2,1.):
                mask=rng.random(shape)<rate;mask[0,0]=True
                rows=np.arange(0,shape[0],2,dtype=np.int64);cols=np.arange(0,shape[1],3,dtype=np.int64)
                expected=R._wrap_edt(mask,3.123)[np.ix_(rows,cols)]
                got=R._wrap_edt(mask,3.123,rows=rows,cols=cols)
                np.testing.assert_array_equal(got,expected)

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
            loaded=None  # release Windows file mappings before deleting the fixture

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
