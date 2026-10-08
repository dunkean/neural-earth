"""Physical and integration checks for explicit GPU Orogen options."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.spatial import cKDTree

import terrain_orogen as orogen
import terrain_orogen_gpu as gpu
from terrain_generation import OROGEN_GPU_PARAMETERS


class GPUOptions(unittest.TestCase):
    def test_reference_defaults_never_initialize_optional_runtime(self):
        with mock.patch.object(gpu,'implementation_identity',side_effect=AssertionError('optional GPU initialized')):
            config=orogen.style_config('earthlike')
            self.assertEqual(config['relief_pipeline'],'orogen')
            self.assertTrue(all(config[k] is False for k in OROGEN_GPU_PARAMETERS))
            self.assertNotIn('orogen_gpu_identity',config)

    def test_unavailable_runtime_has_explicit_cpu_fallback_identity(self):
        identity=dict(available=False,reason='missing NVRTC',fallback='original CPU')
        with mock.patch.object(gpu,'implementation_identity',return_value=identity):
            config=orogen.style_config('earthlike',{'orogen_gpu_climate':True})
        self.assertEqual(config['orogen_gpu_identity'],identity)
        self.assertTrue(config['orogen_gpu_climate'])
        self.assertEqual(config['relief_pipeline'],'orogen')


@unittest.skipUnless(gpu.implementation_identity()['available'],'optional CUDA kernels unavailable')
class GPUExecution(unittest.TestCase):
    def test_nearest_grid_agrees_with_independent_cpu_tree_at_seams_and_poles(self):
        rng=np.random.default_rng(42)
        points=rng.normal(size=(4000,3)).astype(np.float32)
        points/=np.linalg.norm(points,axis=1,keepdims=True)
        query=np.concatenate((orogen.sphere_raster(64,32),np.array([[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]],np.float32)))
        indices=gpu.nearest_regions(points,query)
        expected=cKDTree(points).query(query,k=4)[1]
        np.testing.assert_array_equal(indices,expected)

    def test_climate_only_keeps_relief_and_reference_physical_fields(self):
        reference=orogen.generate_atlas(42,width=128,height=64,include_layers=True,options={'detail':20000})
        accelerated=orogen.generate_atlas(42,width=128,height=64,include_layers=True,
                                        options={'detail':20000,'orogen_gpu_climate':True})
        np.testing.assert_array_equal(reference[0],accelerated[0])
        for name in ('temperature_summer','temperature_winter','precip_summer','precip_winter'):
            np.testing.assert_allclose(reference[4][name],accelerated[4][name],rtol=0,atol=2e-5)
        pipeline=accelerated[2]['gpu_pipeline']
        self.assertGreater(pipeline['timings']['kernels'],100)
        self.assertEqual(pipeline['coverage']['unsupported'],{})
        self.assertLess(pipeline['timings']['readback_bytes'],20_000_000)

    def test_all_stages_repeat_and_preserve_finite_physical_fields(self):
        options=dict(detail=20000,**{key:True for key in OROGEN_GPU_PARAMETERS})
        first=orogen.generate_atlas(43,width=128,height=64,include_layers=True,options=options)
        second=orogen.generate_atlas(43,width=128,height=64,include_layers=True,options=options)
        np.testing.assert_array_equal(first[0],second[0])
        self.assertTrue(np.any(first[0]>0))
        self.assertTrue(np.any(first[0]<0))
        self.assertTrue(np.isfinite(first[0]).all())
        for name,value in first[4].items():
            self.assertTrue(np.isfinite(value).all(),name)
            np.testing.assert_array_equal(value,second[4][name])
        self.assertEqual(first[2]['gpu_pipeline']['coverage']['unsupported'],{})
        self.assertEqual(first[2]['timings']['nearest_backend'],'CUDA-grid')

    def test_city_erosion_and_gpu_climate_use_the_same_retained_world(self):
        options=dict(detail=20000,relief_pipeline='city-gpu',city_erosion_iterations=2,
                     orogen_gpu_climate=True,orogen_gpu_relief=True,orogen_gpu_raster=True)
        result=orogen.generate_atlas(44,width=128,height=64,include_layers=True,options=options)
        from terrain_city_erosion import VERSION as city_version
        self.assertEqual(result[2]['erosion']['engine'],city_version)
        self.assertTrue(result[2]['gpu_pipeline']['flags']['climate'])
        self.assertTrue(np.isfinite(result[4]['precip_summer']).all())
        np.testing.assert_array_equal(result[4]['city_erosion_delta_m'][result[1]],0)

    def test_unavailable_backend_runs_reference_without_changing_outputs(self):
        reference=orogen.generate_atlas(45,width=64,height=32,options={'detail':20000})
        with mock.patch.object(gpu,'implementation_identity',return_value={'available':False,'reason':'missing runtime'}):
            result=orogen.generate_atlas(45,width=64,height=32,
                                        options={'detail':20000,'orogen_gpu_climate':True,'orogen_gpu_raster':True})
        np.testing.assert_array_equal(reference[0],result[0])
        self.assertTrue(result[2]['gpu_pipeline']['fallback'])
        self.assertEqual(result[2]['timings']['nearest_backend'],'CPU-cKDTree')

    def test_graph_thermal_only_preserves_ocean_and_land_mass(self):
        # Four nodes in a ring, all land: conservative simultaneous talus flux.
        import cupy as cp
        height=cp.asarray([.5,.1,.2,.1],dtype=cp.float32)
        fields=dict(adjOffset=cp.asarray([0,2,4,6,8],dtype=cp.int32),
                    adjList=cp.asarray([1,3,0,2,1,3,0,2],dtype=cp.int32),r_elevation=height,
                    r_isOcean=cp.zeros(4,cp.uint8),neighborDist=cp.full(8,.1,cp.float32),
                    r_xyz=cp.zeros(12,cp.float32),hIters=0,tIters=3,gIters=0,K=0,m=.5,dt=1,
                    talusSlope=.6,kThermal=.15,glacialStrength=0)
        before=float(height.sum().get())
        gpu._erode_graph(fields)
        self.assertAlmostEqual(float(height.sum().get()),before,places=6)
        self.assertLess(float(height.max().get()),.5)

    def test_gpu_cache_replay_is_immutable_and_does_not_regenerate(self):
        options=dict(detail=20000,orogen_gpu_climate=True,orogen_gpu_raster=True)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(orogen,'CACHE_ROOT',Path(tmp)), \
             mock.patch.object(orogen,'WIDTH',128),mock.patch.object(orogen,'HEIGHT',64):
            first=orogen.OrogenHeightmap(46,options=options)
            with mock.patch.object(orogen,'generate_atlas',side_effect=AssertionError('cache regenerated')):
                second=orogen.OrogenHeightmap(46,options=options)
            self.assertEqual(first.cache_key,second.cache_key)
            np.testing.assert_array_equal(first.height_m,second.height_m)
            for name,value in second.layers.items():
                np.testing.assert_array_equal(first.layers[name],value)
                self.assertFalse(value.flags.writeable)

    def test_small_plate_count_and_zero_controls_cover_optional_branches(self):
        result=orogen.generate_atlas(47,'gondwana',width=64,height=32,include_layers=True,options=dict(
            detail=20000,plate_count=4,motion_strength=0.,hydraulic=0.,thermal=0.,glacial=0.,
            pressure_noise=0.,temperature_smoothing=0,**{key:True for key in OROGEN_GPU_PARAMETERS}))
        self.assertTrue(np.isfinite(result[0]).all())
        self.assertEqual(result[2]['gpu_pipeline']['coverage']['unsupported'],{})

    def test_gpu_climate_on_imported_source_preserves_source_height(self):
        from terrain_generation import _defaults
        source=np.linspace(-3000,4500,128*64,dtype=np.float32).reshape(64,128)
        options=dict(detail=20000,initial_source='natural',relief_pipeline='original',
                     initial_settings=_defaults('natural'))
        with mock.patch.object(orogen,'initial_raster',return_value=source):
            reference=orogen.generate_atlas(48,width=128,height=64,options=options,include_layers=True)
            result=orogen.generate_atlas(48,width=128,height=64,include_layers=True,
                options=dict(options,**{key:True for key in OROGEN_GPU_PARAMETERS}))
        np.testing.assert_array_equal(result[0],source)
        for name in ('temperature_summer','temperature_winter','precip_summer','precip_winter'):
            np.testing.assert_allclose(result[4][name],reference[4][name],atol=2e-5,rtol=2e-5)
        self.assertEqual(result[2]['gpu_pipeline']['coverage']['unsupported'],{})


if __name__=='__main__':unittest.main()
