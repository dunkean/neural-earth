"""Tectonic geometry, CUDA repeatability, persistence and pipeline routing."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import terrain_generation as generation
import terrain_conditioning as conditioning
import terrain_manifest as manifest
from test_terrain_conditioning import HeightStub, source_fixture


class OrogenRoutingTests(unittest.TestCase):
    def test_profile_and_custom_source_replay_without_native_rust(self):
        default = generation.resolve_generation('orogen')
        self.assertEqual(default.settings['height_source'], 'orogen')
        self.assertTrue(default.needs_bootstrap)
        self.assertEqual(default.bootstrap_generator, 'orogen')
        with tempfile.TemporaryDirectory() as tmp, patch.object(generation, 'REGISTRY_ROOT', Path(tmp)):
            token = generation.register_generation('natural', {'height_source': 'orogen',
                'climate_source': 'natural', 'continental_style': 'archipelago'})
            descriptor = generation.resolve_generation(token)
            self.assertEqual(descriptor.bootstrap_generator, 'orogen')
            self.assertEqual(descriptor.bootstrap_style, 'archipelago')
            provider = HeightStub()
            with patch('terrain_orogen.get_heightmap', return_value=provider) as get_map, \
                 patch('terrain_bootstrap.get_heightmap', side_effect=AssertionError('No Rust')):
                factory = conditioning.GenerationConditioning(2**64-1, descriptor)
                xs, ys = np.array([-100., 100.]), np.array([0., 500.])
                fields = factory.sample(xs, ys)
                np.testing.assert_array_equal(fields[0], provider.sample_height_m(xs, ys))
                self.assertTrue(np.isfinite(fields).all())
            get_map.assert_called_once_with(2**64-1, 'archipelago', options=descriptor.bootstrap_options)

    def test_default_climate_uses_original_seasons(self):
        provider = HeightStub()
        provider.bounds=(-20e6,-10e6,20e6,10e6);provider.width=4;provider.height=2
        provider.layers={name:np.full((2,4),value,np.float32) for name,value in
            [('temperature_summer',.7),('temperature_winter',.3),('precip_summer',.8),('precip_winter',.2)]}
        with patch('terrain_orogen.get_heightmap',return_value=provider):
            factory=conditioning.GenerationConditioning(7,generation.resolve_generation('orogen'))
            fields=factory.sample([0.,500.],[0.,200.])
            self.assertIsNone(factory.native)
            np.testing.assert_allclose(fields[1],0,atol=1e-5)
            monthly_temperature=18*np.cos(np.arange(12)*np.pi/6)
            monthly_rain=np.array([800/6]*6+[200/6]*6)
            np.testing.assert_allclose(fields[2],monthly_temperature.std(ddof=1)*100,rtol=1e-6)
            np.testing.assert_allclose(fields[3],1000)
            np.testing.assert_allclose(fields[4],monthly_rain.std(ddof=1)/(monthly_rain.mean()+1)*100,rtol=1e-6)
            self.assertTrue(np.isfinite(fields).all())

    def test_manifest_uses_orogen_receipt_and_excludes_timings(self):
        receipt = dict(requested_seed_u64='42', selected_seed_u64='42', selected_attempt=0,
            style='earthlike', generator_version='orogen-fixture', native_config={}, attempts=[],
            raster={'width': 2048, 'height': 1024}, hypsometry={}, raw_height_sha256='a',
            height_sha256='a', sign_preserved=True, generation_seconds=1.)
        with patch('terrain_orogen.bootstrap_metadata', return_value=receipt), \
             patch.object(manifest, 'bootstrap_metadata', side_effect=AssertionError('No Rust')):
            a = manifest.build_manifest(42, 'orogen', file_hashes=False)
            receipt['generation_seconds'] = 100
            b = manifest.build_manifest(42, 'orogen', file_hashes=False)
            self.assertEqual(a['world_hash'], b['world_hash'])
            self.assertTrue(a['geography']['bootstrap_periodic_longitude'])
            self.assertEqual(a['geography']['bootstrap_version'], 'orogen-fixture')

    def test_diagnostic_palettes_sample_plate_ids_without_interpolation(self):
        from types import SimpleNamespace
        from terrain_orogen_layers import MODES, render, sample
        atlas = SimpleNamespace(bounds=(-2.,-1.,2.,1.), width=4, height=2,
            layers=dict(plates=np.array([[0,0,1,1]]*2, np.uint16),
                crust=np.array([[0,0,1,1]]*2, np.uint8),
                boundaries=np.array([[0,1,2,3]]*2, np.uint8),
                convergence=np.array([[0.,1.,-1.,0.]]*2, np.float32),
                uplift=np.array([[0.,3000.,6000.,0.]]*2, np.float32)))
        xs, ys = np.array([-1.5,-.5,.5,1.5]), np.array([-.5,.5])
        for mode in ('orogen-plates','orogen-crust','orogen-boundaries','orogen-convergence','orogen-uplift'):
            rgb = render(atlas, mode, xs, ys)
            self.assertEqual(rgb.shape, (2,4,3))
            self.assertTrue(np.isfinite(rgb).all())
            self.assertTrue(((rgb>=0)&(rgb<=1)).all())
        ids = sample(atlas, atlas.layers['plates'], [-2.,2.], ys, categorical=True)
        np.testing.assert_array_equal(ids[:,0],ids[:,1])


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for Orogen relief')
class OrogenCudaTests(unittest.TestCase):
    def test_climate_and_biome_controls_change_fields_without_changing_relief(self):
        from terrain_orogen import generate_atlas
        baseline=generate_atlas(42,width=128,height=64,options={'detail':20000},include_layers=True)
        variant=generate_atlas(42,width=128,height=64,options={
            'detail':20000,'temperature_equator':33.,'biome_tropical_snow':1.,
            'biome_color_af':[.9,.1,.1]},include_layers=True)
        np.testing.assert_array_equal(baseline[0],variant[0])
        self.assertGreater(float(variant[4]['temperature_summer'].mean()),float(baseline[4]['temperature_summer'].mean()))
        self.assertFalse(np.array_equal(baseline[4]['biome_0'],variant[4]['biome_0']))
        for field in variant[4].values():
            self.assertTrue(np.isfinite(field).all())

    def test_spherical_cuda_interpolation_preserves_fields(self):
        from scipy.spatial import ConvexHull, cKDTree
        from terrain_orogen import sphere_raster
        from terrain_orogen_cuda import rasterize
        p=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],np.float32)
        q=sphere_raster(128,64);tri=ConvexHull(p).simplices
        fields={'constant':np.full(6,17,np.float32),'x':p[:,0],'plates':np.arange(6,dtype=np.float32)}
        nearest=cKDTree(p).query(q,k=4)[1]
        out,fallback=rasterize(p,tri,nearest,q,fields,torch.device('cuda'))
        self.assertEqual(fallback,0)
        np.testing.assert_allclose(out['constant'],17,atol=2e-6)
        np.testing.assert_allclose(out['x'],q[:,0]/np.abs(q).sum(axis=1),atol=2e-6)
        np.testing.assert_array_equal(out['plates'],nearest[:,0])

    def test_climate_can_use_other_initial_sources_without_changing_height(self):
        import terrain_orogen as orogen
        from terrain_generation import _defaults
        options=dict(detail=20000,initial_source='natural',relief_pipeline='original',initial_settings=_defaults('natural'))
        # The source raster stays exact when only climate is selected.
        source=np.linspace(-3000,4500,128*64,dtype=np.float32).reshape(64,128)
        with patch.object(orogen,'initial_raster',return_value=source):
            height,_,receipt,_,layers=orogen.generate_atlas(42,width=128,height=64,options=options,include_layers=True)
        np.testing.assert_array_equal(height,source)
        self.assertIn('koppen',layers)
        self.assertIn('precip_summer',layers)
        self.assertEqual(receipt['projection_fallback_pixels'],0)

    def test_stationary_plates_have_no_convergence_or_positive_uplift(self):
        from terrain_orogen import generate_atlas
        _,_,_,_,layers=generate_atlas(42,width=128,height=64,
            options={'detail':20000,'motion_strength':0},include_layers=True)
        self.assertEqual(float(layers['convergence'].max()),0.)
        self.assertLessEqual(float(layers['tectonic'].max()),0.)

    def test_real_u64_seed_persistence_and_periodic_sampling(self):
        import terrain_orogen as orogen
        with tempfile.TemporaryDirectory() as tmp, patch.object(orogen, 'CACHE_ROOT', Path(tmp)), \
             patch.object(orogen, 'WIDTH', 256), patch.object(orogen, 'HEIGHT', 128):
            world = orogen.OrogenHeightmap(2**64-1, options={'detail':20000})
            self.assertFalse(world.height_m.flags.writeable)
            self.assertTrue((world.height_m>0).any() and (world.height_m<0).any())
            with patch.object(orogen, 'generate_atlas', side_effect=AssertionError('Cache must be used')):
                replay = orogen.OrogenHeightmap(2**64-1, options={'detail':20000})
            np.testing.assert_array_equal(world.height_m, replay.height_m)
            x0,y0,x1,y1 = world.bounds
            samples = world.sample_height_m([x0, x1, x0+1000, x1+1000], [y0, 0., y1])
            np.testing.assert_array_equal(samples[:, 0], samples[:, 1])
            np.testing.assert_array_equal(samples[:, 2], samples[:, 3])
            self.assertEqual(world.metadata['requested_seed_u64'], str(2**64-1))
            self.assertTrue({'plates','crust','boundaries','convergence','uplift','koppen','temperature_summer'} <= set(world.layers))
            for name in world.layers:
                np.testing.assert_array_equal(world.layers[name], replay.layers[name])
                self.assertFalse(world.layers[name].flags.writeable)
            # Damaged persisted bytes are regenerated, never admitted.
            world.cache_path.write_bytes(b'bad cache')
            repaired = orogen.OrogenHeightmap(2**64-1, options={'detail':20000})
            np.testing.assert_array_equal(world.height_m, repaired.height_m)


if __name__ == '__main__':
    unittest.main()
