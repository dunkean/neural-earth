"""Physical masks, periodic transport, repeatability and pipeline integration."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

import terrain_city_erosion as city


def surface():
    y,x=np.mgrid[:32,:64]
    result=(800+300*np.cos(x*2*np.pi/64)+100*np.sin(y*np.pi/32)).astype(np.float32)
    result[-6:]=-2000
    return result


class CityValidationTests(unittest.TestCase):
    def test_zero_dose_is_exact_and_does_not_initialize_gpu(self):
        source=surface()
        with patch.object(city,'_runtime',side_effect=AssertionError('No device for dose zero')):
            result,receipt=city.erode(source,strength=0)
        np.testing.assert_array_equal(result,source)
        self.assertIsNot(result,source)
        self.assertEqual(receipt['seconds'],0.)

    def test_invalid_units_and_shapes_fail_before_gpu(self):
        with patch.object(city,'_runtime',side_effect=AssertionError('No invalid GPU requests')):
            for options in ({'iterations':0},{'iterations':1.2},{'strength':float('nan')},
                            {'talus':0},{'motif_km':5001}):
                with self.subTest(options=options),self.assertRaises(ValueError):city.erode(surface(),**options)
            for source in (np.zeros((32,32),np.float32),np.full((32,64),np.nan,np.float32)):
                with self.assertRaises(ValueError):city.erode(source)


@unittest.skipUnless(importlib.util.find_spec('wgpu') and torch.cuda.is_available(),'optional GPU runtime required')
class CityGpuTests(unittest.TestCase):
    def test_physical_masks_repeatability_and_periodic_seam(self):
        source=surface()
        result,receipt=city.erode(source,iterations=3)
        repeated,_=city.erode(source,iterations=3)
        shifted,_=city.erode(np.roll(source,7,axis=1),iterations=3)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[source<=0],source[source<=0])
        self.assertTrue((result[source>0]>=0).all())
        self.assertGreater(float(np.max(np.abs(result-source))),.1)
        np.testing.assert_array_equal(result,repeated)
        # A longitude seam must not become a fixed boundary or a drain.
        np.testing.assert_allclose(shifted,np.roll(result,7,axis=1),atol=.001,rtol=1e-6)
        self.assertEqual(receipt['attempts'][-1]['errors'],0)

    def test_relief_climate_and_original_plates_share_one_world(self):
        import terrain_orogen as orogen
        baseline=orogen.generate_atlas(42,width=128,height=64,include_layers=True,
                    options={'detail':20000,'relief_pipeline':'original'})
        variant=orogen.generate_atlas(42,width=128,height=64,include_layers=True,
                    options={'detail':20000,'relief_pipeline':'city-gpu','city_erosion_iterations':2})
        self.assertFalse(np.array_equal(baseline[0],variant[0]))
        np.testing.assert_array_equal(baseline[1],variant[1])
        # Single-field relief projection and all-field climate projection may
        # round float32 barycentric sums differently (under 1 mm here).
        np.testing.assert_allclose(baseline[0][baseline[1]],variant[0][baseline[1]],atol=.002,rtol=0)
        np.testing.assert_array_equal(variant[4]['city_erosion_delta_m'][baseline[1]],0.)
        for key in ('plates','superPlates','tectonic','crust'):
            np.testing.assert_array_equal(baseline[4][key],variant[4][key])
        self.assertFalse(np.array_equal(baseline[4]['temperature_summer'],variant[4]['temperature_summer']))
        self.assertIn('koppen',variant[4])
        self.assertEqual(variant[2]['erosion']['attempts'][-1]['errors'],0)
        zero=orogen.generate_atlas(42,width=128,height=64,include_layers=True,
                    options={'detail':20000,'relief_pipeline':'city-gpu','city_erosion_strength':0})
        np.testing.assert_array_equal(zero[0],baseline[0])
        for key in baseline[4]:np.testing.assert_array_equal(zero[4][key],baseline[4][key])

    def test_imported_surface_and_cache_replay_preserve_the_gpu_result(self):
        import terrain_orogen as orogen
        options=dict(detail=20000,initial_source='natural',initial_settings={},
                     relief_pipeline='city-gpu',city_erosion_iterations=2)
        source=np.tile(np.linspace(-2000,4500,128,dtype=np.float32),(64,1))
        with tempfile.TemporaryDirectory() as tmp,patch.object(orogen,'CACHE_ROOT',Path(tmp)), \
             patch.object(orogen,'WIDTH',128),patch.object(orogen,'HEIGHT',64), \
             patch.object(orogen,'initial_raster',return_value=source):
            world=orogen.OrogenHeightmap(42,options=options)
            self.assertFalse(np.array_equal(source,world.height_m))
            np.testing.assert_array_equal(world.height_m[source<=0],source[source<=0])
            with patch.object(orogen,'generate_atlas',side_effect=AssertionError('Replay must use cache')):
                replay=orogen.OrogenHeightmap(42,options=options)
            np.testing.assert_array_equal(world.height_m,replay.height_m)
            for key in world.layers:np.testing.assert_array_equal(world.layers[key],replay.layers[key])


if __name__=='__main__':unittest.main()
