"""Physical SNR units, neutral path, driver direction and persisted identities."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path

import hashlib
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

import terrain_generation as generation
from terrain_snr import window_snr, lod_relief


class SnrTests(unittest.TestCase):
    def settings(self, **overrides):
        return generation._normalize('natural', overrides)

    def fields(self, height, temperature=20):
        a = np.zeros((5, 64, 64), np.float32)
        a[0] = np.sign(height)*np.sqrt(abs(height))
        a[1] = temperature
        return a

    def test_neutral_does_not_access_inputs(self):
        self.assertIsNone(window_snr(self.settings(), [.5]*5, object()))

    def test_disabled_adaptive_preserves_controls_and_bypasses_all_ramps(self):
        settings=self.settings(snr_adaptive_enabled=False,snr_altitude_gain=[4,1,1,1,1],
                               snr_driver_gain=[3,1,1,1,1],snr_latitude_gain=4)
        self.assertIsNone(window_snr(settings,[.5]*5,object(),latitude=80))
        self.assertEqual(settings['snr_altitude_gain'][0],4.)
        settings['snr_adaptive_enabled']=True
        self.assertGreater(window_snr(settings,[.5]*5,self.fields(4000,-20),latitude=80)[0][0],.5)
        with self.assertRaises(ValueError):self.settings(snr_adaptive_enabled=1)

    def test_latent_preview_obeys_requested_lod_instead_of_source_lod(self):
        from types import SimpleNamespace
        from scipy.ndimage import gaussian_filter
        tree=ast.parse((source_path('terrain_server.py', root=_REPO_ROOT)).read_text(encoding='utf-8'))
        node=next(node for node in tree.body if getattr(node,'name',None)=='sample_latent_preview')
        def sample_field(world,xs,ys,stage):
            yy,xx=np.mgrid[:len(ys),:len(xs)]
            return (100+np.sin(xx*2)*np.cos(yy*2)).astype(np.float32)
        namespace=dict(np=np,gaussian_filter=gaussian_filter,TILE=256,HALO=24,CLIMATE_SIZE=33,
            sample_field=sample_field,jobs=SimpleNamespace(check_current_interest=lambda:None),
            sample_coarse_climate=lambda *args,**kwargs:np.zeros((5,33,33),np.float32))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'terrain_server.py','exec'),namespace)
        overrides=[0.]*15;overrides[5]=.125  # requested LOD 2; source LOD 3 remains neutral
        settings=self.settings(snr_lod=overrides,snr_detail_mode='per-lod')
        world=SimpleNamespace(_terrain_generation_settings=settings)
        active=namespace['sample_latent_preview'](world,2,0,0)[0]
        settings['snr_detail_mode']='global'
        baseline=namespace['sample_latent_preview'](world,2,0,0)[0]
        self.assertFalse(np.array_equal(active,baseline),'requested LOD 2 override must be visible')
        self.assertLess(float(active.std()),float(baseline.std()))

    def test_latitude_modulates_only_relief_and_matches_both_hemispheres(self):
        s = self.settings(snr_latitude_gain=4, snr_latitude_range=[0, 60])
        equator, _ = window_snr(s, [.5]*5, self.fields(100), latitude=0)
        north, _ = window_snr(s, [.5]*5, self.fields(100), latitude=75)
        south, _ = window_snr(s, [.5]*5, self.fields(100), latitude=-75)
        self.assertEqual(equator, (.5,)*5)
        self.assertEqual(north, (2., .5, .5, .5, .5))
        self.assertEqual(south, north)

    def test_lod_override_scales_residual_and_keeps_overlapping_halos_identical(self):
        values = np.random.default_rng(42).normal(size=(64, 96)).astype(np.float32)
        s = self.settings()
        self.assertIs(lod_relief(s, values, 2), values)
        s['snr_detail_mode']='per-lod';s['snr_lod'][5] = .125
        result = lod_relief(s, values, 2)
        self.assertLess(np.std(result), np.std(values))
        self.assertIs(lod_relief(s, values, 3), values)
        left = lod_relief(s, values[:, :72], 2)
        right = lod_relief(s, values[:, 24:], 2)
        np.testing.assert_array_equal(left[:, 32:64], right[:, 8:40])
        s['snr_detail_mode']='global'
        self.assertIs(lod_relief(s,values,2),values)

    def test_new_settings_replay_and_previous_receipts_remain_resolvable(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(generation,'REGISTRY_ROOT',Path(temporary)):
            s = self.settings(snr_latitude_gain=3, snr_lod=[.125]*15)
            token = generation.register_generation('natural', s)
            self.assertEqual(generation.resolve_generation(token).settings, s)
            old = {k:v for k,v in self.settings(macro_scale_km=800).items()
                   if k not in ('snr_latitude_gain', 'snr_latitude_range', 'snr_lod')}
            payload = generation._payload('natural', old)
            token = 'natural--g'+hashlib.sha256(generation._bytes(payload)).hexdigest()[:24]
            (Path(temporary)/(token+'.json')).write_text(json.dumps(payload))
            resolved = generation.resolve_generation(token).settings
            self.assertEqual(resolved['snr_lod'], [0.]*15)
            self.assertEqual(resolved['snr_latitude_gain'], 1.)

    def test_mountains_and_cold_multiply_only_selected_channels(self):
        s = self.settings(snr_altitude_gain=[4,1,1,1,1], snr_driver_gain=[3,2,1,1,1])
        plain, _ = window_snr(s, [.5]*5, self.fields(100))
        mountain, _ = window_snr(s, [.5]*5, self.fields(4000,-20))
        self.assertEqual(plain, (.5,.5,.5,.5,.5))
        self.assertEqual(mountain, (6.,1.,.5,.5,.5))

    def test_signed_sqrt_height_is_converted_back_to_metres_and_sea_not_mountain(self):
        s = self.settings(snr_altitude_gain=[4,1,1,1,1],snr_bins=3)
        fields = self.fields(1750)
        fields[0,:32] = -100  # bathymetry is not 10 km mountain terrain
        actual,buckets = window_snr(s,[.5]*5,fields)
        self.assertEqual(buckets[0],.5)
        self.assertEqual(actual[0],1.25)
        sea,_ = window_snr(s,[.5]*5,self.fields(-10000))
        self.assertEqual(sea[0],.5)

    def test_other_driver_units_and_decreasing_direction(self):
        s = self.settings(snr_driver_channel='precipitation',snr_driver_range=[100,1000],snr_driver_gain=[1,1,1,2,1])
        f = self.fields(100)
        f[3] = 2000
        actual,_=window_snr(s,[.5]*5,f)
        self.assertEqual(actual,(.5,.5,.5,1.,.5))

    def test_validation_rejects_invalid_ranges_channels_gains_and_bins(self):
        for values in ({'snr_altitude_range_m':[1000,500]}, {'snr_driver_range':[5,5]},
                       {'snr_driver_channel':'foo'}, {'snr_driver_gain':[0]*5},
                       {'snr_bins':1}, {'snr_bins':2.5}, {'snr_altitude_gain':[1]*4}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.settings(**values)

    def test_old_complete_v1_receipt_replays_with_neutral_controls(self):
        settings=self.settings(macro_scale_km=900)
        old={k:v for k,v in settings.items() if not k.startswith('snr_')}
        payload=generation._payload('natural',old)
        token='natural--g'+hashlib.sha256(generation._bytes(payload)).hexdigest()[:24]
        with tempfile.TemporaryDirectory() as temporary, patch.object(generation,'REGISTRY_ROOT',Path(temporary)):
            (Path(temporary)/(token+'.json')).write_text(json.dumps(payload))
            self.assertEqual(generation.resolve_generation(token).settings,settings)
            active=generation.register_generation('natural',{'snr_altitude_gain':[4,1,1,1,1]})
            self.assertNotEqual(active,token)
            self.assertEqual(generation.resolve_generation(active).settings['snr_altitude_gain'][0],4.)


if __name__ == '__main__':
    unittest.main()
