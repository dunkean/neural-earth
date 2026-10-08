"""Physical SNR units, neutral path, driver direction and persisted identities."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

import terrain_generation as generation
from terrain_snr import window_snr


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
