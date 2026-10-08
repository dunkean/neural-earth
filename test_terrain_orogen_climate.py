"""Independent monthly-reference checks for physical Orogen climate adaptation."""
import unittest
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np

from terrain_conditioning import orogen_bioclim
from terrain_orogen_layers import render


class OrogenBioclimTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Orogen Node runtime unavailable')
    def test_zero_global_rainfall_percentile_has_finite_normalization(self):
        module = (Path(__file__).parent / 'native/orogen/vendor/climate-util.js').resolve().as_uri()
        script = f"import {{percentile}} from {json.dumps(module)}; const p=new Float32Array(20); const scale=percentile(p,.95); console.log(JSON.stringify([scale,...p.map(v=>Math.max(0,v/scale))]));"
        result = subprocess.run(['node', '--input-type=module', '-e', script],
                                check=True, capture_output=True, text=True)
        np.testing.assert_array_equal(json.loads(result.stdout), [1] + [0] * 20)

    def test_matches_worldclim_monthly_reference_in_physical_units(self):
        ts = np.array([[28., 15., -20.]], np.float32)
        tw = np.array([[24., -5., -40.]], np.float32)
        ps = np.array([[2700., 1000., 6.]], np.float32)
        pw = np.array([[1600., 0., 0.]], np.float32)
        actual = orogen_bioclim((ts + 45) / 90, (tw + 45) / 90, ps / 1000, pw / 1000)
        months = np.arange(12)[:, None, None]
        temperatures = (ts + tw) / 2 + (ts - tw) / 2 * np.cos(months * np.pi / 6)
        rain = np.concatenate((np.repeat(ps[None] / 6, 6, axis=0),
                               np.repeat(pw[None] / 6, 6, axis=0)))
        expected = np.stack((temperatures.mean(axis=0), temperatures.std(axis=0, ddof=1) * 100,
                             rain.sum(axis=0), rain.std(axis=0, ddof=1) / (rain.mean(axis=0) + 1) * 100))
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-5)
        self.assertEqual(actual.dtype, np.float32)
        self.assertGreater(actual[2, 0, 0], 2000)  # preserved wet tail

    def test_dry_uniform_and_season_swap_are_well_defined(self):
        ts = np.array([[.8, .1, .5]], np.float32)
        tw = np.array([[.6, .3, .5]], np.float32)
        ps = np.array([[0., .8, 1.2]], np.float32)
        pw = np.array([[0., .8, .1]], np.float32)
        first = orogen_bioclim(ts, tw, ps, pw)
        np.testing.assert_array_equal(first, orogen_bioclim(tw, ts, pw, ps))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first[3, 0, :2], [0., 0.])

    def test_more_intra_season_variation_cannot_lower_precipitation_cv(self):
        ps, pw = 1800., 600.
        minimum = orogen_bioclim(.8, .6, ps / 1000, pw / 1000)[3]
        concentrated = np.array([ps, 0, 0, 0, 0, 0, pw, 0, 0, 0, 0, 0])
        cv = concentrated.std(ddof=1) / (concentrated.mean() + 1) * 100
        self.assertGreater(cv, minimum)

    def test_invalid_rain_and_nonfinite_inputs_fail(self):
        for args in ((.5, .5, -1., 1.), (.5, .5, 1., float('nan'))):
            with self.assertRaises(ValueError):
                orogen_bioclim(*args)

    def test_precipitation_diagnostic_distinguishes_values_above_p95(self):
        world = SimpleNamespace(bounds=(-2., -1., 2., 1.), width=2, height=1,
                                layers={'precip_summer': np.array([[1., 3.]], np.float32),
                                        'plates': np.zeros((1, 2), np.float32)})
        colors = render(world, 'orogen-precip-summer', np.array([-1., 1.]), np.array([0.]))
        self.assertFalse(np.array_equal(colors[0, 0], colors[0, 1]))
        self.assertTrue(((colors >= 0) & (colors <= 1)).all())


if __name__ == '__main__':
    unittest.main()
