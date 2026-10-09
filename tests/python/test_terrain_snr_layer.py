
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from terrain_generation import resolve_generation, register_generation
from terrain_snr import window_snr
import terrain_snr_layer as layer


class SnrLayerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        registry = patch('terrain_generation.REGISTRY_ROOT', Path(temporary.name))
        registry.start()
        self.addCleanup(registry.stop)

    def tearDown(self):
        layer._window.cache_clear()

    def test_neutral_policy_never_samples_or_runs_a_model(self):
        baseline = [.05, .5, .25, 1., 2.]
        with patch('terrain_conditioning.make_conditioning_factory', side_effect=AssertionError):
            for channel in range(5):
                result = layer.sample(42, 'natural', baseline, [-1e6, 1e6], [-1e6, 1e6], channel)
                np.testing.assert_allclose(result, baseline[channel])

    def test_policies_match_inference_windows_with_negative_coordinates(self):
        settings = resolve_generation('natural').settings
        settings['snr_altitude_gain'] = [4, 2, 1, 1, 1]
        settings['snr_driver_gain'] = [3, 1, 2, 1, 1]
        profile = register_generation('natural', settings)
        baseline = settings['cond_snr']
        calls = []
        def factory(x0, y0, x1, y1):
            calls.append((x0, y0, x1, y1))
            fields = np.zeros((5, 64, 64), np.float32)
            fields[0] = np.sqrt(1000 if x0 < 0 else 4000)
            fields[1] = -20
            return fields
        xs = np.array([-16, 32, 80]) * layer.RESOLUTION
        ys = np.array([-16, 32]) * layer.RESOLUTION
        with patch('terrain_conditioning.make_conditioning_factory', return_value=factory):
            for channel in range(5):
                actual = layer.sample(7, profile, baseline, xs, ys, channel)
                for i, row in enumerate([-1, 0]):
                    for j, column in enumerate([-1, 0, 1]):
                        fields = factory(column*48, row*48, column*48+64, row*48+64)
                        self.assertAlmostEqual(actual[i,j], window_snr(settings, baseline, fields)[0][channel], places=6)
        self.assertEqual(layer._window.cache_info().misses, 6)
        self.assertIn((-48, -48, 16, 16), calls)

    def test_channel_coloring_and_world_policy_identity(self):
        settings = resolve_generation('natural').settings
        settings['cond_snr'] = [.125, .5, 1, 2, 4]
        profile = register_generation('natural', settings)
        for mode in layer.MODES:
            rgb = layer.render(7, profile, settings['cond_snr'], mode, [0, 1], [0, 1])
            self.assertEqual(rgb.shape, (2, 2, 3))
            self.assertTrue(np.isfinite(rgb).all())
            self.assertTrue(((rgb >= 0) & (rgb <= 1)).all())
        np.testing.assert_allclose(layer.sample(7, profile, settings['cond_snr'], [0], [0], 0), .125)


if __name__ == '__main__':
    unittest.main()
