"""Reference samples cover exact40km physical cell centers without truncation."""
import unittest
import json
from types import SimpleNamespace

import numpy as np

from distill.tile40_reference_data import resample_tile, tile_stats, world_signature, guided_reference


class ReferenceTests(unittest.TestCase):
    def test_resampling_preserves_physical_coordinates_and_altitude(self):
        coordinates = (np.arange(1344)+.5)*30.
        plane = coordinates[:, None]*.02+coordinates[None, :]*.01+100.
        actual = resample_tile(plane)
        expected_axis = (np.arange(1334)+.5)*(40000./1334)
        expected = expected_axis[:, None]*.02+expected_axis[None, :]*.01+100.
        # First center differs by<8mm from the acquisition support boundary.
        np.testing.assert_allclose(actual, expected, atol=.01, rtol=0.)

    def test_reference_must_cover_full_extent(self):
        with self.assertRaisesRegex(ValueError, 'cover the full'):
            resample_tile(np.zeros((512, 512), np.float32))

    def test_stats_use_metre_spacing_for_slope(self):
        height = np.tile(np.arange(100)*30., (100, 1))
        stats = tile_stats(height, 30.)
        self.assertAlmostEqual(stats['slope_p90_degrees'], 45.)

    def test_provenance_is_json_serializable_and_keeps_physical_parameters(self):
        kwargs = {key:[1., 2.] for key in ('cond_snr', 'frequency_mult', 'coarse_means', 'coarse_stds', 'histogram_raw')}
        kwargs.update(residual_mean=0., residual_std=.7, coarse_pooling=1, drop_water_pct=.5,
                      elev_coarse_pool_mode='avg', p5_coarse_pool_mode='avg')
        world = SimpleNamespace(config={'runtime_class':type}, native_resolution=30., T=2,
            latent_compression=8, latents_batch_size=16, decoder_tile_size=512, decoder_tile_stride=384, kwargs=kwargs)
        actual = json.loads(json.dumps(world_signature(world), allow_nan=False))
        self.assertEqual(actual['parameters']['residual_std'], .7)
        self.assertEqual(actual['latent_batch'], 16)
        self.assertNotIn('runtime_class', actual)

    def test_guided_target_keeps_sketch_amplitude_when_teacher_is_flat(self):
        sketch = np.array([[100., 600.], [300., 900.]], np.float32)
        source = np.full((1856, 1856), 400., np.float32)
        actual = guided_reference(source, sketch)
        from distill.tile40_conditioning import sample_sketch
        coordinates = (np.arange(1344)+.5)*30.
        expected = sample_sketch(sketch, coordinates, coordinates)
        np.testing.assert_allclose(actual['native_height'], expected, atol=.001, rtol=0.)
        np.testing.assert_allclose(actual['root_detail'], 0., atol=.00001, rtol=0.)

    def test_guided_target_retains_detail_and_uses_the_same_detail_for_different_sketch_levels(self):
        x = np.arange(1856)
        root = 20.+np.sin(x*.5)[None, :]*.1
        source = np.repeat(root, 1856, axis=0)**2
        first = guided_reference(source, np.full((2, 2), 100., np.float32))
        second = guided_reference(source, np.full((2, 2), 900., np.float32))
        np.testing.assert_array_equal(first['root_detail'], second['root_detail'])
        self.assertGreater(float(first['root_detail'].std()), .06)
        self.assertAlmostEqual(float(np.sqrt(second['native_height']).mean()-np.sqrt(first['native_height']).mean()), 20., places=4)


if __name__ == '__main__':
    unittest.main()
