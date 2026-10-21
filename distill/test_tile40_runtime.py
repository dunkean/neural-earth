import unittest

import torch

from distill.tile40_runtime import (COARSE_ORIGIN, COARSE_SIZE, LATENT_ORIGIN, LATENT_SIZE,
                                  ROOT_ORIGIN, ROOT_SIZE, window_origins, add_window, normalized_canvas)
from distill.features import decoder_features


class RuntimeGeometryTests(unittest.TestCase):
    def test_finite_decoder_features_preserve_world_noise_and_latent_alignment(self):
        # A smaller finite field still crosses the negative512px noise boundary.
        latents = torch.arange(4*80*80,dtype=torch.float32).reshape(4,80,80)
        whole = decoder_features(latents,751,-256,-256,640,noise_tile=512)
        window = decoder_features(latents[:,32:72,32:72],751,0,0,320,noise_tile=512)
        torch.testing.assert_close(whole[:,256:576,256:576],window,atol=0.,rtol=0.)

    def test_all_decoder_windows_fit_the_generated_latents(self):
        starts = window_origins(ROOT_ORIGIN, ROOT_ORIGIN+ROOT_SIZE, 512, 384)
        self.assertEqual(starts, [-384, 0, 384, 768, 1152, 1536])
        for start in starts:
            self.assertGreaterEqual(start//8, LATENT_ORIGIN)
            self.assertLessEqual((start+512)//8, LATENT_ORIGIN+LATENT_SIZE)

    def test_coarse_windows_include_all_negative_context_dependencies(self):
        self.assertEqual(window_origins(COARSE_ORIGIN, COARSE_ORIGIN+COARSE_SIZE, 64, 48), [-48, 0])

    def test_overlap_blending_preserves_a_global_plane_at_negative_coordinates(self):
        canvas = torch.zeros(1, 12, 12)
        weights = torch.zeros(12, 12)
        axis = torch.arange(8)
        weight = torch.outer(axis.float()+1, axis.float()+1)
        for y in window_origins(-3, 9, 8, 4):
            for x in window_origins(-3, 9, 8, 4):
                prediction = (axis[:, None]+y)*2.+(axis[None, :]+x)*3.
                add_window(canvas, weights, prediction[None], weight, y, x, -3)
        actual = normalized_canvas(canvas, weights)[0]
        expected = (torch.arange(-3, 9)[:, None]*2.+torch.arange(-3, 9)[None, :]*3.)
        torch.testing.assert_close(actual, expected)

    def test_incomplete_window_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            normalized_canvas(torch.zeros(1, 8, 8), torch.zeros(8, 8))


if __name__ == '__main__':
    unittest.main()
