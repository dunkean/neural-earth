import unittest

import numpy as np
import torch

from distill.features import COND_MEANS, COND_STDS, noise, base_features
from distill.student import Student, StudentConfig
from distill.tile40_encoder_pilot import encoder_window, encoder_geometry, highpass_delta, fine_relief_terms, ridge_structure_terms


class EncoderPilotTests(unittest.TestCase):
    def test_ridge_loss_detects_wrong_directions_and_has_finite_gradients(self):
        yy,xx=torch.meshgrid(torch.arange(128),torch.arange(128),indexing='ij')
        reference=(1000+10*torch.sin(xx.float()*.3))[None,None].sqrt()
        wrong=(1000+10*torch.sin(yy.float()*.3))[None,None].sqrt().requires_grad_(True)
        identical,_=ridge_structure_terms(reference,reference)
        self.assertEqual(float(identical),0.)
        loss,gap=ridge_structure_terms(wrong,reference)
        self.assertGreater(float(loss.detach()),.1)
        loss.backward()
        self.assertTrue(torch.isfinite(wrong.grad).all())
        self.assertGreater(float(wrong.grad.abs().sum()),0.)

    def test_central_training_crop_matches_whole_tile_convolution(self):
        torch.manual_seed(1431)
        model=Student(StudentConfig(stage='base',width=16,depth=2,dilations=(1,2))).eval()
        with torch.no_grad():
            model.head.weight.normal_(std=.03)
            model.direct.weight.normal_(std=.03)
        coarse=np.broadcast_to(COND_MEANS[:,None,None],(6,24,24)).copy()
        coarse[0]+=np.arange(24,dtype=np.float32)[None,:]
        seed,histogram=8100000,[.2]*5
        y,x,size,oy,ox=encoder_geometry(48,48,model.halo)
        window=base_features(coarse,-8,-8,seed,y,x,size,histogram)[None]
        origin=-64-model.halo
        whole=base_features(coarse,-8,-8,seed,origin,origin,320+2*model.halo,histogram)[None]
        with torch.no_grad():
            actual=model(window)[...,oy:oy+64,ox:ox+64]
            expected=model(whole)[...,48-origin:112-origin,48-origin:112-origin]
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)

    def test_central_crop_retains_aligned_features_and_original_decoder_noise(self):
        yy, xx = torch.meshgrid(torch.arange(320), torch.arange(320), indexing='ij')
        row = dict(case=dict(seed=8100000), latents=(yy*1000+xx)[None].expand(5,-1,-1).float(),
            coarse=np.broadcast_to(COND_MEANS[:,None,None],(6,24,24)).copy(),
            guided_root=torch.zeros(1344,1344),metadata=dict(latent_origin_pixels=[-64,-64],
                coarse_origin_cells=[-8,-8],world_config=dict(parameters=dict(histogram_raw=[.2]*5))))
        inputs,target,gaussian,root=encoder_window(row,48,48,96,'cpu')
        self.assertEqual(encoder_geometry(48,48,96),(-64,-64,288,112,112))
        torch.testing.assert_close(target[0],row['latents'][:,112:176,112:176])
        torch.testing.assert_close(inputs[0,:5],noise(8105819,-64,-64,288,288,5))
        torch.testing.assert_close(gaussian[0],noise(8105819,384,384,512,512,1,tile=512))
        self.assertEqual(tuple(root.shape),(1,1,384,384))

    def test_window_keeps_global_noise_coordinates_and_full_conditioning_halo(self):
        yy, xx = torch.meshgrid(torch.arange(320), torch.arange(320), indexing='ij')
        latents = (yy*1000+xx)[None].expand(5, -1, -1).float()
        coarse = np.broadcast_to(COND_MEANS[:, None, None], (6, 24, 24)).copy()
        coarse[0] += COND_STDS[0]*2.
        row = dict(case=dict(seed=8100000), latents=latents, coarse=coarse,
            guided_root=torch.zeros(1344, 1344), metadata=dict(latent_origin_pixels=[-64, -64],
                coarse_origin_cells=[-8, -8], world_config=dict(parameters=dict(histogram_raw=[.2]*5))))
        inputs, target, gaussian, root = encoder_window(row, 96, 0, 96, 'cpu')
        self.assertEqual(tuple(inputs.shape), (1, 32, 256, 256))
        torch.testing.assert_close(target[0], latents[:, 160:224, 64:128])
        torch.testing.assert_close(inputs[0, :5], noise(8105819, 0, -96, 256, 256, 5))
        torch.testing.assert_close(inputs[0, 10], torch.full((256, 256), 2.))
        self.assertEqual(tuple(gaussian.shape), (1, 1, 512, 512))
        self.assertEqual(tuple(root.shape), (1, 1, 384, 384))

    def test_highpass_removes_constant_macro_height_error(self):
        delta = torch.full((1, 1, 384, 384), 12.)
        torch.testing.assert_close(highpass_delta(delta), torch.zeros_like(delta), atol=1e-5, rtol=0.)

    def test_highpass_retains_fine_relief_and_gradients(self):
        delta = torch.randn(1, 1, 384, 384, requires_grad=True)
        actual = highpass_delta(delta)
        self.assertGreater(float(actual.detach().std()), .9)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(delta.grad).all())
        self.assertGreater(float(delta.grad.abs().sum()), 1.)

    def test_fine_relief_detects_collapse_despite_identical_macro_elevation(self):
        yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='ij')
        height = (1000.+5.*torch.sin(xx*1.2)*torch.cos(yy*1.1))[None, None]
        truth = height.sqrt()
        exact, _ = fine_relief_terms(truth, truth)
        self.assertEqual(float(exact), 0.)
        actual = (1000.+.5*(height-1000.)).sqrt().requires_grad_(True)
        loss, gap = fine_relief_terms(actual, truth)
        self.assertGreater(float(gap.detach()), .15)
        loss.backward()
        self.assertTrue(torch.isfinite(actual.grad).all())
        self.assertGreater(float(actual.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
