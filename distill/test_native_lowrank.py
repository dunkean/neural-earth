import unittest

import torch
from torch.nn import functional as F

from distill.native_lowrank import FactorMPConv
from terrain_diffusion.models.mp_layers import MPConv


class LowrankConvolutionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(82)
        self.conv = MPConv(7, 9, kernel=[3, 3]).eval()
        with torch.no_grad():
            self.conv.weight.mul_(torch.linspace(.2, 2., 9)[:, None, None, None])
        self.inputs = torch.randn(2, 7, 12, 13)

    def test_full_rank_matches_original_global_normalization_and_gain(self):
        expected = self.conv(self.inputs, gain=.37)
        for mode in ('factorized', 'composed'):
            factor = FactorMPConv.from_conv(self.conv, 9, mode)
            torch.testing.assert_close(factor(self.inputs, gain=.37), expected, atol=3e-6, rtol=3e-5)

    def test_composed_norm_matches_dense_weight_and_respects_factor_gauge(self):
        factor = FactorMPConv.from_conv(self.conv, 5)
        expected = factor.composed_weight().square().mean().sqrt()+1e-4
        torch.testing.assert_close(factor.normalization(), expected)
        before = factor(self.inputs)
        with torch.no_grad():
            factor.spatial.mul_(3.); factor.pointwise.div_(3.)
        torch.testing.assert_close(factor(self.inputs), before, atol=3e-6, rtol=3e-5)

    def test_truncated_forward_and_gradients_match_composed_convolution(self):
        factor = FactorMPConv.from_conv(self.conv, 5)
        dense = factor.composed_weight()
        weight = dense/((dense.square().mean().sqrt()+1e-4)*(7*9)**.5)
        direct = F.conv2d(self.inputs, weight, padding=1)
        result = factor(self.inputs)
        torch.testing.assert_close(result, direct, atol=5e-6, rtol=5e-5)
        target = torch.randn_like(result)
        parameters = list(factor.parameters())
        a = torch.autograd.grad((result*target).sum(), parameters, retain_graph=True)
        b = torch.autograd.grad((direct*target).sum(), parameters)
        for actual, expected in zip(a, b):
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=1e-4)

    def test_zero_composed_weight_has_finite_gradients(self):
        factor = FactorMPConv.from_conv(self.conv, 5)
        with torch.no_grad():
            factor.pointwise.zero_()
        result = factor(self.inputs)
        self.assertEqual(float(result.detach().abs().sum()), 0.)
        result.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in factor.parameters()))

    def test_grouped_or_nonspatial_teacher_layers_are_rejected(self):
        with self.assertRaises(ValueError):
            FactorMPConv.from_conv(MPConv(7, 9, kernel=[]), 5)


if __name__ == '__main__':
    unittest.main()
