"""Verify original MP arithmetic at the anchor and a usable scalar gradient."""
import unittest

import torch

from distill.native_scalar_precision import anchored_output_conv
from terrain_diffusion.models.mp_layers import MPConv


class ScalarPrecisionTests(unittest.TestCase):
    def test_anchor_preserves_original_convolution(self):
        torch.manual_seed(73)
        layer = MPConv(16, 5, [3, 3]).eval()
        layer._gain_anchor = -.1943359375
        inputs = torch.randn(2, 16, 8, 8).bfloat16()
        expected = layer(inputs, gain=torch.tensor(layer._gain_anchor).bfloat16())
        actual = anchored_output_conv(layer, inputs, gain=torch.tensor(layer._gain_anchor))
        self.assertTrue(torch.equal(expected, actual))

    def test_fine_gain_has_gradient_and_changes_output(self):
        torch.manual_seed(73)
        layer = MPConv(16, 5, [3, 3]).eval().requires_grad_(False)
        layer._gain_anchor = -.1943359375
        inputs = torch.randn(2, 16, 8, 8).bfloat16()
        gain = torch.tensor(layer._gain_anchor, requires_grad=True)
        initial = anchored_output_conv(layer, inputs, gain=gain)
        initial.float().square().mean().backward()
        self.assertTrue(torch.isfinite(gain.grad))
        self.assertGreater(float(gain.grad.abs()), 0)
        small_gain = gain.detach() + .00006103515625
        self.assertTrue(torch.equal(small_gain.bfloat16(), gain.detach().bfloat16()))
        shifted = anchored_output_conv(layer, inputs, gain=small_gain)
        self.assertFalse(torch.equal(shifted, initial))


if __name__ == '__main__':
    unittest.main()
