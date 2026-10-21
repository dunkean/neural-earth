import unittest

import torch

from distill.tile40_decoder_pilot import physical_loss


class PhysicalLossTests(unittest.TestCase):
    def test_matching_guided_height_has_zero_loss(self):
        root = torch.rand(1, 1, 384, 384)*20.+30.
        loss, values = physical_loss(root, root)
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(values['amplitude_gap']), 0.)

    def test_amplitude_collapse_is_penalized_even_with_matching_mean(self):
        yy, xx = torch.meshgrid(torch.arange(384), torch.arange(384), indexing='ij')
        height = (1000.+100.*torch.sin(xx/12.)*torch.cos(yy/12.))[None, None]
        flat = torch.full_like(height, 1000.)
        loss, values = physical_loss(flat.sqrt(), height.sqrt())
        self.assertGreater(float(loss), .3)
        self.assertGreater(float(values['amplitude_gap']), .8)

    def test_error_is_reported_in_height_metres(self):
        p = torch.full((1, 1, 384, 384), 11.)
        t = torch.full_like(p, 10.)
        _, values = physical_loss(p, t)
        self.assertEqual(float(values['height_mae_m']), 21.)


if __name__ == '__main__':
    unittest.main()
