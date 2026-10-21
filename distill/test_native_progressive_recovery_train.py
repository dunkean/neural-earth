"""Check bounded-memory recomputation against the existing two-pass solver."""
import unittest
from types import SimpleNamespace

import torch

from distill.native_decoded import DecodedRegion, decoded_region_forward
from distill.native_fusion import Region
from distill.native_progressive_recovery_train import paired_forward
from terrain_diffusion.inference.world_pipeline import linear_weight_window


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor(.1))
        self.second = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, inputs, labels, conditions):
        gain = torch.where(labels > 1., self.first, self.second)[:, None, None, None]
        return inputs[:, :5]*gain


class RecomputeTests(unittest.TestCase):
    def test_same_fields_and_parameter_gradients_through_both_passes(self):
        torch.manual_seed(7)
        region = SimpleNamespace(core=(0, 0), init_origin=(-64, -64), dtype=torch.float32,
            init_origins=[(y, x) for y in (-64, -32, 0, 32, 64) for x in (-64, -32, 0, 32, 64)],
            ref_origins=[(y, x) for y in (-32, 0, 32) for x in (-32, 0, 32)])
        region.stages = [dict(inputs=torch.randn(n, 5, 64, 64), labels=torch.full((n,), label),
                              condition=torch.zeros(n, 58)) for n, label in ((25, 1.5625), (9, .609375))]
        region.noise = torch.randn(9, 5, 64, 64)
        region.ref_crops = lambda canvas:Region.ref_crops(region, canvas)
        region.core_crop = lambda refined, weight:DecodedRegion.core_crop(region, refined, weight)
        weight = linear_weight_window(64, 'cpu', torch.float32)
        model = ToyModel()
        original = decoded_region_forward(model, region, weight)['core']
        original.square().mean().backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        actual = paired_forward(model, region, weight)['core']
        self.assertTrue(torch.equal(original.detach(), actual.detach()))
        actual.square().mean().backward()
        for parameter, gradient in zip(model.parameters(), expected):
            self.assertGreater(float(parameter.grad.abs()), 0.)
            torch.testing.assert_close(parameter.grad, gradient, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
