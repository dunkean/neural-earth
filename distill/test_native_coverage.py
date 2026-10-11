"""Recover physical climate from the pinned native conditioning interface."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from distill.native_coverage import climate_from_condition
from terrain_diffusion.inference.world_pipeline import WorldPipeline


class NativeCoverageTests(unittest.TestCase):
    def test_inverse_matches_actual_runtime_conditioning_and_central_cells(self):
        # Outside cells deliberately differ: the vector carries central2x2 means.
        fields = torch.full((1, 7, 4, 4), 1000.)
        expected = torch.tensor([18., 180., 200., 35.])
        fields[:, 2:6, 1:3, 1:3] = expected[None, :, None, None]
        means = torch.tensor([14.99, 11.65, 15.87, 619.26, 833.12, 69.40, .66])
        stds = torch.tensor([21.72, 21.78, 10.40, 452.29, 738.09, 34.59, .47])
        vector = WorldPipeline._process_latent_conditioning(SimpleNamespace(seed=123),
            fields, torch.zeros(1, 5), means, stds, torch.tensor(0.))[0]
        np.testing.assert_allclose(climate_from_condition(vector.numpy()), expected.numpy(), atol=1e-4)
        np.testing.assert_allclose(climate_from_condition(vector.bfloat16().float().numpy()),
                                   expected.numpy(), atol=3.)

    def test_rejects_wrong_interface_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            climate_from_condition(np.zeros(57))
        bad = np.zeros(58); bad[34] = np.nan
        with self.assertRaises(ValueError):
            climate_from_condition(bad)


if __name__ == '__main__':
    unittest.main()
