import unittest

import torch

from distill.native_level_objective import level_terms, protect_scalar_gradients


class LevelObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.variance = torch.ones(5)
        self.floors = dict(height_lf_rms_m=1., c0_c3_lf_normalized_rms=[1.]*4)

    def test_identical_field_zero_loss_and_gradient(self):
        actual = torch.randn(1, 5, 64, 64, requires_grad=True)
        reference = actual.detach().clone().requires_grad_(True)
        loss = level_terms(actual, reference, self.variance, self.floors)['level']
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertEqual(float(actual.grad.abs().sum()), 0)
        self.assertIsNone(reference.grad)

    def test_level_gradient_restores_lowered_land(self):
        reference = torch.zeros(1, 5, 64, 64, requires_grad=True)
        with torch.no_grad():
            reference[:, 4] = 2.
        actual = reference.detach().clone()
        actual[:, 4] -= .01
        actual.requires_grad_(True)
        loss = level_terms(actual, reference, self.variance, self.floors)['level']
        loss.backward()
        self.assertLess(float(actual.grad[:, 4].sum()), 0)
        self.assertEqual(float(actual.grad[:, :4].abs().sum()), 0)
        self.assertIsNone(reference.grad)

    def test_scalar_protection_preserves_kernel_gradient(self):
        model = torch.nn.Module()
        model.register_parameter('gain', torch.nn.Parameter(torch.tensor(1.)))
        model.register_parameter('kernel', torch.nn.Parameter(torch.ones(2, 2)))
        (model.gain*model.kernel.sum()).backward()
        before = model.kernel.grad.clone()
        self.assertEqual(protect_scalar_gradients(model), ['gain'])
        self.assertEqual(float(model.gain.grad), 0)
        self.assertTrue(torch.equal(model.kernel.grad, before))


if __name__ == '__main__':
    unittest.main()
