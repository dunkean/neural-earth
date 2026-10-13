"""Physical units, both stages, position and land/sea gradients for recovery."""
import unittest

import torch

from distill.native_progressive_recovery_loss import height_terms, x0_height_terms, field_terms


def latent_from_root(q):
    result = torch.zeros((q.shape[0], 5, *q.shape[-2:]))
    result[:, 4:5] = (q+31.4)/38.6
    return result


class RecoveryLossTests(unittest.TestCase):
    def test_height_units_match_known_elevations(self):
        target = latent_from_root(torch.full((1, 1, 4, 4), 10.))
        actual = latent_from_root(torch.full((1, 1, 4, 4), 11.))
        terms = height_terms(actual, target)
        self.assertAlmostEqual(float(terms['height_mae_m']), 21., places=3)
        self.assertAlmostEqual(float(terms['height_exact_mse']), 441., places=2)
        self.assertAlmostEqual(float(terms['height_jacobian_mse']), 400., places=2)

    def test_equal_band_power_at_wrong_positions_is_penalized(self):
        grid = (torch.arange(8)[:, None]+torch.arange(8)[None, :])%2*2-1
        pattern = grid.float()[None, None]*.5
        target = latent_from_root(10.+pattern).requires_grad_(True)
        actual = latent_from_root(10.-pattern).requires_grad_(True)
        loss = height_terms(actual, target)['height_jacobian_mse']
        self.assertGreater(float(loss.detach()), 1.)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(actual.grad).all())
        self.assertTrue((actual.grad[:, 4].abs() > 0).all())

    def test_both_native_phases_receive_height_gradients(self):
        labels = torch.tensor([1.5625, .609375])
        inputs = torch.ones((2, 5, 4, 4))
        target = torch.zeros_like(inputs, requires_grad=True)
        actual = torch.zeros_like(inputs)
        actual[:, 4] = .01
        actual.requires_grad_(True)
        terms = x0_height_terms(actual, target, dict(inputs=inputs, labels=labels))
        terms['height_jacobian_mse'].backward()
        self.assertIsNone(target.grad)
        for phase in (0, 1):
            self.assertGreater(float(actual.grad[phase, 4].abs().sum()), 0.)

    def test_shore_gradient_corrects_the_sign_near_zero(self):
        grid = (torch.arange(8)[:, None]+torch.arange(8)[None, :])%2*2-1
        q = grid.float()[None, None]*.5
        target = latent_from_root(q).requires_grad_(True)
        actual = latent_from_root(-q).requires_grad_(True)
        terms = field_terms(actual, target)
        self.assertGreater(float(terms['coast'].detach()), 0.)
        terms['coast'].backward()
        self.assertIsNone(target.grad)
        self.assertTrue((actual.grad[:, 4:5]*q < 0).all())


if __name__ == '__main__':
    unittest.main()
