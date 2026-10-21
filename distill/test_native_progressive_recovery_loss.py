"""Physical units, both stages, position and land/sea gradients for recovery."""
import unittest

import torch

from distill.native_progressive_recovery_loss import (
    height_terms, x0_height_terms, field_terms, multiscale_height_terms, multiscale_x0_terms)


def latent_from_root(q):
    result = torch.zeros((q.shape[0], 5, *q.shape[-2:]))
    result[:, 4:5] = (q+31.4)/38.6
    return result


class RecoveryLossTests(unittest.TestCase):
    def test_multiscale_identity_and_teacher_detachment(self):
        torch.manual_seed(4)
        target = latent_from_root(10.+torch.randn(2, 1, 64, 64)).requires_grad_(True)
        actual = target.detach().clone().requires_grad_(True)
        loss = sum(multiscale_height_terms(actual, target).values())
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertEqual(float(actual.grad.abs().sum()), 0.)

    def test_multiscale_per_scene_scaling_keeps_plain_gradient(self):
        grid = (torch.arange(64)[:, None]+torch.arange(64)[None, :])%2*2-1
        roots = torch.cat([10.+grid[None, None]*.5, 100.+grid[None, None]*.5])
        target = latent_from_root(roots)
        actual = latent_from_root(roots+grid[None, None]*.1).requires_grad_(True)
        multiscale_height_terms(actual, target)['high'].backward()
        norms = actual.grad[:, 4].abs().sum((-2, -1))
        self.assertTrue(torch.isfinite(norms).all())
        self.assertGreater(float(norms[0]), 0.)
        # A tenfold elevation root must not suppress the plain by100x.
        self.assertLess(float(norms[1]/norms[0]), 2.)

    def test_multiscale_requires_both_phases(self):
        values = torch.ones((2, 5, 64, 64))
        with self.assertRaisesRegex(ValueError, 'Both pinned'):
            multiscale_x0_terms(values, values, dict(inputs=values, labels=torch.full((2,), 1.5625)))

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
