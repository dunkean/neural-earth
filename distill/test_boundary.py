"""Check phase labels and masked gradients used to diagnose base grid errors."""
import unittest

import torch

from distill.audit_boundary_gradient import gradient_profiles
from distill.audit_coarse_skip import gradient_regression


class BoundaryDiagnosticTests(unittest.TestCase):
    def test_step_is_assigned_to_right_endpoint_phase_zero(self):
        target = torch.zeros(1, 1, 64, 64)
        prediction = target.clone()
        prediction[..., 32:] = 2.
        values, counts = gradient_profiles(prediction, target, torch.ones_like(target))
        self.assertEqual(float(counts[0]), 128.)
        self.assertTrue(torch.equal(counts[1:], torch.full((31,), 256.)))
        self.assertEqual(float(values[0, 0, 0]), 2.)
        self.assertEqual(float(values[0, 0, 1:].sum()), 0.)

    def test_masked_boundary_does_not_contribute(self):
        target = torch.zeros(1, 1, 64, 64)
        prediction = target.clone()
        prediction[..., 32:] = 2.
        mask = torch.ones_like(target)
        mask[..., 31:33] = 0.
        values, _ = gradient_profiles(prediction, target, mask)
        self.assertTrue(torch.equal(values[0], torch.zeros_like(values[0])))
        self.assertTrue(torch.isfinite(values).all())

    def test_regression_recovers_known_intervention(self):
        coordinate = torch.arange(64, dtype=torch.float32)
        target = (coordinate[None, :] + coordinate[:, None])[None, None]
        delta = target*.2
        prediction = target-.4*delta
        dot, norm, error = gradient_regression(prediction, target, delta, torch.ones_like(target))
        alpha = -dot/norm
        self.assertAlmostEqual(float(alpha), .4, places=5)
        self.assertLess(float((error+2*alpha*dot+alpha.square()*norm).abs()), 1e-7)


if __name__ == '__main__':
    unittest.main()
