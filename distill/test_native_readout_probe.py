import unittest

import torch

from distill.native_readout_probe import RidgeStatistics, apply_readouts, world_folds, combine_statistics


class ReadoutProbeTests(unittest.TestCase):
    def test_fit_recovers_known_residual_on_new_samples(self):
        torch.manual_seed(2026)
        training = torch.randn(1000, 18)
        weight, bias = torch.randn(18), .3
        fit = RidgeStatistics(18, 'cpu')
        fit.add(training, training@weight+bias)
        coefficients = fit.solve(1e-6)
        heldout = torch.randn(100, 18)
        torch.testing.assert_close(heldout@coefficients[:-1]+coefficients[-1],
                                   heldout@weight+bias, rtol=1e-4, atol=1e-4)

    def test_phase_gating_preserves_all_untargeted_outputs_exactly(self):
        torch.manual_seed(7)
        prediction, hidden = torch.randn(4, 5, 16, 16), torch.randn(4, 2, 16, 16)
        coefficients = torch.randn(2, 19)
        labels = torch.tensor([1.5625, .609375, .609375, 1.5625])
        result = apply_readouts(prediction, hidden, labels, coefficients, 'both')
        unchanged = torch.ones_like(result, dtype=torch.bool)
        unchanged[[0, 3], 0] = False; unchanged[[1, 2], 4] = False
        self.assertTrue(torch.equal(result[unchanged], prediction[unchanged]))
        self.assertGreater(float((result-prediction).abs().sum()), 0.)
        first = apply_readouts(prediction, hidden, labels, coefficients, 'first')
        self.assertTrue(torch.equal(first[:, 1:], prediction[:, 1:]))
        self.assertTrue(torch.equal(first[[1, 2]], prediction[[1, 2]]))

    def test_empty_fit_and_invalid_regularization_are_rejected(self):
        fit = RidgeStatistics(2, 'cpu')
        with self.assertRaises(ValueError):
            fit.solve(.01)
        fit.add(torch.ones(3, 2), torch.ones(3))
        for ridge in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                fit.solve(ridge)

    def test_folds_keep_all_conditions_of_a_world_together(self):
        records = [dict(metadata=dict(seed=seed), coarse=variant)
                   for seed in range(12) for variant in ('original', 'coarse4', 'coarse2')]
        assignment = world_folds(records)
        self.assertEqual(set(assignment.values()), {0, 1, 2})
        self.assertEqual(assignment, world_folds(list(reversed(records))))
        self.assertEqual(len(assignment), 12)

    def test_fold_statistics_reproduce_direct_fit_and_residual_mse(self):
        torch.manual_seed(42)
        x, y = torch.randn(100, 3), torch.randn(100)
        parts = [RidgeStatistics(3, 'cpu') for _ in range(2)]
        parts[0].add(x[:40], y[:40]); parts[1].add(x[40:], y[40:])
        combined = combine_statistics(parts)
        coefficients = combined.solve(.01)
        residual = x@coefficients[:-1]+coefficients[-1]-y
        self.assertAlmostEqual(combined.mse(coefficients), float(residual.square().mean()), places=6)


if __name__ == '__main__':
    unittest.main()
