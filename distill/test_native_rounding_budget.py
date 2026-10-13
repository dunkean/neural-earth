import unittest

import torch

from distill.native_rounding_budget import solver_variants


class RoundingBudgetTests(unittest.TestCase):
    def test_zero_angle_returns_input_with_correct_native_scaling(self):
        torch.manual_seed(89)
        inputs = torch.randn(2, 5, 8, 8).bfloat16()
        velocity = torch.randn_like(inputs)
        variants = solver_variants(inputs, velocity, torch.zeros(2))
        for result in variants.values():
            self.assertTrue(torch.equal(result, inputs.float()))

    def test_solver_variants_preserve_simultaneous_sign_symmetry(self):
        torch.manual_seed(89)
        inputs = torch.randn(2, 5, 8, 8).bfloat16()
        velocity = torch.randn_like(inputs)
        labels = torch.tensor([.609375, 1.5625]).bfloat16()
        positive = solver_variants(inputs, velocity, labels)
        negative = solver_variants(-inputs, -velocity, labels)
        for name in positive:
            self.assertTrue(torch.equal(negative[name], -positive[name]))


if __name__ == '__main__':
    unittest.main()
