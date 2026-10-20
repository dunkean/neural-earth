import unittest

import torch

from distill.tile40_adversarial_pilot import Critic, TerrainCritic, critic_hinge, gradient_weight


class CriticTests(unittest.TestCase):
    def test_frozen_critic_still_propagates_to_generator_features(self):
        critic = Critic().eval().requires_grad_(False)
        latent = torch.randn(1, 5, 64, 64, requires_grad=True)
        condition = torch.randn(1, 6, 64, 64)
        score = critic(latent, condition, torch.zeros(1, 5, 1, 1), torch.ones(1, 5, 1, 1))
        score.mean().backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(float(latent.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in critic.parameters()))

    def test_physical_critic_ignores_constant_altitude_and_preserves_relief_amplitude(self):
        critic = TerrainCritic().eval().requires_grad_(False)
        height = (torch.randn(1, 1, 256, 256)*30).requires_grad_(True)
        condition = torch.zeros(1, 6, 256, 256)
        first = critic(height, condition)
        torch.testing.assert_close(first, critic(height+500., condition), atol=2e-5, rtol=2e-5)
        self.assertGreater(float((first-critic(height*2., condition)).detach().abs().max()), .0001)
        first.mean().backward()
        self.assertTrue(torch.isfinite(height.grad).all())
        self.assertGreater(float(height.grad.abs().sum()), 0.)

    def test_matched_critic_comparison_cancels_climate_offsets(self):
        real=torch.tensor([.8,.1]);fake=torch.tensor([.2,-.5]);offset=torch.tensor([40.,-20.])
        torch.testing.assert_close(critic_hinge(real,fake,True),critic_hinge(real+offset,fake+offset,True),atol=2e-6,rtol=0.)

    def test_gradient_balance_limits_large_critic_and_disables_zero_gradient(self):
        parameter=torch.tensor([2.,3.],requires_grad=True)
        physical=parameter.square().sum();adversarial=parameter.sum()*100
        weight,values=gradient_weight(physical,adversarial,(parameter,),.3)
        self.assertAlmostEqual(weight*values['critic_gradient']/values['physical_gradient'],.3)
        zero,_=gradient_weight(physical,parameter.sum()*0,(parameter,),.3)
        self.assertEqual(zero,0.)


if __name__ == '__main__':
    unittest.main()
