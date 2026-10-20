import unittest

import torch

from distill.tile40_flow_pilot import midpoint,midpoint_cropped,colored_prior,FiniteFlowEncoder,coarse_altitude_guide
from distill.student import Student,StudentConfig


class ConstantVelocity(torch.nn.Module):
    def forward(self, inputs):
        return torch.ones_like(inputs[:, :5])*2.


class FlowSolverTests(unittest.TestCase):
    def test_residual_altitude_guide_is_available_and_restored_in_physical_units(self):
        from distill.features import COND_MEANS,COND_STDS
        model=ConstantVelocity()
        model.config=StudentConfig(stage='base',width=16,depth=2)
        model.halo=32
        contract=dict(latent_mean=[0.]*5,latent_std=[1.]*5,residual_altitude=True)
        encoder=FiniteFlowEncoder(model,contract,steps=2)
        fields=torch.zeros(1,32,64,64)
        fields[:,10]=(20.-float(COND_MEANS[0]))/float(COND_STDS[0])
        expected=(20.+31.4)/38.6
        guide=coarse_altitude_guide(fields)
        torch.testing.assert_close(guide,torch.full_like(guide,expected))
        result=encoder(fields)
        torch.testing.assert_close(result[:,:4],torch.full_like(result[:,:4],2.))
        torch.testing.assert_close(result[:,4:5],torch.full_like(result[:,4:5],2.+expected))

    def test_finite_tile_sampling_retains_boundary_and_conditioning(self):
        model = ConstantVelocity()
        model.config = StudentConfig(stage='base', width=16, depth=2)
        model.halo = 32
        contract = dict(latent_mean=[0.]*5, latent_std=[1.]*5)
        encoder = FiniteFlowEncoder(model, contract, steps=8)
        fields = torch.randn(1, 32, 32, 32)
        saved = fields.clone()
        actual = encoder(fields)
        self.assertEqual(encoder.halo, 0)
        torch.testing.assert_close(actual, fields[:, :5]+2.)
        torch.testing.assert_close(fields, saved, atol=0., rtol=0.)

    def test_dependency_crops_match_full_solver_in_the_retained_interior(self):
        torch.manual_seed(627)
        model=Student(StudentConfig(stage='base',width=16,depth=2,dilations=(1,2))).eval()
        with torch.no_grad():model.head.weight.normal_(std=.03)
        fields=torch.randn(1,32,320,320);start=fields[:,:5].clone()
        with torch.no_grad():
            expected=midpoint(model,start,fields)
            actual=midpoint_cropped(model,start,fields)
        margin=4*model.halo
        torch.testing.assert_close(actual[...,margin:-margin,margin:-margin],expected[...,margin:-margin,margin:-margin],atol=2e-5,rtol=2e-5)

    def test_prior_retains_white_channels_and_smooths_only_the_spatial_latents(self):
        torch.manual_seed(237)
        initial=torch.randn(1,5,256,256)
        result=colored_prior(initial)
        torch.testing.assert_close(result[:,1:4],initial[:,1:4],atol=0.,rtol=0.)
        for channel in (0,4):
            interior=result[:,channel,16:-16,16:-16]
            self.assertGreater(float(interior.std()),.85)
            self.assertLess(float(interior.std()),1.2)
        adjacent=(result[:,4,16:-16,16:-17]*result[:,4,16:-16,17:-16]).mean()
        self.assertGreater(float(adjacent),.8)

    def test_midpoint_integrates_a_constant_velocity_and_preserves_seed_shape(self):
        start = torch.randn(1, 5, 32, 32)
        fields = torch.zeros(1, 32, 32, 32)
        for steps in (1, 2, 4):
            result = midpoint(ConstantVelocity(), start, fields, steps)
            torch.testing.assert_close(result, start+2.)
            self.assertEqual(result.shape, start.shape)

    def test_solver_does_not_modify_shared_conditioning_between_calls(self):
        start = torch.zeros(1, 5, 32, 32)
        fields = torch.randn(1, 32, 32, 32)
        saved = fields.clone()
        midpoint(ConstantVelocity(), start, fields)
        torch.testing.assert_close(fields, saved, atol=0., rtol=0.)

    def test_late_schedule_reduces_variance_loss_for_a_stiff_gaussian_bridge(self):
        class GaussianBridge(torch.nn.Module):
            def forward(self, inputs):
                t = inputs[:, 5:6]
                variance = .01
                coefficient = (-(1-t)+t*variance)/((1-t).square()+t.square()*variance)
                return coefficient*inputs[:, :5]
        start = torch.ones(1, 5, 4, 4)
        fields = torch.zeros(1, 32, 4, 4)
        uniform = midpoint(GaussianBridge(), start, fields)
        late = midpoint(GaussianBridge(), start, fields, schedule=[0., .9, 1.])
        # The exact endpoint standard deviation is0.1. Both use four calls.
        self.assertLess(abs(float(late.mean())-.1), abs(float(uniform.mean())-.1))


if __name__ == '__main__':
    unittest.main()
