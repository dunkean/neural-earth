"""Position-sensitive residual addition and exact parent Adam-state retention."""
import unittest

import torch

from distill.native_onepass import noise_windows, load_onepass
from distill.native_onepass_residual import ResidualOnePassBase, ResidualOnePassRuntime, ARCHITECTURE, extend_optimizer_slots
from distill.test_native_onepass import tiny_saved
from distill.native_onepass_residual_loss import level_height_loss


class ResidualTests(unittest.TestCase):
    def test_absolute_level_does_not_disappear_on_high_mountains(self):
        target=torch.zeros(1,5,64,64)
        target[:,4]=(100.+31.4)/38.6
        actual=target.clone();actual[:,4]+=.1
        actual.requires_grad_(True);target.requires_grad_(True)
        loss=level_height_loss(actual,target)
        self.assertGreater(float(loss.detach()),100.)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(float(actual.grad[:,4].abs().sum()),0.)

    def test_level_identity_is_zero(self):
        value=torch.randn(1,5,64,64)
        self.assertEqual(float(level_height_loss(value,value)),0.)

    def test_zero_head_preserves_all_parent_fields(self):
        torch.manual_seed(20)
        parent=tiny_saved();body=load_onepass(parent)
        model=ResidualOnePassBase(parent['config'],parent['gain_anchor'])
        model.load_state_dict(dict(parent['ema'],**{'residual.weight':torch.zeros(5,10,1,1)}))
        noises=noise_windows(617543,[(0,0)])
        condition=torch.randn(1,58)
        with torch.no_grad():
            self.assertTrue(torch.equal(model.windows(noises,condition),body.windows(noises,condition)))

    def test_known_second_noise_is_added_at_matched_positions(self):
        parent=tiny_saved();model=ResidualOnePassBase(parent['config'],parent['gain_anchor'])
        model.load_state_dict(dict(parent['ema'],**{'residual.weight':torch.zeros(5,10,1,1)}))
        noises=noise_windows(617543,[(0,0)]);condition=torch.zeros(1,58)
        with torch.no_grad():
            before=model.windows(noises,condition)
            model.residual.weight[1,6,0,0]=.6
            after=model.windows(noises,condition)
        torch.testing.assert_close(after[:,1]-before[:,1],noises[:,6]*.6,atol=1e-6,rtol=1e-6)
        self.assertTrue(torch.equal(after[:,4],before[:,4]))

    def test_optimizer_extension_preserves_moments_and_parent(self):
        old=dict(state={0:dict(step=torch.tensor(9.),exp_avg=torch.tensor([.1]),exp_avg_sq=torch.tensor([.2]))},
                 param_groups=[dict(params=[0,1],lr=1e-5)])
        new=extend_optimizer_slots(old,2,3)
        self.assertEqual(new['param_groups'][0]['params'],[0,1,2])
        self.assertEqual(old['param_groups'][0]['params'],[0,1])
        self.assertTrue(torch.equal(old['state'][0]['exp_avg'],new['state'][0]['exp_avg']))
        self.assertNotIn(2,new['state'])

    def test_frozen_residual_matches_cpu_training_forward(self):
        parent=tiny_saved();model=ResidualOnePassBase(parent['config'],parent['gain_anchor'])
        state=dict(parent['ema'],**{'residual.weight':torch.randn(5,10,1,1)*.1})
        model.load_state_dict(state)
        saved=dict(parent,architecture=ARCHITECTURE,ema=model.state_dict())
        runtime=ResidualOnePassRuntime(saved)
        noises=noise_windows(617543,[(0,0)]);condition=torch.zeros(1,58)
        with torch.no_grad():
            torch.testing.assert_close(runtime(noises,condition),model.windows(noises,condition),atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
