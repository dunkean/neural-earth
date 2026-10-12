import unittest
import torch
from distill.native_auxiliary_training import root_coast,combine_gradients,gradient_norm
from distill.test_native_field_auxiliary import core


class AuxiliaryTrainingTests(unittest.TestCase):
    def test_root_sign_gradient_survives_zero_and_small_classes_are_excluded(self):
        target=core(torch.full((1,1,16,16),2.));p=core(torch.zeros(1,1,16,16)).requires_grad_(True)
        loss,_,_=root_coast(p,target);loss.backward()
        self.assertTrue(bool((p.grad[:,4]<0).all()))
        tiny=core(torch.full((1,1,16,16),100.));tiny[:,:,-1,-1]=target[:,:,-1,-1]
        p=tiny.clone().requires_grad_(True);loss,_,_=root_coast(p,tiny)
        self.assertEqual(float(loss.detach()),0.)
        loss.backward();self.assertEqual(float(p.grad.abs().max()),0.)

    def test_cap_bounds_raw_and_adam_scaled_increment_and_preserves_base(self):
        params=[torch.nn.Parameter(torch.zeros(2))];base=[torch.tensor([.1,.2])]
        pre=[torch.tensor([100.,1.])];params[0].grad=torch.tensor([1000.,500.])
        raw=gradient_norm(base);scaled=gradient_norm(base,pre)
        row=combine_gradients(params,base,pre,raw,scaled)
        delta=[params[0].grad-base[0]]
        self.assertLessEqual(gradient_norm(delta),.5*raw+1e-6)
        self.assertLessEqual(gradient_norm(delta,pre),.5*scaled+1e-5)
        self.assertLess(row['auxiliary_scale'],1.)
        torch.testing.assert_close(base[0],torch.tensor([.1,.2]))


if __name__=='__main__':unittest.main()
