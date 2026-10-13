import unittest

import torch

from distill.native_field_auxiliary import low_height,low_field_terms


def core(height):
    q=height.sign()*height.abs().sqrt()
    return torch.cat([torch.zeros_like(height).repeat(1,4,1,1),(q+31.4)/38.6],dim=1)


class NativeFieldAuxiliaryTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_exact_teacher_has_zero_auxiliary_loss_and_gradient(self):
        y,x=torch.meshgrid(torch.arange(32.),torch.arange(32.),indexing='ij')
        target=core((.1*x-.2*y)[None,None]);p=target.clone().requires_grad_(True)
        terms=low_field_terms(p,target,flat_scene=True,flat_power_m2=1.)
        (terms['coast']+terms['flat_highpass']).backward()
        self.assertEqual(float(terms['coast'].detach()),0.);self.assertEqual(float(terms['flat_highpass'].detach()),0.)
        self.assertEqual(float(p.grad.abs().max()),0.)

    def test_wrong_sign_gradient_corrects_final_physical_height(self):
        height=torch.full((1,1,16,16),2.);height[...,:8]=-2.
        target=core(height);p=core(-height).requires_grad_(True)
        terms=low_field_terms(p,target,flat_power_m2=1.)
        terms['coast'].backward()
        self.assertGreater(float(terms['coast'].detach()),0.)
        self.assertTrue(bool((p.grad[:,4:5]*(p[:,4:5]-target[:,4:5])>0).all()))
        self.assertEqual(float(p.grad[:,:4].abs().max()),0.)

    def test_paired_highpass_penalizes_noise_without_supervising_nonflat_scenes(self):
        y,x=torch.meshgrid(torch.arange(32.),torch.arange(32.),indexing='ij')
        height=(100.+.1*x+.05*y)[None,None];noise=((x+y)%2*2-1)[None,None]*.2
        target=core(height);p=core(height+noise).requires_grad_(True)
        flat=low_field_terms(p,target,flat_scene=True,flat_power_m2=1.)['flat_highpass']
        self.assertGreater(float(flat.detach()),0.);flat.backward()
        self.assertGreater(float((p.grad[:,4:5]*(p[:,4:5]-target[:,4:5])).sum()),0.)
        other=low_field_terms(p,target,flat_scene=False,flat_power_m2=1.)
        self.assertEqual(float(other['flat_highpass'].detach()),0.)
        self.assertEqual(other['flat_land_pixels'],0)

    def test_no_land_and_invalid_calibration(self):
        t=core(torch.full((1,1,16,16),-100.));p=t.clone().requires_grad_(True)
        terms=low_field_terms(p,t,flat_scene=True,flat_power_m2=1.)
        self.assertEqual(float(terms['flat_highpass'].detach()),0.)
        for value in (0.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):low_field_terms(p,t,flat_power_m2=value)


if __name__=='__main__':unittest.main()
