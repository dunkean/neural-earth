import unittest

import torch

from distill.native_amplitude import amplitude_depth_terms, field_morphology, morphology_degraded


class ValleyAmplitudeTests(unittest.TestCase):
    def terrain(self):
        y,x=torch.meshgrid(torch.arange(64.),torch.arange(64.),indexing='ij')
        return (500.-80.*torch.exp(-((x-32.-3.*torch.sin(y/8.))/3.)**2))[None,None]

    def test_exact_terrain_has_zero_loss(self):
        reference=self.terrain()
        values=amplitude_depth_terms(reference,reference,(1.,2.,4.))
        self.assertEqual(float(values['amplitude']),0.)
        self.assertEqual(float(values['valley_depth']),0.)

    def test_shallow_valleys_produce_finite_corrective_gradients(self):
        reference=self.terrain()
        actual=(500.+.5*(reference-500.)).requires_grad_()
        values=amplitude_depth_terms(actual,reference,(1.,2.,4.))
        self.assertGreater(float(values['amplitude'].detach()),0.)
        self.assertGreater(float(values['valley_depth'].detach()),0.)
        sum(values.values()).backward()
        self.assertTrue(torch.isfinite(actual.grad).all())
        self.assertGreater(float(actual.grad.abs().sum()),0.)
        corrected=amplitude_depth_terms((actual-actual.grad).detach(),reference,(1.,2.,4.))
        self.assertLess(float(sum(corrected.values())),float(sum(values.values()).detach()))

    def test_moving_valleys_cannot_pass_by_matching_amplitude(self):
        reference=self.terrain()
        values=amplitude_depth_terms(torch.roll(reference,12,-1),reference,(1.,2.,4.))
        self.assertGreater(float(values['valley_depth']),.1)

    def test_flat_land_and_empty_sea_are_finite(self):
        for height in (500.,-500.):
            reference=torch.full((1,1,64,64),height)
            actual=reference.clone().requires_grad_()
            values=amplitude_depth_terms(actual,reference,(1.,2.,4.))
            self.assertEqual(float(sum(values.values()).detach()),0.)
            sum(values.values()).backward()
            self.assertTrue(torch.isfinite(actual.grad).all())

    def test_flat_teacher_does_not_reward_added_noise(self):
        reference=torch.full((1,1,64,64),500.)
        x=torch.arange(64.)[None,None,None,:]
        actual=(reference+5.*torch.sin(x)).requires_grad_()
        values=amplitude_depth_terms(actual,reference,(1.,2.,4.))
        self.assertGreater(float(values['amplitude'].detach()),0.)
        self.assertEqual(float(values['valley_depth'].detach()),0.)
        sum(values.values()).backward()
        self.assertTrue(torch.isfinite(actual.grad).all())

    def test_flat_candidate_is_failure_in_a_teacher_valley(self):
        reference=self.terrain()[0,0].numpy()
        actual=torch.full((64,64),500.).numpy()
        result=field_morphology(actual,reference)
        self.assertIsNotNone(result)
        self.assertEqual(result['valley_retention'],0.)
        self.assertEqual(result['tpi_correlation'],0.)

    def test_guard_retains_its_fixed_baseline_and_position_constraint(self):
        before=dict(valley_retention=.4,tpi_correlation=.7)
        baseline={'coarse4:rare:plain':dict(morphology=dict(fine=before,native=before))}
        self.assertFalse(morphology_degraded(baseline,baseline))
        for after in (dict(valley_retention=.2,tpi_correlation=.7),
                      dict(valley_retention=.6,tpi_correlation=.5),None):
            current={'coarse4:rare:plain':dict(morphology=dict(fine=after,native=before))}
            self.assertTrue(morphology_degraded(current,baseline))


if __name__=='__main__':
    unittest.main()
