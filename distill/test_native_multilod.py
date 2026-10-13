import unittest

import torch

from distill.native_multilod import height_terms, relative_tpi


class HeightMorphologyTests(unittest.TestCase):
    def valley(self):
        y,x = torch.meshgrid(torch.linspace(-1,1,64),torch.linspace(-1,1,64),indexing='ij')
        return (100.-30.*torch.exp(-((x-.25*y)**2)/.01))[None,None]

    def test_equal_relief_has_zero_loss(self):
        field = self.valley()
        terms = height_terms(field,field,(1.,2.,4.,8.),.1)
        for value in terms.values():
            self.assertEqual(float(value),0.)

    def test_shifted_valley_is_penalized_and_has_finite_gradients(self):
        target = self.valley()
        shifted = target.roll(5,dims=-1).requires_grad_()
        loss = relative_tpi(shifted,target,(1.,2.,4.,8.))
        self.assertGreater(float(loss.detach()),.1)
        loss.backward()
        self.assertTrue(torch.isfinite(shifted.grad).all())
        self.assertGreater(float(shifted.grad.abs().sum()),0.)

    def test_flat_sea_does_not_divide_by_zero(self):
        target = torch.zeros(1,1,64,64)
        prediction = torch.ones_like(target,requires_grad=True)
        terms = height_terms(prediction,target,(1.,2.,4.,8.),.1)
        loss = sum(terms.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_shape_support_cannot_include_padding_edges(self):
        field = self.valley()
        with self.assertRaisesRegex(ValueError,'interior morphology support'):
            relative_tpi(field,field,(32.,))


if __name__ == '__main__':
    unittest.main()
