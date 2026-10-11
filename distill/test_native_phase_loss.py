import unittest

import torch

from distill.native_phase_loss import band_terms, phase_loss


class PhaseLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.target = torch.randn(4, 5, 64, 64)
        self.inputs = torch.randn_like(self.target)
        self.labels = torch.tensor([1.5625, 1.5625, .609375, .609375])
        self.variance = torch.ones(5)

    def test_identity_has_zero_loss(self):
        loss, terms = phase_loss(self.target, self.inputs, self.labels, self.target, self.variance)
        self.assertEqual(float(loss), 0.)
        self.assertTrue(all(float(value) == 0. for value in terms.values()))

    def test_equal_energy_wrong_positions_are_penalized(self):
        reference = self.target[:2, :4]
        actual = -reference
        position, energy = band_terms(actual, reference, self.variance[:4])
        self.assertGreater(float(position), 3.9)
        self.assertLess(float(energy), 1e-10)

    def test_energy_penalizes_attenuation_and_excess(self):
        reference = self.target[:2, :4]
        errors = [float(band_terms(reference*scale, reference, self.variance[:4])[1]) for scale in (.5, 2.)]
        self.assertGreater(min(errors), 1.)
        self.assertAlmostEqual(*errors, places=3)

    def test_flat_patches_produce_finite_nonzero_student_gradients(self):
        target = torch.zeros_like(self.target, requires_grad=True)
        prediction = (torch.randn_like(target)*.01).requires_grad_(True)
        loss, terms = phase_loss(prediction, torch.zeros_like(target), self.labels, target, self.variance)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.)
        self.assertIsNone(target.grad)

    def test_phase_one_features_and_phase_two_height_are_distinct(self):
        prediction = self.target.clone()
        prediction[0, 0] += .2
        prediction[2, 4] += .2
        _, terms = phase_loss(prediction, self.inputs, self.labels, self.target, self.variance)
        self.assertGreater(float(terms['first_x0']), 0.)
        self.assertGreater(float(terms['second_height']), 0.)
        prediction = self.target.clone()
        prediction[0, 4] += .2
        prediction[2, 0] += .2
        _, terms = phase_loss(prediction, self.inputs, self.labels, self.target, self.variance)
        self.assertEqual(float(terms['first_x0']), 0.)
        self.assertEqual(float(terms['second_height']), 0.)
        self.assertGreater(float(terms['operator']), 0.)

    def test_unknown_labels_are_rejected(self):
        labels = self.labels.clone(); labels[0] = .9
        with self.assertRaises(ValueError):
            phase_loss(self.target, self.inputs, labels, self.target, self.variance)


if __name__ == '__main__':
    unittest.main()
