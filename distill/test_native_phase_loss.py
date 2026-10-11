import unittest

import torch

from distill.native_phase_loss import band_terms, phase_loss, CHANNEL0_WEIGHTS


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

    def fixed_calibration(self):
        return dict(first_power=torch.ones(3, 5), second_power=torch.ones(3, 5),
                    second_variance=torch.ones(5))

    def test_fixed_energy_still_targets_each_teacher_not_train_average(self):
        actual = torch.zeros(2, 1, 64, 64)
        position, energy = band_terms(actual, actual, torch.ones(1), torch.ones(3, 1))
        self.assertEqual(float(position), 0.)
        self.assertEqual(float(energy), 0.)
        wrong = torch.randn_like(actual)
        position, energy = band_terms(wrong, actual, torch.ones(1), torch.ones(3, 1))
        self.assertGreater(float(position), 0.)
        self.assertGreater(float(energy), 0.)

    def test_fixed_channel0_objective_preserves_other_feature_targets(self):
        prediction = self.target.clone(); prediction[0, 0] += .2
        _, terms = phase_loss(prediction, self.inputs, self.labels, self.target, self.variance, self.fixed_calibration())
        extra = CHANNEL0_WEIGHTS['first_x0']*terms['first_x0']+CHANNEL0_WEIGHTS['first_channel0']*terms['first_channel0']
        self.assertAlmostEqual(float(extra/(.5*terms['first_x0'])), 16., places=4)
        prediction = self.target.clone(); prediction[0, 2] += .2
        _, terms = phase_loss(prediction, self.inputs, self.labels, self.target, self.variance, self.fixed_calibration())
        self.assertEqual(float(terms['first_channel0']), 0.)
        self.assertGreater(float(terms['first_x0']), 0.)

    def test_fixed_loss_identity_and_detached_calibration(self):
        calibration = {k: v.requires_grad_(True) for k, v in self.fixed_calibration().items()}
        loss, _ = phase_loss(self.target, self.inputs, self.labels, self.target, self.variance, calibration)
        self.assertEqual(float(loss), 0.)
        prediction = (self.target+.01*torch.randn_like(self.target)).requires_grad_(True)
        loss, _ = phase_loss(prediction, self.inputs, self.labels, self.target, self.variance, calibration)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(all(v.grad is None for v in calibration.values()))

    def test_fixed_position_penalizes_misplaced_equal_power_noise(self):
        reference = self.target[:2, :1]
        position, energy = band_terms(-reference, reference, torch.ones(1), torch.ones(3, 1))
        self.assertGreater(float(position), 1.)
        self.assertEqual(float(energy), 0.)


if __name__ == '__main__':
    unittest.main()
