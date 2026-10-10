import unittest

import torch

from distill.decoded_loss import shore_loss


class ShoreLossTests(unittest.TestCase):
    def test_exact_height_has_zero_loss_and_gradient(self):
        target = torch.tensor([[[[-2., 1., 18., 1000.]]]])
        prediction = target.clone().requires_grad_()
        loss, disagreement = shore_loss(prediction, target, torch.ones_like(target))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(disagreement), 0.)
        torch.testing.assert_close(prediction.grad, torch.zeros_like(target), atol=1e-7, rtol=0.)

    def test_gradient_corrects_wrong_sea_sign_and_ignores_invalid_or_high_relief(self):
        target = torch.tensor([[[[-2., 2., 1000., -2.]]]])
        prediction = torch.tensor([[[[2., -2., -1000., 2.]]]], requires_grad=True)
        mask = torch.tensor([[[[1., 1., 1., 0.]]]])
        loss, disagreement = shore_loss(prediction, target, mask)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.)
        self.assertEqual(float(disagreement), 1.)
        self.assertGreater(float(prediction.grad[0, 0, 0, 0]), 0.)
        self.assertLess(float(prediction.grad[0, 0, 0, 1]), 0.)
        self.assertEqual(float(prediction.grad[0, 0, 0, 2]), 0.)
        self.assertEqual(float(prediction.grad[0, 0, 0, 3]), 0.)

    def test_class_imbalance_and_empty_windows_do_not_dilute_coast_loss(self):
        target = torch.tensor([[[[-2., 2.]]]])
        loss, _ = shore_loss(-target, target, torch.ones_like(target))
        duplicated = torch.tensor([[[[-2., 2., 2., 2., 2., 2.]]]])
        repeat_loss, _ = shore_loss(-duplicated, duplicated, torch.ones_like(duplicated))
        torch.testing.assert_close(loss, repeat_loss)
        batch = torch.cat((target, torch.full_like(target, 1000.)))
        mixed, _ = shore_loss(-batch, batch, torch.ones_like(batch))
        torch.testing.assert_close(loss, mixed)
        empty, _ = shore_loss(-target, target, torch.zeros_like(target))
        self.assertEqual(float(empty), 0.)


if __name__ == '__main__':
    unittest.main()
