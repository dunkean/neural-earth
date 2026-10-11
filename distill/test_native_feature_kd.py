import unittest

import torch
from torch import nn

from distill.native_feature_kd import capture_levels, level_taps, representation_loss


class FeatureKDTests(unittest.TestCase):
    def test_last_block_matches_level_when_depth_changes(self):
        def net(depth):
            model = nn.Module()
            model.enc = nn.ModuleDict({f'512x512_block{i}': nn.Identity() for i in range(depth)})
            model.dec = nn.ModuleDict({f'512x512_block{i}': nn.Identity() for i in range(depth+1)})
            return model
        teacher, student = net(3), net(2)
        self.assertEqual(level_taps(teacher).keys(), level_taps(student).keys())
        self.assertIs(level_taps(teacher)['enc.512x512'], teacher.enc['512x512_block2'])
        self.assertIs(level_taps(student)['dec.512x512'], student.dec['512x512_block2'])

    def test_shrinking_features_is_penalized_and_has_restoring_gradient(self):
        target = torch.randn(2, 4, 8, 8)
        actual = (.5*target).requires_grad_()
        loss, _ = representation_loss({'level': actual}, {'level': target})
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), .25, places=5)
        self.assertLess(float((actual.grad*target).sum()), 0.)

    def test_teacher_is_detached_and_exact_features_have_zero_loss(self):
        target = torch.ones(2, 4, 8, 8, requires_grad=True)
        actual = target.detach().clone().requires_grad_()
        loss, _ = representation_loss({'level': actual}, {'level': target})
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertIsNone(target.grad)

    def test_mismatched_channels_and_missing_levels_refused(self):
        with self.assertRaises(ValueError):
            representation_loss({'a': torch.zeros(1, 2, 8, 8)}, {'a': torch.zeros(1, 3, 8, 8)})
        with self.assertRaises(ValueError):
            representation_loss({}, {})

    def test_hooks_removed_on_exception(self):
        net = nn.Module()
        net.enc = nn.ModuleDict({'512x512_block0': nn.Identity()})
        net.dec = nn.ModuleDict()
        with self.assertRaises(RuntimeError):
            with capture_levels(net) as captured:
                net.enc['512x512_block0'](torch.ones(1))
                self.assertIn('enc.512x512', captured)
                raise RuntimeError('test')
        self.assertEqual(len(net.enc['512x512_block0']._forward_hooks), 0)

    def test_real_native_functional_forward_captures_all_levels_with_gradients(self):
        from terrain_diffusion.models.edm_unet import EDMUnet2D
        from distill.native_base import NativeBase
        config = dict(image_size=16, in_channels=5, out_channels=5, model_channels=8,
                      model_channel_mults=[1, 2, 3, 4], layers_per_block=3,
                      midblock_attention=False, conditional_inputs=[['tensor', 58, 1.]])
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            teacher = EDMUnet2D(**config).eval().requires_grad_(False)
            student = NativeBase(config | {'layers_per_block': 2}).eval()
            inputs, labels, condition = torch.randn(2, 5, 16, 16), torch.tensor([1.56, .61]), torch.randn(2, 58)
            with capture_levels(teacher) as reference:
                teacher(inputs, labels, [condition])
            with capture_levels(student.net) as actual:
                student(inputs, labels, [condition])
            self.assertEqual(len(actual), 8)
            self.assertTrue(all(value.requires_grad for value in actual.values()))
            loss, _ = representation_loss(actual, reference)
            loss.backward()
            for side in ('enc', 'dec'):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in getattr(student.net, side).parameters()))
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
