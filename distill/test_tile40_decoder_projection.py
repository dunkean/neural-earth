"""Reduced skip paths retain both upsampled and encoder concat halves."""
import unittest

import torch

from distill.student import Student, StudentConfig
from distill.tile40_decoder_projection import project_decoder


class ProjectionTests(unittest.TestCase):
    def test_linear_skip_matches_source_with_retained_coordinates(self):
        source = Student(StudentConfig(stage='decoder', width=32, depth=3, dilations=(1, 2)))
        target = project_decoder(source, 16)
        for index, (old, new) in enumerate(zip(source.decoders, target.decoders, strict=True)):
            up = torch.randn(1, 32, 4, 4)
            skip_width = 32 if index == 0 else 16
            skip = torch.randn(1, skip_width, 4, 4)
            actual = new.skip(torch.cat([up, skip], dim=1))
            full = torch.zeros(1, old.skip.in_channels, 4, 4)
            full[:, :32] = up
            old_up_width = 64
            full[:, old_up_width:old_up_width+skip_width] = skip
            expected = old.skip(full)[:, :new.skip.out_channels]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_external_noise_and_latent_direct_path_remains_identical(self):
        source = Student(StudentConfig(stage='decoder', width=32, depth=3))
        with torch.no_grad():
            source.direct.weight.normal_()
        target = project_decoder(source, 16)
        inputs = torch.randn(1, 16, 8, 8)
        torch.testing.assert_close(source.direct(inputs), target.direct(inputs), atol=0., rtol=0.)

    def test_eight_channel_decoder_projection_supports_training_without_narrowing_base(self):
        source = Student(StudentConfig(stage='decoder', width=16, depth=3))
        with torch.no_grad():
            source.head.weight.normal_(std=.03)
        target = project_decoder(source, 8)
        result = target(torch.randn(1, 16, 32, 32))
        self.assertEqual(tuple(result.shape), (1, 1, 32, 32))
        result.square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in target.parameters()))
        with self.assertRaises(ValueError):
            Student(StudentConfig(stage='base', width=8))


if __name__ == '__main__':
    unittest.main()
