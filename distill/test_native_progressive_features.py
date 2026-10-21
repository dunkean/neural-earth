"""Check the teacher-coordinate contract needed after physical compaction."""
import unittest

import torch

from distill.native_progressive_features import compact_feature_loss


class CompactFeatureTests(unittest.TestCase):
    def setUp(self):
        self.name = 'enc.0_block0'
        self.coordinates = [4, 0, 2]
        self.mapping = {self.name+'.conv_res1.weight': {'output': self.coordinates}}
        self.teacher = torch.arange(1., 21.).reshape(1, 5, 2, 2)

    def test_original_coordinates_apply_only_to_teacher(self):
        # Coordinate4 would be out of bounds in the compact three-channel net.
        student = self.teacher[:, self.coordinates].clone()
        loss = compact_feature_loss({self.name: student}, {self.name: self.teacher}, self.mapping)
        self.assertEqual(float(loss), 0.)

    def test_gradient_trains_student_and_keeps_teacher_frozen(self):
        teacher = self.teacher.clone().requires_grad_(True)
        student = (self.teacher[:, self.coordinates]+1).clone().requires_grad_(True)
        compact_feature_loss({self.name: student}, {self.name: teacher}, self.mapping).backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertTrue((student.grad > 0).all())

    def test_corrupt_width_and_duplicate_coordinates_are_rejected(self):
        student = self.teacher[:, self.coordinates]
        for coordinates in ([4, 0], [4, 0, 0]):
            mapping = {self.name+'.conv_res1.weight': {'output': coordinates}}
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                compact_feature_loss({self.name: student}, {self.name: self.teacher}, mapping)


if __name__ == '__main__':
    unittest.main()
