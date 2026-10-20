import unittest

import numpy as np
import torch

from distill.tile40_teacher_quant import pack3,unpack3,encode3,CENTERS


class TeacherQuantTests(unittest.TestCase):
    def test_three_bit_format_known_word_and_roundtrip(self):
        self.assertEqual(pack3(np.arange(8)),bytes.fromhex('88c6fa'))
        expected=np.random.default_rng(291).integers(0,8,1024,dtype=np.uint8)
        np.testing.assert_array_equal(unpack3(pack3(expected)),expected)
        self.assertEqual(len(pack3(expected)),384)

    def test_stored_codes_and_half_scales_reconstruct_the_probe_exactly(self):
        torch.manual_seed(88)
        weight=torch.randn(71,33)
        codes,scales,actual=encode3(weight)
        decoded=torch.tensor(CENTERS)[torch.from_numpy(unpack3(pack3(codes.numpy()))).long()]
        restored=(decoded.reshape(-1,512)*scales.half().float()[:,None]).reshape(-1)[:weight.numel()].reshape(weight.shape)
        torch.testing.assert_close(actual,restored,atol=0.,rtol=0.)
        self.assertLess(float((actual-weight).square().mean()/weight.square().mean()),.06)


if __name__=='__main__':unittest.main()
