import copy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from distill.native_scalar_pack import replace_gain, read_tensor


class ScalarPackTests(unittest.TestCase):
    def test_gain_expansion_preserves_unrelated_payload_and_scale_offsets(self):
        original = b'PREF\x00\x00CODE' + b'SCALES!!' + b'LAST'
        manifest = dict(bytes=len(original), tensors={
            'net.first': dict(offset=0, bytes=4, dtype='bfloat16', shape=[2]),
            'net.out_gain': dict(offset=4, bytes=2, dtype='bfloat16', shape=[]),
            'net.hidden': dict(offset=6, bytes=4, dtype='nf4', shape=[8],
                               scale_offset=10, scale_count=2)})
        unchanged = copy.deepcopy(manifest)
        gain = torch.tensor(-.19427490234375)
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory)/'old.bin', Path(directory)/'new.bin'
            old.write_bytes(original)
            converted = replace_gain(old, manifest, new, gain)
            actual = new.read_bytes()
            self.assertEqual(actual[:4], original[:4])
            self.assertEqual(actual[8:], original[6:])
            self.assertEqual(actual[4:8], gain.numpy().astype('<f4').tobytes())
            self.assertEqual(converted['tensors']['net.hidden']['offset'], 8)
            self.assertEqual(converted['tensors']['net.hidden']['scale_offset'], 12)
            self.assertEqual(converted['bytes'], len(original)+2)
            self.assertEqual(old.read_bytes(), original)
            self.assertEqual(manifest, unchanged)

    def test_read_preserves_gain_that_bf16_would_round_away(self):
        gain = np.array(-.19427490234375, dtype='<f4')
        row = dict(offset=0, bytes=4, dtype='float32', shape=[])
        value = read_tensor(io.BytesIO(gain.tobytes()), 'net.out_gain', row, {})
        self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(value.shape, torch.Size([]))
        self.assertEqual(float(value), float(gain))
        self.assertNotEqual(float(value.bfloat16()), float(value))

    def test_rejects_truncation_or_fp32_regular_tensor(self):
        row = dict(offset=0, bytes=4, dtype='float32', shape=[])
        with self.assertRaises(ValueError):
            read_tensor(io.BytesIO(b'\x00\x00'), 'net.out_gain', row, {})
        with self.assertRaises(ValueError):
            read_tensor(io.BytesIO(b'\x00'*4), 'net.some_kernel', row, {})


if __name__ == '__main__':
    unittest.main()
