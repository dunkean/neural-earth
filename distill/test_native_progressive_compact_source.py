"""Prevent an accidental BF16 restart of the compact recovery masters."""
import unittest

import torch

from distill.native_progressive_compact import select_source


class CompactSourceTests(unittest.TestCase):
    def test_raw_master_selection_preserves_sub_bf16_updates(self):
        raw = torch.tensor([1.+1e-4], dtype=torch.float32)
        average = torch.tensor([1.+2e-4], dtype=torch.float32)
        source = dict(model={'net.enc.0_conv.weight':raw},
                      ema={'net.enc.0_conv.weight':average})
        selected = select_source(source, 'raw', True)
        self.assertTrue(torch.equal(selected['ema']['net.enc.0_conv.weight'], raw))
        self.assertFalse(torch.equal(raw, raw.bfloat16().float()))
        self.assertTrue(torch.equal(source['ema']['net.enc.0_conv.weight'], average))

    def test_inference_snapshot_cannot_claim_master_precision(self):
        source = dict(ema={'net.enc.0_conv.weight':torch.ones(1, dtype=torch.bfloat16)})
        with self.assertRaises(ValueError):
            select_source(source, keep_master=True)

    def test_raw_request_requires_actual_raw_state(self):
        source = dict(ema={'net.enc.0_conv.weight':torch.ones(1)})
        with self.assertRaises(ValueError):
            select_source(source, 'raw', True)


if __name__ == '__main__':
    unittest.main()
