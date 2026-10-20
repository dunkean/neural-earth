import unittest

import torch

from distill.student import Student, StudentConfig
from distill.tile40_export_bundle import compact_decoder_input


class CompactInputTests(unittest.TestCase):
    def test_removing_zero_channels_preserves_both_input_paths(self):
        torch.manual_seed(751)
        source = Student(StudentConfig(stage='decoder', width=16, depth=2)).eval()
        with torch.no_grad():
            source.head.weight.normal_(std=.03)
            source.direct.weight.normal_(std=.1)
        folded = compact_decoder_input(source)
        for scale in (.5, 4.):
            five = torch.randn(1, 5, 32, 32)*scale
            full = torch.cat((five, torch.zeros(1, 11, 32, 32)), dim=1)
            with torch.no_grad():
                torch.testing.assert_close(folded(five), source(full), atol=2e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
