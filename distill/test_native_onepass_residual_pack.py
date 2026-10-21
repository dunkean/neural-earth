"""The compact residual artifact preserves all five quantized window outputs."""
from pathlib import Path
import tempfile
import unittest

import torch

from distill.native_onepass import noise_windows
from distill.native_onepass_residual_pack import (
    ResidualQuantOnePass, ResidualPackedRuntime, export_residual_packed,
)
from distill.test_native_onepass import tiny_saved


class PackedResidualTests(unittest.TestCase):
    def saved(self):
        torch.manual_seed(12)
        parent = tiny_saved()
        model = ResidualQuantOnePass(parent['config'], parent['gain_anchor']).eval()
        model.load_state_dict(parent['ema'] | {'residual.weight': torch.randn(5, 10, 1, 1)*.1})
        saved = parent | dict(architecture='native-two-noise-onepass-residual-nf4', step=4,
            ema=model.state_dict(), teacher_revision='test', teacher_weights_sha256='test')
        return saved, model

    def test_export_round_trip_preserves_quantized_fields_and_counts_head(self):
        saved, model = self.saved()
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            output = Path(directory)/'packed'
            manifest = export_residual_packed(saved, output, device='cpu')
            runtime = ResidualPackedRuntime(output)
            noises = noise_windows(615123, [(0, 0), (0, 32)])
            condition = torch.zeros(2, 58)
            self.assertTrue(torch.equal(model.windows(noises, condition), runtime(noises, condition)))
            self.assertEqual(manifest['onepass_wrapper']['download_weight_bytes'],
                             (output/'weights.bin').stat().st_size+(output/'residual.bin').stat().st_size)
            self.assertEqual((output/'residual.bin').stat().st_size, 200)
            self.assertEqual(manifest['quantization_device'], 'cpu')

    def test_loader_rejects_corrupted_residual_payload(self):
        saved, _ = self.saved()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'packed'
            export_residual_packed(saved, output)
            with (output/'residual.bin').open('r+b') as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes([first[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'Residual payload'):
                ResidualPackedRuntime(output)


if __name__ == '__main__':
    unittest.main()
