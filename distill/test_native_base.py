"""Teacher transfer and native solver contracts, without GPU or model downloads."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from distill.coarse_solver import CoarseSolver
from distill.native_base import NativeBase, NativePairs, pruned_config, transfer_state
from distill.export import export_stage
from distill.student import Student, StudentConfig, load_student


class NativeTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_every_pruned_tensor_inherits_the_matching_teacher_skip(self):
        config = dict(image_size=64, in_channels=5, out_channels=5, model_channels=16,
                      layers_per_block=3, model_channel_mults=[1, 2, 3, 4],
                      conditional_inputs=[['tensor', 58, 1.]], concat_balance=.5)
        teacher = NativeBase(config)
        student = NativeBase(pruned_config(config))
        state, mapping = transfer_state(teacher.net.state_dict(), student.net.state_dict(), student.config)
        student.net.load_state_dict(state, strict=True)
        self.assertEqual(set(mapping), set(student.net.state_dict()))
        self.assertEqual(mapping['enc.32x32_block1.emb_gain'], 'enc.32x32_block2.emb_gain')
        self.assertEqual(mapping['dec.32x32_block1.emb_gain'], 'dec.32x32_block2.emb_gain')
        self.assertEqual(mapping['dec.32x32_block2.emb_gain'], 'dec.32x32_block3.emb_gain')
        for name, parent in mapping.items():
            torch.testing.assert_close(state[name], teacher.net.state_dict()[parent], rtol=0, atol=0)
        inputs = torch.randn(1, 5, 64, 64)
        output = student(inputs, torch.ones(1), [torch.randn(1, 58)])
        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in student.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None))

    def test_short_solvers_evaluate_exactly_the_requested_number_of_times(self):
        class FakeNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.tensor(.1))
                self.calls = 0

            def forward(self, inputs, noise_labels, conditional_inputs):
                self.calls += 1
                return inputs[:, :6]*self.gain

        for count in (1, 2):
            solver = CoarseSolver.__new__(CoarseSolver)
            torch.nn.Module.__init__(solver)
            solver.steps, solver.delta_ratio = count, .05
            solver.net = FakeNetwork()
            output = solver(torch.randn(1, 16, 64, 64))
            self.assertEqual(solver.net.calls, count)
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(solver.net.gain.grad))

    def test_refinement_and_initialization_have_equal_sampling_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = np.array([1.5]*8+[.6]*2, np.float16)
            np.savez(Path(directory)/'train-0000.npz', inputs=np.zeros((10,5,64,64),np.float16),
                     labels=labels, condition=np.zeros((10,58),np.float16),
                     target=np.zeros((10,5,64,64),np.float16))
            pairs = NativePairs(directory, 'train')
            probabilities = pairs.stage_probabilities()
            self.assertAlmostEqual(float(probabilities[:8].sum()), .5)
            self.assertAlmostEqual(float(probabilities[8:].sum()), .5)
            self.assertEqual(pairs[(0,123)][0].shape, (5,64,64))

    def test_single_stage_export_never_requires_or_emits_a_base(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as directory:
            model = Student(StudentConfig(stage='decoder', width=16, depth=2))
            path = Path(directory)/'source.pt'
            state = model.state_dict()
            torch.save(dict(config=asdict(model.config), model=state, ema=state, step=5), path)
            output = Path(directory)/'exports'
            manifest = export_stage(path, output)
            self.assertEqual(manifest['stage'], 'decoder')
            self.assertFalse((output/'base.pt').exists())
            loaded, _ = load_student(output/'decoder.pt')
            for name, value in loaded.state_dict().items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0)
            self.assertEqual(export_stage(path, output), manifest)
            (output/'decoder.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'immutable'):
                export_stage(path, output)


if __name__ == '__main__':
    unittest.main()
