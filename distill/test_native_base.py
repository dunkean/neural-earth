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
from distill.native_compression import compressed_config, project_state, select_channels
from distill.native_bank import SceneCapture, scene_proposals


class NativeTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def compression_teacher(self):
        config = dict(image_size=64, in_channels=5, out_channels=5, model_channels=16,
                      layers_per_block=3, model_channel_mults=[1, 2, 3, 4],
                      conditional_inputs=[['tensor', 58, 1.]], concat_balance=.5,
                      block_kwargs=dict(channels_per_head=4))
        return NativeBase(config)

    def test_uncompressed_projection_reconstructs_teacher_with_nonzero_output(self):
        teacher = self.compression_teacher()
        teacher.net.out_gain.data.fill_(1.)
        student = NativeBase(compressed_config(teacher.config, 16, 3))
        state, mapping = project_state(teacher.net, student.net)
        student.net.load_state_dict(state, strict=True)
        self.assertEqual(set(mapping), set(student.net.state_dict()))
        inputs, labels, conditions = torch.randn(1, 5, 64, 64), torch.ones(1), [torch.randn(1, 58)]
        expected = teacher(inputs, labels, conditions)
        self.assertGreater(float(expected.abs().max().detach()), .01)
        torch.testing.assert_close(student(inputs, labels, conditions), expected, rtol=0, atol=0)

    def test_compression_preserves_interleaved_qkv_and_both_skip_groups(self):
        teacher = self.compression_teacher()
        teacher.net.out_gain.data.fill_(1.)
        for blocks in (1, 2, 3):
            student = NativeBase(compressed_config(teacher.config, 8, blocks))
            state, mapping = project_state(teacher.net, student.net)
            student.net.load_state_dict(state, strict=True)
            key = 'dec.8x8_in0.attn_qkv.weight'
            selected = select_channels(64, 32, 4)
            qkv = (selected[:, None]*3+torch.arange(3)).flatten()
            expected = teacher.net.state_dict()[key].index_select(0, qkv).index_select(1, selected)
            torch.testing.assert_close(state[key], expected, rtol=0, atol=0)
            self.assertEqual(mapping[key]['output'], qkv.tolist())
            key = 'dec.8x8_block0.conv_res0.weight'
            self.assertEqual(mapping[key]['inputs'], selected.tolist()+(selected+64).tolist())
            # Embedding coordinates and their learned Fourier basis remain intact.
            torch.testing.assert_close(state['noise_linear.weight'], teacher.net.state_dict()['noise_linear.weight'], rtol=0, atol=0)
            output = student(torch.randn(1, 5, 64, 64), torch.ones(1), [torch.randn(1, 58)])
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None))
            self.assertGreater(float(student.net.out_conv.weight.grad.abs().sum()), 0.)

    def test_calibrated_selection_keeps_complete_attention_heads(self):
        scores = torch.arange(16).float()
        self.assertEqual(select_channels(16, 8, 4, scores).tolist(), list(range(8, 16)))
        with self.assertRaises(ValueError):
            select_channels(16, 8, 4, torch.full((16,), float('nan')))
        with self.assertRaises(ValueError):
            config = self.compression_teacher().config | dict(block_kwargs=dict(channels_per_head=8))
            compressed_config(config, 7, 2)

    def test_scene_selection_keeps_distinct_land_coast_and_sea_proposals(self):
        rows = [dict(index=i, land_fraction=value) for i, value in enumerate((1., .48, 0., .8))]
        self.assertEqual([row['index'] for row in scene_proposals(rows)], [0, 1, 2])

    def test_scene_capture_retains_coordinates_small_values_and_blended_targets(self):
        class Teacher(torch.nn.Module):
            def forward(self, inputs, **kwargs):
                return inputs*2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scene.npz'
            recorder = SceneCapture(Teacher(), path, dict(seed=10000, split='train'))
            for label, context in ((1.5, (0, -2, 3)), (.6, (0, -1, 2))):
                recorder.contexts = [context]
                recorder(torch.full((1, 5, 64, 64), 1e-9), torch.tensor([label]), [torch.zeros(1,58)])
            field = torch.ones(6,64,64)
            field[-1].fill_(.5)
            recorder.save_scene(field)
            with np.load(path, allow_pickle=False) as stored:
                self.assertEqual(stored['inputs'].dtype, np.float32)
                self.assertGreater(float(stored['inputs'].min()), 0.)
                np.testing.assert_array_equal(stored['contexts'], [[0,-2,3],[0,-1,2]])
                np.testing.assert_array_equal(stored['final_latents'], np.full((5,64,64), 2.))

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
