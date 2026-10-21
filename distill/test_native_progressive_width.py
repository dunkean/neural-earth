"""Regression controls for retained-coordinate normalization and frozen runtime."""
import unittest

import torch

from distill.native_compression import compressed_config, project_state
from distill.native_progressive_width import ProgressiveWidthBase, compact_state, progressive_normalize
from distill.native_progressive_runtime import runtime_progressive
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.models.mp_layers import normalize


class ProgressiveWidthTests(unittest.TestCase):
    def test_removed_embedding_ones_do_not_change_retained_normalization(self):
        indices = torch.tensor([0, 2])
        values = torch.tensor([[3., 1., 4., 1.]], dtype=torch.bfloat16)
        actual = progressive_normalize(values, indices, 1., embedding=True)
        reference = values[:, indices]/torch.sqrt((values[:, indices]**2).mean(1, keepdim=True)+1e-8)
        self.assertTrue(torch.equal(actual[:, indices], reference))
        self.assertEqual(float(actual[:, [1, 3]].abs().sum()), 0.)

    def test_encoder_normalization_uses_retained_channel_count(self):
        indices = torch.tensor([1, 3])
        values = torch.arange(1., 17.).reshape(1, 4, 2, 2).bfloat16()
        actual = progressive_normalize(values, indices, 1.)
        self.assertTrue(torch.equal(actual[:, indices], normalize(values[:, indices], dim=1)))
        self.assertEqual(float(actual[:, [0, 2]].abs().sum()), 0.)

    def test_source_identity_and_compact_runtime(self):
        old_threads = torch.get_num_threads();torch.set_num_threads(1)
        try:
            torch.manual_seed(143)
            config = dict(image_size=16, in_channels=5, out_channels=5, model_channels=32,
                          model_channel_mults=[1, 2], layers_per_block=3, emb_channels=64,
                          noise_emb_dims=32, block_kwargs=dict(channels_per_head=16),
                          conditional_inputs=[['tensor', 58, 1.]], midblock_attention=True)
            teacher = EDMUnet2D.from_config(config).bfloat16().eval().requires_grad_(False)
            teacher.out_gain.fill_(.125)
            target_config = compressed_config(dict(teacher.config), 16, 3)
            target = EDMUnet2D.from_config(target_config)
            _, mapping = project_state(teacher, target)
            model = ProgressiveWidthBase(config, target_config, mapping, .125).eval()
            model.net.load_state_dict(teacher.state_dict())
            x, labels, conditions = torch.randn(1, 5, 16, 16).bfloat16(), torch.tensor([.609375]).bfloat16(), torch.randn(1, 58).bfloat16()
            with torch.no_grad():
                reference = teacher(x, labels, [conditions]).float()
                self.assertTrue(torch.equal(reference, model(x, labels, [conditions])))
                model.amount = 1.
                state = compact_state(model)
                identity = {name:dict(source=plan['source'],
                    output=list(range(len(plan['output']))) if plan['output'] is not None else None,
                    inputs=list(range(len(plan['inputs']))) if plan['inputs'] is not None else None)
                    for name, plan in mapping.items()}
                saved = dict(architecture='native-progressive-width', config=target_config,
                             target_config=target_config, mapping=identity, gain_anchor=.125,
                             amount=0., ema=state, phase='compact160')
                compact = ProgressiveWidthBase(target_config, target_config, identity, .125).eval()
                compact.load_state_dict(state)
                runtime = runtime_progressive(saved, 'cpu')
                self.assertTrue(torch.equal(compact(x, labels, [conditions]), runtime(x, labels, [conditions]).float()))
                with self.assertRaises(ValueError):
                    runtime_progressive(saved|dict(amount=.5), 'cpu')
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
