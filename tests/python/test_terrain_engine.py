"""Backend admission, model sharing and attention layout/scale regression gates."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(_REPO_ROOT / 'terrain-diffusion'))
import terrain_nn_constants as constants
from terrain_diffusion.models.unet_block import UNetBlock


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.runtime = patch.object(constants, '_runtime_config', None)
        self.errors = patch.object(constants, '_kernel_errors', {})
        self.runtime.start()
        self.errors.start()
        self.world = SimpleNamespace(device='cpu', coarse_model=UNetBlock(64,64,0,attention=True),
            base_model=UNetBlock(64,64,0,attention=True), decoder_model=UNetBlock(64,64,0,attention=True))

    def tearDown(self):
        constants.restore_constants(self.world)
        self.errors.stop()
        self.runtime.stop()

    def test_shared_backend_cannot_switch_even_after_restore(self):
        constants.prepare_constants(self.world, exact_kernels=True)
        constants.restore_constants(self.world)
        with self.assertRaises(ValueError):
            constants.validate_engine_config(exact_kernels=False)
        with self.assertRaises(ValueError):
            constants.validate_engine_config(exact_kernels=True, attention_backend='sdpa-base')

    def test_preparation_failure_records_reason_and_keeps_reference(self):
        self.world.device='cuda:0'
        with patch('terrain_cuda_kernels.prepare', side_effect=OSError('missing NVRTC')):
            constants.prepare_constants(self.world, exact_kernels=True)
        self.assertIn('missing NVRTC', constants.kernel_status()['preparation_errors'][0])
        with torch.inference_mode():
            x=torch.randn(1,64,8,8)
            self.assertTrue(torch.equal(constants._silu(x), constants._original_silu(x)))
        self.assertIs(self.world.base_model.activation, constants._silu)
        constants.restore_constants(self.world)
        self.assertIs(self.world.base_model.activation, constants._original_silu)

    def test_invalid_profile_does_not_commit_or_mutate_and_off_path_checks_sharing(self):
        from terrain_inference import configure_world, InferenceProfile
        profile=InferenceProfile('cpu-test',cuda_graphs=False,exact_kernels=False)
        world=SimpleNamespace(torch_compile=True)
        with self.assertRaises(ValueError):
            configure_world(world,profile)
        self.assertIsNone(constants._runtime_config)
        self.assertFalse(hasattr(world,'_terrain_generation_settings'))
        constants.validate_engine_config(exact_kernels=True)
        world=SimpleNamespace(torch_compile=False,_terrain_profile=profile,_terrain_world_profile='natural')
        with self.assertRaises(ValueError):
            configure_world(world,profile)
        self.assertFalse(hasattr(world,'_terrain_generation_settings'))

    def test_sdpa_uses_tokens_and_single_scale_base_only(self):
        constants.prepare_constants(self.world, attention_backend='sdpa-base')
        self.assertEqual(self.world.coarse_model._terrain_attention_backend, 'reference')
        self.assertEqual(self.world.decoder_model._terrain_attention_backend, 'reference')
        block=self.world.base_model.eval()
        x=torch.randn(2,64,4,5)
        with torch.inference_mode():
            y=block.attn_qkv(x).reshape(2,1,64,3,20)
            q,k,v=constants.layers.normalize(y,dim=2).unbind(3)
            expected_k=(k/torch.sqrt(torch.tensor(64.,dtype=k.dtype))).transpose(-2,-1).contiguous()
            original=torch.nn.functional.scaled_dot_product_attention
            with patch('torch.nn.attention.sdpa_kernel', return_value=contextlib.nullcontext()), \
                    patch('torch.nn.functional.scaled_dot_product_attention', wraps=original) as call:
                result=block.attn(x)
            self.assertEqual(result.shape,x.shape)
            args,kwargs=call.call_args
            self.assertEqual(args[0].shape,(2,1,20,64))
            self.assertTrue(torch.equal(args[1],expected_k))
            self.assertEqual(kwargs,dict(dropout_p=0.,scale=1.))


if __name__=='__main__':
    unittest.main()
