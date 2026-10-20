import builtins
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from distill.coarse_solver import CoarseSolver


class CoarseCpuLoadingTests(unittest.TestCase):
    def test_cached_initialization_does_not_import_cuda_demo_or_change_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'coarse_model').mkdir()
            (root/'config.json').write_text(json.dumps({'coarse_stds':[2., 1.]}))
            (root/'coarse_model/config.json').write_text('{}')
            (root/'coarse_model/diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
            net = torch.nn.Linear(3, 3)
            expected = net.weight.detach().bfloat16().float()
            before = Path.cwd()
            original_import = builtins.__import__
            def guarded_import(name, *args, **kwargs):
                if name == 'terrain_app':
                    raise AssertionError('CPU loading must not import the CUDA-only server.')
                return original_import(name, *args, **kwargs)
            with patch('terrain_paths.model_snapshot', return_value=root), \
                 patch('terrain_diffusion.models.edm_unet.EDMUnet2D.from_pretrained', return_value=net), \
                 patch('builtins.__import__', side_effect=guarded_import):
                model = CoarseSolver(2)
            self.assertEqual(Path.cwd(), before)
            self.assertEqual(model.delta_ratio, .5)
            self.assertEqual(model.net.weight.dtype, torch.float32)
            torch.testing.assert_close(model.net.weight, expected, atol=0., rtol=0.)


if __name__ == '__main__':
    unittest.main()
