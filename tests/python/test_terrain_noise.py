"""Cached noise tiles return the upstream field's exact bytes, and selected decoders reach refinement."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(_REPO_ROOT / 'terrain-diffusion'))
from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch
import terrain_noise
from terrain_noise import noise_patch, upload


class NoiseCacheTests(unittest.TestCase):
    def setUp(self):
        terrain_noise.clear()

    def test_cached_patches_equal_upstream_bytes_across_tiles_and_signs(self):
        cases = [(5819, 384*7, 384*11, 512, 512, 1, 512), (5819, -384*3, 384*2, 512, 512, 1, 512),
                 (42, -32*5, -32*9, 64, 64, 5, 64), (43, 48*3, -48*7, 64, 64, 6, 64),
                 (2**63+17, 1000, -999, 37, 91, 2, 64)]
        for seed, y, x, h, w, channels, tile in cases:
            for _ in range(2):  # miss, then hit
                expected = gaussian_noise_patch(seed, y, x, h, w, channels=channels, tile_h=tile, tile_w=tile)
                actual = noise_patch(seed, y, x, h, w, channels=channels, tile_h=tile, tile_w=tile)
                self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertGreater(terrain_noise.stats['hits'], 0)

    def test_cached_tiles_cannot_be_mutated_by_callers(self):
        out = noise_patch(1, 0, 0, 64, 64, channels=1, tile_h=64, tile_w=64)
        out[:] = 0
        again = noise_patch(1, 0, 0, 64, 64, channels=1, tile_h=64, tile_w=64)
        self.assertEqual(again.tobytes(), gaussian_noise_patch(1, 0, 0, 64, 64, channels=1, tile_h=64, tile_w=64).tobytes())
        self.assertFalse(terrain_noise._tiles[(1, 0, 0, 1, 64, 64)].flags.writeable)

    def test_budget_bounds_resident_tiles(self):
        with patch.object(terrain_noise, '_budget', 3*64*64*4):
            for tx in range(8):
                noise_patch(7, 0, tx*64, 64, 64, channels=1, tile_h=64, tile_w=64)
            self.assertLessEqual(terrain_noise._bytes, 3*64*64*4)
            self.assertEqual(len(terrain_noise._tiles), 3)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_nonblocking_upload_keeps_bf16_rounding(self):
        value = noise_patch(9, 13, -77, 512, 512, channels=1, tile_h=512, tile_w=512)
        for dtype in (torch.bfloat16, torch.float32):
            expected = torch.from_numpy(value).to('cuda', dtype=dtype)
            self.assertTrue(torch.equal(upload(value, 'cuda', dtype), expected))


class DecoderDispatchTests(unittest.TestCase):
    def test_selected_student_receives_every_decoder_window_with_refinement_seed(self):
        from terrain_inference import _decoder_batch
        calls = []
        world = SimpleNamespace(_terrain_decoder_windows=lambda *args: calls.append(args) or ['student'])
        weight = torch.ones(512, 512)
        result = _decoder_batch(world, [(0, 1, 2)], ['latent'], None, weight, [], 512, 384, noise_seed=99)
        self.assertEqual(result, ['student'])
        self.assertEqual(calls, [([(0, 1, 2)], ['latent'], weight, 512, 384, 99)])

    def test_refinement_octave_uses_the_world_decoder_dispatch(self):
        import terrain_refinement
        world = SimpleNamespace(decoder_tile_size=512, decoder_tile_stride=384, latent_compression=8,
                                device='cpu', seed=5, tile_store=None,
                                _terrain_profile=SimpleNamespace(gpu_windows=False))
        with patch('terrain_inference._decoder_batch', return_value=['window']) as batch:
            tensor = terrain_refinement._detail_field(world, 1)
            world.latents = torch.ones(6, 4096, 4096)
            self.assertEqual(tensor._f([(0, 0, 0)]), ['window'])
        self.assertEqual(batch.call_args.kwargs['noise_seed'], 5 + 104729)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class StudentGraphTests(unittest.TestCase):
    def test_graph_slots_equal_eager_for_every_window_and_slot(self):
        from distill.inference import autocast, prepare_students, run_graphs
        from distill.student import Student, StudentConfig
        torch.manual_seed(0)
        model = Student(StudentConfig('decoder', 16, 2, (1,))).cuda().eval().requires_grad_(False)
        torch.nn.init.normal_(model.head.weight, std=.1)
        prepare_students(dict(decoder=model), 'cuda')
        inputs = [torch.randn(1, 16, 64, 64, device='cuda').contiguous(memory_format=torch.channels_last)
                  for _ in range(5)]
        with torch.inference_mode():
            with autocast('decoder'):
                expected = [model(value) for value in inputs]
            for slots in (1, 2, 4):
                actual = run_graphs(model, 'decoder', inputs, slots)
                for a, e in zip(actual, expected):
                    self.assertTrue(torch.equal(a, e))
        self.assertEqual(len(model._distill_graphs.slots), 4)

    def test_non_finite_output_is_refused_without_waiting_per_window(self):
        from distill.inference import FiniteGuard
        guard = FiniteGuard(limit=2)
        guard.add('decoder', [torch.ones(4, device='cuda')])
        with self.assertRaises(FloatingPointError):
            for _ in range(4):
                guard.add('decoder', [torch.full((4,), float('nan'), device='cuda')])
                torch.cuda.synchronize()


if __name__ == '__main__':
    unittest.main()
