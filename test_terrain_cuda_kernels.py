"""CUDA admission, exact-rounding and replay gates for native pointwise kernels."""
import os
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

import torch

import terrain_cuda_kernels as kernels


class AdmissionTests(unittest.TestCase):
    def test_cpu_training_and_shapes_are_not_admitted(self):
        x = torch.ones(2, 3, 4, 4, dtype=torch.bfloat16)
        self.assertIsNone(kernels.binary_sum(x, x, torch.ones(2), torch.ones(())))
        self.assertIsNone(kernels.silu_scaled(x))
        self.assertIsNone(kernels.binary_concat(x, torch.ones(1), [torch.ones(()), torch.ones(())], 1))

    def test_prepare_capture_guard_selects_requested_device(self):
        with patch.object(kernels, '_packs', {}), patch('torch.cuda.device') as device, \
                patch('torch.cuda.is_current_stream_capturing', return_value=True):
            with self.assertRaises(kernels.KernelUnavailable): kernels.prepare('cuda:1')
            device.assert_called_once_with(1)


@unittest.skipUnless(os.environ.get('TERRAIN_TEST_CUDA') == '1' and torch.cuda.is_available(), 'Exclusive CUDA gate')
class KernelTests(unittest.TestCase):
    @torch.inference_mode()
    def test_sum_rounding_signed_zero_and_contiguous_admission(self):
        for batch in (1, 9, 16):
            a = torch.randn((batch, 192, 16, 16), device='cuda', dtype=torch.bfloat16)
            b = torch.randn_like(a)
            a.flatten()[:4] = torch.tensor([0., -0., 0., -0.], device='cuda', dtype=a.dtype)
            b.flatten()[:4] = torch.tensor([0., 0., -0., -0.], device='cuda', dtype=a.dtype)
            for values in ([.5, .5], [.1, .9], [1., -1.]):
                w = torch.tensor(values, device='cuda', dtype=a.dtype)
                norm = torch.linalg.vector_norm(w)
                expected = ((a * w[0] + b * w[1]) + 0.) / norm
                actual = kernels.binary_sum(a, b, w, norm)
                self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
            self.assertIsNone(kernels.binary_sum(a[..., ::2], b[..., ::2], w, norm))
        self.assertIsNone(kernels.binary_sum(a.float(), b.float(), w.float(), norm.float()))

    @torch.inference_mode()
    def test_concat_dynamic_inputs_capture_and_private_output(self):
        kernels.prepare('cuda')
        for batch in (1, 9, 16):
            a = torch.randn((batch, 192, 16, 16), device='cuda', dtype=torch.bfloat16)
            b = torch.randn((batch, 384, 16, 16), device='cuda', dtype=torch.bfloat16)
            factors = [torch.tensor(v, device='cuda', dtype=a.dtype) for v in (.375, .75)]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = kernels.binary_concat(a, b, factors, 1)
            for _ in range(2):
                a.normal_(); b.normal_()
                expected = torch.cat([a * factors[0], b * factors[1]], dim=1)
                graph.replay()
                self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
            self.assertIsNone(kernels.binary_concat(a, b, factors, 2))

    @torch.inference_mode()
    def test_silu_exhaustive_bf16_candidates(self):
        # Exhaustive finite BF16 inputs gate decides whether this can be exact.
        bits = torch.arange(65536, dtype=torch.int32, device='cuda').to(torch.int16)
        x = bits.view(torch.bfloat16)
        x = x[torch.isfinite(x)].contiguous()
        expected = torch.nn.functional.silu(x) / .596
        actual = kernels.silu_scaled(x)
        mismatch = torch.count_nonzero(actual.view(torch.int16) != expected.view(torch.int16)).item()
        self.assertEqual(mismatch, 0, f'MP-SiLU is not exact for {mismatch} BF16 inputs')

    @torch.inference_mode()
    def test_sum_and_silu_replay_on_worker_stream(self):
        kernels.prepare('cuda')
        def worker():
            with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
                x = torch.randn((9, 192, 16, 16), device='cuda', dtype=torch.bfloat16)
                y = torch.randn_like(x)
                weights = torch.tensor([.3, .7], device='cuda', dtype=x.dtype)
                norm = torch.linalg.vector_norm(weights)
                for forward, reference in (
                    (lambda: kernels.binary_sum(x, y, weights, norm),
                     lambda: ((x * weights[0] + y * weights[1]) + 0.) / norm),
                    (lambda: kernels.silu_scaled(x), lambda: torch.nn.functional.silu(x) / .596)):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): actual = forward()
                    for _ in range(2):
                        x.normal_(); y.normal_(); expected = reference(); graph.replay()
                        self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(worker).result(timeout=30)


if __name__ == '__main__':
    unittest.main()
