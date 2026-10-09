"""GPU contracts for isolated coarse solves, dynamic inputs and ordered joins."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(_REPO_ROOT / 'terrain-diffusion'))
import torch
from terrain_coarse_graph import CoarseSolverAdapter
from terrain_coarse_streams import CoarseSolverInputs, CoarseStreamPool
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler


class Network(torch.nn.Module):
    config = {}

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(.125))

    def forward(self, x, noise_labels, conditional_inputs, precomputed_embeds=None):
        result = x[:, :6]*self.gain + x[:, 6:7]*.25
        result = result + noise_labels.reshape(-1,1,1,1)*.01
        for value in conditional_inputs:
            result = result + value.reshape(-1,1,1,1)*.003
        if precomputed_embeds is not None:
            result = result + precomputed_embeds[:, :1, None, None]*.002
        return result


class CoarseStreamsTests(unittest.TestCase):
    def test_cpu_and_invalid_slot_counts_refused(self):
        for count in (0,3,5):
            with self.assertRaisesRegex(ValueError, '1, 2, 4, 8 or 16'):
                CoarseStreamPool(Network(), EDMDPMSolverMultistepScheduler(), 'cpu', streams=count)
        with self.assertRaisesRegex(ValueError, 'requires CUDA'):
            CoarseStreamPool(Network(), EDMDPMSolverMultistepScheduler(), 'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    @torch.inference_mode()
    def test_independent_solvers_dynamic_conditions_order_and_output_lifetime(self):
        # Ordinary parameters need version counters for graph invalidation.
        with torch.inference_mode(False):
            model = Network().to('cuda')
        scheduler = EDMDPMSolverMultistepScheduler()
        scheduler.set_timesteps(20)
        inputs = []
        reference = CoarseSolverAdapter(model, scheduler, 'cuda')
        expected = []
        for offset in (0., 2., -1., 5.):
            item = CoarseSolverInputs(torch.full((1,6,2,2), offset, device='cuda'),
                 torch.stack([scheduler.trigflow_precondition_noise(s.to('cuda').view(1)) for s in scheduler.sigmas[:-1]]),
                 (torch.full((1,5,2,2), offset+.2, device='cuda'), torch.tensor([offset], device='cuda')),
                 torch.full((20,1,3), offset-.1, device='cuda'))
            inputs.append(item)
            expected.append(reference(item.sample, item.labels, list(item.conditions), precomputed_embeds=item.embeds))
        with CoarseStreamPool(model, scheduler, 'cuda', streams=2) as pool:
            with self.assertRaisesRegex(RuntimeError, 'Warm'):
                pool.run(inputs[:2])
            pool.warm(inputs[0], expected[0])
            self.assertIsNot(pool.slots[0][1].model.scheduler, pool.slots[1][1].model.scheduler)
            self.assertIs(pool.slots[0][1].model.network, pool.slots[1][1].model.network)
            first, _ = pool.run(inputs[:2])
            second, _ = pool.run([inputs[3], inputs[2]])
            # Results can be consumed on the caller stream without a device-wide
            # synchronization; old output clones survive subsequent slot reuse.
            combined = torch.cat(first+second)
            self.assertTrue(torch.equal(combined, torch.cat([*expected[:2], expected[3], expected[2]])))
            for _, graph in pool.slots:
                self.assertEqual(graph.capture_calls, 1)
                self.assertEqual(graph.fallback_calls, 0)
            replays = [graph.replay_calls for _, graph in pool.slots]
            bad = CoarseSolverInputs(inputs[0].sample.expand(2,-1,-1,-1), inputs[0].labels,
                                     inputs[0].conditions, inputs[0].embeds)
            with self.assertRaisesRegex(ValueError, 'batch size 1'):
                pool.run([inputs[0], bad])
            self.assertEqual(replays, [graph.replay_calls for _, graph in pool.slots])
            model.gain.add_(.1)
            with self.assertRaisesRegex(RuntimeError, 'weights changed'):
                pool.run(inputs[:1])
        with self.assertRaisesRegex(RuntimeError, 'Warm'):
            pool.run(inputs[:1])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    @torch.inference_mode()
    def test_capture_memory_refusal_is_not_silent_fallback(self):
        with torch.inference_mode(False):
            model = Network().to('cuda')
        scheduler = EDMDPMSolverMultistepScheduler()
        scheduler.set_timesteps(20)
        inputs = CoarseSolverInputs(torch.ones(1,6,2,2,device='cuda'),
                    torch.stack([s.to('cuda').view(1) for s in scheduler.sigmas[:-1]]),
                    (torch.zeros(1,5,2,2,device='cuda'),), None)
        expected = CoarseSolverAdapter(model, scheduler, 'cuda')(
                    inputs.sample, inputs.labels, list(inputs.conditions))
        with CoarseStreamPool(model, scheduler, 'cuda', max_bytes=1) as pool:
            with self.assertRaisesRegex(RuntimeError, 'capture refused'):
                pool.warm(inputs, expected)
            self.assertFalse(pool.stats()['ready'])


if __name__ == '__main__':
    unittest.main()
