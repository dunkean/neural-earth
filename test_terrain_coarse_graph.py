"""Whole-solver adapter contracts using the real scheduler, with no GPU."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parent / 'terrain-diffusion'))
from terrain_coarse_graph import CoarseSolverAdapter, run_coarse_solver, clear_coarse_solver
from terrain_cuda_graphs import CudaGraphModel
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler


class Model(torch.nn.Module):
    config = {}

    def forward(self, x, noise_labels, conditional_inputs, precomputed_embeds=None):
        result = x[:, :6] * .125 + x[:, 6:7] * .25
        result = result + noise_labels.reshape(-1, 1, 1, 1) * .01
        for value in conditional_inputs:
            result = result + value.reshape(-1, 1, 1, 1) * .003
        if precomputed_embeds is not None:
            result = result + precomputed_embeds[:, :1, None, None] * .002
        return result


def reference(model, sample, cond, labels, conditions, embeds, scheduler):
    scheduler.set_timesteps(20)
    for index, (time, sigma) in enumerate(zip(scheduler.timesteps, scheduler.sigmas)):
        scaled = scheduler.precondition_inputs(sample, sigma.to(sample.device))
        prediction = model(torch.cat([scaled, cond], dim=1).to(sample.dtype),
                           noise_labels=labels[index], conditional_inputs=conditions,
                           precomputed_embeds=embeds[index] if embeds is not None else None)
        sample = scheduler.step(prediction, time, sample).prev_sample
    return sample


class CoarseGraphTests(unittest.TestCase):
    @torch.inference_mode()
    def test_shared_graph_lifetime_and_logical_forward_callback(self):
        model = CudaGraphModel(Model())
        calls = []
        model.__dict__['_terrain_note_graph_forwards'] = lambda value, count: calls.append((value.shape, count))
        first, second = SimpleNamespace(coarse_model=model), SimpleNamespace(coarse_model=model)
        scheduler = EDMDPMSolverMultistepScheduler()
        scheduler.set_timesteps(20)
        sample, cond = torch.ones(1, 6, 2, 2), torch.zeros(1, 5, 2, 2)
        labels = [scheduler.trigflow_precondition_noise(s.view(1)) for s in scheduler.sigmas[:-1]]
        expected = run_coarse_solver(first, scheduler, sample, cond, labels, [], None)
        graph = first._terrain_coarse_solver_graph
        clear_coarse_solver(first)
        actual = run_coarse_solver(second, scheduler, sample, cond, labels, [], None)
        self.assertIs(second._terrain_coarse_solver_graph, graph)
        self.assertNotIn('_terrain_coarse_solver_graph', first.__dict__)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(calls, [(sample.shape, 20), (sample.shape, 20)])
        self.assertEqual(len(model._terrain_solver_graph_cache), 1)
        different = EDMDPMSolverMultistepScheduler(sigma_min=.003)
        different.set_timesteps(20)
        run_coarse_solver(first, different, sample, cond, labels, [], None)
        self.assertEqual(len(model._terrain_solver_graph_cache), 1)
        self.assertIsNot(first._terrain_coarse_solver_graph, graph)

    @torch.inference_mode()
    def test_actual_scheduler_reset_dynamic_conditions_and_no_nested_graph(self):
        model = Model()
        wrapped = CudaGraphModel(model)
        scheduler = EDMDPMSolverMultistepScheduler(sigma_min=.002, sigma_max=80, sigma_data=.5)
        scheduler.set_timesteps(20)
        adapter = CoarseSolverAdapter(wrapped, scheduler, 'cpu')
        sample = torch.arange(24, dtype=torch.float32).reshape(1, 6, 2, 2) * .1
        cond = torch.ones(1, 5, 2, 2)
        labels = torch.stack([scheduler.trigflow_precondition_noise(sigma.view(1))
                              for sigma in scheduler.sigmas[:-1]])
        with patch.object(wrapped, 'forward', side_effect=AssertionError('Nested graph')):
            with patch('torch.cuda.synchronize', side_effect=AssertionError('CPU test used CUDA')):
                for offset in (0., 1., -2., 0.):
                    conditions = [torch.tensor([offset])] * 5
                    for embeds in (None, torch.full((20, 1, 3), offset)):
                        expected = reference(model, sample, cond+offset, labels, conditions, embeds, scheduler)
                        actual = adapter(sample, labels, [cond+offset, *conditions], precomputed_embeds=embeds)
                        self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
                        self.assertEqual(adapter.scheduler.model_outputs, [None, None])
                        self.assertEqual(adapter.scheduler.step_index, 20)

    def test_stochastic_or_thresholded_solver_refused(self):
        for options in ({'algorithm_type': 'sde-dpmsolver++'}, {'thresholding': True}):
            scheduler = EDMDPMSolverMultistepScheduler(**options)
            with self.assertRaisesRegex(ValueError, 'deterministic'):
                CoarseSolverAdapter(Model(), scheduler, 'cpu')


if __name__ == '__main__':
    unittest.main()
