"""Optional whole twenty-step coarse graph, preserving upstream scalar maths.

The CPU scheduler coefficients become fixed scalar kernel arguments during
capture. Moving their log/exp arithmetic to CUDA would change rounding, so
only the per-forward sigma inputs are copied to CUDA ahead of capture. Dynamic
conditioning and all twenty embeddings remain graph inputs (including regional
SNR changes). No nested CudaGraphModel is invoked by the adapter.
"""
from collections import OrderedDict
import json
import threading

import torch

from terrain_cuda_graphs import CudaGraphModel
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler


class CoarseSolverAdapter(torch.nn.Module):
    def __init__(self, model, scheduler, device):
        super().__init__()
        self.network = model.model if isinstance(model, CudaGraphModel) else model
        self.scheduler = EDMDPMSolverMultistepScheduler.from_config(scheduler.config)
        self.scheduler.set_timesteps(20)
        if (self.scheduler.config.algorithm_type != 'dpmsolver++'
                or self.scheduler.config.thresholding):
            raise ValueError('Whole coarse graph requires the deterministic non-thresholded solver')
        self.register_buffer('input_sigmas', self.scheduler.sigmas.to(device).clone(), persistent=False)
        self.eval()

    @property
    def config(self):
        return self.network.config

    def forward(self, x, noise_labels, conditional_inputs,
                return_logvar=False, precomputed_embeds=None):
        if return_logvar:
            raise ValueError('Whole coarse solver does not produce log variance')
        scheduler = self.scheduler
        # Reset all mutable solver history on warmup, eager fallback and capture.
        # Replay executes the unrolled graph and cannot retain Python history.
        scheduler.model_outputs = [None] * scheduler.config.solver_order
        scheduler.lower_order_nums = 0
        scheduler._step_index = 0
        scheduler._begin_index = None
        cond_img, *conditions = conditional_inputs
        n = x.shape[0]
        sample = x
        for idx, t in enumerate(scheduler.timesteps):
            scaled = scheduler.precondition_inputs(sample, self.input_sigmas[idx])
            model_in = torch.cat([scaled, cond_img], dim=1).to(x.dtype)
            output = self.network(
                model_in, noise_labels=noise_labels[idx].expand(n),
                conditional_inputs=conditions,
                precomputed_embeds=(precomputed_embeds[idx].expand(n, -1)
                                    if precomputed_embeds is not None else None))
            sample = scheduler.step(output, t, sample).prev_sample
        # Do not pin intermediates from the last replay/warmup in Python.
        scheduler.model_outputs = [None] * scheduler.config.solver_order
        return sample


def run_coarse_solver(world, scheduler, sample, cond_img, labels, conditions, embeds):
    """Share one bounded solver graph across seeds borrowing the same weights.

    A plain __dict__ cache avoids registering the solver as a nested model
    child. The lock covers selection, eviction and use; a world never replays
    a stale graph reference after another configuration replaced the cache.
    """
    owner = world.coarse_model
    lock = owner.__dict__.setdefault('_terrain_solver_graph_lock', threading.RLock())
    key = (json.dumps(dict(scheduler.config), sort_keys=True), str(sample.device), sample.dtype)
    with lock:
        cache = owner.__dict__.setdefault('_terrain_solver_graph_cache', OrderedDict())
        graph = cache.get(key)
        if graph is None:
            while cache:
                _, previous = cache.popitem(last=False)
                previous._clear_buckets()
            graph = CudaGraphModel(CoarseSolverAdapter(owner, scheduler, sample.device),
                                   max_buckets=1, max_batch=1)
            cache[key] = graph
        world.__dict__['_terrain_coarse_solver_graph'] = graph
        output = graph(sample, noise_labels=torch.stack(labels),
                       conditional_inputs=[cond_img, *conditions],
                       precomputed_embeds=torch.stack(embeds) if embeds is not None else None)
    note = getattr(owner, '_terrain_note_graph_forwards', None)
    if note is not None:
        note(sample, 20)
    return output


def clear_coarse_solver(world):
    # Shared immutable weights own the bounded pool until eviction or process
    # teardown. Rebuilding/closing one seed must not invalidate another's pool.
    world.__dict__.pop('_terrain_coarse_solver_graph', None)
