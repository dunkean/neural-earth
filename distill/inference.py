"""Offline stage hooks for accepted-candidate evaluation, with unchanged upstream."""
from __future__ import annotations

from pathlib import Path
from types import MethodType
import threading

import torch
import torch.nn.functional as F
from infinite_tensor import InfiniteTensor, TensorWindow

from distill.features import COND_MEANS, COND_STDS, noise, phase
from distill.student import load_student


def normalized(value):
    return (value[:-1]/value[-1:].clamp_min(1e-12)).float()


def base_inputs(coarse, coarse_y, coarse_x, seed, y, x, size, histogram, device):
    """Assemble the CPU feature schema directly on the inference device."""
    if y % 32 or x % 32 or size % 32:
        raise ValueError('Base inputs must align to the 32-latent coarse grid.')
    cy, cx = y//32-int(coarse_y), x//32-int(coarse_x)
    field = torch.as_tensor(coarse, dtype=torch.float32)[:, cy:cy+size//32, cx:cx+size//32]
    if field.shape != (6, size//32, size//32):
        raise ValueError('Stored conditioning does not cover the requested halo.')
    # Retain the original CPU normalization and trigonometric arithmetic.
    field = (field-torch.from_numpy(COND_MEANS)[:, None, None])/torch.from_numpy(COND_STDS)[:, None, None]
    inputs = torch.zeros(32, size, size, device=device, dtype=torch.float32)
    inputs[:5].copy_(noise(seed+5819, y, x, size, size, 5))
    inputs[5:10].copy_(noise(seed+5820, y, x, size, size, 5))
    inputs[10:16].copy_(F.interpolate(field[None].to(device), size=(size, size), mode='nearest')[0])
    inputs[16].fill_((1-.66)/.47)
    inputs[17:21].copy_(phase(y, x, size, size))
    inputs[21:26].copy_(torch.as_tensor(histogram, dtype=torch.float32).to(device)[:, None, None])
    return inputs


def decoder_inputs(latents, seed, y, x, size, device):
    inputs = torch.zeros(16, size, size, device=device, dtype=torch.float32)
    inputs[:1].copy_(noise(seed+5819, y, x, size, size, 1, tile=size))
    fields = torch.as_tensor(latents, dtype=torch.float32, device=device)
    inputs[1:5].copy_(F.interpolate(fields[None], size=(size, size), mode='nearest')[0])
    return inputs


def coarse_inputs(condition, labels, seed, y, x, device):
    condition = torch.as_tensor(condition, dtype=torch.float32, device=device)
    h, w = condition.shape[-2:]
    inputs = torch.zeros(16, h, w, device=device, dtype=torch.float32)
    inputs[:6].copy_(noise(seed+1, y, x, h, w, 6))
    inputs[6:11].copy_(condition)
    inputs[11:].copy_(torch.as_tensor(labels, dtype=torch.float32, device=device)[:, None, None])
    return inputs


def coarse_graph_forward(model, inputs):
    """Share a bounded whole-solver capture across worlds borrowing this model."""
    if not inputs.is_cuda or torch.is_grad_enabled():
        return model(inputs)
    lock = model.__dict__.setdefault('_distill_graph_lock', threading.Lock())
    with lock:
        key = (tuple(inputs.shape), tuple(inputs.stride()), inputs.dtype, inputs.device,
               tuple((id(p), p._version) for p in model.parameters()))
        cached = model.__dict__.get('_distill_coarse_graph')
        if cached is None or cached[0] != key:
            model.__dict__.pop('_distill_coarse_graph', None)
            cached = None
            static = inputs.clone()
            stream = torch.cuda.Stream(device=inputs.device)
            stream.wait_stream(torch.cuda.current_stream(inputs.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    reference = model(static)
            torch.cuda.current_stream(inputs.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
                output = model(static)
            graph.replay()
            if not torch.equal(reference, output):
                raise RuntimeError('Coarse student graph differs from its eager output.')
            cached = (key, graph, static, output)
            model.__dict__['_distill_coarse_graph'] = cached
        _, graph, static, output = cached
        static.copy_(inputs)
        graph.replay()
        return output.clone()


def install(world, checkpoints, tile_size=512):
    """Install before bind. Mapping is stage -> checkpoint; combinations supported."""
    if tile_size < 32 or tile_size % 32:
        raise ValueError('Base tile size must align to 32.')
    students = {}
    for stage, path in checkpoints.items():
        if isinstance(path, torch.nn.Module):
            model = path
        else:
            model, checkpoint = load_student(path, world.device)
        if model.config.stage != stage:
            raise ValueError(f'{path}: expected {stage}, found {model.config.stage}.')
        if stage == 'coarse' and model.config.solver_steps and torch.device(world.device).type == 'cuda':
            profile = getattr(world, '_terrain_profile', None)
            model.solver.prepare_capture(exact_kernels=getattr(profile, 'exact_kernels', False),
                                         attention_backend=getattr(profile, 'attention_backend', 'reference'))
        students[stage] = model
    world._distill_students = students
    world._distill_counts = {stage: dict(calls=0, output_pixels=0) for stage in students}

    def forward(stage, features):
        model = students[stage]
        inputs = features[None].to(world.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = (coarse_graph_forward(model, inputs) if stage == 'coarse' else model(inputs))[0].float()
        if not torch.isfinite(output).all():
            raise FloatingPointError(f'Non-finite {stage} student output.')
        world._distill_counts[stage]['calls'] += 1
        return output

    if 'base' in students:
        halo = students['base'].halo
        total = tile_size+2*halo

        def build_base(self):
            def predict(ctxs, conditions):
                outputs = []
                for (_, iy, ix), condition in zip(ctxs, conditions):
                    y, x = iy*tile_size-halo, ix*tile_size-halo
                    field = normalized(condition).cpu().numpy()
                    features = base_inputs(field, y//32, x//32, self.seed, y, x,
                                           total, self.kwargs['histogram_raw'], self.device)
                    pred = forward('base', features)[:, halo:halo+tile_size, halo:halo+tile_size]
                    self._distill_counts['base']['output_pixels'] += tile_size**2
                    outputs.append(torch.cat([pred, pred.new_ones(1, tile_size, tile_size)]))
                return outputs
            return InfiniteTensor(shape=(6, None, None), f=predict,
                output_window=TensorWindow(size=(6, tile_size, tile_size)), args=(self.coarse,),
                args_windows=(TensorWindow(size=(7, total//32, total//32),
                    stride=(7, tile_size//32, tile_size//32), offset=(0, -halo//32, -halo//32)),),
                device=self.device, tile_store=self.tile_store, batch_size=1,
                tensor_id='distill_final_latent')
        world._build_latent_stage = MethodType(build_base, world)

    if 'coarse' in students:
        original = world._build_coarse_stage

        def build_coarse(self):
            from distill.teacher import physical_coarse_scaling
            from terrain_inference import _coarse_batch
            from terrain_diffusion.inference.world_pipeline import linear_weight_window
            from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
            tensor = original()
            if self.kwargs['coarse_pooling'] != 1:
                raise ValueError('Coarse student currently requires pool=1.')
            scheduler = EDMDPMSolverMultistepScheduler(sigma_min=.002, sigma_max=80, sigma_data=.5)
            weight = linear_weight_window(64, 'cpu', torch.float32).to(self.device)
            t_cond = torch.atan(torch.tensor(self.kwargs['cond_snr'])).to(self.device, dtype=torch.bfloat16)
            labels = [v.detach().view(-1) for v in torch.log(torch.tan(t_cond)/8)]
            means, stds = physical_coarse_scaling(self)
            means = torch.as_tensor(means, device=self.device)[:, None, None]
            stds = torch.as_tensor(stds, device=self.device)[:, None, None]

            def predict(ctxs):
                outputs = []
                for ctx in ctxs:
                    prepared = _coarse_batch(self, [ctx], scheduler, weight, t_cond, labels, 1, prepare_only=True)
                    condition = prepared.conditions[0][0]
                    scalars = torch.stack([v[0] for v in prepared.conditions[1:]])
                    features = coarse_inputs(condition, scalars, self.seed, ctx[1]*48, ctx[2]*48, self.device)
                    pred = forward('coarse', features)*stds+means
                    self._distill_counts['coarse']['output_pixels'] += 64**2
                    outputs.append(torch.cat([pred*weight[None], weight[None]]))
                return outputs
            tensor._f = predict
            return tensor
        world._build_coarse_stage = MethodType(build_coarse, world)

    if 'decoder' in students:
        original_decoder = world._build_decoder_stage

        def build_decoder(self):
            from terrain_diffusion.inference.world_pipeline import linear_weight_window
            tensor = original_decoder()
            size, stride = self.decoder_tile_size, self.decoder_tile_stride
            weight = linear_weight_window(size, 'cpu', torch.float32).to(self.device)

            def predict(ctxs, latents):
                outputs = []
                for ctx, latent in zip(ctxs, latents):
                    features = decoder_inputs(normalized(latent)[:4], self.seed,
                                              ctx[1]*stride, ctx[2]*stride, size, self.device)
                    pred = forward('decoder', features)
                    self._distill_counts['decoder']['output_pixels'] += size**2
                    outputs.append(torch.cat([pred*weight[None], weight[None]]))
                return outputs
            tensor._f = predict
            return tensor
        world._build_decoder_stage = MethodType(build_decoder, world)
    return world
