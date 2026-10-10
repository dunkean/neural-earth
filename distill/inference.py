"""Offline stage hooks for accepted-candidate evaluation, with unchanged upstream."""
from __future__ import annotations

from pathlib import Path
from types import MethodType

import numpy as np
import torch
from infinite_tensor import InfiniteTensor, TensorWindow

from distill.features import base_features, coarse_features, decoder_features
from distill.student import load_student


def normalized(value):
    return (value[:-1]/value[-1:].clamp_min(1e-12)).float()


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
        students[stage] = model
    world._distill_students = students
    world._distill_counts = {stage: dict(calls=0, output_pixels=0) for stage in students}

    def forward(stage, features):
        model = students[stage]
        inputs = features[None].to(world.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(inputs)[0].float()
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
                    features = base_features(field, y//32, x//32, self.seed, y, x,
                                             total, self.kwargs['histogram_raw'])
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
                    condition = prepared.conditions[0][0].float().cpu().numpy()
                    scalars = np.array([float(v[0]) for v in prepared.conditions[1:]], np.float32)
                    features = coarse_features(condition, scalars, self.seed, ctx[1]*48, ctx[2]*48)
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
                    features = decoder_features(normalized(latent)[:4].cpu(), self.seed,
                                                ctx[1]*stride, ctx[2]*stride, size)
                    pred = forward('decoder', features)
                    self._distill_counts['decoder']['output_pixels'] += size**2
                    outputs.append(torch.cat([pred*weight[None], weight[None]]))
                return outputs
            tensor._f = predict
            return tensor
        world._build_decoder_stage = MethodType(build_decoder, world)
    return world
