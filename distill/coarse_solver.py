"""Coarse distillation through a short differentiable solver initialized by teacher.

The convolutional one-pass students remain separate. This trial preserves the
teacher architecture and shortens its 20 solver evaluations, optimizing against
the same final targets. It makes no claim of parameter compression or acceptance.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from distill.common import bootstrap
bootstrap()


class CoarseSolver(nn.Module):
    def __init__(self, steps):
        super().__init__()
        from terrain_app import resolve_model_source
        from terrain_diffusion.models.edm_unet import EDMUnet2D
        if not 2 <= steps <= 20:
            raise ValueError('The coarse solver trial requires 2–20 evaluations.')
        source = Path(resolve_model_source())
        self.net = EDMUnet2D.from_pretrained(str(source/'coarse_model'))
        # Production casts weights and Fourier buffers to BF16 before any
        # functional normalization. Reproduce that initial state, then retain
        # FP32 masters for the optimizer instead of changing the teacher at init.
        nn.Module.to(self.net, dtype=torch.bfloat16)
        nn.Module.to(self.net, dtype=torch.float32)
        self.steps = steps
        cfg = json.loads((source/'config.json').read_text())
        self.delta_ratio = cfg['coarse_stds'][1] / cfg['coarse_stds'][0]

    def prepare_capture(self, *, exact_kernels=False, attention_backend='reference'):
        from terrain_nn_constants import prepare_constants
        # Reuse the runtime's exact constant caches for this separate network.
        prepare_constants(SimpleNamespace(device=next(self.net.parameters()).device,
            coarse_model=self.net, base_model=nn.Identity(), decoder_model=nn.Identity()),
            exact_kernels=exact_kernels, attention_backend=attention_backend)

    def forward(self, features):
        # Autocast changes FP32 scalar/embedding operations inside this solver:
        # measured 20-step error was ~16 m, versus ~0.21 m without it on a
        # stored teacher pair. Kernels still use explicitly cast BF16 tensors.
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self._forward(features)

    def _forward(self, features):
        from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
        if features.shape[1:] != (16, 64, 64):
            raise ValueError('The coarse solver operates on complete 16×64×64 feature windows.')
        # MP layers normally normalize their Parameter in place in train mode.
        # Across several differentiable calls this changes versions needed by
        # backward. Eval mode retains functional weight normalization and gradients,
        # without those parameter mutations; no upstream code is patched.
        self.net.eval()
        dtype = torch.bfloat16 if features.device.type == 'cuda' else torch.float32
        scheduler = EDMDPMSolverMultistepScheduler(sigma_min=.002, sigma_max=80, sigma_data=.5)
        scheduler.set_timesteps(self.steps)
        input_sigmas = scheduler.sigmas
        if features.is_cuda:
            # Keep scheduler coefficients on CPU to preserve their arithmetic.
            # Only the network's sigma inputs already used CUDA in the original
            # path. Prepare their identical copies before graph capture.
            key = (features.device, self.steps)
            cached = self.__dict__.get('_input_sigma_cache')
            if cached is None or cached[0] != key:
                cached = (key, scheduler.sigmas.to(features.device))
                self.__dict__['_input_sigma_cache'] = cached
            input_sigmas = cached[1]
        sample = features[:, :6].to(dtype) * scheduler.sigmas[0]
        condition = features[:, 6:11].to(dtype)
        labels = [features[:, i, 0, 0].to(dtype) for i in range(11, 16)]
        # Scalar gains and Fourier buffers also participate in BF16 rounding.
        # A functional cast preserves that production computation while gradients
        # still reach the FP32 masters, including through all solver evaluations.
        state = {name: value.to(dtype) for name, value in
                 list(self.net.named_parameters()) + list(self.net.named_buffers())}
        for t, sigma in zip(scheduler.timesteps, input_sigmas):
            inputs = torch.cat((scheduler.precondition_inputs(sample, sigma), condition), dim=1).to(dtype)
            noise_labels = scheduler.trigflow_precondition_noise(sigma.view(-1)).to(dtype).expand(features.shape[0])
            result = torch.func.functional_call(self.net, state, (inputs,),
                         dict(noise_labels=noise_labels, conditional_inputs=labels))
            sample = scheduler.step(result, t, sample).prev_sample
        normalized = sample.float() / scheduler.config.sigma_data
        # Stored targets normalize physical mean and p5 with the same height
        # standard deviation. Teacher network channel 1 is their smaller delta.
        return torch.cat((normalized[:, :1],
                          normalized[:, :1] - normalized[:, 1:2] * self.delta_ratio,
                          normalized[:, 2:]), dim=1)
