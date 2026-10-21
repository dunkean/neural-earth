"""Frozen one-evaluation base windows and production InfiniteTensor installation."""
from types import MethodType, SimpleNamespace

import torch
from torch import nn

from distill.native_onepass import load_onepass, noise_windows
from distill.native_phase_loss import PHASE_LABELS
from distill.native_fusion import window_outputs
from distill.native_progressive_width import anchored_scale
from distill.native_progressive_runtime import frozen_output
from terrain_diffusion.models.mp_layers import normalize


class OnePassRuntime(nn.Module):
    def __init__(self, saved, device='cpu'):
        super().__init__()
        owner = load_onepass(saved, device).requires_grad_(False)
        self.config = owner.config
        gain = owner.net.out_gain.detach().float().clone()
        dtype = torch.bfloat16 if torch.device(device).type == 'cuda' else torch.float32
        self.net = owner.net.to(dtype=dtype).eval().requires_grad_(False)
        self.net.out_gain = nn.Parameter(gain, requires_grad=False)
        self.register_buffer('label', torch.tensor(PHASE_LABELS[0], device=device, dtype=dtype))
        out = self.net.out_conv
        kernel = normalize(out.weight.float())*anchored_scale(gain, owner.gain_anchor, out.weight[0].numel())
        from terrain_inference import prepare_models
        prepare_models(SimpleNamespace(coarse_model=self.net, base_model=self.net, decoder_model=self.net))
        out._runtime_out_weight = kernel.to(dtype)
        out.forward = MethodType(frozen_output, out)

    def forward(self, noises, condition):
        dtype = self.label.dtype
        first = torch.sin(self.label)*noises[:, :5].to(dtype)
        inputs = torch.cat((first, noises[:, 5:].to(dtype)), dim=1)
        labels = self.label.expand(len(inputs))
        velocity = self.net(inputs, noise_labels=labels, conditional_inputs=[condition.to(dtype)])
        return window_outputs(first, velocity, labels, dtype)


def install_onepass(world, model):
    """Install before bind; retain original condition construction and64/32 fusion."""
    world._distill_onepass_base = model
    world._distill_onepass_counts = dict(calls=0, windows=0)
    def build(self):
        from infinite_tensor import InfiniteTensor, TensorWindow
        from terrain_diffusion.inference.world_pipeline import linear_weight_window
        means = torch.tensor([14.99, 11.65, 15.87, 619.26, 833.12, 69.40, .66])
        stds = torch.tensor([21.72, 21.78, 10.40, 452.29, 738.09, 34.59, .47])
        histogram = torch.tensor([self.kwargs['histogram_raw']])
        weight = linear_weight_window(64, 'cpu', torch.float32)
        @torch.no_grad()
        def predict(ctxs, conditions):
            vectors = []
            origins = []
            for ctx, condition in zip(ctxs, conditions):
                image = condition[:-1]/condition[-1:]
                image = torch.cat((image, image.new_ones(1, 4, 4)))[None]
                vector = self._process_latent_conditioning(image, histogram, means, stds,
                    torch.tensor(0.), seed_offset=ctx[1]*65536+ctx[2])
                vectors.append(vector)
                origins.append((ctx[1]*32, ctx[2]*32))
            if not origins:
                return []
            noise = noise_windows(self.seed, origins, self.device)
            condition = torch.cat(vectors).to(self.device)
            fields = model(noise, condition).cpu().float()
            if not torch.isfinite(fields).all():
                raise FloatingPointError('Nonfinite one-pass base output.')
            self._distill_onepass_counts['calls'] += 1
            self._distill_onepass_counts['windows'] += len(fields)
            return [torch.cat((field*weight[None], weight[None])) for field in fields]
        return InfiniteTensor(shape=(6, None, None), f=predict,
            output_window=TensorWindow(size=(6, 64, 64), stride=(6, 32, 32)),
            args=(self.coarse,), args_windows=(TensorWindow(size=(7, 4, 4), stride=(7, 1, 1), offset=(0, -1, -1)),),
            batch_size=self.latents_batch_size, tile_store=self.tile_store, tensor_id='distill_native_onepass_latents')
    world._build_latent_stage = MethodType(build, world)
