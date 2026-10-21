"""Frozen compact-width inference with cached native MP kernels and output gain."""
from types import MethodType, SimpleNamespace

import torch
from torch.nn import functional as F

from distill.native_progressive_width import load_progressive, effective_kernel


def frozen_output(layer, inputs, gain=1):
    return F.conv2d(inputs, layer._runtime_out_weight,
                    padding=0 if layer.no_padding else layer._runtime_out_weight.shape[-1]//2,
                    groups=layer.groups)


@torch.no_grad()
def runtime_progressive(saved, device='cpu'):
    if saved.get('phase') != 'compact160' or saved['amount'] != 0 or saved['config'] != saved['target_config']:
        raise ValueError('Runtime loading requires the physically compacted checkpoint.')
    owner = load_progressive(saved, device)
    gain = owner.net.out_gain.detach().float().clone()
    net = owner.net.bfloat16()
    net.out_gain = torch.nn.Parameter(gain, requires_grad=False)
    out = net.out_conv
    kernel = effective_kernel(out, out.weight, net.out_gain).bfloat16()
    from terrain_inference import prepare_models
    prepare_models(SimpleNamespace(coarse_model=net, base_model=net, decoder_model=net))
    # Compute the anchored output kernel once; this also avoids materializing
    # its BF16 anchor constant during CUDA graph capture.
    out._runtime_out_weight = kernel
    out.forward = MethodType(frozen_output, out)
    net._terrain_custom_base = True
    return net.eval().requires_grad_(False)
