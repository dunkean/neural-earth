"""Frozen native inference adapter with baked NF4 kernels, retaining native geometry."""
import math

import torch

from distill.native_weight_quant import encode, load_weight_quant
from terrain_diffusion.models.mp_layers import normalize


@torch.no_grad()
def runtime_model(saved,device='cpu'):
    owner=load_weight_quant(saved,device)
    # Match QAT's BF16 parameter cast BEFORE FP32 global normalization.
    owner.net.bfloat16()
    for name in owner.quantized_layers:
        layer=owner.net.get_submodule(name)
        effective=normalize(layer.weight.float())/math.sqrt(layer.weight[0].numel())
        layer.weight.copy_(encode(effective,owner.scheme,owner.block)[2].bfloat16())
    owner._packed=True
    net=owner.net.eval().requires_grad_(False)
    # Harnesses must preserve these forwards: generic MP preparation would
    # normalize the already normalized/quantized kernels a second time.
    net._terrain_custom_base=True
    net._terrain_weight_quant_scheme=owner.scheme
    return net
