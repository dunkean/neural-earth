"""Immutable cached BF16 copies for compression profiling, never training.

Factor normalization is computed once from FP32 masters. Ordinary parameters
are rounded as in NativeBase; the live training network is never patched.
"""
import copy
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from distill.native_lowrank import FactorMPConv


class FrozenFactor(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.padding = 0 if factor.no_padding else 1
        self.mode = factor.mode
        with torch.no_grad():
            if self.mode == 'factorized':
                scale = factor.normalization()*math.sqrt(factor.in_channels*9)
                self.register_buffer('spatial', factor.spatial.detach().bfloat16().clone())
                self.register_buffer('pointwise', (factor.pointwise.detach().float()*(1/scale)).bfloat16())
            else:
                weight = factor.composed_weight().bfloat16().float()
                scale = (weight.square().mean().sqrt()+1e-4)*math.sqrt(factor.in_channels*9)
                self.register_buffer('weight', (weight*(1/scale)).bfloat16())

    def forward(self, inputs, gain=1):
        if isinstance(gain, torch.Tensor) or gain != 1:
            raise ValueError('Only ordinary internal constant-gain convolutions are frozen.')
        if self.mode == 'composed':
            return F.conv2d(inputs, self.weight, padding=self.padding)
        return F.conv2d(F.conv2d(inputs, self.spatial, padding=self.padding), self.pointwise)


@torch.no_grad()
def freeze(model, constants=False):
    from terrain_inference import prepare_models
    net = copy.deepcopy(model.net).eval().requires_grad_(False)
    # Factor FP32 masters must survive until their composed RMS is computed.
    for name, value in net.named_parameters():
        if not name.endswith(('.spatial', '.pointwise')):
            value.data = value.data.bfloat16()
    for name, value in list(net.named_buffers()):
        if value.is_floating_point():
            parent, leaf = name.rsplit('.', 1) if '.' in name else ('', name)
            net.get_submodule(parent)._buffers[leaf] = value.bfloat16()
    for name, module in list(net.named_modules()):
        if isinstance(module, FactorMPConv):
            parent, leaf = name.rsplit('.', 1)
            net.get_submodule(parent)._modules[leaf] = FrozenFactor(module)
    world = SimpleNamespace(device=next(net.parameters()).device,
                            coarse_model=net, base_model=net, decoder_model=net)
    prepare_models(world)
    if constants:
        from terrain_nn_constants import prepare_constants
        prepare_constants(world, exact_kernels=False)
    return net
