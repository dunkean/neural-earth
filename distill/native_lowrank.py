"""Keep the teacher's feature basis while factorizing internal MP convolutions.

Both factors share the original composed-weight RMS normalization. The dense
mode is an algebra/control path; only the two-convolution mode saves compute.
Neither singular-value energy nor a full-rank control proves terrain quality.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from distill.native_base import NativeBase
from terrain_diffusion.models.mp_layers import MPConv


class FactorMPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, rank, no_padding=False, mode='factorized'):
        super().__init__()
        if kernel != (3, 3) or not 0 < rank <= min(out_channels, in_channels*9):
            raise ValueError('A positive internal 3x3 convolution rank is required.')
        if mode not in ('factorized', 'composed'):
            raise ValueError('Unknown convolution execution mode.')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.rank, self.no_padding, self.mode = rank, no_padding, mode
        self.spatial = nn.Parameter(torch.empty(rank, in_channels, *kernel))
        self.pointwise = nn.Parameter(torch.empty(out_channels, rank, 1, 1))

    @classmethod
    @torch.no_grad()
    def from_conv(cls, conv, rank, mode='factorized'):
        if not isinstance(conv, MPConv) or conv.groups != 1 or tuple(conv.weight.shape[-2:]) != (3, 3):
            raise ValueError('Only ordinary internal MPConv 3x3 layers can be factorized.')
        # Use FP64 only for the offline decomposition. FP32 eigenvectors of
        # these wide kernels introduced visible full-network roundoff before
        # any rank reduction; factors/forward/optimizer masters remain FP32.
        weight = conv.weight.double().flatten(1)
        layer = cls(weight.shape[1]//9, weight.shape[0], (3, 3), rank, conv.no_padding, mode).to(weight.device)
        # Eigh of the small output Gram matrix avoids materializing the wide
        # right singular-vector matrix. Reconstruct A as U^T W, without inverse
        # singular values; retain near-zero directions safely at full rank.
        _, vectors = torch.linalg.eigh(weight@weight.T)
        left = vectors[:, -rank:].flip(1).contiguous()
        right = left.T@weight
        singular = right.square().sum(1).sqrt()
        balance = singular.clamp_min(1e-12).sqrt()
        layer.pointwise.copy_((left*balance[None]).reshape_as(layer.pointwise))
        layer.spatial.copy_((right/balance[:, None]).reshape_as(layer.spatial))
        return layer

    def composed_weight(self):
        a, b = self.spatial.float().flatten(1), self.pointwise.float().flatten(1)
        return (b@a).reshape(self.out_channels, self.in_channels, 3, 3)

    def normalization(self):
        a, b = self.spatial.float().flatten(1), self.pointwise.float().flatten(1)
        squared = ((b.T@b)*(a@a.T)).sum().clamp_min(torch.finfo(torch.float32).tiny)
        return (squared/(self.out_channels*self.in_channels*9)).sqrt()+1e-4

    def forward(self, inputs, gain=1):
        padding = 0 if self.no_padding else 1
        if self.mode == 'composed':
            # The original wrapper casts masters before MP normalization.
            weight = self.composed_weight().to(inputs.dtype).float()
            scale = (weight.square().mean().sqrt()+1e-4)*math.sqrt(self.in_channels*9)
            return F.conv2d(inputs, (weight*(gain/scale)).to(inputs.dtype), padding=padding)
        scale = self.normalization()*math.sqrt(self.in_channels*9)
        hidden = F.conv2d(inputs, self.spatial.to(inputs.dtype), padding=padding)
        return F.conv2d(hidden, (self.pointwise.float()*(gain/scale)).to(inputs.dtype))


def eligible(name, module):
    return (name.startswith(('enc.', 'dec.')) and isinstance(module, MPConv)
            and module.groups == 1 and module.weight.ndim == 4
            and tuple(module.weight.shape[-2:]) == (3, 3)
            and min(module.weight.shape[:2]) >= 64)


def rank_for(module, fraction):
    if not 0 < fraction <= 1:
        raise ValueError('Rank fraction must be in (0,1].')
    output, inputs = module.weight.shape[:2]
    return min(output, inputs*9, max(16, math.ceil(output*fraction/16)*16))


class NativeLowrankBase(NativeBase):
    def __init__(self, config, ranks, mode='factorized'):
        super().__init__(config)
        self.ranks, self.mode = dict(ranks), mode
        modules = dict(self.net.named_modules())
        for name, rank in self.ranks.items():
            conv = modules[name]
            if not eligible(name, conv):
                raise ValueError(f'Layer is outside the permitted teacher basis: {name}')
            replacement = FactorMPConv(conv.weight.shape[1], conv.weight.shape[0], (3, 3), rank,
                                       conv.no_padding, mode)
            self.net.get_submodule(name.rsplit('.', 1)[0])._modules[name.rsplit('.', 1)[1]] = replacement

    @classmethod
    @torch.no_grad()
    def from_teacher(cls, teacher, fraction, mode='factorized'):
        ranks = {name: rank_for(module, fraction) for name, module in teacher.named_modules() if eligible(name, module)}
        result = cls(dict(teacher.config), ranks, mode).to(next(teacher.parameters()).device)
        source = teacher.state_dict()
        ordinary = {f'net.{name}': value for name, value in source.items()
                    if name.rsplit('.', 1)[0] not in ranks}
        result.load_state_dict(ordinary, strict=False)
        for name, rank in ranks.items():
            conv = FactorMPConv.from_conv(teacher.get_submodule(name), rank, mode)
            result.net.get_submodule(name.rsplit('.', 1)[0])._modules[name.rsplit('.', 1)[1]] = conv
        return result.eval()

    @classmethod
    def from_basis(cls, saved, fraction, mode='factorized'):
        ranks = {name: min(rank, max(16, math.ceil(rank*fraction/16)*16))
                 for name, rank in saved['factor_ranks'].items()}
        if not 0 < fraction <= 1:
            raise ValueError('Rank fraction must be in (0,1].')
        model = cls(saved['config'], ranks, mode)
        state = {}
        for name, value in saved['ema'].items():
            if name.endswith('.spatial'):
                value = value[:ranks[name[len('net.'):].rsplit('.', 1)[0]]]
            elif name.endswith('.pointwise'):
                value = value[:, :ranks[name[len('net.'):].rsplit('.', 1)[0]]]
            state[name] = value
        model.load_state_dict(state)
        return model.eval()

    def forward(self, inputs, noise_labels, conditional_inputs):
        self.net.eval()
        dtype = torch.bfloat16 if inputs.is_cuda else torch.float32
        # Keep factors in FP32 until global composed-weight normalization;
        # independent BF16 factor casts still belong to the deployed path.
        state = {name: value if name.endswith(('.spatial', '.pointwise')) else value.to(dtype)
                 for name, value in list(self.net.named_parameters())+list(self.net.named_buffers())}
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return torch.func.functional_call(self.net, state, (inputs.to(dtype),),
                dict(noise_labels=noise_labels.to(dtype),
                     conditional_inputs=[value.to(dtype) for value in conditional_inputs])).float()
