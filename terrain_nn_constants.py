"""Exact eval-only constant caches making Terrain Diffusion CUDA-capturable.

The upstream helpers construct CPU tensors from Python lists then copy them to
CUDA at every block. CUDA graph capture rejects those pageable host copies.
Warmup fills these small per-device caches first; capture only reads CUDA data.
"""
from types import MethodType
import numpy as np
import torch
import terrain_diffusion.models.mp_layers as layers
import terrain_diffusion.models.edm_unet as edm
import terrain_diffusion.models.unet_block as blocks

_original_sum = layers.mp_sum
_original_concat = layers.mp_concat
_original_resample = layers.resample
_weights = {}
_sums = {}
_scales = {}
_sum_norms = {}
_concat_factors = {}
_resample_weights = {}


def _weight(args, w):
    if w is None:
        values = tuple([1 / len(args)] * len(args))
    elif isinstance(w, float):
        values = (1 - w, w)
    elif isinstance(w, (tuple, list)):
        values = tuple(w)
    else:
        return None
    key = (values, args[0].device, args[0].dtype)
    if key not in _weights:
        _weights[key] = torch.tensor(values, device=args[0].device, dtype=args[0].dtype)
    return _weights[key]


def _sum(args, w=None):
    if torch.is_grad_enabled():
        return _original_sum(args, w)
    weights = _weight(args, w)
    if weights is None:
        return _original_sum(args, w)
    key = id(weights)
    if key not in _sum_norms:
        _sum_norms[key] = torch.linalg.vector_norm(weights)
    return torch.sum(torch.stack([value * weight for value, weight in zip(args, weights)]), dim=0) / _sum_norms[key]


def _concat(args, dim=1, w=None):
    if torch.is_grad_enabled():
        return _original_concat(args, dim, w)
    weights = _weight(args, w)
    if weights is None:
        return _original_concat(args, dim, w)
    sizes = tuple(value.shape[dim] for value in args)
    key = (sizes, id(weights))
    if key not in _concat_factors:
        total_key = (sum(sizes), args[0].device, args[0].dtype)
        if total_key not in _sums:
            _sums[total_key] = torch.tensor(sum(sizes), device=args[0].device, dtype=args[0].dtype)
        c = torch.sqrt(_sums[total_key] / torch.sum(torch.square(weights)))
        # Cache the original complete scalar expression, including its BF16
        # rounding after division and multiplication. Do not combine factors.
        _concat_factors[key] = [c / np.sqrt(size) * weights[i] for i, size in enumerate(sizes)]
    return torch.concat([value * factor for value, factor in zip(args, _concat_factors[key])], dim=dim)


def _resample(x, mode='keep', factor=2):
    if torch.is_grad_enabled() or mode in ('keep', 'up_bilinear'):
        return _original_resample(x, mode, factor)
    key = (x.shape[1], mode, factor, x.device, x.dtype)
    if key not in _resample_weights:
        size = 1 if mode == 'down' else factor
        _resample_weights[key] = torch.ones((x.shape[1], 1, size, size), device=x.device, dtype=x.dtype)
    if mode == 'down':
        return torch.nn.functional.conv2d(x, _resample_weights[key], groups=x.shape[1], stride=factor)
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, _resample_weights[key], groups=x.shape[1], stride=factor)


def _attention(self, x):
    if self.training or torch.is_grad_enabled():
        return self.__dict__['_terrain_original_attn'](x)
    y = self.attn_qkv(x)
    y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
    q, k, v = layers.normalize(y, dim=2).unbind(3)
    key = (q.shape[2], q.device, q.dtype)
    if key not in _scales:
        _scales[key] = torch.sqrt(torch.tensor(q.shape[2], dtype=q.dtype, device=q.device))
    attention = torch.einsum('nhcq,nhck->nhqk', q, k / _scales[key]).softmax(dim=3)
    y = torch.einsum('nhqk,nhck->nhcq', attention, v)
    return self.attn_proj(y.reshape(*x.shape))


def prepare_constants(world):
    # Match each import alias. The process is the optional local inference
    # agent; no upstream file or checkpoint is rewritten.
    layers.mp_sum = edm.mp_sum = blocks.mp_sum = _sum
    layers.mp_concat = edm.mp_concat = _concat
    layers.resample = blocks.resample = _resample
    for model in (world.coarse_model, world.base_model, world.decoder_model):
        for module in model.modules():
            if isinstance(module, blocks.UNetBlock) and module.num_heads and '_terrain_original_attn' not in module.__dict__:
                module.__dict__['_terrain_original_attn'] = module.attn
                module.attn = MethodType(_attention, module)


def restore_constants(world):
    layers.mp_sum = edm.mp_sum = blocks.mp_sum = _original_sum
    layers.mp_concat = edm.mp_concat = _original_concat
    layers.resample = blocks.resample = _original_resample
    for model in (world.coarse_model, world.base_model, world.decoder_model):
        for module in model.modules():
            original = module.__dict__.pop('_terrain_original_attn', None)
            if original is not None:
                module.attn = original
