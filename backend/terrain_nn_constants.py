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
_original_silu = layers.mp_silu
_weights = {}
_sums = {}
_scales = {}
_sum_norms = {}
_concat_factors = {}
_resample_weights = {}
_runtime_config = None
_kernel_errors = {}


def _kernel_ready(x):
    return (_runtime_config is not None and _runtime_config[0] and x.is_cuda and
            x.device.index not in _kernel_errors)


def _silu(x):
    if _kernel_ready(x):
        from terrain_cuda_kernels import silu_scaled
        result = silu_scaled(x)
        if result is not None:
            return result
    return _original_silu(x)


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
    if len(args) == 2 and args[0].shape == args[1].shape:
        if _kernel_ready(args[0]):
            from terrain_cuda_kernels import binary_sum
            result = binary_sum(args[0], args[1], weights, _sum_norms[key])
            if result is not None:
                return result
        # Each product still rounds in the original dtype before addition.
        # Binary reduction needs no stacked tensor; +0 matches sum's zero sign.
        return ((args[0] * weights[0] + args[1] * weights[1]) + 0.) / _sum_norms[key]
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
    if len(args) == 2 and _kernel_ready(args[0]):
        from terrain_cuda_kernels import binary_concat
        result = binary_concat(args[0], args[1], _concat_factors[key], dim)
        if result is not None:
            return result
    return torch.concat([value * factor for value, factor in zip(args, _concat_factors[key])], dim=dim)


def _resample(x, mode='keep', factor=2):
    if torch.is_grad_enabled() or mode in ('keep', 'up_bilinear'):
        return _original_resample(x, mode, factor)
    # The unit grouped kernels have exactly one contributing value per output.
    # Add +0 to reproduce the convolution accumulator's canonical signed zero.
    # Never change the train path or bilinear interpolation.
    if mode == 'down':
        return x[:, :, ::factor, ::factor] + 0.
    assert mode == 'up'
    return torch.nn.functional.interpolate(x, scale_factor=factor, mode='nearest') + 0.


def _attention(self, x):
    if self.training or torch.is_grad_enabled():
        return self.__dict__['_terrain_original_attn'](x)
    y = self.attn_qkv(x)
    y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
    q, k, v = layers.normalize(y, dim=2).unbind(3)
    key = (q.shape[2], q.device, q.dtype)
    if key not in _scales:
        _scales[key] = torch.sqrt(torch.tensor(q.shape[2], dtype=q.dtype, device=q.device))
    if self.__dict__.get('_terrain_attention_backend', 'reference') == 'sdpa-base':
        # Experimental: float accumulation changes BF16 score/softmax rounding.
        # Keep the original rounded key scale; SDPA must not scale a second time.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        # Local Windows Torch lacks FlashAttention; cuDNN is also a fused engine.
        # Never silently select the unoptimized math fallback in this experiment.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            y = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(-2, -1).contiguous(),
                (k / _scales[key]).transpose(-2, -1).contiguous(),
                v.transpose(-2, -1).contiguous(), dropout_p=0., scale=1.)
        y = y.transpose(-2, -1).contiguous()
    else:
        attention = torch.einsum('nhcq,nhck->nhqk', q, k / _scales[key]).softmax(dim=3)
        y = torch.einsum('nhqk,nhck->nhcq', attention, v)
    return self.attn_proj(y.reshape(*x.shape))


def validate_engine_config(*, exact_kernels=False, attention_backend='reference'):
    global _runtime_config
    if attention_backend not in ('reference', 'sdpa-base'):
        raise ValueError('Unknown attention backend')
    requested = (bool(exact_kernels), attention_backend)
    if _runtime_config is not None and _runtime_config != requested:
        raise ValueError('Pointwise engine changes require a fresh process; captured models may be shared')
    _runtime_config = requested


def prepare_constants(world, *, exact_kernels=False, attention_backend='reference'):
    validate_engine_config(exact_kernels=exact_kernels, attention_backend=attention_backend)
    device = torch.device(world.device)
    index = device.index if device.index is not None else torch.cuda.current_device() if device.type == 'cuda' else None
    if exact_kernels and device.type == 'cuda' and index not in _kernel_errors:
        from terrain_cuda_kernels import prepare
        try:
            prepare(device)
        except Exception as error:
            # Only preparation can fall back. A launch error must propagate.
            _kernel_errors[index] = f'{type(error).__name__}: {error}'[:400]
    # Match each import alias. The process is the optional local inference
    # agent; no upstream file or checkpoint is rewritten.
    layers.mp_sum = edm.mp_sum = blocks.mp_sum = _sum
    layers.mp_concat = edm.mp_concat = _concat
    layers.resample = blocks.resample = _resample
    for role, model in (('coarse', world.coarse_model), ('base', world.base_model), ('decoder', world.decoder_model)):
        for module in model.modules():
            if isinstance(module, blocks.UNetBlock):
                module.__dict__['_terrain_attention_backend'] = attention_backend if role == 'base' else 'reference'
            if isinstance(module, blocks.UNetBlock) and module.activation is _original_silu:
                module.activation = _silu
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
            if isinstance(module, blocks.UNetBlock) and module.activation is _silu:
                module.activation = _original_silu
            module.__dict__.pop('_terrain_attention_backend', None)


def kernel_status():
    requested = bool(_runtime_config and _runtime_config[0])
    runtime = __import__('terrain_cuda_kernels').status() if requested else None
    devices = set(_kernel_errors) | set(runtime['devices'] if runtime else [])
    return dict(requested=requested, preparation_errors=dict(_kernel_errors),
                effective_by_device={index:bool(requested and index not in _kernel_errors and
                                               index in runtime['devices']) for index in devices},
                runtime=runtime)
