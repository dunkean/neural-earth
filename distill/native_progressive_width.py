"""Gradually remove teacher channels before building the narrower native U-Net.

At amount0 the source forward is unchanged. At amount1 each MP kernel uses
the explicit projected tensor's normalization/fan-in, scattered into the old
shape. Encoder and embedding normalizations use the same retained coordinates.
This avoids abruptly changing width, skip layout and MP normalization together.
"""
import math
from contextlib import contextmanager
from types import MethodType

import torch
from torch.nn import functional as F

from distill.native_base import NativeBase
from distill.native_compression import compressed_config, project_state
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.models.mp_layers import MPConv, normalize, mp_sum, mp_concat
from terrain_diffusion.models.unet_block import UNetBlock


def scatter_channels(values, indices, count):
    shape = list(values.shape)
    shape[1] = count
    return values.new_zeros(shape).index_copy(1, indices, values)


def anchored_scale(gain, anchor, count):
    fan = math.sqrt(count)
    reference = gain.new_tensor(anchor, dtype=torch.bfloat16)
    return (reference/fan).float() + (gain-reference.float())/fan


def effective_kernel(layer, weight, gain):
    owner = layer._progressive_owner
    if layer._progressive_name == 'out_conv':
        factor = anchored_scale(gain, owner.gain_anchor, weight[0].numel())
    else:
        factor = gain/math.sqrt(weight[0].numel())
    return normalize(weight.float())*factor


def progressive_conv(layer, inputs, gain=1):
    owner = layer._progressive_owner
    amount = owner.amount
    if amount == 0 and layer._progressive_name != 'out_conv':
        return MPConv.forward(layer, inputs, gain)
    cached = owner._cache.get(layer._progressive_name)
    if cached is not None:
        if owner._cache_amount != amount or torch.is_grad_enabled():
            raise ValueError('Invalid progressive frozen-weight cache.')
        if cached.ndim == 2:
            return F.linear(inputs, cached)
        return F.conv2d(inputs, cached, padding=0 if layer.no_padding else cached.shape[-1]//2,
                        groups=layer.groups)
    weight = layer.weight
    if amount == 0:
        kernel = effective_kernel(layer, weight, gain)
    else:
        out, inc = layer._progressive_out, layer._progressive_in
        projected = weight.index_select(0, out).index_select(1, inc)
        compact = effective_kernel(layer, projected, gain)
        embedded = compact.new_zeros(weight.shape)
        # index_copy preserves gradients back to the retained source coordinates.
        rows = compact.new_zeros((len(out), weight.shape[1], *weight.shape[2:])).index_copy(1, inc, compact)
        embedded = embedded.index_copy(0, out, rows)
        kernel = embedded if amount == 1 else effective_kernel(layer, weight, gain).lerp(embedded, amount)
    kernel = kernel.to(inputs.dtype)
    if owner._cache_enabled:
        if torch.is_grad_enabled():
            raise ValueError('Frozen progressive cache cannot be used in backward.')
        owner._cache[layer._progressive_name] = kernel
    if kernel.ndim == 2:
        return F.linear(inputs, kernel)
    return F.conv2d(inputs, kernel, padding=0 if layer.no_padding else kernel.shape[-1]//2,
                    groups=layer.groups)


def progressive_normalize(values, indices, amount, embedding=False):
    selected = values.index_select(1, indices)
    if embedding:
        compact = selected/torch.sqrt(torch.mean(selected**2, dim=1, keepdim=True)+1e-8)
        complete = values/torch.sqrt(torch.mean(values**2, dim=1, keepdim=True)+1e-8)
    else:
        compact, complete = normalize(selected, dim=1), normalize(values, dim=1)
    embedded = scatter_channels(compact, indices, values.shape[1])
    return embedded if amount == 1 else complete.lerp(embedded, amount)


def progressive_block(block, inputs, emb):
    owner, amount = block._progressive_owner, block._progressive_owner.amount
    if amount == 0:
        return UNetBlock.forward(block, inputs, emb)
    x = block.resample(inputs)
    if block.mode == 'enc':
        if block.conv_skip is not None:
            x = block.conv_skip(x)
        x = progressive_normalize(x, block._progressive_channels, amount)
    y = block.conv_res0(block.activation(x))
    if block.emb_linear is not None:
        c = block.emb_linear(emb, gain=block.emb_gain)+1
        c = progressive_normalize(c, block._progressive_channels, amount, embedding=True)
        y = block.activation(y*c.unsqueeze(2).unsqueeze(3).to(y.dtype))
    else:
        y = block.activation(y)
    y = block.conv_res1(y)
    if block.mode == 'dec' and block.conv_skip is not None:
        x = block.conv_skip(x)
    if x.shape[2:] != y.shape[2:]:
        raise ValueError('Progressive native blocks require unchanged padded geometry.')
    x = mp_sum([x, y], w=block.res_balance)
    if block.num_heads:
        x = mp_sum([x, block.attn(x)], w=block.attn_balance)
    return torch.clip(x, -block.clip_act, block.clip_act) if block.clip_act is not None else x


def progressive_unet(net, inputs, noise_labels, conditional_inputs,
                     return_logvar=False, precomputed_embeds=None):
    owner, amount = net._progressive_owner, net._progressive_owner.amount
    if amount == 0:
        return EDMUnet2D.forward(net, inputs, noise_labels, conditional_inputs,
                                 return_logvar, precomputed_embeds)
    if return_logvar:
        raise ValueError('Progressive training uses the base velocity interface only.')
    emb = precomputed_embeds if precomputed_embeds is not None else net.compute_embeddings(noise_labels, conditional_inputs)
    x = torch.cat((inputs, torch.ones_like(inputs[:, :1])), dim=1)
    skips = []
    for name, block in net.enc.items():
        x = block(x) if 'conv' in name else block(x, emb)
        indices = block._progressive_out if 'conv' in name else block._progressive_channels
        skips.append((x, indices))
    current = indices
    for name, block in net.dec.items():
        if 'block' in name:
            skip, skip_indices = skips.pop()
            joined_indices = torch.cat((current, skip_indices+x.shape[1]))
            narrow = mp_concat((x.index_select(1, current), skip.index_select(1, skip_indices)), w=net.concat_balance)
            embedded = scatter_channels(narrow, joined_indices, x.shape[1]+skip.shape[1])
            x = embedded if amount == 1 else mp_concat((x, skip), w=net.concat_balance).lerp(embedded, amount)
        x = block(x, emb)
        current = block._progressive_channels
    return net.out_conv(x, gain=net.out_gain)


class ProgressiveWidthBase(NativeBase):
    def __init__(self, config, target_config, mapping, gain_anchor):
        super().__init__(config)
        self.target_config, self.mapping = dict(target_config), mapping
        self.gain_anchor, self.amount = float(gain_anchor), 0.
        self._cache, self._cache_enabled, self._cache_amount = {}, False, None
        object.__setattr__(self.net, '_progressive_owner', self)
        self.net.forward = MethodType(progressive_unet, self.net)
        for name, layer in self.net.named_modules():
            if isinstance(layer, MPConv):
                object.__setattr__(layer, '_progressive_owner', self)
                layer._progressive_name = name
                plan = mapping[name+'.weight']
                if plan['output'] is not None:
                    layer.register_buffer('_progressive_out', torch.tensor(plan['output'], dtype=torch.long), persistent=False)
                    layer.register_buffer('_progressive_in', torch.tensor(plan['inputs'], dtype=torch.long), persistent=False)
                    layer.forward = MethodType(progressive_conv, layer)
                elif name == 'out_conv':
                    raise ValueError('Output input projection is missing.')
            elif isinstance(layer, UNetBlock):
                object.__setattr__(layer, '_progressive_owner', self)
                channels = mapping[name+'.conv_res1.weight']['output']
                layer.register_buffer('_progressive_channels', torch.tensor(channels, dtype=torch.long), persistent=False)
                layer.forward = MethodType(progressive_block, layer)
        # The output scalar uses its anchored FP32 expression. Embedding gains
        # are frozen during this first channel-removal stage.
        for name, parameter in self.net.named_parameters():
            if parameter.ndim == 0 and name != 'out_gain':
                parameter.requires_grad_(False)

    @contextmanager
    def cached_weights(self):
        if self._cache_enabled:
            raise ValueError('Nested progressive caches are not supported.')
        self._cache_enabled, self._cache_amount = True, self.amount
        try:
            yield
        finally:
            self._cache.clear()
            self._cache_enabled = False

    def forward(self, inputs, noise_labels, conditional_inputs):
        self.net.eval()
        state = {name:value.to(torch.float32 if name == 'out_gain' else torch.bfloat16)
                 for name, value in list(self.net.named_parameters())+list(self.net.named_buffers())
                 if not name.rsplit('.', 1)[-1].startswith('_progressive_')}
        # Integer selection buffers must retain their integer dtype.
        state.update({name:value for name, value in self.net.named_buffers()
                      if name.rsplit('.', 1)[-1].startswith('_progressive_')})
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return torch.func.functional_call(self.net, state, (inputs.bfloat16(),),
                dict(noise_labels=noise_labels.bfloat16(),
                     conditional_inputs=[value.bfloat16() for value in conditional_inputs])).float()


def load_progressive(saved, device='cpu', raw=False):
    if saved['architecture'] != 'native-progressive-width':
        raise ValueError('Explicit progressive-width checkpoint required.')
    model = ProgressiveWidthBase(saved['config'], saved['target_config'], saved['mapping'], saved['gain_anchor'])
    model.load_state_dict(saved['model'] if raw else saved['ema'])
    model.amount = float(saved['amount'])
    return model.to(device).eval().requires_grad_(False)


def compact_state(model):
    result = {}
    for name, tensor in model.state_dict().items():
        plan = model.mapping[name[4:]]
        if plan['output'] is not None:
            tensor = tensor.index_select(0, tensor.new_tensor(plan['output'], dtype=torch.long))
            tensor = tensor.index_select(1, tensor.new_tensor(plan['inputs'], dtype=torch.long))
        result[name] = tensor.detach().clone()
    return result
