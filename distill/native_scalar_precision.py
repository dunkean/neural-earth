"""Separate NF4 experiment: FP32 output gain with optional exact BF16 anchor.

The legacy QAT, runtime and packed artifacts remain unchanged. Merely promoting
out_gain changes the rounding of gain/sqrt(fan_in). The anchored mode retains
that original rounded factor at initialization and adds FP32 gain increments.
All convolutions and activations still execute in BF16.
"""
import math
from types import MethodType

import torch
from torch.nn import functional as F

from distill.native_weight_quant import NativeWeightQuantBase
from terrain_diffusion.models.mp_layers import MPConv, normalize


def anchored_output_conv(layer, inputs, gain=1):
    if not torch.is_tensor(gain) or gain.dtype != torch.float32:
        raise ValueError('Anchored output gain must remain FP32.')
    anchor = gain.new_tensor(layer._gain_anchor, dtype=torch.bfloat16)
    fan_in = math.sqrt(layer.weight[0].numel())
    # This division is deliberately BF16, exactly as in the original MPConv.
    scale = (anchor / fan_in).float() + (gain - anchor.float()) / fan_in
    weight = (normalize(layer.weight.float()) * scale).to(inputs.dtype)
    return F.conv2d(inputs, weight,
                    padding=0 if layer.no_padding else weight.shape[-1] // 2,
                    groups=layer.groups)


class ScalarPrecisionBase(NativeWeightQuantBase):
    def __init__(self, config, scheme='nf4', block=512, layers=None):
        super().__init__(config, scheme, block, layers)
        self.precision_mode = 'bf16'
        self.gain_anchor = None

    def set_precision(self, mode, anchor=None):
        if mode not in ('bf16', 'fp32', 'anchored'):
            raise ValueError('Unknown scalar precision experiment.')
        if mode == 'anchored':
            if anchor is None:
                raise ValueError('Explicit saved BF16 effective anchor required.')
            rounded = float(torch.tensor(anchor).bfloat16().float())
            if rounded != float(anchor):
                raise ValueError('Anchor must be exactly representable in BF16.')
            self.gain_anchor = rounded
            self.net.out_conv._gain_anchor = rounded
            self.net.out_conv.forward = MethodType(anchored_output_conv, self.net.out_conv)
        else:
            self.net.out_conv.forward = MethodType(MPConv.forward, self.net.out_conv)
        self.precision_mode = mode

    def forward(self, inputs, noise_labels, conditional_inputs):
        self.net.eval()
        state = {name: value.to(torch.float32 if name == 'out_gain' and
                 self.precision_mode != 'bf16' else torch.bfloat16)
                 for name, value in list(self.net.named_parameters()) +
                 list(self.net.named_buffers())}
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return torch.func.functional_call(self.net, state,
                (inputs.to(torch.bfloat16),),
                dict(noise_labels=noise_labels.to(torch.bfloat16),
                     conditional_inputs=[v.to(torch.bfloat16) for v in conditional_inputs])).float()


def load_scalar_precision(saved, device='cpu'):
    if saved['architecture'] != 'native-weight-quant':
        raise ValueError('A legacy NF4 source checkpoint is required for this probe.')
    model = ScalarPrecisionBase(saved['config'], saved['scheme'], saved['block'],
                                saved['quantized_layers']).to(device).eval()
    model.load_state_dict(saved['ema'])
    return model.requires_grad_(False)
