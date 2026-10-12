"""Native teacher topology with simulated E4M3 convolutions for weight training.

BF16 convolutions perform the compute. This is precision-aware distillation,
not a claim of accelerated FP8 inference. MP normalization, conditioning,
embeddings, attention logits/softmax, skips and entry/exit heads stay native.
"""
import math
from types import MethodType

import torch
from torch.nn import functional as F

from distill.native_base import NativeBase
from terrain_diffusion.models.mp_layers import MPConv,normalize


def fake_e4m3(value,scale):
    scale=scale.detach().float().clamp_min(1e-12)
    rounded=(value.float()/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale
    # Preserve the rounded forward exactly, including clipped BF16 values.
    # x + (rounded-x).detach() loses low bits through cancellation on clipping.
    return rounded.to(value.dtype).detach()+(value-value.detach())


def quantized_conv(layer,inputs,gain=1):
    if not layer._fp8_enabled:return MPConv.forward(layer,inputs,gain)
    # This is the original global MP weight normalization, before any rounding.
    weight=normalize(layer.weight.float())*(gain/math.sqrt(layer.weight[0].numel()))
    dims=tuple(range(1,weight.ndim))
    scale=weight.detach().abs().amax(dim=dims,keepdim=True).clamp_min(1e-12)/448
    weight=fake_e4m3(weight,scale).to(inputs.dtype)
    inputs=fake_e4m3(inputs,layer._fp8_scale)
    return F.conv2d(inputs,weight,padding=(0 if layer.no_padding else weight.shape[-1]//2),groups=layer.groups)


class NativeFP8Base(NativeBase):
    def __init__(self,config,layers=None):
        super().__init__(config)
        first='enc.'+next(iter(self.net.enc))
        eligible=[n for n,m in self.net.named_modules() if isinstance(m,MPConv) and m.weight.ndim==4 and
                  n not in (first,'out_conv')]
        self.quantized_layers=list(eligible if layers is None else layers)
        if not self.quantized_layers or any(n not in eligible for n in self.quantized_layers):
            raise ValueError('Only native internal convolutions may be quantized.')
        for name in self.quantized_layers:
            layer=self.net.get_submodule(name)
            layer.register_buffer('_fp8_scale',torch.zeros((),dtype=torch.float32))
            layer._fp8_enabled=False
            layer.forward=MethodType(quantized_conv,layer)

    def load_teacher(self,state):
        missing,unexpected=self.load_state_dict({'net.'+n:v for n,v in state.items()},strict=False)
        if set(missing)!={'net.'+n+'._fp8_scale' for n in self.quantized_layers} or unexpected:
            raise ValueError('Only static activation scales may be missing from the original teacher.')

    def set_quantized(self,enabled):
        for name in self.quantized_layers:
            layer=self.net.get_submodule(name)
            if enabled and (not torch.isfinite(layer._fp8_scale) or layer._fp8_scale<=0):
                raise ValueError('Frozen positive TRAIN activation scales are required.')
            layer._fp8_enabled=bool(enabled)

    def forward(self,inputs,noise_labels,conditional_inputs):
        self.net.eval();dtype=torch.bfloat16 if inputs.is_cuda else torch.float32
        state={name:value.to(torch.float32 if name.endswith('._fp8_scale') else dtype)
               for name,value in list(self.net.named_parameters())+list(self.net.named_buffers())}
        with torch.autocast(device_type=inputs.device.type,enabled=False):
            return torch.func.functional_call(self.net,state,(inputs.to(dtype),),dict(
                noise_labels=noise_labels.to(dtype),conditional_inputs=[v.to(dtype) for v in conditional_inputs])).float()


def load_fp8(saved,device='cpu'):
    model=NativeFP8Base(saved['config'],saved['quantized_layers']).to(device).eval()
    model.load_state_dict(saved['ema']);model.set_quantized(True)
    return model.requires_grad_(False)
