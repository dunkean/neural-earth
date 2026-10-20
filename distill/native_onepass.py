"""Native conditional U-Net predicting the teacher's fused two-pass field.

The two world-coordinate noise streams remain explicit. Window64/stride32,
condition58 and original modulation/attention are retained; the solver evaluates
one network once per output window. This is a distinct student, never teacherT1.
"""
from types import MethodType

import numpy as np
import torch

from distill.native_base import NativeBase
from distill.native_compression import block_source, compressed_config
from distill.native_fusion import window_outputs
from distill.native_scalar_precision import anchored_output_conv
from distill.native_phase_loss import PHASE_LABELS

ARCHITECTURE = 'native-two-noise-onepass'


def extend_noise_kernel(weight):
    if weight.ndim != 4 or weight.shape[1] != 6:
        raise ValueError('Five noise channels plus the native constant are required.')
    extended = weight.new_zeros((weight.shape[0], 11, *weight.shape[2:]))
    extended[:, :5] = weight[:, :5]
    extended[:, 10:11] = weight[:, 5:6]
    return extended


class OnePassBase(NativeBase):
    def __init__(self, config, gain_anchor):
        if config['in_channels'] != 10 or config['out_channels'] != 5:
            raise ValueError('The one-pass model requires both five-channel noise streams.')
        super().__init__(config)
        self.gain_anchor = float(gain_anchor)
        self.net.out_conv._gain_anchor = self.gain_anchor
        self.net.out_conv.forward = MethodType(anchored_output_conv, self.net.out_conv)
        for name, parameter in self.net.named_parameters():
            if parameter.ndim == 0 and name != 'out_gain':
                parameter.requires_grad_(False)

    def forward(self, inputs, noise_labels, conditional_inputs):
        self.net.eval()
        dtype = torch.bfloat16 if inputs.is_cuda else torch.float32
        state = {name:value.to(torch.float32 if name == 'out_gain' else dtype)
                 for name, value in list(self.net.named_parameters())+list(self.net.named_buffers())}
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return torch.func.functional_call(self.net, state, (inputs.to(dtype),),
                dict(noise_labels=noise_labels.to(dtype),
                     conditional_inputs=[value.to(dtype) for value in conditional_inputs])).float()

    def windows(self, noises, condition):
        if noises.ndim != 4 or noises.shape[1:] != (10, 64, 64):
            raise ValueError('Two complete native64px noise fields are required.')
        dtype = torch.bfloat16 if noises.is_cuda else torch.float32
        label = torch.full((len(noises),), PHASE_LABELS[0], device=noises.device, dtype=dtype)
        first = (torch.sin(label[:, None, None, None])*noises[:, :5].to(dtype)).float()
        inputs = torch.cat((first, noises[:, 5:].float()), dim=1)
        velocity = self(inputs, label, [condition])
        return window_outputs(first, velocity, label, dtype)


@torch.no_grad()
def initialize(source, blocks=2):
    if source.get('phase') != 'compact160' or source['amount'] != 0:
        raise ValueError('Completed compact160 parent required.')
    config = compressed_config(source['config'], 160, blocks) | dict(in_channels=10)
    model = OnePassBase(config, source['gain_anchor'])
    parent = source.get('recovery_master', source['ema'])
    first = 'enc.'+next(iter(model.net.enc))+'.weight'
    state, mapping = {}, {}
    for name, expected in model.state_dict().items():
        parent_name = 'net.'+block_source(name[4:], blocks)
        value = parent[parent_name]
        if name == 'net.'+first:
            value = extend_noise_kernel(value)
        if value.shape != expected.shape:
            raise ValueError(f'Unmapped one-pass transfer:{name}:{value.shape}!={expected.shape}')
        state[name] = value.float().clone()
        mapping[name] = parent_name
    model.load_state_dict(state)
    return model, mapping


def noise_windows(seed, origins, device='cpu'):
    from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch
    if not origins:
        raise ValueError('At least one window origin is required.')
    arrays = [np.concatenate([gaussian_noise_patch(int(seed)+offset, int(y), int(x), 64, 64,
                 channels=5, tile_h=64, tile_w=64) for offset in (5819, 5820)], axis=0)
              for y, x in origins]
    return torch.from_numpy(np.stack(arrays)).to(device)


def load_onepass(saved, device='cpu', masters=False):
    if saved.get('architecture') != ARCHITECTURE or saved.get('solver_evaluations') != 1:
        raise ValueError('Explicit one-pass checkpoint required.')
    if saved.get('teacher_solver_evaluations') != 2:
        raise ValueError('The one-pass target must be the original fused two-pass teacher.')
    model = OnePassBase(saved['config'], saved['gain_anchor'])
    model.load_state_dict(saved['model'] if masters else saved['ema'])
    return model.to(device).eval()
