"""One-pass native body with a cheap learned, position-matched noise residual.

A50-parameter FP32 head carries known components of the two deterministic noise
streams. It is fitted against teacher minus body, never against an energy-only
objective. The body retains its original one-evaluation solver and geometry.
"""
import torch
from torch import nn
from torch.nn import functional as F

from distill.native_onepass import OnePassBase, ARCHITECTURE as BODY_ARCHITECTURE
from distill.native_onepass_runtime import OnePassRuntime

ARCHITECTURE = 'native-two-noise-onepass-residual'


class ResidualOnePassBase(OnePassBase):
    def __init__(self, config, gain_anchor):
        super().__init__(config, gain_anchor)
        self.residual = nn.Conv2d(10, 5, 1, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.residual.weight)

    def windows(self, noises, condition):
        base = super().windows(noises, condition)
        with torch.autocast(device_type=noises.device.type, enabled=False):
            return base+F.conv2d(noises.float(), self.residual.weight.float())


class ResidualOnePassRuntime(nn.Module):
    def __init__(self, saved, device='cpu'):
        super().__init__()
        if saved.get('architecture') != ARCHITECTURE:
            raise ValueError('Explicit residual one-pass checkpoint required.')
        state = saved['ema']
        body = dict(saved, architecture=BODY_ARCHITECTURE,
                    ema={name:value for name,value in state.items() if name.startswith('net.')})
        owner = OnePassRuntime(body, device)
        self.net = owner.net
        self.label = owner.label
        self.config = owner.config
        # Keep the existing frozen body logic and expose net for graph constants.
        self.body = owner
        self.residual = nn.Conv2d(10, 5, 1, bias=False, dtype=torch.float32, device=device)
        with torch.no_grad():
            self.residual.weight.copy_(state['residual.weight'])
        self.eval().requires_grad_(False)

    def forward(self, noises, condition):
        with torch.autocast(device_type=noises.device.type, enabled=False):
            return self.body(noises, condition)+F.conv2d(noises.float(), self.residual.weight.float())


def load_residual_onepass(saved, device='cpu', masters=False):
    if (saved.get('architecture') != ARCHITECTURE or saved.get('solver_evaluations') != 1 or
            saved.get('teacher_solver_evaluations') != 2):
        raise ValueError('A native residual one-pass/T2-target checkpoint is required.')
    model = ResidualOnePassBase(saved['config'], saved['gain_anchor'])
    model.load_state_dict(saved['model'] if masters else saved['ema'])
    return model.to(device).eval()


def extend_optimizer_slots(saved, old_count, new_count):
    """Retain every Adam moment and append lazily initialized residual slots."""
    import copy
    state = copy.deepcopy(saved)
    if len(state['param_groups']) != 1 or len(state['param_groups'][0]['params']) != old_count:
        raise ValueError('Expected one ordered parent optimizer group.')
    if new_count < old_count:
        raise ValueError('Residual expansion cannot remove parent parameters.')
    old_ids = state['param_groups'][0]['params']
    if old_ids != list(range(old_count)):
        raise ValueError('Parent Adam parameter order is not contiguous.')
    state['param_groups'][0]['params'] = list(range(new_count))
    return state
