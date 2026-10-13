"""Zero-initialized world-phase adapters preserve the resumed field function."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from distill.field_base import FieldBase


def phase_features(origins,height,width,spacing,device,dtype):
    ys=torch.arange(height,device=device,dtype=torch.float32)*spacing
    xs=torch.arange(width,device=device,dtype=torch.float32)*spacing
    k=torch.arange(1,5,device=device,dtype=torch.float32)
    result=[]
    for y,x in origins:
        a=((ys+(y%64)).remainder(64))[:,None]*(2*math.pi/64)*k[None]
        b=((xs+(x%64)).remainder(64))[:,None]*(2*math.pi/64)*k[None]
        fy=torch.cat((a.sin(),a.cos()),dim=-1)[:,None].expand(height,width,8)
        fx=torch.cat((b.sin(),b.cos()),dim=-1)[None].expand(height,width,8)
        result.append(torch.cat((fy,fx),dim=-1))
    return torch.stack(result).to(dtype)


def add_phase_embedding(net,embedding,origins,height,width,spacing):
    # Actual field scenes use a spatial7x7 condition grid. Scalar conditions are
    # retained for the original64-window numerical control and do not add phase.
    if embedding.ndim==2:return embedding
    phase=phase_features(origins,height,width,spacing,embedding.device,embedding.dtype)
    return embedding+F.linear(phase,net.field_phase_embedding.to(embedding.dtype))


class FieldPhaseBase(FieldBase):
    def __init__(self,config):
        super().__init__(config)
        first=next(iter(self.net.enc.values()))
        self.net.register_parameter('field_phase_stem',nn.Parameter(torch.zeros(first.out_channels,16,1,1)))
        self.net.register_parameter('field_phase_embedding',nn.Parameter(torch.zeros(self.net.emb_channels,16)))

    def load_parent(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if set(missing)!={'net.field_phase_stem','net.field_phase_embedding'} or unexpected:
            raise ValueError('The parent may differ only by the two zero-initialized adapters.')
