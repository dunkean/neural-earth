"""Four first-phase heads, identified by world64 window-origin parity."""
import copy

import numpy as np
import torch
from torch import nn

from distill.field_phase import FieldPhaseBase
from distill.native_fusion import window_outputs


def is_head(name):
    return name=='net.field_window_gains' or name.startswith('net.field_window_heads.')


def head_weights(origins,height,width,device):
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    window=linear_weight_window(64,device,torch.float32)
    result=[]
    for y,x in origins:
        ys=torch.arange(height,device=device)+y;xs=torch.arange(width,device=device)+x
        result.append(torch.stack([window[(ys-32*py).remainder(64)[:,None],
                                         (xs-32*px).remainder(64)[None,:]]
                                  for py in (0,1) for px in (0,1)]))
    return torch.stack(result)[:,:,None]


def combine_heads(values,origins):
    weights=head_weights(origins,*values.shape[-2:],values.device)
    return (values.float()*weights).sum(dim=1)/weights.sum(dim=1)


@torch.no_grad()
def recorded_heads(path,scene):
    with np.load(path,allow_pickle=False) as sample:
        ids=np.flatnonzero(sample['labels']==sample['labels'].max())
        origins=[tuple(int(v)*32 for v in sample['contexts'][i,1:]) for i in ids]
        values={k:torch.from_numpy(sample[k][ids].copy()).to(scene.initial.device)
                for k in ('inputs','target','labels')}
    outputs=window_outputs(values['inputs'],values['target'],values['labels'],scene.dtype)
    target=outputs.new_zeros((4,5,256,256));mask=outputs.new_zeros((4,1,256,256))
    for value,(y,x) in zip(outputs,origins):
        index=2*((y//32)%2)+(x//32)%2;dy,dx=y-scene.origin[0],x-scene.origin[1]
        if not (0<=dy<=192 and 0<=dx<=192):raise ValueError('Unshifted native49 first windows required.')
        if mask[index,:,dy:dy+64,dx:dx+64].any():raise ValueError('Same-parity windows must not overlap.')
        target[index,:,dy:dy+64,dx:dx+64]=value;mask[index,:,dy:dy+64,dx:dx+64]=1
    if not mask[:,:,32:224,32:224].all():raise ValueError('All four central192 targets must be complete.')
    blended=combine_heads(target[None],[scene.origin])[0,:,32:224,32:224]
    if float((blended-scene.initial).square().mean())>1e-8:
        raise ValueError('World-parity heads do not reconstruct the exact initial192 target.')
    return target,mask


def head_loss(predicted,target,mask,variance):
    error=(predicted-target[None]).square()/variance[None,None,:,None,None]
    per_head=(error*mask[None]).sum((2,3,4))/(5*mask.sum((1,2,3))[None])
    return per_head.mean(),per_head


class FieldHeadsBase(FieldPhaseBase):
    def __init__(self,config):
        super().__init__(config)
        self.net.field_window_heads=nn.ModuleList([copy.deepcopy(self.net.out_conv) for _ in range(4)])
        self.net.register_parameter('field_window_gains',nn.Parameter(self.net.out_gain.detach().expand(4).clone()))

    def load_parent(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if set(missing)!={n for n in self.state_dict() if is_head(n)} or unexpected:
            raise ValueError('Only the four new output heads may differ from the phase parent.')
        with torch.no_grad():
            for head in self.net.field_window_heads:head.load_state_dict(self.net.out_conv.state_dict())
            self.net.field_window_gains.copy_(self.net.out_gain.expand(4))

    def forward(self,inputs,noise_labels,conditions,origins,centres,return_features=False,return_head_velocities=False):
        if not return_head_velocities:
            return super().forward(inputs,noise_labels,conditions,origins,centres,return_features)
        self.net.eval();dtype=torch.bfloat16 if inputs.is_cuda else torch.float32
        state={name:value.to(dtype) for name,value in list(self.net.named_parameters())+list(self.net.named_buffers())}
        with torch.autocast(device_type=inputs.device.type,enabled=False):
            result,features,heads=torch.func.functional_call(self.net,state,(inputs.to(dtype),),dict(
                noise_labels=noise_labels.to(dtype),conditions=conditions.to(dtype),origins=origins,centres=centres,
                return_features=return_features,return_head_velocities=True))
        return result.float(),features,heads.float()
