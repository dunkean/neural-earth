"""Absolute height level against each scene's variation, with TRAIN-scale floors."""
import torch

from distill.native_decoded import blur


def level_height_loss(actual,target,floor_m=10.):
    if actual.shape!=target.shape or actual.ndim!=4 or actual.shape[1]!=5:
        raise ValueError('Matching five-channel fields required.')
    if floor_m<=0:raise ValueError('A positive fixed height floor required.')
    reference=target[:,4:5].detach().float()*38.6-31.4
    height=reference.sign()*reference.square()
    delta=(actual[:,4:5].float()*38.6-31.4-reference)*(2*reference.abs().clamp_min(1.))
    smooth=torch.cat([blur(value[None],8.) for value in delta])
    scale=height[...,16:48,16:48].std(dim=(-2,-1),unbiased=False,keepdim=True).clamp_min(floor_m)
    return ((smooth[...,16:48,16:48]/scale).square().mean())
