"""Paired objectives on complete native final fields; no change to the100k run."""
import math

import torch

from distill.decoded_loss import shore_loss
from distill.native_decoded import blur


def low_height(core):
    if core.ndim!=4 or core.shape[1]!=5:raise ValueError('Complete five-channel final fields required.')
    q=core[:,4:5].float()*38.6-31.4
    return q.sign()*q.square()


def low_field_terms(prediction,target,*,flat_scene=False,flat_power_m2,
                    band_m=20.,temperature_m=2.):
    """Use final fused fields, not individual noisy diffusion-window outputs.

    flat_power_m2 must be fixed from TRAIN teacher fields before the experiment.
    High-pass error matches the same teacher pixels; no stand-alone energy target.
    This lowLOD proxy does not certify the final decodedLOD0 coastline.
    """
    if prediction.shape!=target.shape:raise ValueError('Paired final field shapes must agree.')
    if not math.isfinite(flat_power_m2) or flat_power_m2<=0:
        raise ValueError('Fixed positive TRAIN flat-scene power required.')
    if min(prediction.shape[-2:])<=6:raise ValueError('Interior support required.')
    actual,reference=low_height(prediction),low_height(target.detach()).detach()
    shore,disagreement=shore_loss(actual,reference,torch.ones_like(reference),band_m,temperature_m)
    flat=actual.sum()*0
    count=0
    if flat_scene:
        actual_band=(actual-blur(actual,1.))[...,3:-3,3:-3]
        target_band=(reference-blur(reference,1.))[...,3:-3,3:-3]
        land=(reference[...,3:-3,3:-3]>0).float()
        counts=land.sum((-3,-2,-1));active=(counts>0).float()
        per_window=((actual_band-target_band).square()*land).sum((-3,-2,-1))
        per_window=per_window/counts.clamp_min(1)/flat_power_m2
        flat=(per_window*active).sum()/active.sum().clamp_min(1)
        count=int(counts.sum())
    return dict(coast=shore,flat_highpass=flat,coast_disagreement=disagreement,
                height_mae_m=(actual-reference).abs().mean(),flat_land_pixels=count)
