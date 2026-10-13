"""Position-sensitive physical objectives for the compact-base recovery.

These losses are prepared separately from the running channel-removal recipe.
Their coefficients must be calibrated on TRAIN gradients before recovery.
"""
import math

import torch

from distill.native_auxiliary_training import root_coast
from distill.native_phase_loss import PHASE_LABELS, x0_from_velocity


def height_terms(actual, target, scale_m=1.):
    if actual.shape != target.shape or actual.ndim != 4 or actual.shape[1] != 5:
        raise ValueError('Matching five-channel native fields are required.')
    if not math.isfinite(scale_m) or scale_m <= 0:
        raise ValueError('A positive fixed TRAIN height scale is required.')
    reference = target.detach().float()
    q_target = reference[:, 4:5]*38.6-31.4
    q_actual = actual[:, 4:5].float()*38.6-31.4
    # Detached teacher Jacobian: a matched position and amplitude is required.
    # No blur or stand-alone power objective can hide uncorrelated grain.
    jacobian = 2*q_target.abs().clamp_min(1.)
    linearized = (q_actual-q_target)*jacobian
    exact = q_actual.sign()*q_actual.square()-q_target.sign()*q_target.square()
    return dict(height_jacobian_mse=(linearized/scale_m).square().mean(),
                height_exact_mse=(exact/scale_m).square().mean(),
                height_mae_m=exact.abs().mean())


def x0_height_terms(prediction, target, batch, scale_m=1.):
    labels = batch['labels'].float()
    first, second = [torch.isclose(labels, labels.new_tensor(value), atol=.004, rtol=0.)
                     for value in PHASE_LABELS]
    if not first.any() or not second.any() or not (first | second).all():
        raise ValueError('Both pinned native phases are required.')
    actual = x0_from_velocity(batch['inputs'], prediction, labels)
    reference = x0_from_velocity(batch['inputs'], target.detach(), labels).detach()
    phases = [height_terms(actual[mask], reference[mask], scale_m) for mask in (first, second)]
    # Each phase receives equal weight, independent of incidental batch counts.
    result = {key:torch.stack([terms[key] for terms in phases]).mean() for key in phases[0]}
    result.update(first_height_jacobian_mse=phases[0]['height_jacobian_mse'],
                  second_height_jacobian_mse=phases[1]['height_jacobian_mse'])
    return result


def field_terms(actual, target, scale_m=1.):
    result = height_terms(actual, target, scale_m)
    result['coast'], result['coast_disagreement'], result['coast_counts'] = root_coast(actual, target)
    return result
