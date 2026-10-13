"""Low-frequency-only auxiliary: detached teacher Jacobian, full TT target."""
from pathlib import Path

import torch

from distill.common import HOLDOUT_SEEDS
from distill.native_decoded import DecodedRegion, blur, decoded_region_forward


def level_terms(actual, target, variance, floors):
    if actual.shape != target.shape or actual.shape != (1, 5, 64, 64):
        raise ValueError('Complete, paired64x64 native cores required.')
    reference = target.detach().float()
    q_actual = actual[:, 4:5].float()*38.6-31.4
    q_reference = reference[:, 4:5]*38.6-31.4
    delta = blur(q_actual-q_reference, 8.)
    jacobian = (2*blur(q_reference, 8.).abs().clamp_min(1.)).detach()
    meters = ((delta*jacobian)[..., 16:48, 16:48].square().mean() /
              floors['height_lf_rms_m']**2)
    channels = blur((actual[:, :4].float()-reference[:, :4]).reshape(4, 1, 64, 64), 8.)
    channel_mse = channels[..., 16:48, 16:48].square().mean((-3, -2, -1))
    tau = actual.new_tensor(floors['c0_c3_lf_normalized_rms'], dtype=torch.float32)
    features = (channel_mse/variance[:4].detach()/tau.square()).mean()
    return dict(level=meters+.1*features, height_lf=meters, features_lf=features)


def full_level_fields(model, teacher, record, noise_seed):
    # Dedicated fresh objects: never mutate the main KD sampler's cache.
    region = DecodedRegion(Path(record['root'])/record['file'], 'cuda')
    if region.metadata['split'] != 'train' or region.metadata['seed'] in HOLDOUT_SEEDS:
        raise ValueError('Independent TRAIN scenes only.')
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    region.refresh_noise(teacher, weight, noise_seed)
    # Gradient through BOTH student stages; TT target independent of the pupil.
    actual = decoded_region_forward(model, region, weight, checkpointed=True)['core']
    return actual[None], region.target.detach()[None]


def protect_scalar_gradients(model):
    names = []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 0:
            if parameter.grad is not None:
                parameter.grad.zero_()
            names.append(name)
    return names
