"""Preserve terrain amplitude and depth at the teacher's valley positions.

Amplitude alone can reward misplaced noise. Teacher valley masks anchor the
extra depth term to the same pixels; ordinary position-sensitive losses remain.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from distill.decoded_loss import reconstructed_height
from distill.native_decoded import blur
from distill.native_multilod import low_height
from distill.terrain_shape import compare_shape


def amplitude_depth_terms(prediction, reference, sigmas, floor_m=1.):
    if prediction.shape != reference.shape or not math.isfinite(floor_m) or floor_m <= 0:
        raise ValueError('Matching fields and a positive finite amplitude floor required.')
    reference = reference.detach().float()
    prediction = prediction.float()
    amplitudes, depths = [], []
    for sigma in sigmas:
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('Positive finite morphology scales required.')
        margin = int(math.ceil(3*sigma))
        if min(reference.shape[-2:]) <= 2*margin:
            raise ValueError('Insufficient interior morphology support.')
        target = (reference-blur(reference,sigma))[...,margin:-margin,margin:-margin]
        actual = (prediction-blur(prediction,sigma))[...,margin:-margin,margin:-margin]
        land = reference[...,margin:-margin,margin:-margin] > 0
        if int(land.sum()) < 16:
            # Connected zero keeps gradients defined for sea-only crops.
            amplitudes.append(actual.sum()*0)
            depths.append(actual.sum()*0)
            continue
        target_values, actual_values = target[land], actual[land]
        target_power = target_values.square().mean()
        actual_power = actual_values.square().mean()
        denominator = target_power+floor_m**2
        amplitudes.append(torch.log((actual_power+floor_m**2)/denominator).square())
        cutoff = torch.quantile(target_values,.1).clamp_max(0.).detach()
        valleys = land & (target < cutoff)
        if valleys.any():
            depths.append((actual[valleys]-target[valleys]).square().mean()/denominator)
        else:
            depths.append(actual.sum()*0)
    if not amplitudes:
        raise ValueError('At least one morphology scale is required.')
    return dict(amplitude=torch.stack(amplitudes).mean(), valley_depth=torch.stack(depths).mean())


def amplitude_terms(result, region, decoder):
    with torch.no_grad():
        reference = reconstructed_height(decoder(region.target[None],region.decoder_noise),region.target[None])
    height = reconstructed_height(decoder(result['core'][None],region.decoder_noise),result['core'][None])
    fine = amplitude_depth_terms(height,reference,(2.,8.))
    native = amplitude_depth_terms(low_height(result['core']),low_height(region.target),(1.,2.,4.))
    return dict(height_amplitude=fine['amplitude'], height_valley_depth=fine['valley_depth'],
        low_height_amplitude=native['amplitude'], low_height_valley_depth=native['valley_depth'])


# Equal allocation across the two terrain scales and the two extra terms.
AMPLITUDE_WEIGHTS = dict(height_amplitude=.075, height_valley_depth=.075,
    low_height_amplitude=.075, low_height_valley_depth=.075)


def field_morphology(candidate, reference):
    rows = [row for row in compare_shape(candidate,reference)['topographic_position']
            if row['sigma_pixels'] in (2.,8.) and row['available'] and row['valley_pixels'] >= 16]
    if not rows:
        return None
    # An undefined candidate correlation (flat prediction) is failure, rather
    # than a reason to discard a teacher-supported landscape from the guard.
    return {key:float(np.mean([row[key] if row[key] is not None else 0. for row in rows]))
            for key in ('valley_iou','valley_retention','tpi_correlation','tpi_rms_ratio')}


@torch.no_grad()
def morphology_summary(result, region, decoder):
    heights=[]
    for latent in (result['core'],region.target):
        heights.append((reconstructed_height(decoder(latent[None],region.decoder_noise),latent[None])[0,0].cpu().numpy(),
                        low_height(latent)[0,0].cpu().numpy()))
    return dict(fine=field_morphology(heights[0][0],heights[1][0]),
                native=field_morphology(heights[0][1],heights[1][1]))


def morphology_degraded(groups, baseline_groups):
    for name, baseline in baseline_groups.items():
        current=groups[name]
        # Fixed baseline, not the previously selected checkpoint. IoU selection
        # penalizes false valleys; these guards also protect recall and position.
        for scale in ('fine','native'):
            before=baseline.get('morphology',{}).get(scale)
            if before is None:
                continue
            after=current.get('morphology',{}).get(scale)
            if after is None or after['valley_retention']+.01 < before['valley_retention'] or \
                    after['tpi_correlation']+.02 < before['tpi_correlation']:
                return True
    return False
