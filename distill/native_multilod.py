"""Native-height morphology plus frozen original-decoder physical supervision.

LOD3 consumes channel4 directly, so decoded relief alone cannot supervise it.
Both height scales are balanced by teacher relief, including low plains; fixed
train-only feature variance and on-policy operator anchors resist decoder tricks.
"""
from __future__ import annotations

import math

import torch

from distill.decoded_loss import reconstructed_height, relative_height_gradient_loss
from distill.native_decoded import blur
from distill.native_fusion import forward_batches, overlap_loss, window_outputs

PHYSICAL_WEIGHTS = dict(height_relative=1., low_height_relative=1.,
    height_gradient=.3, low_height_gradient=.5, height_tpi=.15, low_height_tpi=.35)
REGULARIZATION_WEIGHTS = dict(normalized_latent=.3, feature_relative_mse=.4,
    operator=.05, overlap=.02)


def low_height(latent):
    q = latent[4:5].float()*38.6-31.4
    return (q.sign()*q.square())[None]


def relative_tpi(prediction, reference, sigmas, floor_m=1.):
    if prediction.shape != reference.shape or floor_m <= 0 or not math.isfinite(floor_m):
        raise ValueError('Matching fields and a positive finite shape floor required.')
    losses = []
    for sigma in sigmas:
        margin = int(math.ceil(3*sigma))
        if min(reference.shape[-2:]) <= 2*margin:
            raise ValueError('The field is smaller than the interior morphology support.')
        actual = (prediction-blur(prediction,sigma))[...,margin:-margin,margin:-margin]
        target = (reference-blur(reference,sigma))[...,margin:-margin,margin:-margin].detach()
        scale = target.square().mean().clamp_min(floor_m**2)
        losses.append((actual-target).square().mean()/scale)
    return torch.stack(losses).mean()


def height_terms(prediction, target, sigmas, gradient_floor):
    target = target.detach()
    mae = (prediction-target).abs().mean()
    return dict(mae_m=mae, relative=mae/target.std(unbiased=False).clamp_min(10.),
        gradient=relative_height_gradient_loss(prediction,target,torch.ones_like(target),floor_m=gradient_floor),
        tpi=relative_tpi(prediction,target,sigmas))


def physical_terms(result, region, decoder, variance):
    with torch.no_grad():
        reference = reconstructed_height(decoder(region.target[None],region.decoder_noise),region.target[None])
    height = reconstructed_height(decoder(result['core'][None],region.decoder_noise),result['core'][None])
    fine = height_terms(height,reference,(2.,8.,32.),.02)
    native = height_terms(low_height(result['core']),low_height(region.target),(1.,2.,4.,8.),.1)
    errors = (result['core']-region.target).square()/variance.to(height.device).clamp_min(1e-3)[:,None,None]
    return dict(height_mae_m=fine['mae_m'], low_height_mae_m=native['mae_m'],
        height_relative=fine['relative'], low_height_relative=native['relative'],
        height_gradient=fine['gradient'], low_height_gradient=native['gradient'],
        height_tpi=fine['tpi'], low_height_tpi=native['tpi'],
        normalized_latent=errors.mean(), feature_relative_mse=errors[:4].mean())


def physical_score(terms):
    return sum(weight*terms[name] for name,weight in PHYSICAL_WEIGHTS.items())


def multilod_loss(result, region, decoder, teacher, variance):
    terms = physical_terms(result,region,decoder,variance)
    first,second = region.stages
    with torch.no_grad():
        target = forward_batches(teacher,result['inputs1'].detach().to(region.dtype),
            second['labels'].to(region.dtype),second['condition'].to(region.dtype)).float()
    terms['operator'] = (result['velocity0']-first['target']).square().mean()+\
                        (result['velocity1']-target).square().mean()
    terms['overlap'] = overlap_loss(result['output1'],window_outputs(result['inputs1'].detach(),target,
        second['labels'],region.dtype),region.ref_origins)
    loss = physical_score(terms)+sum(weight*terms[name] for name,weight in REGULARIZATION_WEIGHTS.items())
    return loss,terms
