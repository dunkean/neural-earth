"""Direct teacher x0 detail targets with distinct objectives for the two phases.

No decoder or physical-height gradient participates in this objective. Band
energy is paired with position-sensitive error: noise with correct power must
not pass. Denominators depend only on the teacher and fixed TRAIN variance.
"""
import torch

from distill.native_decoded import blur

PHASE_LABELS = (1.5625, .609375)
WEIGHTS = dict(first_x0=.5, first_band_position=.1, first_band_energy=.05,
               second_height=.5, second_band_position=.1, second_band_energy=.05)


def x0_from_velocity(inputs, velocity, labels):
    # FP32 loss arithmetic keeps small-gradient objectives resolvable. Runtime
    # BF16 expression/fusion is evaluated independently, never replaced here.
    angle = labels.float()[:, None, None, None]
    return torch.cos(angle)*inputs.float()+torch.sin(angle)*velocity.float()


def band_terms(prediction, reference, variance):
    variance = variance.to(reference.device).reshape(1, -1).detach()
    floor = variance.clamp_min(1e-8)*1e-5
    position, energy = [], []
    for sigma in (1., 2., 4.):
        batch, channels = prediction.shape[:2]
        actual = (prediction-blur(prediction.reshape(-1, 1, *prediction.shape[-2:]), sigma)
                  .reshape_as(prediction))
        target = (reference-blur(reference.reshape(-1, 1, *reference.shape[-2:]), sigma)
                  .reshape_as(reference)).detach()
        margin = int(3*sigma)
        actual, target = actual[..., margin:-margin, margin:-margin], target[..., margin:-margin, margin:-margin]
        target_power = target.square().mean(dim=(-2, -1))
        actual_power = actual.square().mean(dim=(-2, -1))
        position.append(((actual-target).square().mean(dim=(-2, -1))/(target_power+floor)).mean())
        energy.append(torch.log((actual_power+floor)/(target_power+floor)).square().mean())
    return torch.stack(position).mean(), torch.stack(energy).mean()


def phase_loss(prediction, inputs, labels, target, variance):
    if prediction.shape != target.shape or prediction.shape != inputs.shape or prediction.shape[1] != 5:
        raise ValueError('Matching native five-channel fields required.')
    if min(prediction.shape[-2:]) <= 24:
        raise ValueError('Complete native windows required for interior band support.')
    first = torch.isclose(labels.float(), labels.new_tensor(PHASE_LABELS[0]).float(), atol=.004, rtol=0.)
    second = torch.isclose(labels.float(), labels.new_tensor(PHASE_LABELS[1]).float(), atol=.004, rtol=0.)
    if not first.any() or not second.any() or not (first | second).all():
        raise ValueError('Balanced native two-phase supervision required; unknown label rejected.')
    if variance.numel() != 5 or not torch.isfinite(variance).all() or not (variance > 0).all():
        raise ValueError('Five positive fixed TRAIN variances required.')
    target = target.detach()
    residual = prediction-target
    terms = dict(operator=torch.stack([residual[mask].square().mean() for mask in (first, second)]).mean())
    terms['operator_gradient'] = torch.stack([
        residual[mask].diff(dim=axis).square().mean() for mask in (first, second) for axis in (-1, -2)]).mean()
    actual, reference = x0_from_velocity(inputs, prediction, labels), x0_from_velocity(inputs, target, labels).detach()
    scale = variance.to(actual.device).detach().reshape(1, 5, 1, 1)
    terms['first_x0'] = ((actual[first, :4]-reference[first, :4]).square()/scale[:, :4]).mean()
    # Local teacher variation makes low plains visible to the height objective;
    # the TRAIN floor prevents infinite weights on constant patches.
    height_scale = reference[second, 4:5].var(dim=(-2, -1), unbiased=False, keepdim=True)
    height_scale = height_scale.clamp_min(scale[:, 4:5]*1e-3).detach()
    terms['second_height'] = ((actual[second, 4:5]-reference[second, 4:5]).square()/height_scale).mean()
    terms['first_band_position'], terms['first_band_energy'] = band_terms(actual[first, :4], reference[first, :4], variance[:4])
    terms['second_band_position'], terms['second_band_energy'] = band_terms(actual[second, 4:5], reference[second, 4:5], variance[4:5])
    loss = terms['operator']+.05*terms['operator_gradient']+sum(WEIGHTS[key]*terms[key] for key in WEIGHTS)
    return loss, terms
