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
CHANNEL0_WEIGHTS = dict(first_x0=.5, first_channel0=1.875,
    first_band_position=.1, first_band_energy=.05,
    second_height=.05, second_band_position=.01, second_band_energy=.005)


def x0_from_velocity(inputs, velocity, labels):
    # FP32 loss arithmetic keeps small-gradient objectives resolvable. Runtime
    # BF16 expression/fusion is evaluated independently, never replaced here.
    angle = labels.float()[:, None, None, None]
    return torch.cos(angle)*inputs.float()+torch.sin(angle)*velocity.float()


def band_terms(prediction, reference, variance, train_power=None):
    variance = variance.to(reference.device).reshape(1, -1).detach()
    floor = variance.clamp_min(1e-8)*1e-5
    position, energy = [], []
    if train_power is not None:
        train_power = train_power.to(reference.device).detach()
        if train_power.shape != (3, prediction.shape[1]) or not torch.isfinite(train_power).all() or not (train_power > 0).all():
            raise ValueError('Positive fixed TRAIN powers per band and channel required.')
    for index, sigma in enumerate((1., 2., 4.)):
        batch, channels = prediction.shape[:2]
        actual = (prediction-blur(prediction.reshape(-1, 1, *prediction.shape[-2:]), sigma)
                  .reshape_as(prediction))
        target = (reference-blur(reference.reshape(-1, 1, *reference.shape[-2:]), sigma)
                  .reshape_as(reference)).detach()
        margin = int(3*sigma)
        actual, target = actual[..., margin:-margin, margin:-margin], target[..., margin:-margin, margin:-margin]
        target_power = target.square().mean(dim=(-2, -1))
        actual_power = actual.square().mean(dim=(-2, -1))
        denominator = target_power+floor if train_power is None else train_power[index][None]
        energy_floor = floor if train_power is None else denominator
        position.append(((actual-target).square().mean(dim=(-2, -1))/denominator).mean())
        # TRAIN power stabilizes the ratio; the target remains THIS window's
        # teacher power. Flat teachers must never be asked to add average noise.
        energy.append(torch.log((actual_power+energy_floor)/(target_power+energy_floor)).square().mean())
    return torch.stack(position).mean(), torch.stack(energy).mean()


def phase_loss(prediction, inputs, labels, target, variance, calibration=None):
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
    height_scale = (reference[second, 4:5].var(dim=(-2, -1), unbiased=False, keepdim=True)
                    .clamp_min(scale[:, 4:5]*1e-3).detach() if calibration is None else
                    calibration['second_variance'][4].to(actual.device).detach())
    if not torch.isfinite(height_scale).all() or not (height_scale > 0).all():
        raise ValueError('Positive fixed TRAIN spatial variance required.')
    terms['second_height'] = ((actual[second, 4:5]-reference[second, 4:5]).square()/height_scale).mean()
    if calibration is None:
        terms['first_band_position'], terms['first_band_energy'] = band_terms(actual[first, :4], reference[first, :4], variance[:4])
        terms['second_band_position'], terms['second_band_energy'] = band_terms(actual[second, 4:5], reference[second, 4:5], variance[4:5])
        weights = WEIGHTS
    else:
        terms['first_channel0'] = ((actual[first, :1]-reference[first, :1]).square()/scale[:, :1]).mean()
        terms['first_band_position'], terms['first_band_energy'] = band_terms(actual[first, :1], reference[first, :1], variance[:1], calibration['first_power'][:, :1])
        terms['second_band_position'], terms['second_band_energy'] = band_terms(actual[second, 4:5], reference[second, 4:5], variance[4:5], calibration['second_power'][:, 4:5])
        weights = CHANNEL0_WEIGHTS
    loss = terms['operator']+.05*terms['operator_gradient']+sum(weights[key]*terms[key] for key in weights)
    return loss, terms
