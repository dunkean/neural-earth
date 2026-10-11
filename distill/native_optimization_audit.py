"""TRAIN-only task-gradient and source-relative movement diagnostics."""
import numpy as np
import torch

from distill.native_diverse_windows import diverse_batch
from distill.native_phase_loss import CHANNEL0_WEIGHTS, phase_loss


def block(name):
    for prefix, label in (('net.enc.512x512', 'encoder512'), ('net.dec.64x64', 'decoder64'),
                          ('net.dec.512x512', 'decoder512'), ('net.out_', 'output')):
        if name.startswith(prefix):
            return label
    return 'other'


def gradient_audit(model, records, root, variance, calibration, batches=16):
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    rows = []
    for index in range(batches):
        generator = np.random.default_rng(170030+index)
        batch, selected = diverse_batch(records, root, generator, index)
        prediction = model(batch['inputs'], batch['labels'], [batch['condition']])
        _, terms = phase_loss(prediction, batch['inputs'], batch['labels'], batch['target'], variance, calibration)
        first = sum(CHANNEL0_WEIGHTS[key]*terms[key] for key in
                    ('first_channel0', 'first_band_position', 'first_band_energy'))
        second = sum(CHANNEL0_WEIGHTS[key]*terms[key] for key in
                     ('second_height', 'second_band_position', 'second_band_energy'))
        first_grad = torch.autograd.grad(first, [p for _, p in parameters], retain_graph=True, allow_unused=True)
        second_grad = torch.autograd.grad(second, [p for _, p in parameters], allow_unused=True)
        sums = {name: torch.zeros(3, device='cuda', dtype=torch.float64)
                for name in ('global', 'encoder512', 'decoder64', 'decoder512', 'output', 'other')}
        for (name, _), a, b in zip(parameters, first_grad, second_grad):
            values = torch.stack((a.square().sum() if a is not None else first.new_zeros(()),
                                  b.square().sum() if b is not None else first.new_zeros(()),
                                  (a*b).sum() if a is not None and b is not None else first.new_zeros(()))).double()
            sums['global'] += values; sums[block(name)] += values
        details = {}
        for name, values in sums.items():
            aa, bb, ab = values.cpu().tolist()
            details[name] = dict(first_norm=aa**.5, second_norm=bb**.5,
                cosine=ab/(aa*bb)**.5 if aa > 0 and bb > 0 else None)
        rows.append(dict(index=index, worlds=[r['metadata']['seed'] for _, r in selected], groups=details))
        del first_grad, second_grad, prediction, terms, first, second
    return rows


@torch.no_grad()
def movement(model, reference):
    total, delta, changed, count = 0., 0., 0., 0
    outputs = []
    for name, parameter in model.named_parameters():
        original = reference[name].to(parameter.device)
        aa = float(original.square().sum()); dd = float((parameter-original).square().sum())
        cc = float((parameter.bfloat16() != original.bfloat16()).sum())
        total += aa; delta += dd; changed += cc; count += parameter.numel()
        if name == 'net.out_conv.weight':
            for channel in range(5):
                a, b = original[channel], parameter[channel]
                outputs.append(dict(channel=channel,
                    relative_update=float((a-b).square().sum().sqrt()/a.square().sum().sqrt()),
                    changed_bf16_fraction=float((a.bfloat16() != b.bfloat16()).float().mean())))
    return dict(relative_update=(delta/total)**.5, changed_bf16_fraction=changed/count, output_channels=outputs)


def morphology_guard(candidate, baseline):
    """Stop for supported raw regression; small groups remain diagnostic warnings."""
    before = {row['file']: row for row in baseline['rows']}
    groups = {}
    for after in candidate['rows']:
        prior = before[after['file']]
        name = after['coarse_variant']+':'+(after['case_kind'] or after['profile'])
        for scale in ('fine', 'native'):
            a, b = prior.get('morphology', {}).get(scale), after.get('morphology', {}).get(scale)
            if a is None:
                continue
            if b is None:
                return [dict(group=name, scale=scale, reason='defined source morphology became undefined')]
            for metric, tolerance in (('valley_retention', .01), ('tpi_correlation', .02)):
                key = (name, scale, metric, tolerance)
                groups.setdefault(key, {}).setdefault(after['seed'], []).append(a[metric]-b[metric])
    reasons = []
    rng = np.random.default_rng(17004)
    for (name, scale, metric, tolerance), worlds in sorted(groups.items()):
        values = np.array([np.mean(v) for v in worlds.values()])
        if len(values) < 3:
            if values.mean() > .10:
                reasons.append(dict(group=name, scale=scale, metric=metric, reason='large raw regression', mean_drop=float(values.mean())))
            continue
        means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
        lower = float(np.quantile(means, .05))
        if lower > tolerance:
            reasons.append(dict(group=name, scale=scale, metric=metric, worlds=len(values),
                                lower95_drop=lower, tolerance=tolerance))
    return reasons
