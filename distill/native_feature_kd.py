"""Supervise matching encoder/decoder level outputs without normalizing away depth.

Only equal-width native networks are supported. The last block at each level
has the same semantic position despite a different number of retained blocks.
Hooks are scoped to one ordinary forward (never a checkpointed recomputation).
"""
from contextlib import contextmanager

import torch


def level_taps(net):
    result = {}
    for side in ('enc', 'dec'):
        for name, module in getattr(net, side).items():
            if '_block' in name:
                level, index = name.rsplit('_block', 1)
                key = f'{side}.{level}'
                previous = result.get(key)
                if previous is None or int(index) > previous[0]:
                    result[key] = (int(index), module)
    return {key: value[1] for key, value in result.items()}


@contextmanager
def capture_levels(net):
    values, handles = {}, []
    try:
        for name, module in level_taps(net).items():
            def capture(module, inputs, output, name=name):
                if name in values:
                    raise RuntimeError('Feature capture requires an ordinary, single forward.')
                values[name] = output
            handles.append(module.register_forward_hook(capture))
        yield values
    finally:
        for handle in handles:
            handle.remove()


def representation_loss(actual, reference):
    if not actual or actual.keys() != reference.keys():
        raise ValueError('Teacher and student must expose the same nonempty levels.')
    losses = {}
    for name, value in actual.items():
        target = reference[name].detach().float()
        if value.shape != target.shape:
            raise ValueError('Feature KD requires matching channel coordinates and shapes.')
        # Preserve amplitude as well as spatial/channel position. Per-channel
        # teacher energy balances levels; candidate features are never rescaled.
        energy = target.square().mean(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4)
        losses[name] = ((value.float()-target).square()/energy).mean()
    return torch.stack(list(losses.values())).mean(), losses


def feature_step(model, teacher, inputs, labels, condition, feature_weight=.5):
    with torch.no_grad(), capture_levels(teacher) as reference:
        target = teacher(inputs.to(torch.bfloat16), noise_labels=labels.to(torch.bfloat16),
                         conditional_inputs=[condition.to(torch.bfloat16)]).float()
    with capture_levels(model.net) as actual:
        prediction = model(inputs, labels, [condition])
    feature, levels = representation_loss(actual, reference)
    operator = (prediction-target).square().mean()
    gradient = sum((prediction.diff(dim=axis)-target.diff(dim=axis)).square().mean()
                   for axis in (-2, -1))
    loss = operator+.05*gradient+feature_weight*feature
    return loss, dict(operator=operator, gradient=gradient, representation=feature,
                     **{f'feature:{key}': value for key, value in levels.items()})
