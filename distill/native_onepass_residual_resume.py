"""Explicit extension of residual one-pass training without resetting Adam."""
import math


def validate_parent_contract(parent, current):
    for key in ('source_sha256', 'seed', 'solver_evaluations', 'teacher_solver_evaluations',
                'noise_offsets', 'ema_half_life', 'loss', 'sampler', 'teacher_revision'):
        if parent.get(key) != current.get(key):
            raise ValueError(f'Resume changes the parent recipe: {key}')
    old_code = parent['code_sha256']
    new_code = current['code_sha256']
    for name, digest in old_code.items():
        if name != 'native_onepass_residual_train.py' and new_code.get(name) != digest:
            raise ValueError(f'Resume changes a parent implementation: {name}')


def continuation_lr(step, start, end, initial, minimum=1e-6):
    if not 0 <= start <= step <= end or start >= end or not initial >= minimum > 0:
        raise ValueError('Invalid explicit continuation schedule.')
    progress = (step-start)/(end-start)
    return minimum+(initial-minimum)*.5*(1+math.cos(math.pi*progress))
