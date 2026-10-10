"""Initialize a wider local student from a frozen EMA, with recorded provenance.

Pad learned channels and compensate PixelNorm, including concatenated U-Net
skips. Small new weights activate the additional capacity. This is a new trial
with a fresh optimizer; it never overwrites the source checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import io
from pathlib import Path

import torch

from distill.common import atomic_json, atomic_write, external_path
from distill.student import Student, StudentConfig


def expanded_config(config, width):
    if config.solver_steps or width <= config.width or width % 16:
        raise ValueError('Widening requires a larger local convolutional student width.')
    return replace(config, width=width, norm_eps=config.norm_eps * config.width / width)


def expand_state(old, new, state, jitter=1e-4):
    if (new.config.stage != old.config.stage or new.config.depth != old.config.depth or
            new.config.dilations != old.config.dilations):
        raise ValueError('Widening preserves the stage, depth and receptive field.')
    result = new.state_dict()
    old_widths = [old.config.width * min(2**i, 2) for i in range(old.config.depth)]
    new_widths = [new.config.width * min(2**i, 2) for i in range(new.config.depth)]
    for name, source in state.items():
        dest = result[name]
        if dest.shape == source.shape:
            dest.copy_(source)
            continue
        dest.zero_()
        if source.ndim == 1:
            dest[:source.shape[0]].copy_(source)
            continue
        if source.ndim != 4:
            raise ValueError(f'Unsupported widened parameter: {name}')
        if jitter:
            dest.normal_(std=jitter)
        inputs = torch.arange(source.shape[1])
        if name.startswith('decoders.') and ('.layers.1.' in name or '.skip.' in name):
            index = int(name.split('.')[1])
            a, b = old_widths[-1-index], old_widths[-2-index]
            new_a = new_widths[-1-index]
            inputs = torch.cat((torch.arange(a), torch.arange(b) + new_a))
        outputs = torch.arange(source.shape[0])
        scale = (source.shape[1] / dest.shape[1]) ** .5 if ('.layers.1.' in name or '.layers.4.' in name) else 1.
        dest[outputs[:, None], inputs[None, :]] = source * scale
    return result


def initialize(checkpoint, output, width=128, seed=314159):
    output = external_path(output)
    if (output/'latest.pt').exists():
        raise ValueError('Widening initializes a new trial; its checkpoint already exists.')
    payload = Path(checkpoint).read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    source_copy = output/'source.pt'
    atomic_write(source_copy, lambda handle: handle.write(payload))
    cfg = dict(saved['config'])
    cfg['dilations'] = tuple(cfg['dilations'])
    old_config = StudentConfig(**cfg)
    torch.manual_seed(seed)
    old, new = Student(old_config), Student(expanded_config(old_config, width))
    expanded = expand_state(old, new, saved['ema'])
    new.load_state_dict(expanded)
    optimizer = torch.optim.AdamW(new.parameters(), lr=2e-4, betas=(.9, .99), weight_decay=1e-4, fused=True)
    provenance = dict(source_checkpoint=str(source_copy), source_sha256=hashlib.sha256(payload).hexdigest(),
                      source_step=saved['step'], source_width=old_config.width, width=width, seed=seed,
                      jitter=1e-4, optimizer_reset=True, source_weights='EMA',
                      widening_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      student_code_sha256=hashlib.sha256(Path(__file__).with_name('student.py').read_bytes()).hexdigest(),
                      note='New architecture trial; later resumes retain this trial optimizer/RNG/sampler.')
    state = dict(saved, config=asdict(new.config), model=expanded, ema=expanded,
                 optimizer=optimizer.state_dict(), step=0, best_score=float('inf'), validation=None,
                 rng=torch.get_rng_state(), warm_start=provenance)
    state['arguments'] = dict(saved['arguments'], width=width, output=str(output))
    atomic_write(output/'latest.pt', lambda handle: torch.save(state, handle))
    atomic_json(output/'warm-start.json', provenance)
    atomic_json(output/'status.json', dict(status='initialized', step=0, accepted=False))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--width', type=int, default=128)
    args = parser.parse_args()
    print(initialize(args.checkpoint, args.output, args.width))


if __name__ == '__main__':
    main()
