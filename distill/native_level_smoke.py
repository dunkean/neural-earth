"""Exact A/B-zero state control and three real scalar-runtime forward controls."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from distill.common import atomic_json, external_path
from distill.native_auxiliary_smoke import cpu_state, equal
from distill.native_level_train import Engine, code_hashes, configure
from distill.native_scalar_precision import load_scalar_precision, runtime_scalar_model
from distill.native_weight_quant import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'calibration', 'floors', 'moments', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve prior smoke proof.')
    configure()
    source = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    moments = torch.load(args.moments, map_location='cpu', weights_only=True, mmap=True)
    calibration, floors = json.loads(args.calibration.read_text()), json.loads(args.floors.read_text())
    if calibration['source_sha256'] != file_sha(args.source) or moments['source_sha256'] != file_sha(args.source):
        raise ValueError('Smoke sources differ.')
    rows, runtime_checks = {}, []
    reference = None
    scalar_saved = None
    runtime_batches = []
    for mode in ('a', 'zero', 'b'):
        engine = Engine(source, calibration, floors, moments, 'a' if mode == 'a' else 'b', zero=mode == 'zero')
        started = time.monotonic()
        updates = [engine.update() for _ in range(3)]
        rows[mode] = dict(updates=updates, elapsed_seconds=time.monotonic()-started,
                          teacher_frozen=all(p.grad is None for p in engine.teacher.parameters()))
        if mode == 'a':
            reference = cpu_state(engine)
        elif mode == 'zero':
            if not equal(reference, cpu_state(engine)):
                raise ValueError('A and B-zero raw/EMA/Adam/RNG must be identical.')
            if not all(a['loss'] == b['loss'] and a['grad_norm'] == b['grad_norm']
                       for a, b in zip(rows['a']['updates'], updates, strict=True)):
                raise ValueError('A and B-zero regular losses/gradients differ.')
            del reference
        else:
            scalar_saved = {key: source[key] for key in ('config', 'scheme', 'block', 'quantized_layers')}
            scalar_saved.update(architecture='native-weight-quant-scalar', scalar_precision='anchored-fp32-v1',
                gain_anchor=engine.anchor, ema={n: v.detach().to(dtype=torch.float32 if n == 'net.out_gain'
                else torch.bfloat16, device='cpu') for n, v in engine.ema.state_dict().items()})
            with torch.no_grad(), engine.model.cached_weights():
                for step in (78000, 78001, 78002):
                    batch, _ = engine.bank.batch(engine.records, np.random.default_rng(121+step),
                                                step, engine.model, on_policy=step%2 == 0)
                    runtime_batches.append({name: batch[name].cpu() for name in ('inputs', 'labels', 'condition')})
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    owner = load_scalar_precision(scalar_saved, 'cuda')
    runtime = runtime_scalar_model(scalar_saved, 'cuda')
    with torch.no_grad(), owner.cached_weights():
        for index, batch in enumerate(runtime_batches):
            batch = {name: value.cuda() for name, value in batch.items()}
            actual = owner(batch['inputs'], batch['labels'], [batch['condition']])
            baked = runtime(batch['inputs'].bfloat16(), noise_labels=batch['labels'].bfloat16(),
                            conditional_inputs=[batch['condition'].bfloat16()]).float()
            if not torch.equal(actual, baked):
                raise ValueError('Scalar snapshot QAT and baked runtime differ.')
            runtime_checks.append(dict(index=index, max_difference=0., windows=len(actual),
                                       gain_dtype=str(runtime.out_gain.dtype)))
    atomic_json(output, dict(status='complete', source_sha256=file_sha(args.source),
        calibration_sha256=file_sha(args.calibration), floors_sha256=file_sha(args.floors),
        moments_sha256=file_sha(args.moments), code_sha256=code_hashes(), rows=rows,
        zero_auxiliary_full_path_exact=True, runtime_checks=runtime_checks,
        gpu=torch.cuda.get_device_name(), smoke_updates_only=True, served_weights_changed=False))


if __name__ == '__main__':
    main()
