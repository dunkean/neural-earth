"""Idle BF16 numerical-wrapper timings to choose a compression curriculum.

This uses the same NativeBase/NativeLowrankBase path as the training controls.
It is not the application's cached inference path or a deployment speed claim.
"""
import argparse
import gc
import hashlib
import json
import time

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, external_path
from distill.native_base import NativeBase, teacher_source
from distill.native_lowrank import NativeLowrankBase
from terrain_diffusion.models.edm_unet import EDMUnet2D


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basis', default=str(DATA/'eval/native-lowrank-fp64-basis.pt'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.basis, map_location='cpu', weights_only=True)
    bank = DATA/'crops/native-mixed-scenes-v2'
    row = next(r for r in json.loads((bank/'manifest.json').read_text())['scenes']
               if r['metadata']['split'] == 'train' and r['coarse_variant'] == 'coarse4')
    with np.load(bank/row['file']) as sample:
        inputs = torch.from_numpy(sample['inputs'][:args.batch].copy()).cuda()
        labels = torch.from_numpy(sample['labels'][:args.batch].copy()).cuda()
        cond = torch.from_numpy(sample['condition'][:args.batch].copy()).cuda()
    results = []
    for fraction in (None, 1., 5/6, .75, 2/3, .5):
        if fraction is None:
            source = EDMUnet2D.from_pretrained(str(teacher_source()))
            model = NativeBase(source.config)
            model.net.load_state_dict(source.state_dict()); del source
        else:
            model = NativeLowrankBase.from_basis(saved, fraction)
        model = model.cuda().eval().requires_grad_(False)
        def forward():
            return model(inputs, labels, [cond])
        for _ in range(5):
            output = forward()
        if not torch.isfinite(output).all():
            raise FloatingPointError('Nonfinite timing forward.')
        torch.cuda.synchronize()
        samples = []
        for _ in range(3):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                forward()
            end.record(); torch.cuda.synchronize()
            samples.append(start.elapsed_time(end)/args.iterations/len(inputs))
        result = dict(rank_fraction=fraction, params=sum(p.numel() for p in model.parameters()),
            milliseconds_per_window=float(np.median(samples)), repeats=samples, batch=len(inputs))
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, output; gc.collect(); torch.cuda.empty_cache()
    reference = results[0]['milliseconds_per_window']
    for result in results:
        result['speedup_vs_same_wrapper'] = reference/result['milliseconds_per_window']
    atomic_json(external_path(args.output), dict(status='complete', gpu=torch.cuda.get_device_name(),
        finished_at=time.time(), producer_sha256={name: hashlib.sha256((REPO/'distill'/name).read_bytes()).hexdigest()
        for name in ('native_lowrank_speed.py', 'native_lowrank.py', 'native_base.py')},
        results=results, accepted=False, note=__doc__))


if __name__ == '__main__':
    main()
