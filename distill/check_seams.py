"""Compare a trained base field under two tile partitions, including its halo.

The numerical test cannot certify the appearance of physical terrain. Its
report leaves visual review pending; inspect the physical comparison sheets
before using it as acceptance evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from distill.common import DATA, REPO, atomic_json, external_path
from distill.dataset import Crops
from distill.student import load_student


@torch.inference_mode()
def compare_partition(model, inputs, core=128, autocast=True):
    halo = model.halo
    height, width = inputs.shape[-2]-2*halo, inputs.shape[-1]-2*halo
    if min(height, width, core) <= 0 or height % core or width % core or core % 32:
        raise ValueError('The useful field must divide into positive, 32-aligned cores.')
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
        full = model(inputs[None])[0, :, halo:halo+height, halo:halo+width].float()
        tiled = torch.empty_like(full)
        for y in range(0, height, core):
            for x in range(0, width, core):
                patch = inputs[:, y:y+core+2*halo, x:x+core+2*halo]
                tiled[:, y:y+core, x:x+core] = model(patch[None])[0, :, halo:halo+core, halo:halo+core].float()
    if not torch.isfinite(full).all() or not torch.isfinite(tiled).all():
        raise FloatingPointError('Non-finite tiled or full field.')
    difference = (tiled-full).abs()
    seam = torch.zeros(height, width, device=device, dtype=torch.bool)
    jump_errors = []
    for x in range(core, width, core):
        seam[:, x-1:x+1] = True
        jump_errors.append(((tiled[:, :, x]-tiled[:, :, x-1])-
                            (full[:, :, x]-full[:, :, x-1])).abs().flatten())
    for y in range(core, height, core):
        seam[y-1:y+1, :] = True
        jump_errors.append(((tiled[:, y]-tiled[:, y-1])-
                            (full[:, y]-full[:, y-1])).abs().flatten())
    if not jump_errors:
        raise ValueError('At least one internal seam is required.')
    jumps = torch.cat(jump_errors)
    return dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
                seam_mean_abs=float(difference[:, seam].mean()),
                interior_mean_abs=float(difference[:, ~seam].mean()),
                max_jump_error=float(jumps.max()), per_channel_max=difference.flatten(1).max(1).values.cpu().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/main')
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--core', type=int, default=128)
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Absolute tolerance in normalized latent units.')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or not 0 < args.tolerance <= 1e-4:
        parser.error('Positive sample count and tolerance at most 1e-4 required.')
    torch.set_num_threads(4)
    device = 'cpu' if args.cpu else 'cuda:0'
    payload = args.checkpoint.read_bytes()
    model, saved = load_student(io.BytesIO(payload), device)
    if model.config.stage != 'base':
        parser.error('This partition test applies to the non-blended base field.')
    dataset = Crops(args.dataset, 'base', args.split, halo=model.halo)
    count = min(args.count, len(dataset))
    indices = torch.linspace(0, len(dataset)-1, count).round().long().tolist()
    rows = []
    for index in indices:
        inputs, _, _ = dataset[index]
        values = compare_partition(model, inputs, args.core)
        values.update(file=dataset.paths[index].name,
                      passed=values['max_abs'] <= args.tolerance and values['max_jump_error'] <= 2*args.tolerance)
        rows.append(values)
        print(json.dumps(values), flush=True)
    report = dict(stage='base', checkpoint_digest=hashlib.sha256(payload).hexdigest(), step=saved['step'],
                  student_source_digests={
                      'code:'+name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                      for name in ('distill/student.py', 'distill/features.py', 'distill/inference.py', 'distill/coarse_solver.py')},
                  device=device, gpu=None if args.cpu else torch.cuda.get_device_name(),
                  dtype='bf16 autocast with FP32 output heads', core=args.core, halo=model.halo,
                  tolerance=args.tolerance, split=args.split, rows=rows,
                  numerical_passed=all(row['passed'] for row in rows),
                  visual_review=dict(passed=False, reason='Physical terrain sheets have not been reviewed.'),
                  passed=False)
    atomic_json(external_path(args.output), report)
    raise SystemExit(0 if report['numerical_passed'] else 1)


if __name__ == '__main__':
    main()
