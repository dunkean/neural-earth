"""Common, read-only paired validation for the finite shoreline loss trial.

This is an interior height proxy, not full physical acceptance. All audited
validation windows are measured; no reserved evaluation seed enters training.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from distill.common import DATA, REPO, atomic_json, external_path
from distill.decoded_loss import PairedCrops, decode, frozen_decoder, reconstructed_height, shore_loss
from distill.student import load_student
from distill.train import centre


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(checkpoint, decoder_path, audit, output):
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    payload = Path(checkpoint).read_bytes()
    model, saved = load_student(io.BytesIO(payload), device)
    if model.config.stage != 'base' or saved['step'] <= 0:
        raise ValueError('A trained base checkpoint is required.')
    decoder, decoder_identity = frozen_decoder(decoder_path, device)
    dataset = PairedCrops(saved['arguments']['dataset'], 'val', model.halo, audit)
    rows = []
    with torch.inference_mode():
        for index, path in enumerate(dataset.paths):
            inputs, target, mask, gaussian, residual, valid = (v[None].to(device) for v in dataset[index])
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                prediction = centre(model(inputs), target)
                decoded = decode(decoder, prediction, gaussian)
            ph = reconstructed_height(decoded, prediction)
            th = reconstructed_height(residual, target)
            valid = valid[..., 64:-64, 64:-64]
            if not torch.isfinite(ph).all() or not torch.isfinite(th).all():
                raise ValueError('Non-finite proxy heights.')
            near = (valid > 0) & (th.abs() <= 20.)
            coast_kl, coast_disagreement = shore_loss(ph, th, valid)
            valid_count, near_count = int((valid > 0).sum()), int(near.sum())
            if not valid_count:
                raise ValueError('An audited validation window has no valid interior.')
            error = (ph-th).abs()
            def slope(height):
                mx = (valid[..., 1:] > 0) & (valid[..., :-1] > 0)
                my = (valid[..., 1:, :] > 0) & (valid[..., :-1, :] > 0)
                return ((height.diff(dim=-1).abs()*mx).sum() +
                        (height.diff(dim=-2).abs()*my).sum())/(mx.sum()+my.sum()).clamp_min(1)
            rows.append(dict(file=path.name, valid_pixels=valid_count, near_pixels=near_count,
                height_mae_m_proxy=float((error*valid).sum()/valid.sum().clamp_min(1)),
                near_error_sum_m=float(error[near].sum()),
                near_wrong_pixels=int((((ph > 0) != (th > 0)) & near).sum()),
                coast_kl=float(coast_kl), coast_balanced_disagreement=float(coast_disagreement),
                slope_ratio_proxy=float(slope(ph)/slope(th).clamp_min(1e-8))))
    coastal = [row for row in rows if row['near_pixels']]
    if not coastal:
        raise ValueError('No coastal validation windows; primary metric undefined.')
    near_count = sum(row['near_pixels'] for row in coastal)
    metrics = dict(coast_balanced_disagreement=sum(r['coast_balanced_disagreement'] for r in coastal)/len(coastal),
        coast_kl=sum(r['coast_kl'] for r in coastal)/len(coastal),
        near_mae_m_proxy=sum(r['near_error_sum_m'] for r in coastal)/near_count,
        near_pixel_disagreement=sum(r['near_wrong_pixels'] for r in coastal)/near_count,
        height_mae_m_proxy=sum(r['height_mae_m_proxy'] for r in rows)/len(rows))
    result = dict(metrics=metrics, rows=rows, windows=len(rows), coastal_windows=len(coastal),
        checkpoint=dict(path=str(Path(checkpoint).resolve()), sha256=hashlib.sha256(payload).hexdigest(),
                        step=saved['step'], weights='EMA'), decoder=decoder_identity,
        audit_sha256=fingerprint(audit), dataset_manifest_sha256=fingerprint(dataset.root/'manifest.json'),
        code_sha256={name: fingerprint(REPO/'distill'/name) for name in
                     ('shore_probe.py', 'decoded_loss.py', 'student.py', 'features.py')},
        accepted=False, note='Validation-only interior proxy; full physical views and continuity are still required.')
    atomic_json(external_path(output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--selection', type=Path, help='JSON specifying checkpoint, decoder, audit and output.')
    parser.add_argument('--decoder', type=Path, default=DATA/'ckpt/candidates/decoder-step200000-3a6303b22b45.pt')
    parser.add_argument('--audit', type=Path, default=DATA/'ckpt/base-decoded/paired-audit.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.selection:
        selected = json.loads(args.selection.read_text())
        args.checkpoint, args.decoder, args.audit, args.output = (
            Path(selected[key]) for key in ('checkpoint', 'decoder', 'audit', 'output'))
    if not args.checkpoint or not args.output:
        parser.error('Checkpoint and output required, directly or through a selection.')
    result = measure(args.checkpoint, args.decoder, args.audit, args.output)
    for name, value in result['metrics'].items():
        print(f'METRIC {name}={value:.12g}', flush=True)
    print(f'ARTIFACT shore_validation={args.output}', flush=True)


if __name__ == '__main__':
    main()
