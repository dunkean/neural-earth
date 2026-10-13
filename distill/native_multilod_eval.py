"""Evaluate native EMA checkpoints on the frozen, matched three-coarse bank.

Uses both physical scales and position-sensitive rare-case morphology. This
interior proxy does not replace full pipeline, cold seams or visual acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from distill.common import DATA, HOLDOUT_SEEDS, REPO, REVISION, atomic_json, external_path
from distill.native_base import NativeBase, teacher_source
from distill.native_decoded import OriginalDecoder
from distill.native_decoded_eval import shape_validation
from distill.native_multilod_train import validate


def validation_records(manifest, statistics):
    train = [row for row in manifest['scenes'] if row['metadata']['split'] == 'train']
    records = [row for row in manifest['scenes'] if row['metadata']['split'] == 'val']
    train_seeds = {row['metadata']['seed'] for row in train}
    val_seeds = {row['metadata']['seed'] for row in records}
    if not train or not records or train_seeds & val_seeds or (train_seeds | val_seeds) & HOLDOUT_SEEDS:
        raise ValueError('Matched validation must contain independent, reserved-seed-free worlds.')
    if statistics['split'] != 'train' or statistics['dataset_manifest_sha256'] != manifest['original_manifest_sha256']:
        raise ValueError('Use the fixed original TRAIN variance for every coarse branch.')
    return records


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/native-mixed-scenes-v2')
    parser.add_argument('--statistics', type=Path, default=DATA/'eval/native-decoded-probe/channel-statistics.json')
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-job')
    parser.add_argument('--minimum-completed-step', type=int)
    parser.add_argument('--rare-shape', action='store_true')
    args = parser.parse_args()
    for name in ('dataset', 'statistics', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.checkpoints = [path.expanduser().resolve() for path in args.checkpoints]
    if args.wait_job:
        from distill.run_training import wait_job
        wait_job(args.wait_job)
    root = external_path(args.dataset)
    manifest = json.loads((root/'manifest.json').read_text())
    if hashlib.sha256((teacher_source()/'diffusion_pytorch_model.safetensors').read_bytes()).hexdigest() != manifest['teacher_weights_sha256']:
        raise ValueError('The matched bank and installed teacher weights differ.')
    if json.loads((root/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Finish the matched bank before evaluating.')
    records = validation_records(manifest, json.loads(args.statistics.read_text()))
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    decoder = OriginalDecoder('cuda')
    variance = torch.tensor(json.loads(args.statistics.read_text())['variance'], device='cuda')
    results = []
    for path in args.checkpoints:
        if args.minimum_completed_step is not None:
            status = json.loads((path.parent/'status.json').read_text())
            if status['status'] != 'complete' or status['step'] < args.minimum_completed_step:
                raise ValueError('Training did not complete the requested palier.')
        payload = path.read_bytes()
        saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        if (saved.get('teacher_revision') or saved.get('dataset_manifest', {}).get('revision')) != REVISION:
            raise ValueError('Native checkpoint uses a different or missing teacher revision.')
        model = NativeBase(saved['config']).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['ema'])
        details = validate(model, records, root, weight, decoder, variance,
            preserve_valleys=bool(saved.get('contract',{}).get('preserve_valleys',False)))
        rare = [row for row in records if row['case_kind'] is not None]
        shapes = shape_validation(model, [root/row['file'] for row in rare], weight, decoder) if args.rare_shape else None
        if shapes is not None:
            for record, row in zip(rare, shapes):
                row.update(coarse_variant=record['coarse_variant'], case_kind=record['case_kind'],
                    file=record['file'])
        results.append(dict(checkpoint=str(path), sha256=hashlib.sha256(payload).hexdigest(),
            step=saved['step'], rare_shapes=shapes, **details))
        print(json.dumps(dict(step=saved['step'], selection_score=details['selection_score'],
            feature_relative_mse=details['feature_relative_mse'])), flush=True)
        del model, saved, payload
        torch.cuda.empty_cache()
    atomic_json(external_path(args.output), dict(status='complete', gpu=torch.cuda.get_device_name(),
        dataset_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        statistics_sha256=hashlib.sha256(args.statistics.read_bytes()).hexdigest(),
        decoder_sha256=hashlib.sha256((teacher_source().parent/'decoder_model/diffusion_pytorch_model.safetensors').read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in (
            'distill/native_multilod_eval.py', 'distill/native_multilod_train.py',
            'distill/native_multilod.py', 'distill/native_decoded_eval.py', 'distill/native_decoded.py',
            'distill/native_base.py', 'distill/terrain_shape.py')},
        results=results, accepted=False,
        note='Matched three-coarse validation interior proxy; full pipeline, cold seams and visual acceptance pending.'))


if __name__ == '__main__':
    main()
