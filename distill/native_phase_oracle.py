"""Separate phase-one drift from phase-two fit on independent captured worlds.

All four configurations share noise, conditioning, fusion and frozen decoder.
Teacher-forced phase two uses the captured inputs (the training distribution).
Teacher/teacher replay controls the floating-point floor against stored targets.
This is an interior diagnostic, not an application or seam certification.
"""
import argparse
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA, REPO, REVISION, atomic_json, external_path
from distill.decoded_loss import reconstructed_height
from distill.native_base import NativeBase, teacher_source
from distill.native_decoded import DecodedRegion, OriginalDecoder, decoded_region_forward
from distill.native_fusion import forward_batches, window_outputs
from distill.native_multilod import low_height
from distill.native_multilod_eval import validation_records
from distill.terrain_shape import compare_shape


def phase_cores(student_result, region, student, teacher, weight):
    second = region.stages[1]
    cores = {'student_student': student_result['core']}
    for name, model, inputs in (
        ('student_teacher', teacher, student_result['inputs1']),
        ('teacher_student', student, second['inputs']),
        ('teacher_teacher', teacher, second['inputs']),
    ):
        velocity = forward_batches(model, inputs.to(region.dtype), second['labels'].to(region.dtype),
                                   second['condition'].to(region.dtype)).float()
        cores[name] = region.core_crop(window_outputs(inputs, velocity, second['labels'], region.dtype), weight)
    return cores


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/native-mixed-scenes-v2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coarse', choices=('original', 'coarse4', 'coarse2'), default='coarse4')
    parser.add_argument('--split', choices=('train', 'val'), default='val', help='Train is diagnostic only; never acceptance evidence.')
    parser.add_argument('--limit', type=int, help='Small smoke only; omitted evaluates all selected independent worlds.')
    args = parser.parse_args()
    root, output, checkpoint = (external_path(p) for p in (args.dataset, args.output, args.checkpoint))
    manifest = json.loads((root/'manifest.json').read_text())
    stats_path = DATA/'eval/native-decoded-probe/channel-statistics.json'
    independent_validation = validation_records(manifest, json.loads(stats_path.read_text()))
    pool = independent_validation if args.split == 'val' else [r for r in manifest['scenes'] if r['metadata']['split'] == 'train']
    records = [r for r in pool if r['coarse_variant'] == args.coarse]
    if json.loads((root/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Complete independent matched bank required.')
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('Positive smoke limit required.')
        if args.split == 'train':
            buckets = {}
            for record in sorted(records, key=lambda r: r['file']):
                buckets.setdefault((record['metadata']['profile'], record['case_kind']), []).append(record)
            records = [r for stripe in zip_longest(*buckets.values()) for r in stripe if r is not None]
        records = records[:args.limit]
    source = teacher_source()
    with (source/'diffusion_pytorch_model.safetensors').open('rb') as handle:
        teacher_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    if teacher_sha != manifest['teacher_weights_sha256']:
        raise ValueError('Captured bank and installed teacher differ.')
    with checkpoint.open('rb') as handle:
        checkpoint_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (saved.get('teacher_revision') or saved.get('dataset_manifest', {}).get('revision')) != REVISION:
        raise ValueError('Checkpoint teacher revision differs.')
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    student = NativeBase(saved['config']).cuda().eval().requires_grad_(False)
    student.load_state_dict(saved['ema'])
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    from types import SimpleNamespace
    from terrain_inference import prepare_models
    teacher = EDMUnet2D.from_pretrained(str(source)).cuda().bfloat16().eval().requires_grad_(False)
    prepare_models(SimpleNamespace(coarse_model=teacher, base_model=teacher, decoder_model=teacher))
    decoder = OriginalDecoder('cuda')
    weight = linear_weight_window(64, 'cuda', torch.float32)
    rows = []
    for index, record in enumerate(records):
        region = DecodedRegion(root/record['file'], 'cuda')
        result = decoded_region_forward(student, region, weight)
        reference = {
            3: low_height(region.target)[0, 0].cpu().numpy(),
            0: reconstructed_height(decoder(region.target[None], region.decoder_noise),
                                    region.target[None])[0, 0, 64:-64, 64:-64].cpu().numpy(),
        }
        for name, latent in phase_cores(result, region, student, teacher, weight).items():
            heights = {
                3: low_height(latent)[0, 0].cpu().numpy(),
                0: reconstructed_height(decoder(latent[None], region.decoder_noise),
                                        latent[None])[0, 0, 64:-64, 64:-64].cpu().numpy(),
            }
            for lod, actual in heights.items():
                shape = compare_shape(actual, reference[lod])
                tpi = next(t for t in shape['topographic_position'] if t['sigma_pixels'] == 2.)
                rows.append(dict(configuration=name, lod=lod, file=record['file'],
                    kind=record['case_kind'], seed=record['metadata']['seed'],
                    profile=record['metadata']['profile'],
                    mae_m=float(np.abs(actual-reference[lod]).mean()),
                    coast_disagreement=float(np.mean((actual > 0) != (reference[lod] > 0))),
                    tpi_correlation=tpi.get('tpi_correlation'),
                    tpi_rms_ratio=tpi.get('tpi_rms_ratio'), valley_retention=tpi.get('valley_retention')))
        if index % 10 == 0 or index+1 == len(records):
            print(json.dumps(dict(completed=index+1, total=len(records))), flush=True)
    summary = []
    for name in dict.fromkeys(r['configuration'] for r in rows):
        for lod in (3, 0):
            group = [r for r in rows if r['configuration'] == name and r['lod'] == lod and r['kind'] is not None]
            summary.append(dict(configuration=name, lod=lod, rare_count=len(group), **{
                key: float(np.median([r[key] for r in group if r[key] is not None])) if any(r[key] is not None for r in group) else None
                for key in ('mae_m', 'tpi_correlation', 'tpi_rms_ratio', 'valley_retention')}))
    atomic_json(output, dict(status='complete', step=saved['step'], checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha, teacher_sha256=teacher_sha, coarse=args.coarse,
        decoder='original', records=len(records), smoke=args.limit is not None,
        split=args.split, validation_evidence=args.split == 'val' and args.limit is None,
        dataset_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summary=summary, rows=rows, accepted=False,
        note='Interior phase diagnostic; teacher-forced inputs are captured; no app/seam/performance acceptance.'))


if __name__ == '__main__':
    main()
