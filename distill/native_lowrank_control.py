"""Measure teacher-basis factorization before considering a new training recipe."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write, external_path
from distill.native_base import NativeBase, teacher_source
from distill.native_decoded import DecodedRegion, OriginalDecoder
from distill.native_feature_train import balanced_windows
from distill.native_lowrank import NativeLowrankBase
from distill.native_multilod_eval import validation_records
from distill.native_multilod_train import digest, validate
from terrain_diffusion.models.edm_unet import EDMUnet2D


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check'))
    parser.add_argument('--basis', type=Path, default=DATA/'eval/native-lowrank-full-basis.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fractions', type=float, nargs='+', default=[1.])
    parser.add_argument('--mode', choices=('composed', 'factorized'), default='factorized')
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--wait-job', help='Wait for a verified successful basis/control job before loading CUDA.')
    args = parser.parse_args()
    if args.wait_job:
        from distill.run_training import wait_job
        wait_job(args.wait_job)
    output, basis_path = external_path(args.output), external_path(args.basis)
    source = teacher_source()
    teacher_sha = digest(source/'diffusion_pytorch_model.safetensors')
    torch.set_num_threads(8); torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    teacher = EDMUnet2D.from_pretrained(str(source)).cuda().eval().requires_grad_(False)
    producers = {f'distill/{name}': digest(REPO/'distill'/name) for name in
                 ('native_lowrank.py', 'native_lowrank_control.py', 'native_progress_images.py',
                  'native_base.py', 'native_multilod_train.py', 'native_feature_train.py', 'native_decoded.py')}
    if args.action == 'prepare':
        if basis_path.exists():
            raise ValueError('Keep an existing basis; do not overwrite the diagnostic.')
        model = NativeLowrankBase.from_teacher(teacher, 1.).eval().requires_grad_(False)
        reconstruction = {}
        for name in model.ranks:
            target = teacher.get_submodule(name).weight
            actual = model.net.get_submodule(name).composed_weight()
            reconstruction[name] = dict(relative_mse=float((actual-target).square().mean()/target.square().mean()),
                                        max_abs=float((actual-target).abs().max()))
        saved = dict(config=dict(teacher.config), ema={k: v.cpu() for k, v in model.state_dict().items()},
            architecture='native-lowrank', factor_ranks=model.ranks, factor_mode='factorized', rank_fraction=1.,
            step=0, teacher_revision=REVISION, teacher_weights_sha256=teacher_sha,
            evaluation_only=True, inference_only=True, producer_sha256=producers,
            note='Full teacher basis, no training. Factorized inference is a numerical control, not an accepted student.')
        atomic_write(basis_path, lambda handle: torch.save(saved, handle))
        atomic_json(output, dict(status='complete', basis=str(basis_path), basis_sha256=digest(basis_path),
            teacher_weights_sha256=teacher_sha, teacher_revision=REVISION, gpu=torch.cuda.get_device_name(),
            layers=len(model.ranks), parameters=sum(p.numel() for p in model.parameters()),
            source_parameters=sum(p.numel() for p in teacher.parameters()), decomposition_dtype='float64',
            reconstruction=reconstruction, producer_sha256=producers, accepted=False))
        print(json.dumps(dict(basis=str(basis_path), layers=len(model.ranks))), flush=True)
        return

    saved = torch.load(basis_path, map_location='cpu', weights_only=True)
    if saved['teacher_revision'] != REVISION or saved['teacher_weights_sha256'] != teacher_sha:
        raise ValueError('The basis and installed teacher differ.')
    dataset = DATA/'crops/native-mixed-scenes-v2'
    manifest = json.loads((dataset/'manifest.json').read_text())
    stats = json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())
    if manifest['teacher_weights_sha256'] != teacher_sha or json.loads((dataset/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Complete matched bank with this teacher required.')
    records = validation_records(manifest, stats)
    # First six world/case triples, all coarse branches; no fitting or selection.
    fixed = records[:18]
    reference = NativeBase(saved['config']).cuda().eval().requires_grad_(False)
    reference.net.load_state_dict(teacher.state_dict())
    del teacher
    decoder, weight, baseline = None, None, None
    rare = [r for r in records if r['case_kind']]
    variance = torch.tensor(stats['variance'], device='cuda')
    if args.physical:
        from terrain_diffusion.inference.world_pipeline import linear_weight_window
        decoder = OriginalDecoder('cuda'); weight = linear_weight_window(64, 'cuda', torch.float32)
        baseline = validate(reference, rare, dataset, weight, decoder, variance, True)
    results = []
    for fraction in args.fractions:
        model = NativeLowrankBase.from_basis(saved, fraction, args.mode).cuda().eval().requires_grad_(False)
        rows, fp32_rows = [], []
        for index, record in enumerate(fixed):
            batch = balanced_windows(DecodedRegion(dataset/record['file'], 'cuda'), np.random.default_rng(18210+index))
            target = reference(batch['inputs'], batch['labels'], [batch['condition']])
            actual = model(batch['inputs'], batch['labels'], [batch['condition']])
            residual = actual-target
            rows.append(dict(file=record['file'], coarse_variant=record['coarse_variant'],
                relative_mse=float(residual.square().mean()/target.square().mean().clamp_min(1e-12)),
                channel_mse=torch.stack([residual[:2].square().mean((0, 2, 3)), residual[2:].square().mean((0, 2, 3))]).cpu().tolist(),
                reference_channel_power=torch.stack([target[:2].square().mean((0, 2, 3)), target[2:].square().mean((0, 2, 3))]).cpu().tolist()))
            if fraction == 1. and index < 3:
                target32 = reference.net(batch['inputs'].float(), batch['labels'].float(), [batch['condition'].float()])
                actual32 = model.net(batch['inputs'].float(), batch['labels'].float(), [batch['condition'].float()])
                error = actual32-target32
                fp32_rows.append(dict(file=record['file'], relative_mse=float(error.square().mean()/target32.square().mean().clamp_min(1e-12)),
                    max_abs=float(error.abs().max()),
                    channel_mse=torch.stack([error[:2].square().mean((0, 2, 3)), error[2:].square().mean((0, 2, 3))]).cpu().tolist(),
                    reference_channel_power=torch.stack([target32[:2].square().mean((0, 2, 3)), target32[2:].square().mean((0, 2, 3))]).cpu().tolist()))
        fp32_pass = all(r['relative_mse'] < 1e-6 and r['max_abs'] < .002 for r in fp32_rows) if fp32_rows else None
        if fp32_rows and not fp32_pass:
            atomic_json(output, dict(status='failed', reason='Full-rank FP32 control failed', fp32_rows=fp32_rows, accepted=False))
            raise ValueError('Full-rank algebra control failed; do not interpret compression quality.')
        details = validate(model, rare, dataset, weight, decoder, variance, True) if args.physical else None
        label = f'lowrank{round(1000*fraction):04d}-{args.mode}'
        checkpoint = output.parent/f'{label}-control.pt'
        atomic_write(checkpoint, lambda handle: torch.save(dict(saved,
            ema={k: v.cpu() for k, v in model.state_dict().items()}, factor_ranks=model.ranks,
            factor_mode=args.mode, rank_fraction=fraction, producer_sha256=producers), handle))
        if args.physical:
            subprocess.run([sys.executable, '-m', 'distill.native_progress_images', '--checkpoints', str(checkpoint),
                            '--label', label], cwd=REPO, check=True)
        results.append(dict(rank_fraction=fraction, mode=args.mode, checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint), parameters=sum(p.numel() for p in model.parameters()),
            operator_rows=rows, fp32_rows=fp32_rows, fp32_control_pass=fp32_pass, physical=details,
            mean_relative_mse=float(np.mean([r['relative_mse'] for r in rows]))))
        atomic_json(output, dict(status='running', gpu=torch.cuda.get_device_name(), results=results,
            baseline=baseline, producer_sha256=producers, accepted=False))
        print(json.dumps(dict(fraction=fraction, mean_relative_mse=results[-1]['mean_relative_mse'])), flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
    atomic_json(output, dict(status='complete', gpu=torch.cuda.get_device_name(), results=results, baseline=baseline,
        basis_sha256=digest(basis_path), dataset_manifest_sha256=digest(dataset/'manifest.json'),
        teacher_weights_sha256=teacher_sha, producer_sha256=producers, accepted=False,
        note='Untrained factorization controls only; no deployment, cold seams or idle speed acceptance.'))


if __name__ == '__main__':
    main()
