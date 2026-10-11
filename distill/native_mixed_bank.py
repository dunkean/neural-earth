"""Capture matched original-base targets with original, coarse4 and coarse2.

World splits and rare locations are frozen before capture. Every coarse branch
uses the same original T2 base for its target, on its own conditioning. Original
scenes are hard-linked without altering existing provenance or their metadata.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal

import numpy as np
import torch

from distill.common import DATA, REPO, REVISION, HOLDOUT_SEEDS, atomic_json, external_path, source_digest
from distill.native_base import teacher_source
from distill.native_bank import SceneCapture

KINDS = ('desert-plain', 'temperate-plain', 'wet-complex-coast', 'low-plain-sea')
VARIANTS = ('original', 'coarse4', 'coarse2')
STOP = False


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stop(signum, frame):
    global STOP
    STOP = True


def validate_inventory(rows):
    worlds = {split: {row['metadata']['seed'] for row in rows if row['metadata']['split'] == split}
              for split in ('train', 'val')}
    if worlds['train'] & worlds['val'] or (worlds['train'] | worlds['val']) & HOLDOUT_SEEDS:
        raise ValueError('Mixed bank train, validation or reserved worlds overlap.')
    addresses = [(r['metadata']['seed'], r['metadata']['profile'], r['metadata']['y'],
                  r['metadata']['x']) for r in rows]
    if len(addresses) != len(set(addresses)):
        raise ValueError('A source location occurs more than once in the mixed bank.')
    for split in ('train', 'val'):
        kinds = {r['case_kind'] for r in rows if r['metadata']['split'] == split and r['case_kind']}
        if kinds != set(KINDS):
            raise ValueError(f'Physically qualified {split} cases must cover all four categories.')
    return worlds


def inventory(original, rare):
    for root in (original, rare):
        if json.loads((root/'status.json').read_text())['status'] != 'complete':
            raise ValueError('Both original and rare captures must be complete.')
    rows = []
    manifest = json.loads((original/'manifest.json').read_text())
    for index, item in enumerate(manifest['scenes']):
        path = original/f'{item["metadata"]["split"]}-{index:04d}.npz'
        with np.load(path, allow_pickle=False) as sample:
            meta = json.loads(str(sample['metadata']))
        rows.append(dict(source=str(path), source_sha256=digest(path), metadata=meta, case_kind=None))
    qualification = json.loads((rare/'qualification.json').read_text())
    for row in qualification['rows']:
        if not row['physical_qualified']:
            continue
        path = rare/row['file']
        with np.load(path, allow_pickle=False) as sample:
            meta = json.loads(str(sample['metadata']))
        if meta['seed'] != row['seed'] or meta['split'] != row['split']:
            raise ValueError('Rare qualification and captured metadata disagree.')
        rows.append(dict(source=str(path), source_sha256=digest(path), metadata=meta,
                         case_kind=row['requested'], climate_archetype=row['climate_archetype']))
    validate_inventory(rows)
    for row in rows:
        meta = row['metadata']
        if meta['native_size'] != 128 or meta['y'] % 32 or meta['x'] % 32:
            raise ValueError('Complete original128 scenes on the base32 grid required.')
        if meta['teacher_revision'] != REVISION:
            raise ValueError('Source scenes use another teacher revision.')
    return rows


def make_plan(original, rare):
    rows = inventory(original, rare)
    weights = {}
    bundled = REPO/'distill/weights'
    weight_manifest = json.loads((bundled/'manifest.json').read_text())
    for variant in VARIANTS[1:]:
        entry = weight_manifest['exports'][variant]
        path = bundled/entry['file']
        if digest(path) != entry['sha256']:
            raise ValueError('Coarse inference weights differ from their manifest.')
        if variant == 'coarse2' and (entry['training_status'] != 'complete' or entry['last_training_step'] != 200000):
            raise ValueError('Finish coarse2 200k before recapturing base targets.')
        weights[variant] = dict(path=str(path), sha256=entry['sha256'], selected_step=entry['training_step'])
    scenes = []
    for index, row in enumerate(rows):
        for variant in VARIANTS:
            scenes.append(row | dict(coarse_variant=variant,
                file=f'{row["metadata"]["split"]}-{index:04d}-{variant}.npz'))
    return dict(schema=3, revision=REVISION, scenes=scenes, coarse_weights=weights,
        original_manifest_sha256=digest(original/'manifest.json'),
        rare_manifest_sha256=digest(rare/'manifest.json'),
        rare_qualification_sha256=digest(rare/'qualification.json'),
        teacher_weights_sha256=digest(teacher_source()/'diffusion_pytorch_model.safetensors'),
        teacher_code=source_digest(), size=128, stride=32, complete_contexts=True,
        note='Same original base teacher, same coarse branch for teacher/student. Rare categories refer to original-teacher qualification; no held-out test worlds.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'capture'))
    parser.add_argument('--original', type=Path, default=DATA/'crops/native-scenes-v1')
    parser.add_argument('--rare', type=Path, default=DATA/'crops/native-rare-scenes-v1')
    parser.add_argument('--output', type=Path, default=DATA/'crops/native-mixed-scenes-v2')
    args = parser.parse_args()
    root = external_path(args.output)
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        plan = json.loads(manifest_path.read_text())
    else:
        plan = make_plan(args.original, args.rare)
        atomic_json(manifest_path, plan)
    if args.action == 'plan':
        print(json.dumps(dict(scenes=len(plan['scenes']), variants=VARIANTS)), flush=True)
        return
    if plan['teacher_code'] != source_digest() or plan['teacher_weights_sha256'] != digest(teacher_source()/'diffusion_pytorch_model.safetensors'):
        raise ValueError('Original teacher changed after mixed-bank planning.')
    for entry in plan['coarse_weights'].values():
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('A frozen coarse branch changed after planning.')
    contract = dict(manifest_sha256=digest(manifest_path), gpu=torch.cuda.get_device_name(),
        producer_sha256={name:digest(REPO/name) for name in ('distill/native_mixed_bank.py',
            'distill/native_bank.py', 'distill/native_base.py', 'distill/teacher.py',
            'distill/inference.py', 'distill/coarse_solver.py', 'distill/student.py')})
    contract_path = root/'capture-contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Mixed capture producer changed; use a separate bank.')
    atomic_json(contract_path, contract)
    from tools.verification.compare_base_variants import Count
    from distill.teacher import Worlds
    from distill.inference import install
    from distill.student import load_student
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    worlds = Worlds(plan['scenes'][0]['metadata']['seed'])
    models = {name:load_student(entry['path'], 'cuda')[0] for name,entry in plan['coarse_weights'].items()}
    for index, item in enumerate(plan['scenes']):
        if STOP:
            break
        path = root/item['file']
        if path.exists():
            continue
        variant = item['coarse_variant']
        if variant == 'original':
            source = Path(item['source'])
            if digest(source) != item['source_sha256']:
                raise ValueError('Original captured scene changed after planning.')
            os.link(source, path)
        else:
            meta = dict(item['metadata'], coarse_variant=variant,
                        coarse_weights_sha256=plan['coarse_weights'][variant]['sha256'])
            world = worlds.create(meta['seed'], meta['profile'])
            install(world, dict(coarse=models[variant]))
            world.bind()
            recorder = SceneCapture(world.base_model, path, meta)
            world.base_model = recorder
            original = world._latent_inference

            def capture(ctxs, *values, **options):
                recorder.contexts = ctxs
                return original(ctxs, *values, **options)

            world._latent_inference = capture
            try:
                y, x = meta['y'], meta['x']
                with torch.inference_mode():
                    field = world.latents[:,y:y+128,x:x+128]
                    initial = world.latents.args[0][:,y-32:y+160,x-32:x+160]
                recorder.save_scene(field, initial)
                if len(recorder.rows) != 74:
                    raise ValueError('Unexpected native128 capture dependencies.')
            finally:
                world.close()
                del world, recorder, original
                gc.collect()
                torch.cuda.empty_cache()
        if index % 30 == 0 or index+1 == len(plan['scenes']):
            atomic_json(root/'status.json', dict(status='running', completed=index+1, target=len(plan['scenes'])))
            print(json.dumps(dict(completed=index+1, target=len(plan['scenes']), variant=variant)), flush=True)
    present = sum((root/row['file']).is_file() for row in plan['scenes'])
    atomic_json(root/'status.json', dict(status='interrupted' if STOP else 'complete',
        completed=present, target=len(plan['scenes']), train_shards=sum(row['metadata']['split']=='train' for row in plan['scenes']),
        val_shards=sum(row['metadata']['split']=='val' for row in plan['scenes']), accepted=False))
    if not STOP and present != len(plan['scenes']):
        raise RuntimeError('Mixed native bank is incomplete.')


if __name__ == '__main__':
    main()
