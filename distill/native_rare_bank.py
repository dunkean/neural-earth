"""Select and capture independent native rare scenes from teacher-only proposals.

The old large teacher bank proposes locations. Each proposal is requalified on
the aligned64-latent core, then recaptured with complete native T2 dependencies.
Existing native scenes and live trainer contracts are never changed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import signal

import numpy as np
import torch

from distill.common import (DATA, REPO, REVISION, HOLDOUT_SEEDS, atomic_json,
                            external_path, source_digest)
from distill.native_bank import SceneCapture
from distill.native_base import teacher_source
from distill.native_coverage import audit
from distill.rare_cases import field_stats, qualifies, relief_stats

BUCKETS = (('desert-plain', 'warm-arid'), ('temperate-plain', 'mild-temperate'),
           ('wet-complex-coast', None), ('low-plain-sea', None))
STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_candidates(path, kinds):
    """No student and no reserved test world participates in proposal ranking."""
    with np.load(path, allow_pickle=False) as sample:
        meta = json.loads(str(sample['metadata']))
        if meta['seed'] in HOLDOUT_SEEDS or meta['split'] not in ('train', 'val'):
            raise ValueError('Rare native proposals must exclude the reserved test worlds.')
        target, coarse = sample['target'], sample['coarse']
        y, x, crop = meta['y'], meta['x'], meta['crop']
        if y % 32 or x % 32 or target.shape[-2:] != (crop, crop):
            raise ValueError('Teacher proposal geometry changed.')
        rows = []
        source_sha256 = digest(path)
        for cy in range(((y+95)//96)*96, y+crop-63, 96):
            for cx in range(((x+95)//96)*96, x+crop-63, 96):
                oy, ox = cy-y, cx-x
                q = target[4, oy:oy+64, ox:ox+64].astype(np.float64)*38.6-31.4
                height = np.sign(q)*q**2
                iy, ix = cy//32-meta['coarse_y'], cx//32-meta['coarse_x']
                climate = coarse[2:, iy:iy+2, ix:ix+2].astype(np.float64)
                if climate.shape != (4, 2, 2):
                    raise ValueError('Complete climate cells for the aligned core required.')
                fields = np.concatenate((height[None], climate.repeat(32, 1).repeat(32, 2)))
                c, s = field_stats(fields), relief_stats(height, 3)
                if c is None:
                    continue
                for kind, archetype in BUCKETS:
                    if kind not in kinds or not qualifies(kind, s, c, archetype):
                        continue
                    score = (-s['coast_complexity'] if kind == 'wet-complex-coast' else
                             s['land_height_p90'] if kind == 'low-plain-sea' else s['land_height_std'])
                    rows.append(dict(source=str(path), source_sha256=source_sha256,
                        proposal_kind=kind, climate_archetype=archetype, selection_score=score,
                        proposal_lowfreq=s, proposal_climate=c,
                        metadata=dict(seed=meta['seed'], profile=meta['profile'], split=meta['split'],
                                      y=cy, x=cx, teacher_revision=REVISION)))
    return rows


def select_candidates(rows, quotas):
    selected, counts = [], {}
    for split, quota in quotas.items():
        for kind, archetype in BUCKETS:
            pool = sorted((row for row in rows if row['metadata']['split'] == split and
                           row['proposal_kind'] == kind),
                          key=lambda row: (row['selection_score'], row['metadata']['seed'],
                                           row['metadata']['y'], row['metadata']['x']))
            chosen, seeds = [], set()
            for row in pool:
                m = row['metadata']
                if m['seed'] in seeds or any(p['metadata']['seed'] == m['seed'] and
                    abs(p['metadata']['y']-m['y']) < 128 and abs(p['metadata']['x']-m['x']) < 128
                    for p in selected):
                    continue
                chosen.append(row); selected.append(row); seeds.add(m['seed'])
                if len(chosen) == quota:
                    break
            counts[f'{split}:{kind}'] = dict(selected=len(chosen), requested=quota, proposals=len(pool))
            if not chosen:
                raise ValueError(f'No independent aligned teacher proposal for {split}/{kind}.')
    train = {row['metadata']['seed'] for row in selected if row['metadata']['split'] == 'train'}
    val = {row['metadata']['seed'] for row in selected if row['metadata']['split'] == 'val'}
    if train & val or (train | val) & HOLDOUT_SEEDS:
        raise ValueError('Rare native train, validation or reserved worlds overlap.')
    return selected, counts


def plan(main, coverage, native, train_count, val_count):
    existing = json.loads((native/'manifest.json').read_text())
    data = json.loads(coverage.read_text())
    if data['dataset_manifest_digest'] != digest(main/'manifest.json'):
        raise ValueError('Teacher coverage does not describe the current source bank.')
    rows = []
    for row in data['rows']:
        # A large256-square mixed scene may contain a qualifying low64-square
        # coastal core even when its global land-height percentile is too high.
        kinds = {kind for kind, _ in BUCKETS} if row['split'] == 'val' else set(row['kinds'])
        if kinds:
            rows.extend(core_candidates(main/'base'/row['file'], kinds))
    selected, counts = select_candidates(rows, dict(train=train_count, val=val_count))
    worlds = {split: {row['metadata']['seed'] for row in existing['scenes']
                     if row['metadata']['split'] == split} for split in ('train', 'val')}
    for row in selected:
        meta = row['metadata']
        if meta['seed'] in worlds['val' if meta['split'] == 'train' else 'train']:
            raise ValueError('Rare extension conflicts with the existing native split.')
    return dict(schema=2, revision=REVISION, scenes=selected, counts=counts,
        size=128, stride=32, complete_contexts=True, store_dtype='float32',
        main_manifest_sha256=digest(main/'manifest.json'), source_coverage_sha256=digest(coverage),
        native_manifest_sha256=digest(native/'manifest.json'), teacher_code=source_digest(),
        teacher_sha256=digest(teacher_source()/'diffusion_pytorch_model.safetensors'),
        note='Teacher-only aligned64-core proposals. Native recapture and requalification still required. No student or reserved world used.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'capture'])
    parser.add_argument('--main', type=Path, default=DATA/'crops/main')
    parser.add_argument('--native', type=Path, default=DATA/'crops/native-scenes-v1')
    parser.add_argument('--coverage', type=Path, default=DATA/'eval/rare-training-coverage.json')
    parser.add_argument('--output', type=Path, default=DATA/'crops/native-rare-scenes-v1')
    parser.add_argument('--train-per-kind', type=int, default=16)
    parser.add_argument('--val-per-kind', type=int, default=8)
    args = parser.parse_args()
    if min(args.train_per_kind, args.val_per_kind) < 1:
        parser.error('Positive rare scene quotas required.')
    root = external_path(args.output)
    manifest = root/'manifest.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
    else:
        data = plan(args.main, args.coverage, args.native, args.train_per_kind, args.val_per_kind)
        atomic_json(manifest, data)
    if args.action == 'plan':
        print(json.dumps(dict(scenes=len(data['scenes']), counts=data['counts'])), flush=True)
        return
    if data['teacher_code'] != source_digest() or data['teacher_sha256'] != digest(teacher_source()/'diffusion_pytorch_model.safetensors'):
        raise ValueError('Pinned teacher changed after rare scene planning.')
    contract = dict(manifest_sha256=digest(manifest), producer_sha256=digest(Path(__file__)),
                    recorder_sha256=digest(REPO/'distill/native_bank.py'), gpu=torch.cuda.get_device_name())
    if (root/'capture-contract.json').exists() and json.loads((root/'capture-contract.json').read_text()) != contract:
        raise ValueError('Rare native capture contract changed; inspect before resuming.')
    atomic_json(root/'capture-contract.json', contract)
    from distill.teacher import Worlds
    torch.set_num_threads(8)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    worlds = Worlds(data['scenes'][0]['metadata']['seed'])
    for index, item in enumerate(data['scenes']):
        if STOP:
            break
        meta = dict(item['metadata'], native_size=128, native_init_halo=32)
        path = root/f'{meta["split"]}-{index:04d}.npz'
        physical_path = root/f'physical-{index:04d}.json'
        if path.exists() and physical_path.exists():
            continue
        world = worlds.create(meta['seed'], meta['profile'])
        recorder = SceneCapture(world.base_model, path, meta)
        world.base_model = recorder
        original = world._latent_inference

        def capture(contexts, *values, **options):
            recorder.contexts = contexts
            return original(contexts, *values, **options)

        world._latent_inference = capture
        try:
            y, x = meta['y'], meta['x']
            if not path.exists():
                with torch.inference_mode():
                    field = world.latents[:, y:y+128, x:x+128]
                    initial = world.latents.args[0][:, y-32:y+160, x-32:x+160]
                recorder.save_scene(field, initial)
            # Independent full-pipeline qualification, after the native calls
            # have been frozen. Extra decoder dependencies must not enter them.
            with torch.inference_mode():
                height = world.get(y*8+64, x*8+64, y*8+448, x*8+448,
                                   with_climate=False)['elev'].float().cpu().numpy()
            physical = relief_stats(height, 0)
            c = item['proposal_climate']
            passed = qualifies(item['proposal_kind'], physical, c, item['climate_archetype'])
            atomic_json(physical_path, dict(seed=meta['seed'], split=meta['split'],
                kind=item['proposal_kind'], climate_archetype=item['climate_archetype'],
                teacher_decoded=physical, proxy_climate=c, physical_qualified=passed,
                note='Full original decoder/height pipeline on aligned384-pixel interior; climate comes from source teacher coarse cells.'))
            atomic_json(root/'status.json', dict(status='running', completed=index+1, target=len(data['scenes'])))
            print(json.dumps(dict(scene=index, split=meta['split'], kind=item['proposal_kind'])), flush=True)
        finally:
            world.close()
            del world, recorder, original
            gc.collect()
            torch.cuda.empty_cache()
    if STOP:
        atomic_json(root/'status.json', dict(status='interrupted', target=len(data['scenes'])))
        return
    coverage = audit(root)
    atomic_json(root/'native-coverage.json', coverage)
    qualified = []
    for row in coverage['rows']:
        # Coverage filenames encode the original plan index, avoiding split-order assumptions.
        index = int(Path(row['file']).stem.split('-')[-1])
        proposal = data['scenes'][index]
        present = (proposal['proposal_kind'] in row['kinds'] and
                   (proposal['climate_archetype'] is None or proposal['climate_archetype'] in row['archetypes']))
        qualified.append(dict(file=row['file'], split=row['split'], seed=row['seed'],
            requested=proposal['proposal_kind'], climate_archetype=proposal['climate_archetype'],
            proxy_qualified=present, physical_qualified=json.loads((root/f'physical-{index:04d}.json').read_text())['physical_qualified']))
    atomic_json(root/'qualification.json', dict(rows=qualified, accepted=False,
        note='Captured native low-frequency proxy and conditioning, not full decoded qualification.'))
    atomic_json(root/'status.json', dict(status='complete', target_shards=len(data['scenes']),
        train_shards=len(list(root.glob('train-*.npz'))), val_shards=len(list(root.glob('val-*.npz'))),
        complete_contexts=True, accepted=False))


if __name__ == '__main__':
    main()
