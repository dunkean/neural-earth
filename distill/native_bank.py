"""Native teacher scenes with complete window contexts and final blended targets.

Train worlds come from the audited bank; validation gets separate new worlds.
Each train world contributes land, coast and sea proposals. FP32 storage retains
the BF16 calls exactly. Capture runs only after the full coarse jobs finish.
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

from distill.common import DATA, REVISION, HOLDOUT_SEEDS, atomic_json, atomic_write, external_path, source_digest
from distill.native_base import CaptureCalls, teacher_source
from distill.teacher import PROFILES

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def scene_proposals(rows):
    """Distinct representatives; a proposal is not a guaranteed rare landscape."""
    selected = []
    for ordered in (sorted(rows, key=lambda row: -row['land_fraction']),
                    sorted(rows, key=lambda row: abs(row['land_fraction']-.5)),
                    sorted(rows, key=lambda row: row['land_fraction'])):
        for row in ordered:
            if row['index'] not in {item['index'] for item in selected}:
                selected.append(row)
                break
    return selected


def make_plan(root, train_worlds=32, val_worlds=8, size=128):
    main = DATA/'crops/main'
    rows = {}
    for line in (main/'generation.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('split') == 'train' and row.get('land_fraction') is not None:
            rows[row['index']] = row
    groups = {}
    for row in rows.values():
        if row['seed'] in HOLDOUT_SEEDS:
            raise ValueError('Reserved evaluation world in native scene proposals.')
        groups.setdefault((row['profile'], row['seed']), []).append(row)
    chosen = []
    for profile in PROFILES:
        worlds = sorted(key for key in groups if key[0] == profile)
        if len(worlds) < train_worlds:
            raise ValueError('Insufficient independent train worlds.')
        indices = np.linspace(0, len(worlds)-1, train_worlds).round().astype(int)
        for index in indices:
            for row in scene_proposals(groups[worlds[index]]):
                path = main/'base'/f'train-{row["index"]:07d}.npz'
                with np.load(path, allow_pickle=False) as sample:
                    metadata = json.loads(str(sample['metadata']))
                if any(metadata[key] != row[key] for key in ('seed', 'profile', 'split', 'y', 'x')):
                    raise ValueError('Generation log and current scene metadata disagree.')
                chosen.append(dict(metadata=metadata, source=str(path), land_fraction=row['land_fraction']))
    train_seeds = {item['metadata']['seed'] for item in chosen}
    all_seeds = {row['seed'] for row in rows.values()}
    for index in range(val_worlds*len(PROFILES)):
        seed = 200000+index
        if seed in all_seeds or seed in HOLDOUT_SEEDS:
            raise ValueError('New native validation world overlaps training.')
        chosen.append(dict(metadata=dict(seed=seed, split='val', profile=PROFILES[index % len(PROFILES)],
            position_index=index//len(PROFILES), teacher_revision=REVISION), source=None))
    assert not train_seeds & {item['metadata']['seed'] for item in chosen if item['metadata']['split'] == 'val'}
    return dict(schema=2, revision=REVISION, scenes=chosen, size=size, stride=32,
                train_worlds_per_profile=train_worlds, val_worlds_per_profile=val_worlds,
                store_dtype='float32', complete_contexts=True,
                source_manifest_sha256=hashlib.sha256((main/'manifest.json').read_bytes()).hexdigest(),
                teacher_sha256=hashlib.sha256((teacher_source()/'diffusion_pytorch_model.safetensors').read_bytes()).hexdigest(),
                teacher_code=source_digest())


class SceneCapture(CaptureCalls):
    def __init__(self, teacher, path, metadata):
        super().__init__(teacher, path, metadata, limit=1000000)
        self.contexts = None
        self.call_index = 0

    def forward(self, inputs, noise_labels, conditional_inputs, **kwargs):
        if self.contexts is None or len(self.contexts) != inputs.shape[0]:
            raise ValueError('Missing native window coordinates.')
        begin = len(self.rows)
        output = super().forward(inputs, noise_labels, conditional_inputs, **kwargs)
        for row, context in zip(self.rows[begin:], self.contexts):
            row['context'] = np.asarray(context, np.int64)
            row['call_id'] = self.call_index
        self.call_index += 1
        return output

    def save_scene(self, field, initial_weighted=None):
        if len(self.counts) != 2:
            raise ValueError('A complete scene requires both teacher stages.')
        from distill.teacher import unweight
        target, mask = unweight(field)
        arrays = {key: np.stack([row[key] for row in self.rows]).astype(np.float32)
                  for key in ('inputs', 'labels', 'condition', 'target')}
        arrays['contexts'] = np.stack([row['context'] for row in self.rows])
        arrays['call_ids'] = np.array([row['call_id'] for row in self.rows], np.int32)
        arrays['final_latents'], arrays['final_mask'] = target, mask
        if initial_weighted is not None:
            arrays['initial_weighted'] = initial_weighted.detach().float().cpu().numpy()
            arrays['initial_origin'] = np.array([self.metadata['y']-32, self.metadata['x']-32], np.int64)
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise FloatingPointError('Non-finite native scene.')
        arrays['metadata'] = np.array(json.dumps(self.metadata | dict(stage_counts=self.counts)))
        atomic_write(self.path, lambda handle: np.savez(handle, **arrays))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'capture'))
    parser.add_argument('--output', type=Path, default=DATA/'crops/native-scenes-v1')
    args = parser.parse_args()
    root = external_path(args.output)
    manifest = root/'manifest.json'
    plan = json.loads(manifest.read_text()) if manifest.exists() else make_plan(root)
    atomic_json(manifest, plan)
    if args.action == 'plan':
        print(json.dumps(dict(scenes=len(plan['scenes']), size=plan['size'])), flush=True)
        return
    if plan['teacher_code'] != source_digest():
        raise ValueError('Teacher capture code changed since scene planning.')
    if plan['teacher_sha256'] != hashlib.sha256((teacher_source()/'diffusion_pytorch_model.safetensors').read_bytes()).hexdigest():
        raise ValueError('Teacher weights changed since scene planning.')
    capture_contract = dict(producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    contract_path = root/'capture-contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != capture_contract:
        raise ValueError('Native capture producer changed; use a new bank or audit the change before resuming.')
    atomic_json(contract_path, capture_contract)
    from distill.teacher import Worlds, position
    torch.set_num_threads(8)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    worlds = Worlds(plan['scenes'][0]['metadata']['seed'])
    for index, item in enumerate(plan['scenes']):
        if STOP:
            break
        metadata = dict(item['metadata'])
        metadata.update(native_size=plan['size'], native_init_halo=32)
        path = root/f'{metadata["split"]}-{index:04d}.npz'
        if path.exists():
            continue
        world = worlds.create(metadata['seed'], metadata['profile'])
        if item['source'] is None:
            metadata['y'], metadata['x'] = position(world, metadata['position_index'], metadata['seed'], plan['size'])
        recorder = SceneCapture(world.base_model, path, metadata)
        world.base_model = recorder
        original = world._latent_inference

        def capture(ctxs, *values, **options):
            recorder.contexts = ctxs
            return original(ctxs, *values, **options)

        world._latent_inference = capture
        try:
            with torch.inference_mode():
                field = world.latents[:, metadata['y']:metadata['y']+plan['size'],
                                      metadata['x']:metadata['x']+plan['size']]
                initial = world.latents.args[0][:, metadata['y']-32:metadata['y']+plan['size']+32,
                                                 metadata['x']-32:metadata['x']+plan['size']+32]
            recorder.save_scene(field, initial)
            atomic_json(root/'status.json', dict(status='running', completed=index+1, target=len(plan['scenes'])))
            print(json.dumps(dict(scene=index, split=metadata['split'], windows=len(recorder.rows))), flush=True)
        finally:
            world.close()
            del world, recorder, original
            gc.collect()
            torch.cuda.empty_cache()
    atomic_json(root/'status.json', dict(status='interrupted' if STOP else 'complete',
        train_shards=len(list(root.glob('train-*.npz'))), val_shards=len(list(root.glob('val-*.npz'))),
        target_shards=len(plan['scenes']), complete_contexts=True, store_dtype='float32'))


if __name__ == '__main__':
    main()
