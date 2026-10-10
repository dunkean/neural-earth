"""Teacher-derived base: prune blocks, retain its native conditional interface.

prepare records real teacher calls from both production stages. train regresses
those calls, keeping the 64/32 windows, conditioning58, attention and T=2 fusion
available to the runtime. This pilot is deliberately separate from the rejected
dense CNN students. It produces no intermediate comparison plates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from distill.common import (DATA, REPO, REVISION, HOLDOUT_SEEDS, atomic_json,
                            atomic_write, append_json, external_path, source_digest)
from distill.dataset import StepBatches
from terrain_diffusion.models.edm_unet import EDMUnet2D

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def pruned_config(config):
    config = {key: value for key, value in dict(config).items() if not key.startswith('_')}
    if config['layers_per_block'] != 3:
        raise ValueError('The native base pilot requires the pinned three-block teacher.')
    return config | dict(layers_per_block=2)


def source_name(name, config):
    """Retain encoder blocks0/2 and map their matching decoder skip inputs."""
    parts = name.split('.')
    if len(parts) > 1 and parts[0] in ('enc', 'dec') and '_block' in parts[1]:
        prefix, index = parts[1].rsplit('_block', 1)
        old_index = ((0, 2) if parts[0] == 'enc' else (0, 2, 3))[int(index)]
        parts[1] = f'{prefix}_block{old_index}'
    return '.'.join(parts)


def transfer_state(teacher_state, student_state, config):
    result, mapping = {}, {}
    for name, tensor in student_state.items():
        parent = source_name(name, config)
        if parent not in teacher_state or teacher_state[parent].shape != tensor.shape:
            raise ValueError(f'Teacher transfer cannot initialize {name} from {parent}.')
        result[name] = teacher_state[parent].detach().clone()
        mapping[name] = parent
    return result, mapping


class NativeBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.net = EDMUnet2D.from_config(config)

    def forward(self, inputs, noise_labels, conditional_inputs):
        # Functional BF16 casts preserve the original computation and FP32
        # optimizer masters. Eval keeps MP layers from mutating saved parameters.
        self.net.eval()
        dtype = torch.bfloat16 if inputs.is_cuda else torch.float32
        state = {name: value.to(dtype) for name, value in
                 list(self.net.named_parameters()) + list(self.net.named_buffers())}
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return torch.func.functional_call(self.net, state, (inputs.to(dtype),),
                dict(noise_labels=noise_labels.to(dtype),
                     conditional_inputs=[value.to(dtype) for value in conditional_inputs])).float()


def teacher_source():
    from terrain_app import resolve_model_source
    return Path(resolve_model_source())/'base_model'


def source_samples(root, per_profile):
    """Choose independent train/val worlds from the existing, reserved-seed-free bank."""
    from distill.teacher import PROFILES
    selected = []
    for split, quota in (('train', per_profile), ('val', 1)):
        counts = {profile: 0 for profile in PROFILES}
        worlds = set()
        for path in sorted((root/'base').glob(f'{split}-*.npz')):
            with np.load(path, allow_pickle=False) as sample:
                meta = json.loads(str(sample['metadata']))
            seed, profile = meta['seed'], meta['profile']
            if seed in HOLDOUT_SEEDS:
                raise ValueError('A reserved evaluation seed entered the native training bank.')
            if profile not in counts or counts[profile] >= quota or seed in worlds:
                continue
            selected.append(dict(path=str(path), metadata=meta))
            worlds.add(seed)
            counts[profile] += 1
            if all(count >= quota for count in counts.values()):
                break
        if any(count < quota for count in counts.values()):
            raise ValueError(f'Not enough independent {split} worlds for the native base pilot.')
    train = {item['metadata']['seed'] for item in selected if item['metadata']['split'] == 'train'}
    val = {item['metadata']['seed'] for item in selected if item['metadata']['split'] == 'val'}
    if train & val:
        raise ValueError('Native train and validation worlds overlap.')
    return selected


class CaptureCalls(nn.Module):
    def __init__(self, teacher, path, metadata, limit=16):
        super().__init__()
        self.teacher = teacher
        self.path, self.metadata, self.limit = path, metadata, limit
        self.counts = {}
        self.rows = []

    def forward(self, inputs, noise_labels, conditional_inputs, **kwargs):
        result = self.teacher(inputs, noise_labels=noise_labels,
                              conditional_inputs=conditional_inputs, **kwargs)
        label = round(float(noise_labels[0]), 5)
        count = self.counts.get(label, 0)
        keep = min(inputs.shape[0], self.limit-count)
        for index in range(keep):
            self.rows.append(dict(inputs=inputs[index].detach().float().cpu().numpy(),
                labels=noise_labels[index].detach().float().cpu().numpy(),
                condition=conditional_inputs[0][index].detach().float().cpu().numpy(),
                target=result[index].detach().float().cpu().numpy()))
        self.counts[label] = count+keep
        return result

    def save(self):
        if len(self.counts) != 2 or not self.rows:
            raise ValueError('Native teacher capture must contain both production stages.')
        arrays = {key: np.stack([row[key] for row in self.rows]).astype(np.float16)
                  for key in self.rows[0]}
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise FloatingPointError('Non-finite native teacher pairs.')
        arrays['metadata'] = np.array(json.dumps(self.metadata | dict(stage_counts=self.counts)))
        atomic_write(self.path, lambda handle: np.savez(handle, **arrays))
        return len(self.rows)


def prepare(args):
    from distill.teacher import Worlds
    output = external_path(args.dataset)
    source = teacher_source()
    selected = source_samples(DATA/'crops/main', args.worlds_per_profile)
    contract = dict(schema=1, teacher_revision=REVISION, source_samples=selected,
                    teacher_sha256=hashlib.sha256((source/'diffusion_pytorch_model.safetensors').read_bytes()).hexdigest(),
                    teacher_code=source_digest(),
                    production_stages=2, window=64, stride=32, conditioning=58,
                    per_stage_cap=16, capture_size=64)
    if (output/'manifest.json').exists() and json.loads((output/'manifest.json').read_text()) != contract:
        raise ValueError('Native teacher dataset provenance changed.')
    atomic_json(output/'manifest.json', contract)
    worlds = Worlds(selected[0]['metadata']['seed'])
    for index, item in enumerate(selected):
        if STOP:
            break
        meta = item['metadata']
        path = output/f'{meta["split"]}-{index:04d}.npz'
        if path.exists():
            continue
        world = worlds.create(meta['seed'], meta['profile'])
        recorder = CaptureCalls(world.base_model, path, meta)
        world.base_model = recorder
        try:
            with torch.inference_mode():
                world.latents[:, meta['y']:meta['y']+64, meta['x']:meta['x']+64]
            count = recorder.save()
            print(json.dumps(dict(captured_world=index, split=meta['split'], windows=count)), flush=True)
        finally:
            world.close()
            del world, recorder
            gc.collect()
            torch.cuda.empty_cache()
    atomic_json(output/'status.json', dict(status='interrupted' if STOP else 'complete',
                train_shards=len(list(output.glob('train-*.npz'))), val_shards=len(list(output.glob('val-*.npz')))))


class NativePairs(Dataset):
    def __init__(self, root, split):
        self.arrays = {key: [] for key in ('inputs', 'labels', 'condition', 'target')}
        self.files = sorted(Path(root).glob(f'{split}-*.npz'))
        if not self.files:
            raise ValueError(f'No native {split} pairs.')
        for path in self.files:
            with np.load(path, allow_pickle=False) as sample:
                for key in self.arrays:
                    self.arrays[key].append(torch.from_numpy(sample[key].astype(np.float32)))
        self.arrays = {key: torch.cat(values) for key, values in self.arrays.items()}

    def stage_probabilities(self):
        # Give refinement as much supervision as initialization, even though
        # its dependency graph requires more first-stage windows per crop.
        labels = self.arrays['labels'].numpy()
        unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        if len(unique) != 2:
            raise ValueError('The native pilot must supervise both production stages.')
        return 1./counts[inverse]/len(unique)

    def __len__(self):
        return self.arrays['inputs'].shape[0]

    def __getitem__(self, address):
        index = address[0] if isinstance(address, tuple) else address
        return tuple(self.arrays[key][index] for key in ('inputs', 'labels', 'condition', 'target'))


@torch.no_grad()
def validate(model, data, device):
    # A short native-operator validation selects checkpoints. It is not a new
    # morphology diagnosis or an acceptance claim about decoded terrain.
    errors = []
    for index in torch.linspace(0, len(data)-1, min(12, len(data))).round().long().tolist():
        inputs, labels, condition, target = (v[None].to(device) for v in data[index])
        prediction = model(inputs, labels, [condition])
        errors.append(float((prediction-target).square().mean()))
    return float(np.mean(errors))


def train(args):
    source = teacher_source()
    dataset = external_path(args.dataset)
    output = external_path(args.output)
    data_status = json.loads((dataset/'status.json').read_text())
    if data_status['status'] != 'complete':
        raise ValueError('Finish native teacher capture before training.')
    manifest = json.loads((dataset/'manifest.json').read_text())
    train_pairs, val_pairs = NativePairs(dataset, 'train'), NativePairs(dataset, 'val')
    device = torch.device('cuda:0')
    saved = torch.load(output/'latest.pt', map_location='cpu', weights_only=True) if (output/'latest.pt').exists() else None
    teacher_config = json.loads((source/'config.json').read_text())
    config = pruned_config(teacher_config)
    torch.manual_seed(args.seed)
    model = NativeBase(config)
    if saved:
        if (saved['config'] != config or saved['dataset_manifest'] != manifest or saved['seed'] != args.seed or
                saved['arguments']['batch'] != args.batch or saved['arguments']['lr'] != args.lr):
            raise ValueError('Native base resume provenance changed.')
        model.load_state_dict(saved['model'])
        mapping = saved['teacher_mapping']
    else:
        teacher = EDMUnet2D.from_pretrained(str(source))
        teacher.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        state, mapping = transfer_state(teacher.state_dict(), model.net.state_dict(), config)
        model.net.load_state_dict(state, strict=True)
        del teacher, state
    model.to(device).eval()
    ema = NativeBase(config).to(device).eval().requires_grad_(False)
    ema.load_state_dict(saved['ema'] if saved else model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=1e-4, fused=True)
    start = saved['step'] if saved else 0
    if start > args.steps:
        raise ValueError('Cannot resume native base to an earlier step.')
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    torch.backends.cudnn.benchmark = True
    best = saved['best_score'] if saved else float('inf')
    step = start
    started = time.monotonic()
    last_save = started
    atomic_json(output/'run.json', dict(config=config, start_step=start, target_steps=args.steps,
                teacher_sha256=manifest['teacher_sha256'], teacher_revision=REVISION,
                parameters=sum(p.numel() for p in model.parameters()),
                transfer_complete=True, retained_blocks=[0, 2],
                train_windows=len(train_pairs), val_windows=len(val_pairs), batch=args.batch,
                lr=args.lr, inference_stages=2, window=64, stride=32, conditioning=58))

    def checkpoint(name):
        state = dict(config=config, model=model.state_dict(), ema=ema.state_dict(),
            optimizer=optimizer.state_dict(), step=step, best_score=best,
            dataset_manifest=manifest, teacher_mapping=mapping, seed=args.seed,
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            arguments=dict(steps=args.steps, lr=args.lr, batch=args.batch))
        atomic_write(output/name, lambda handle: torch.save(state, handle))

    if not saved:
        baseline = validate(ema, val_pairs, device)
        best = baseline
        checkpoint('initial.pt')
        append_json(output/'validation.jsonl', dict(step=0, native_velocity_mse=baseline))
        print(json.dumps(dict(step=0, native_velocity_mse=baseline)), flush=True)
    loader = DataLoader(train_pairs, batch_sampler=StepBatches(len(train_pairs), args.batch,
        start, args.steps, args.seed, train_pairs.stage_probabilities()), num_workers=0, pin_memory=True)
    try:
        for batch in loader:
            if STOP:
                break
            inputs, labels, condition, target = (v.to(device, non_blocking=True) for v in batch)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs, labels, [condition])
            loss = (prediction-target).square().mean()
            for axis in (-1, -2):
                loss = loss+.05*(prediction.diff(dim=axis)-target.diff(dim=axis)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite native base loss at step{step}.')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            with torch.no_grad():
                decay = min(.999, (1+step)/(10+step))
                for average, parameter in zip(ema.parameters(), model.parameters()):
                    average.lerp_(parameter, 1-decay)
            if step % 25 == 0 or step == start+1:
                item = dict(step=step, loss=float(loss), grad_norm=float(norm),
                            elapsed_seconds=time.monotonic()-started)
                append_json(output/'train.jsonl', item)
                print(json.dumps(item), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                score = validate(ema, val_pairs, device)
                append_json(output/'validation.jsonl', dict(step=step, native_velocity_mse=score))
                if score < best:
                    best = score
                    checkpoint('best.pt')
                if step == args.eval_every or step == args.steps:
                    checkpoint(f'step{step}.pt')
            if time.monotonic()-last_save >= 120:
                checkpoint('latest.pt')
                last_save = time.monotonic()
                atomic_json(output/'status.json', dict(status='running', step=step, best_score=best))
        checkpoint('latest.pt')
        atomic_json(output/'status.json', dict(status='interrupted' if STOP else 'complete',
                    step=step, best_score=best, elapsed_seconds=time.monotonic()-started,
                    accepted=False))
    except Exception as exc:
        checkpoint('latest.pt')
        atomic_json(output/'status.json', dict(status='failed', step=step, error=f'{type(exc).__name__}: {exc}'))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train'])
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/base-native-pilot')
    parser.add_argument('--output', type=Path, default=DATA/'ckpt/base-native-pilot')
    parser.add_argument('--worlds-per-profile', type=int, default=2)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--seed', type=int, default=20261010)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.eval_every, args.worlds_per_profile) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('Positive finite native pilot settings required.')
    torch.set_num_threads(8)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    (prepare if args.action == 'prepare' else train)(args)


if __name__ == '__main__':
    main()
