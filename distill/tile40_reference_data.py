"""Resumable full40km teacher tiles paired with explicit sketches and climates."""
import argparse
import gc
import json
from pathlib import Path
import signal
import shutil
import time

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
import torch

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write, external_path, source_digest
from distill.native_base import teacher_source
from distill.native_progressive_prepare import configure
from distill.native_weight_quant import file_sha
from distill.teacher import Worlds, unweight
from distill.tile40_conditioning import CLIMATES, SIDE_M, SketchConditioning, cases, sketch, sample_sketch
from distill.tile40_height import resample_tile,guided_reference



def world_signature(world):
    """Record physical/numerical parameters,without ConfigMixin runtime classes."""
    fields = ('cond_snr', 'frequency_mult', 'coarse_means', 'coarse_stds', 'histogram_raw')
    parameters = {key:[float(value) for value in world.kwargs[key]] for key in fields}
    parameters.update(residual_mean=float(world.kwargs['residual_mean']),
        residual_std=float(world.kwargs['residual_std']), coarse_pooling=int(world.kwargs['coarse_pooling']),
        drop_water_pct=float(world.kwargs['drop_water_pct']),
        elev_coarse_pool_mode=str(world.kwargs['elev_coarse_pool_mode']),
        p5_coarse_pool_mode=str(world.kwargs['p5_coarse_pool_mode']))
    return dict(native_resolution=float(world.native_resolution), T=int(world.T),
        latent_compression=int(world.latent_compression), latent_batch=int(world.latents_batch_size),
        decoder_tile_size=int(world.decoder_tile_size), decoder_tile_stride=int(world.decoder_tile_stride),
        parameters=parameters)




def tile_stats(height, spacing):
    values = np.asarray(height, np.float64)
    gy, gx = np.gradient(values, spacing)
    return dict(min_m=float(values.min()), max_m=float(values.max()), mean_m=float(values.mean()),
        p10_m=float(np.quantile(values, .1)), p90_m=float(np.quantile(values, .9)),
        relief_p90_p10_m=float(np.quantile(values, .9)-np.quantile(values, .1)),
        slope_p90_degrees=float(np.quantile(np.degrees(np.arctan(np.hypot(gx, gy))), .9)),
        land_fraction=float(np.mean(values > 0)))




@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA/'tile40-20261014/reference-v1')
    parser.add_argument('--limit', type=int, default=5, help='Maximum new tiles in this bounded segment.')
    parser.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=['train', 'val'])
    parser.add_argument('--guided', action='store_true', help='Anchor macro relief to the sketch while retaining original decoded detail.')
    parser.add_argument('--coarse-checkpoint', type=Path, help='Use a frozen deployable coarse context for teacher base/decoder targets.')
    parser.add_argument('--expanded-context', action='store_true', help='Retain all decoder-window and encoder-halo dependencies.')
    parser.add_argument('--train-per-relief', type=int, default=8, help='Independent TRAIN geometry seeds per relief;VAL/TEST remain fixed.')
    parser.add_argument('--compact-train', action='store_true', help='Keep FP16 root details/latents and conditioning for TRAIN;retain full VAL/TEST DEMs.')
    args = parser.parse_args()
    # Teacher loading may changecwd;freeze the user's checkpoint location first.
    if args.coarse_checkpoint:
        args.coarse_checkpoint = args.coarse_checkpoint.resolve()
    root = external_path(args.output)
    if args.limit < 1 or not 8 <= args.train_per_relief <= 10000 or (args.compact_train and not args.guided):
        raise ValueError('Positive bounded generation count required.')
    root.mkdir(parents=True, exist_ok=True)
    configure()
    source = teacher_source().parent
    specs = cases(train_per_relief=args.train_per_relief)
    signature = dict(schema=1, teacher_revision=REVISION,
        teacher_weights={name:file_sha(source/name/'diffusion_pytorch_model.safetensors')
                         for name in ('coarse_model', 'base_model', 'decoder_model')},
        teacher_code=source_digest(), producer_sha256=file_sha(Path(__file__)),
        conditioning_sha256=file_sha(REPO/'distill/tile40_conditioning.py'),
        side_m=SIDE_M, pixels=1334, spacing_m=SIDE_M/1334, native_acquisition_spacing_m=30.,
        acquisition_pixels=1856 if args.guided else 1344, sketch_pixels=16, climate_prototypes=CLIMATES,
        target='sketch-root+teacher-decoded-highpass-root' if args.guided else 'original-teacher',
        lowpass_sigma_m=1920. if args.guided else None,
        noise='Same geometry seed for all climates;two base noises+decoder noise remain explicit.',
        splits='Independent geometry/world seeds;all five climates grouped by seed.', cases=specs,
        note='Teacher paired references;not trained product weights or browser performance evidence.')
    if args.train_per_relief != 8 or args.compact_train:
        signature.update(train_per_relief=args.train_per_relief, compact_train=args.compact_train,
            compact_train_schema='Sketch metres,conditioning,FP16 latents+root_detail;full VAL/TEST untouched.')
    latent_origin, latent_end = (-64, 256) if args.expanded_context else (-32, 224)
    coarse_origin, coarse_end = (-8, 16) if args.expanded_context else (-2, 10)
    signature.update(latent_origin_pixels=[latent_origin]*2, latent_size=latent_end-latent_origin,
        coarse_origin_cells=[coarse_origin]*2, coarse_size=coarse_end-coarse_origin,
        internal_coarse_checkpoint=None if args.coarse_checkpoint is None else
            dict(file=args.coarse_checkpoint.name, sha256=file_sha(args.coarse_checkpoint)))
    manifest = root/'manifest.json'
    # JSON serialization canonicalizes tuples to lists for stable restart checks.
    signature = json.loads(json.dumps(signature))
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Reference contract changed;preserve this dataset and use a new version.')
    atomic_json(manifest, signature)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    worlds, coarse_model, completed = None, None, 0
    started = time.monotonic()
    selected = [row for row in specs if row['split'] in args.splits]
    try:
        for row in selected:
            destination = root/row['file']
            if destination.exists():
                with np.load(destination, allow_pickle=False) as saved:
                    meta = json.loads(str(saved['metadata']))
                    if meta['case'] != row or meta['manifest_sha256'] != file_sha(manifest):
                        raise ValueError('Stored tile differs from this reference contract.')
                continue
            if stopped or completed >= args.limit:
                break
            if args.compact_train and shutil.disk_usage(root).free < 2*2**30:
                raise ValueError('Preserve2GiB free space for atomic saves;stop this bounded acquisition.')
            if worlds is None:
                worlds = Worlds(row['seed'])
            world = worlds.create(row['seed'], 'natural')
            if world.native_resolution != 30. or world.T != 2:
                raise ValueError('Pinned30m/T2 teacher required.')
            elevation = sketch(row['seed'], row['relief'])
            factory = SketchConditioning(elevation, row['climate'])
            if args.coarse_checkpoint:
                from distill.student import load_student
                from distill.inference import install
                if coarse_model is None:
                    coarse_model, _ = load_student(args.coarse_checkpoint, world.device)
                install(world, dict(coarse=coarse_model))
                world.bind()
            # Fresh world's caches are empty:override before any coarse/base read.
            world.synthetic_map_factory = factory
            began = time.monotonic()
            try:
                torch.cuda.synchronize()
                bounds = (-256, -256, 1600, 1600) if args.guided else (0, 0, 1344, 1344)
                native = world.get(*bounds, with_climate=False)['elev']
                if isinstance(native, torch.Tensor):
                    native = native.float().cpu().numpy()
                guided = guided_reference(native, elevation) if args.guided else None
                height = resample_tile(guided['native_height'] if guided else native)
                latents, latent_mask = unweight(world.latents[:, latent_origin:latent_end, latent_origin:latent_end])
                coarse, coarse_mask = unweight(world.coarse[:, coarse_origin:coarse_end, coarse_origin:coarse_end])
                if latents.shape != (5, latent_end-latent_origin, latent_end-latent_origin) or not latent_mask.all() or not coarse_mask.all():
                    raise ValueError('Full teacher latent/conditioning coverage required.')
                torch.cuda.synchronize()
                elapsed = time.monotonic()-began
                metadata = dict(case=row, manifest_sha256=file_sha(manifest),
                    teacher_revision=REVISION, world_config=world_signature(world),
                    shape=[1334, 1334], bounds_m=[0., 0., SIDE_M, SIDE_M], spacing_m=SIDE_M/1334,
                    latent_origin_pixels=[latent_origin]*2, latent_spacing_m=240.,
                    coarse_origin_cells=[coarse_origin]*2, coarse_spacing_m=7680.,
                    height_units='metres', generation_seconds=elapsed, gpu=torch.cuda.get_device_name())
                metadata.update(target=signature['target'], native_training_spacing_m=30.,
                    native_training_pixels=1344, root_detail_units='signed-sqrt-metres')
                arrays = dict(sketch_m=elevation, climate_vector=np.asarray(CLIMATES[row['climate']], np.float32),
                              height_m=height, latents=latents.astype(np.float16), coarse=coarse.astype(np.float32))
                if guided:
                    arrays.update(root_detail_native=guided['root_detail'], teacher_height_native_m=guided['teacher_native'])
                precision = None
                if args.compact_train and row['split'] == 'train':
                    detail = guided['root_detail']
                    arrays['root_detail_native'] = detail.astype(np.float16)
                    arrays.pop('height_m')
                    arrays.pop('teacher_height_native_m')
                    original_root = np.sign(guided['native_height'])*np.sqrt(np.abs(guided['native_height']))
                    stored_root = original_root+arrays['root_detail_native'].astype(np.float32)-detail
                    difference = np.abs(stored_root*np.abs(stored_root)-guided['native_height'])
                    precision = dict(height_mae_m=float(difference.mean()), height_max_abs_m=float(difference.max()))
                    if precision['height_mae_m'] > .25 or precision['height_max_abs_m'] > 8.:
                        raise ValueError(f'Compact TRAIN precision exceeds its physical budget:{precision}')
                atomic_write(destination, lambda handle:np.savez_compressed(handle,
                    metadata=np.array(json.dumps(metadata)), **arrays))
                stats = tile_stats(height, SIDE_M/1334)
                atomic_json(destination.with_suffix('.json'), dict(case=row, file_sha256=file_sha(destination),
                    stats=stats, generation_seconds=elapsed, original_teacher=args.guided is False,
                    guided_target=args.guided, target=signature['target'], compact_train_precision=precision))
                print(json.dumps(dict(case=row, seconds=elapsed, stats=stats)), flush=True)
                completed += 1
                present = sum((root/item['file']).exists() for item in selected)
                atomic_json(root/'status.json', dict(status='running', generated_this_segment=completed,
                    available=present, requested=len(selected), full_contract_cases=len(specs),
                    elapsed_seconds=time.monotonic()-started))
            finally:
                world.close()
                del world
                gc.collect()
                torch.cuda.empty_cache()
        present = sum((root/item['file']).exists() for item in selected)
        atomic_json(root/'status.json', dict(status='interrupted' if stopped else
            'complete' if present == len(selected) else 'segment_complete',
            generated_this_segment=completed, available=present, requested=len(selected),
            full_contract_cases=len(specs), elapsed_seconds=time.monotonic()-started))
    except Exception as exc:
        atomic_json(root/'status.json', dict(status='failed', error=str(exc), generated_this_segment=completed))
        raise


if __name__ == '__main__':
    main()
