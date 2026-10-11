"""Teacher-only wet high-mountain bank, separate from the frozen four-case bank.

Climate and decoded relief qualify a proposal. Incised, branched valleys still
require inspection of the teacher preview; TPI is not a hydrology simulation.
New validation worlds and reserved final worlds never enter training.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import (DATA, HOLDOUT_SEEDS, REPO, REVISION, atomic_json,
                            atomic_write, external_path, source_digest)
from distill.native_base import teacher_source
from distill.rare_cases import field_stats, relief_stats

ROOT = DATA/'crops/native-wet-mountains-v1'
KIND = 'wet-high-mountain'
CRITERIA = dict(rain_min_mm=1200., land_min=.95, height_p90_min_m=2000.,
                relief_p90_p10_min_m=500., slope_p90_min_degrees=15.,
                decoded_tpi8_rms_min_m=10.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mountain_stats(height, lod):
    from scipy.ndimage import gaussian_filter
    height = np.asarray(height, np.float64)
    if height.ndim != 2 or min(height.shape) < 32 or not np.isfinite(height).all():
        raise ValueError('A finite two-dimensional teacher terrain is required.')
    result = relief_stats(height, lod)
    land = height > 0
    result['height_p10_m'] = float(np.quantile(height[land], .1)) if land.any() else 0.
    result['relief_p90_p10_m'] = result['land_height_p90']-result['height_p10_m']
    tpi = height-gaussian_filter(height, 8., mode='reflect')
    interior = land.copy()
    interior[:16] = interior[-16:] = False
    interior[:, :16] = interior[:, -16:] = False
    result['tpi8_rms_m'] = float(np.sqrt(np.mean(tpi[interior]**2))) if interior.any() else 0.
    return result


def qualifies(stats, climate, decoded=False):
    if climate is None:
        return False
    return bool(climate['rain'] >= CRITERIA['rain_min_mm'] and
        stats['land'] >= CRITERIA['land_min'] and
        stats['land_height_p90'] >= CRITERIA['height_p90_min_m'] and
        stats['relief_p90_p10_m'] >= CRITERIA['relief_p90_p10_min_m'] and
        stats['slope_land_p90'] >= CRITERIA['slope_p90_min_degrees'] and
        (not decoded or stats['tpi8_rms_m'] >= CRITERIA['decoded_tpi8_rms_min_m']))


def check_splits(rows, existing):
    assigned = {}
    for row in [*existing, *rows]:
        meta = row['metadata']
        seed, split = meta['seed'], meta['split']
        if split not in ('train', 'val', 'final'):
            raise ValueError('Explicit train, val or final split required.')
        if (split == 'final') != (seed in HOLDOUT_SEEDS):
            raise ValueError('Reserved final worlds must never enter TRAIN or VAL.')
        if seed in assigned and assigned[seed] != split:
            raise ValueError('World leakage between training, validation and final tests.')
        assigned[seed] = split


def survey(root, per_world):
    from terrain_conditioning import WORLD_BOUNDS, make_conditioning_factory
    existing = json.loads((DATA/'crops/native-mixed-scenes-v2/manifest.json').read_text())['scenes']
    # TRAIN worlds already used for the original teacher bank. VAL are new,
    # checked against the complete existing mixed inventory, not just a crop.
    worlds = [('train', seed) for seed in (10060, 10258, 10324)]
    worlds += [('val', seed) for seed in (620001, 620002, 620003)]
    worlds += [('final', seed) for seed in (101, 303, 404)]
    check_splits([dict(metadata=dict(seed=s, split=t)) for t, s in worlds], existing)
    x0, y0, x1, y1 = WORLD_BOUNDS
    spacing = 30*8*128
    xs = np.arange(x0+spacing/2, x1, spacing)
    ys = np.arange(y0+spacing/2, y1, spacing)
    rows = []
    for split, seed in worlds:
        factory = make_conditioning_factory(seed, 'natural')
        fields = factory.sample(xs, ys)
        mask = (fields[0] >= 2000.) & (fields[3] >= 1200.)
        indices = np.flatnonzero(mask)
        # All ranking uses original conditioning, before loading any student.
        indices = indices[np.argsort(-fields[0].ravel()[indices], kind='stable')]
        chosen = []
        for index in indices:
            iy, ix = np.unravel_index(index, mask.shape)
            y, x = (int(np.floor(v/(240*96)))*96 for v in (ys[iy], xs[ix]))
            if any(abs(y-p['metadata']['y']) < 256 and abs(x-p['metadata']['x']) < 256 for p in chosen):
                continue
            offsets = (np.arange(64)+.5)*240
            local = factory.sample(x*240+offsets, y*240+offsets)
            climate, stats = field_stats(local), mountain_stats(local[0], 3)
            if qualifies(stats, climate):
                chosen.append(dict(metadata=dict(seed=seed, split=split, profile='natural',
                    y=y, x=x, teacher_revision=REVISION), case_kind=KIND,
                    proposal_climate=climate, proposal_relief=stats))
                if len(chosen) == per_world:
                    break
        rows.extend(chosen)
        print(json.dumps(dict(split=split, seed=seed, proposals=len(chosen))), flush=True)
    check_splits(rows, existing)
    missing = [dict(split=t, seed=s) for t, s in worlds if not any(
        r['metadata']['seed'] == s for r in rows)]
    if missing:
        raise ValueError(f'No qualifying conditioning proposal in {missing}; expand the survey.')
    return dict(schema=1, revision=REVISION, scenes=rows, criteria=CRITERIA,
        source_manifest_sha256=digest(DATA/'crops/native-mixed-scenes-v2/manifest.json'),
        teacher_sha256=digest(teacher_source()/'diffusion_pytorch_model.safetensors'),
        teacher_code=source_digest(), frozen=False, accepted=False,
        note='Conditioning proposals only; original teacher decoded qualification and visual valley review pending.')


def preview(root, record, low, fine, qualification):
    from PIL import Image, ImageDraw, ImageFont
    from distill.native_progress_images import shade
    meta = record['metadata']
    image = Image.new('RGB', (1560, 912), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    draw.text((12, 12), f'Teacher original | montagne humide | {meta["split"]} | seed {meta["seed"]} | {record["coarse_variant"]}', font=font, fill='white')
    draw.text((12, 43), f'Pluie {qualification["climate"]["rain"]:.0f} mm/an | altitude p90 {qualification["decoded"]["land_height_p90"]:.0f} m | relief p90-p10 {qualification["decoded"]["relief_p90_p10_m"]:.0f} m', font=font, fill='white')
    draw.text((12, 75), 'Qualification numerique : '+str(qualification['physical_qualified'])+' | Vallees ramifiees : inspection visuelle requise', font=font, fill='white')
    for col, (e, lod) in enumerate(((low, 3), (fine, 0))):
        draw.text((12+col*780, 112), f'LOD {lod} | ombrage fixe, sans lissage, relief x1.5', font=font, fill='white')
        panel = shade(e, 30*2**lod, 1.5).resize((768, 768), Image.Resampling.NEAREST)
        image.paste(panel, (col*780, 144))
    name = f'mountains-teacher-{Path(record["file"]).stem}'
    output = external_path(DATA/'_base'); output.mkdir(parents=True, exist_ok=True)
    atomic_write(output/f'{name}.png', lambda f: image.save(f, format='PNG'))
    atomic_json(output/f'{name}.json', qualification | dict(record=record))
    (output/f'{name}.txt').write_text(json.dumps(qualification, indent=2)+'\n')


@torch.no_grad()
def capture(root):
    from distill.teacher import Worlds
    from distill.native_bank import SceneCapture
    from distill.inference import install
    from distill.student import load_student
    from distill.native_decoded import DecodedRegion, OriginalDecoder
    from distill.decoded_loss import reconstructed_height
    from distill.native_multilod import low_height
    from distill.native_coverage import climate_from_condition
    plan = json.loads((root/'proposals.json').read_text())
    if plan['teacher_code'] != source_digest() or plan['teacher_sha256'] != digest(teacher_source()/'diffusion_pytorch_model.safetensors'):
        raise ValueError('Original teacher changed after proposal selection.')
    weight_entries = json.loads((DATA/'crops/native-mixed-scenes-v2/manifest.json').read_text())['coarse_weights']
    weights = {name: Path(entry['path']) for name, entry in weight_entries.items()}
    if any(digest(weights[name]) != entry['sha256'] for name, entry in weight_entries.items()):
        raise ValueError('Coarse weights changed from the frozen matched bank.')
    contract = dict(proposals_sha256=digest(root/'proposals.json'),
        producer_sha256=digest(Path(__file__)), recorder_sha256=digest(REPO/'distill/native_bank.py'),
        coarse_weights_sha256={k: digest(v) for k, v in weights.items()})
    if (root/'capture-contract.json').exists() and json.loads((root/'capture-contract.json').read_text()) != contract:
        raise ValueError('Capture contract changed; preserve this bank and use a new directory.')
    atomic_json(root/'capture-contract.json', contract)
    torch.set_num_threads(8)
    worlds, decoder = Worlds(plan['scenes'][0]['metadata']['seed']), OriginalDecoder('cuda')
    models = {name: load_student(path, 'cuda')[0] for name, path in weights.items()}
    records, qualifications = [], []
    for index, proposal in enumerate(plan['scenes']):
        for variant in ('original', 'coarse4', 'coarse2'):
            meta = proposal['metadata'] | dict(native_size=128, native_init_halo=32, coarse_variant=variant)
            name = f'{meta["split"]}-{index:04d}-{variant}.npz'
            record = dict(file=name, metadata=meta, coarse_variant=variant, case_kind=KIND)
            records.append(record)
            path, physical_path = root/name, root/f'{Path(name).stem}-qualification.json'
            if path.exists() and physical_path.exists():
                qualifications.append(json.loads(physical_path.read_text())); continue
            world = worlds.create(meta['seed'], meta['profile'])
            try:
                if variant != 'original':
                    install(world, dict(coarse=models[variant]))
                    world.bind()
                recorder = SceneCapture(world.base_model, path, meta)
                world.base_model = recorder
                original = world._latent_inference
                def intercept(contexts, *values, **options):
                    recorder.contexts = contexts
                    return original(contexts, *values, **options)
                world._latent_inference = intercept
                y, x = meta['y'], meta['x']
                if not path.exists():
                    field = world.latents[:, y:y+128, x:x+128]
                    initial = world.latents.args[0][:, y-32:y+160, x-32:x+160]
                    recorder.save_scene(field, initial)
                region = DecodedRegion(path, 'cuda')
                first, second = region.stages
                # Center refinement window conditions reflect this coarse branch.
                climate_values = climate_from_condition(second['condition'][4].float().cpu().numpy())
                low = low_height(region.target)[0, 0].cpu().numpy()
                climate = field_stats(np.concatenate((low[None],
                    np.broadcast_to(climate_values[:, None, None], (4, 64, 64)))))
                fine = reconstructed_height(decoder(region.target[None], region.decoder_noise), region.target[None])[0, 0].cpu().numpy()[64:-64, 64:-64]
                low_stats, fine_stats = mountain_stats(low, 3), mountain_stats(fine, 0)
                qualified = qualifies(low_stats, climate) and qualifies(fine_stats, climate, decoded=True)
                qualification = dict(file=name, split=meta['split'], seed=meta['seed'],
                    requested=KIND, coarse_variant=variant, climate=climate, native=low_stats,
                    decoded=fine_stats, physical_qualified=qualified, visual_review='pending',
                    note='Original base teacher T2 on this coarse branch, frozen original decoder; TPI is morphology, not proof of drainage.')
                atomic_json(physical_path, qualification)
                preview(root, record, low, fine, qualification)
                qualifications.append(qualification)
                atomic_json(root/'status.json', dict(status='running', completed=len(qualifications), target=len(plan['scenes'])*3))
                print(json.dumps(dict(file=name, physical_qualified=qualified)), flush=True)
            finally:
                world.close(); del world
                gc.collect(); torch.cuda.empty_cache()
    check_splits(records, [])
    atomic_json(root/'manifest.json', dict(schema=1, revision=REVISION, scenes=records,
        original_manifest_sha256=json.loads((DATA/'crops/native-mixed-scenes-v2/manifest.json').read_text())['original_manifest_sha256'],
        proposals_sha256=digest(root/'proposals.json'), criteria=CRITERIA,
        teacher_weights_sha256=plan['teacher_sha256'], accepted=False))
    atomic_json(root/'qualification.json', dict(rows=qualifications, accepted=False))
    atomic_json(root/'status.json', dict(status='complete', captured=len(records),
        qualified=sum(r['physical_qualified'] for r in qualifications), visual_review_pending=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'capture'))
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--per-world', type=int, default=2)
    args = parser.parse_args()
    root = external_path(args.output)
    if args.per_world < 1:
        parser.error('Positive proposal quota required.')
    if args.action == 'plan':
        if (root/'proposals.json').exists():
            raise ValueError('Existing proposals are immutable; use a new output directory.')
        atomic_json(root/'proposals.json', survey(root, args.per_world))
    else:
        capture(root)


if __name__ == '__main__':
    main()
