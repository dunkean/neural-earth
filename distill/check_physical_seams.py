"""Compare final terrain from four viewer tiles with one larger request.

Fresh worlds prevent a shared cache from hiding partition effects. The teacher
is checked the same way; reports, raw elevations, images and a CSV are retained.
This is continuity evidence, not an automatic artistic-quality decision.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import gc
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
import torch

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write, external_path, source_digest
from distill.inference import install
from distill.rare_cases import KINDS, load_sites
from distill.student import load_student


def historical_sites(names):
    """Use the unchanged held-out screenshot coordinates, not student-selected views."""
    path = REPO/'docs/images/screenshots.json'
    source = json.loads(path.read_text())
    if source['modelRevision'] != REVISION:
        raise ValueError('Historical sites refer to a different teacher revision.')
    kinds = {'snow': 'mountains-snow', 'desert': 'mountains-arid', 'coast': 'historical-coast'}
    result = []
    for name in names:
        item = source['sites'][name]
        if int(item['seed']) != 42:
            raise ValueError('Historical screenshot world must remain held out.')
        result.append(dict(name=name, seed=42, profile='orogen', x=item['x'], y=item['y'],
            kind=kinds[name], conditioning=dict(temp=item['source_summer_temperature_c'],
                                               rain=item['source_annual_rain_mm'])))
    return result, path


def partition_stats(full, tiles, halo=24):
    """Use cores for the mosaic and shared halo pixels for overlap checks."""
    size = tiles[(0, 0)].shape[0] - 2 * halo
    expected_shape = (size + 2 * halo, size + 2 * halo)
    if full.shape != (2 * size, 2 * size) or any(t.shape != expected_shape for t in tiles.values()):
        raise ValueError('A 2x2 tile grid and matching larger field are required.')
    if not all(np.isfinite(t).all() for t in [full, *tiles.values()]):
        raise FloatingPointError('Non-finite physical terrain.')
    mosaic = np.block([[tiles[(y, x)][halo:halo+size, halo:halo+size] for x in range(2)] for y in range(2)])
    difference = mosaic.astype(np.float64) - full
    jumps = np.concatenate([
        (difference[:, size] - difference[:, size-1]).ravel(),
        (difference[size] - difference[size-1]).ravel()])
    seam = np.zeros(full.shape, bool)
    seam[:, size-1:size+1] = True
    seam[size-1:size+1] = True
    overlaps = []
    # Four pixels at each request edge exclude the display Gaussian's support.
    edge = 4
    if halo <= edge:
        raise ValueError('The shared halo must exceed display-filter support.')
    for y in range(2):
        a, b = tiles[(y, 0)], tiles[(y, 1)]
        overlaps.append((a[halo:halo+size, size+edge:size+2*halo-edge].astype(np.float64) -
                         b[halo:halo+size, edge:2*halo-edge]).ravel())
    for x in range(2):
        a, b = tiles[(0, x)], tiles[(1, x)]
        overlaps.append((a[size+edge:size+2*halo-edge, halo:halo+size].astype(np.float64) -
                         b[edge:2*halo-edge, halo:halo+size]).ravel())
    overlap = np.concatenate(overlaps)
    metrics = dict(mae_m=float(np.abs(difference).mean()), max_m=float(np.abs(difference).max()),
                   seam_mae_m=float(np.abs(difference[seam]).mean()),
                   interior_mae_m=float(np.abs(difference[~seam]).mean()),
                   max_jump_error_m=float(np.abs(jumps).max()),
                   overlap_mae_m=float(np.abs(overlap).mean()), overlap_max_m=float(np.abs(overlap).max()),
                   coast_sign_disagreement=float(np.mean((mosaic > 0) != (full > 0))))
    return metrics, mosaic


def shade(e, lod):
    gy, gx = np.gradient(e.astype(np.float64) * (3 if lod == 3 else 1.5), 30 * 2**lod)
    slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    light = np.clip(np.sin(np.pi/4)*np.cos(slope) + np.cos(np.pi/4)*np.sin(slope)*np.cos(7*np.pi/4-aspect), 0, 1)
    colour = np.where((e > 0)[..., None], [200, 185, 150], [120, 150, 190])
    return (colour * (.25 + .75 * light[..., None])).astype(np.uint8)


def preview(path, full, mosaic, lod, title):
    size = full.shape[0]
    difference = mosaic.astype(float) - full
    scale = max(float(np.abs(difference).max()), .001)
    error = np.clip(127.5 + difference / scale * 127.5, 0, 255).astype(np.uint8)
    image = Image.new('RGB', (3*size, size+42), (25, 25, 25))
    draw = ImageDraw.Draw(image)
    draw.text((4, 3), title, fill='white')
    for x, data, label in [(0, shade(full, lod), 'one larger request'),
                           (size, shade(mosaic, lod), 'four tiles'),
                           (2*size, np.repeat(error[..., None], 3, axis=-1), f'error +/-{scale:.4g} m')]:
        image.paste(Image.fromarray(data), (x, 42))
        draw.text((x+4, 23), label, fill='white')
    image.save(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--coarse', type=Path)
    parser.add_argument('--decoder', type=Path)
    parser.add_argument('--site-manifest', type=Path, default=DATA/'eval/rare-sites-complete.json')
    parser.add_argument('--sites', nargs='+', help='Otherwise one site per category; prefer warmer plains.')
    parser.add_argument('--historical-sites', nargs='+', choices=('snow', 'desert', 'coast'),
                        help='Capture the original held-out screenshot locations instead of the rare bank.')
    parser.add_argument('--include-rare-sites', nargs='+',
                        help='Also capture explicitly named frozen rare cases alongside historical sites.')
    parser.add_argument('--lods', nargs='+', type=int, choices=(0, 3), default=[3, 0])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {k:v for k,v in dict(base=args.base, coarse=args.coarse, decoder=args.decoder).items() if v}
    if paths and set(paths) != {'base', 'coarse', 'decoder'}:
        parser.error('Supply all three candidate stages together, or run teacher only.')
    if args.historical_sites and args.sites:
        parser.error('Choose historical sites or names from the rare bank.')
    if args.include_rare_sites and not args.historical_sites:
        parser.error('--include-rare-sites requires --historical-sites; otherwise use --sites.')
    if args.historical_sites:
        if len(set(args.historical_sites)) != len(args.historical_sites):
            parser.error('Historical site names must be distinct.')
        sites, manifest_path = historical_sites(args.historical_sites)
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if args.include_rare_sites:
            rare = load_sites(args.site_manifest)
            if len(set(args.include_rare_sites)) != len(args.include_rare_sites) or set(args.include_rare_sites)-{s['name'] for s in rare}:
                parser.error('Additional rare site names must be distinct and present in the frozen bank.')
            sites += [s for s in rare if s['name'] in args.include_rare_sites]
            manifest_digest = dict(historical=manifest_digest,
                rare=hashlib.sha256(args.site_manifest.read_bytes()).hexdigest())
    else:
        sites = load_sites(args.site_manifest)
        manifest_path = args.site_manifest
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if args.sites:
        if set(args.sites) - {s['name'] for s in sites}:
            parser.error('Unknown site in the frozen bank.')
        sites = [s for s in sites if s['name'] in args.sites]
    elif not args.historical_sites:
        selected = []
        for kind in KINDS:
            choices = [s for s in sites if s['kind'] == kind]
            selected.append(next((s for s in choices if s.get('climate_archetype')), choices[0]))
        sites = selected
    models, fingerprint = {}, {}
    for stage, path in paths.items():
        payload = path.read_bytes()
        fingerprint[stage] = hashlib.sha256(payload).hexdigest()
        with torch.inference_mode(False):
            models[stage], _ = load_student(io.BytesIO(payload), 'cuda')
    for name in ('distill/student.py', 'distill/features.py', 'distill/inference.py', 'distill/coarse_solver.py'):
        fingerprint['code:'+name] = hashlib.sha256((REPO/name).read_bytes()).hexdigest()
    output = external_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(gpu=torch.cuda.get_device_name(), sites=sites, lods=args.lods, rows=[],
                  checkpoint_digests=fingerprint, teacher_sources=source_digest(),
                  site_manifest_digest=manifest_digest,
                  fresh_world_per_partition=True, fresh_world_per_viewer_tile=True,
                  tile_size=256, halo=24, accepted=False,
                  visual_review=dict(passed=False, reason='Review whole/tiled physical images before selection.'))
    if (output/'report.json').exists():
        old = json.loads((output/'report.json').read_text())
        for key in ('gpu', 'sites', 'lods', 'checkpoint_digests', 'teacher_sources', 'site_manifest_digest',
                    'fresh_world_per_viewer_tile'):
            if old.get(key) != report[key]:
                raise ValueError('Seam evidence changed; use a new output directory.')
        report = old
    from terrain_app import load_pipeline
    from terrain_inference import configure_world
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    import terrain_server as server
    server.existing_final_mip = lambda *unused: None
    from terrain_snr import lod_relief
    from terrain_generation import resolve_generation
    loaded = load_pipeline(sites[0]['seed'])
    config = {k:v for k,v in dict(loaded.config).items() if not k.startswith('_')}
    def world_for(site, variant):
        world = WorldPipeline(**(config | dict(seed=site['seed'], dtype='bf16', latents_batch_size=16,
            cache_limit=1024**3, torch_compile=False, log_mode='silent')))
        world.coarse_model, world.base_model, world.decoder_model = (
            loaded.coarse_model, loaded.base_model, loaded.decoder_model)
        configure_world(world, replace(loaded._terrain_profile, cuda_graphs=False), world_profile=site['profile'])
        if variant == 'student_all':
            install(world, models)
        world.bind()
        return world
    for site in sites:
        for lod in args.lods:
            span = 256 * 30 * 2**lod
            tx, ty = int(np.floor(site['x']/span)), int(np.floor(site['y']/span))
            for variant in ['reference'] + (['student_all'] if paths else []):
                if any(r['site'] == site['name'] and r['lod'] == lod and r['variant'] == variant for r in report['rows']):
                    continue
                name = f"{site['name']}-lod{lod}-{variant}"
                whole_world = world_for(site, variant)
                # 816-wide request spans [-280, 536); its central 512 samples
                # exactly cover the cores of the four 304-wide viewer requests.
                whole, _ = server.sample_elevation(whole_world, lod, tx, ty, halo=280)
                if lod >= 3:
                    whole = gaussian_filter(whole, sigma=.65, mode='reflect').astype(np.float32)
                whole = lod_relief(resolve_generation(site['profile']).settings, whole, lod)[280:792, 280:792]
                del whole_world
                gc.collect()
                torch.cuda.empty_cache()
                tiles = {}
                # Each viewer tile starts cold too: a shared neighbouring cache
                # must not hide a disagreement caused by request partitioning.
                for y, x in [(1, 1), (1, 0), (0, 1), (0, 0)]:
                    tile_world = world_for(site, variant)
                    value, _, stage = server.sample_physical(tile_world, site['seed'], site['profile'], lod, tx+x, ty+y)
                    if stage not in ('latent', 'decoder'):
                        raise ValueError(f'Unexpected cached/preview stage: {stage}')
                    tiles[(y, x)] = value
                    del tile_world
                    gc.collect()
                    torch.cuda.empty_cache()
                metrics, mosaic = partition_stats(whole, tiles)
                row = dict(site=site['name'], lod=lod, variant=variant, **metrics)
                report['rows'].append(row)
                atomic_write(output/(name+'.npz'), lambda f: np.savez_compressed(f, whole=whole, tiled=mosaic,
                             **{f'tile_{y}_{x}':v for (y,x),v in tiles.items()}))
                preview(output/(name+'.png'), whole, mosaic, lod, name)
                atomic_json(output/'report.json', report)
                print(json.dumps(row), flush=True)
    with (output/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report['rows'][0]))
        writer.writeheader()
        writer.writerows(report['rows'])
    atomic_json(output/'report.json', report | dict(status='complete'))


if __name__ == '__main__':
    main()
