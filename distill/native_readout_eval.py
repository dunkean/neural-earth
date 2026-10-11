"""Compare autonomous frozen readouts against the source on matched rare scenes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import torch

from distill.common import DATA, atomic_json, atomic_write
from distill.native_decoded import DecodedRegion, OriginalDecoder, decoded_region_forward
from distill.native_decoded_eval import shape_validation
from distill.native_multilod import low_height
from distill.native_multilod_eval import validation_records
from distill.native_mixed_bank import KINDS
from distill.native_progress_images import LABELS, shade
from distill.native_readout_probe import ReadoutProbe, digest, load_source
from distill.decoded_loss import reconstructed_height
from distill.terrain_shape import compare_shape


def paired_interval(records, baseline, candidate):
    worlds = {}
    for record, before, after in zip(records, baseline, candidate):
        values = []
        for item in (before, after):
            row = next(v for v in item['decoded_lod0']['topographic_position'] if v['sigma_pixels'] == 2.)
            values.append(row['valley_retention'] if row['available'] else None)
        if None not in values:
            worlds.setdefault(record['metadata']['seed'], []).append(values[1]-values[0])
    if len(worlds) < 2:
        return dict(worlds=len(worlds), available=False)
    grouped = np.array([np.mean(values) for values in worlds.values()])
    rng = np.random.default_rng(20261017)
    bootstraps = grouped[rng.integers(len(grouped), size=(5000, len(grouped)))].mean(axis=1)
    return dict(worlds=len(worlds), available=True, mean_change=float(grouped.mean()),
        interval95=np.quantile(bootstraps, [.025, .975]).tolist(),
        method='Paired mean recall change, resampling independent worlds; diagnostic only.')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--mode', choices=('first', 'second', 'both'), default='first')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--operator-report', type=Path, default=DATA/'eval/native160-readout-operator-validation.json')
    parser.add_argument('--label', default='readout')
    args = parser.parse_args()
    operator = json.loads(args.operator_report.read_text())
    if operator['status'] != 'complete' or operator['probe_sha256'] != digest(args.probe) or \
            operator['sanity_relative_mse'] > .001:
        raise ValueError('Exact independently validated probe with a passing reconstruction control required.')
    saved = torch.load(args.probe, map_location='cpu', weights_only=True)
    contract = saved['contract']
    source, step = load_source(Path(contract['source']))
    if contract['source_sha256'] != digest(contract['source']) or contract['smoke']:
        raise ValueError('Full TRAIN probe and frozen source required.')
    candidate = ReadoutProbe(source, saved['coefficients'].cuda(), args.mode,
                            features=contract.get('features', 'last')).eval()
    root = DATA/'crops/native-mixed-scenes-v2'
    manifest = json.loads((root/'manifest.json').read_text())
    if contract['dataset_manifest_sha256'] != digest(root/'manifest.json'):
        raise ValueError('Frozen matched bank required.')
    stats = json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())
    records = [r for r in validation_records(manifest, stats)
               if r['coarse_variant'] == 'coarse4' and r['case_kind'] is not None]
    if set(saved['train_files']) & {r['file'] for r in records}:
        raise ValueError('TRAIN/VAL overlap.')
    torch.set_num_threads(8); torch.backends.cudnn.benchmark = True
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    decoder = OriginalDecoder('cuda')
    files = [root/r['file'] for r in records]
    baseline = shape_validation(source, files, weight, decoder)
    shapes = shape_validation(candidate, files, weight, decoder)
    for record, before, after in zip(records, baseline, shapes):
        for row in (before, after):
            row.update(seed=record['metadata']['seed'], case_kind=record['case_kind'],
                       coarse_variant=record['coarse_variant'])
    report = dict(status='complete', source_step=step, probe_sha256=digest(args.probe),
        source_sha256=contract['source_sha256'], mode=args.mode, gpu=torch.cuda.get_device_name(),
        baseline=baseline, candidate=shapes, paired_recall=paired_interval(records, baseline, shapes),
        dataset_manifest_sha256=contract['dataset_manifest_sha256'], producer_sha256=digest(Path(__file__)),
        accepted=False, note='Autonomous interior proxy on20 coarse4 rare scenes; full pipeline, cold seams and idle performance pending.')
    atomic_json(args.output, report)
    selected = [sorted((r for r in records if r['case_kind'] == kind),
                       key=lambda r: (r['metadata']['seed'], r['file']))[0] for kind in KINDS]
    rows, fields = [], []
    for record in selected:
        region = DecodedRegion(root/record['file'], 'cuda')
        latents = [region.target, decoded_region_forward(source, region, weight)['core'],
                   decoded_region_forward(candidate, region, weight)['core']]
        heights = []
        for latent in latents:
            fine = reconstructed_height(decoder(latent[None], region.decoder_noise), latent[None])[0, 0]
            heights.append({3: low_height(latent)[0, 0].cpu().numpy(), 0: fine[64:-64, 64:-64].cpu().numpy()})
        for lod in (3, 0):
            pair = [h[lod] for h in heights]; fields.append(pair)
            for label, elevation in zip(('source', 'readout'), pair[1:]):
                shape = compare_shape(elevation, pair[0])
                tpi = next(v for v in shape['topographic_position'] if v['sigma_pixels'] == 2.)
                rows.append(dict(kind=record['case_kind'], seed=record['metadata']['seed'], lod=lod,
                    model=label, mae_m=float(np.abs(elevation-pair[0]).mean()),
                    coast_disagreement=float(np.mean((elevation > 0) != (pair[0] > 0))),
                    tpi_correlation=tpi.get('tpi_correlation'), valley_retention=tpi.get('valley_retention'),
                    tpi_rms_ratio=tpi.get('tpi_rms_ratio')))
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.label):
        parser.error('Use a simple lowercase strategy label.')
    stem = f'{args.label}-{args.mode}-000000'
    output = DATA/'_base'; output.mkdir(exist_ok=True)
    image = Image.new('RGB', (250+3*524, 140+len(fields)*556), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    draw.text((12, 10), f'Ridge TRAIN {args.mode}, features {contract.get("features", "last")} | source160x3 etape{step} | coarse4', fill='white', font=font)
    draw.text((12, 39), 'Validation autonome interne, decoder original fige. Meme bruit. Aucun lissage.', fill='white', font=small)
    draw.text((12, 66), 'LOD3 :64 pixels natifs ; LOD0 :interieur384 pixels. Probe '+report['probe_sha256'][:12], fill='white', font=small)
    for column, label in enumerate(('Teacher original', 'Source figee', 'Source + readout TRAIN')):
        draw.text((250+524*column, 102), label, fill='white', font=font)
    for index, triple in enumerate(fields):
        record = selected[index//2]; lod = 3 if index % 2 == 0 else 0; y = 140+index*556
        exaggeration = 3. if lod == 3 else 1.5
        draw.multiline_text((12, y+125), LABELS[record['case_kind']]+f'\nLOD {lod}\nSeed {record["metadata"]["seed"]}\nRelief exagere x{exaggeration:g}', fill='white', font=small, spacing=8)
        for column, elevation in enumerate(triple):
            image.paste(shade(elevation, 30*2**lod, exaggeration).resize((512, 512), Image.Resampling.NEAREST), (250+column*524, y))
    meta = PngImagePlugin.PngInfo(); meta.add_text('ProbeSHA256', report['probe_sha256'])
    meta.add_text('Description', 'Teacher, frozen source and TRAIN-only readout, identical scenes and noise. Metrics in adjacent CSV/JSON. Interior proxy, not final acceptance.')
    if (output/f'{stem}.png').exists():
        raise ValueError('Refuse to overwrite a previous diagnostic image.')
    atomic_write(output/f'{stem}.png', lambda f: image.save(f, format='PNG', pnginfo=meta))
    with (output/f'{stem}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(output/f'{stem}.json', dict(**report, rendered_rows=rows))
    (output/f'{stem}.txt').write_text('Teacher, source160x3 step15000 and TRAIN-only phase-gated readout. Eight matched panels, four rare types, LOD3/0. Exact values in adjacent CSV/JSON; acceptance remains pending.\n')
    print(json.dumps(dict(status='complete', image=str(output/f'{stem}.png'), paired_recall=report['paired_recall'])), flush=True)


if __name__ == '__main__':
    main()
