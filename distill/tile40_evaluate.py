"""Evaluate complete40km numerical-reference tiles,with clearly marked controls."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import scipy
import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, external_path
from distill.native_progress_images import shade
from distill.native_weight_quant import file_sha
from distill.student import load_student
from distill.tile40_export_coarse import ExportCoarse
from distill.tile40_reference_data import guided_reference, resample_tile, tile_stats
from distill.tile40_runtime import Tile40
from distill.tile40_conditioning import CLIMATES, RELIEFS


def directional_coherence(height):
    """Energy-weighted ridge orientation after removing the960m background."""
    field=np.asarray(height,np.float64)
    high=field-gaussian_filter(field,32.)
    dy,dx=np.gradient(high)
    aa=gaussian_filter(dx*dx,5.)[96:-96,96:-96]
    bb=gaussian_filter(dy*dy,5.)[96:-96,96:-96]
    ab=gaussian_filter(dx*dy,5.)[96:-96,96:-96]
    energy=aa+bb
    coherence=np.sqrt((aa-bb)**2+4*ab**2)/(energy+1e-8)
    return float((coherence*energy).sum()/max(float(energy.sum()),1e-8))


def _reference_fields(target, cache_root):
    """Reuse exact FP64 reference bands across checkpoints;verify derived data."""
    tag=f'{np.__version__}:{scipy.__version__}:{file_sha(Path(__file__))}'
    digest=hashlib.sha256(tag.encode())
    digest.update(target.tobytes())
    key=digest.hexdigest()
    path=external_path(cache_root)/f'{key}.npz'
    sigmas=(2.,8.,32.,64.)
    array_sha=lambda value:hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
    if path.exists():
        try:
            with np.load(path,allow_pickle=False) as saved:
                metadata=json.loads(str(saved['metadata']))
                bands=[saved[f'band{index}'].copy() for index in range(4)]
            json.dumps(metadata,allow_nan=False)
            if metadata['key']!=key or any(
                band.dtype!=np.float64 or band.shape!=(1334-6*int(sigma),)*2 or
                array_sha(band)!=metadata['band_sha256'][index]
                for index,(sigma,band) in enumerate(zip(sigmas,bands))):
                raise ValueError('Invalid reference metric cache.')
            return bands,metadata
        except (OSError,ValueError,KeyError,IndexError,EOFError,zipfile.BadZipFile):
            # This is derived data;rebuild damaged entries from the true target.
            pass
    bands=[]
    for sigma in sigmas:
        margin=int(3*sigma)
        bands.append((target-gaussian_filter(target,sigma))[margin:-margin,margin:-margin])
    metadata=dict(key=key,band_sha256=[array_sha(band) for band in bands],
        coherence=directional_coherence(target),stats=tile_stats(target,40000./1334))
    atomic_write(path,lambda handle:np.savez(handle,metadata=np.array(json.dumps(metadata,allow_nan=False)),
        **{f'band{index}':band for index,band in enumerate(bands)}))
    return bands,metadata


def compare_height(prediction, target, reference_cache=None):
    p, t = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    if p.shape != (1334, 1334) or t.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Both physical comparisons must cover the complete40km tile.')
    delta = p-t
    cached=_reference_fields(t,reference_cache) if reference_cache is not None else None
    bands = []
    for index,sigma in enumerate((2., 8., 32., 64.)):
        margin = int(3*sigma)
        a = (p-gaussian_filter(p, sigma))[margin:-margin, margin:-margin]
        b = cached[0][index] if cached is not None else (t-gaussian_filter(t, sigma))[margin:-margin, margin:-margin]
        aa, bb = float(np.sqrt(np.mean(a*a))), float(np.sqrt(np.mean(b*b)))
        norm = float(np.linalg.norm(a)*np.linalg.norm(b))
        bands.append(dict(sigma_pixels=sigma, sigma_m=sigma*40000./1334,
            prediction_rms_m=aa, reference_rms_m=bb, amplitude_ratio=aa/max(bb, 1e-5),
            correlation=float(np.sum(a*b)/norm) if norm > 1e-8 else None))
    return dict(height_mae_m=float(np.mean(np.abs(delta))), height_rmse_m=float(np.sqrt(np.mean(delta*delta))),
                directional_coherence=dict(prediction=directional_coherence(p),reference=cached[1]['coherence'] if cached is not None else directional_coherence(t),
                    scope='960m background removed;150m local gradient tensor;energy-weighted;diagnostic,not an acceptance score.'),
                prediction_stats=tile_stats(p, 40000./1334), reference_stats=cached[1]['stats'] if cached is not None else tile_stats(t, 40000./1334), bands=bands)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA/'tile40-20261014/reference-coarse2-guided-v4')
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--decoder', type=Path, default=DATA/'tile40-20261014/decoder16-pilot-v1/step-020000.pt')
    parser.add_argument('--encoder-weights', choices=('raw', 'ema'), default='ema')
    parser.add_argument('--decoder-weights', choices=('raw', 'ema'), default='ema')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--reference-metric-cache',type=Path,default=DATA/'cache/tile40-reference-metrics-v1')
    parser.add_argument('--no-reference-metric-cache',dest='reference_metric_cache',action='store_const',const=None)
    parser.add_argument('--oracle-controls', action='store_true', help='Separate decoder-only controls with supplied teacher latents.')
    parser.add_argument('--original-decoder-control',action='store_true',help='Diagnostic:original decoder on generated latents;not a browser product.')
    parser.add_argument('--flow', action='store_true', help='Generate an independent conditional sample with four small velocity calls.')
    parser.add_argument('--finite-flow', action='store_true', help='Single finite tile boundary;no infinite-world dependency halo.')
    parser.add_argument('--whole-decoder', action='store_true', help='Decode the1856px finite root field in one call.')
    parser.add_argument('--flow-split', type=float, default=.5, help='Middle integration boundary;two midpoint intervals still use four calls.')
    parser.add_argument('--flow-steps', type=int, default=2, help='Midpoint intervals;larger values measure the sampling quality/cost tradeoff.')
    args = parser.parse_args()
    if args.finite_flow and not args.flow:
        parser.error('--finite-flow requires --flow.')
    output = external_path(args.output)
    if output.exists() or args.limit < 1:
        raise ValueError('A fresh output and a positive bounded count are required.')
    output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    if manifest['internal_coarse_checkpoint']['sha256'] != file_sha(REPO/'distill/weights/coarse2.pt'):
        raise ValueError('Reference conditioning uses different coarse weights.')
    constant_case = next(row for row in manifest['cases'] if row['split']=='train')
    with np.load(args.dataset/constant_case['file'], allow_pickle=False) as item:
        parameters = json.loads(str(item['metadata']))['world_config']['parameters']
    coarse, _ = load_student(REPO/'distill/weights/coarse2.pt')
    encoder, checkpoint = load_student(args.encoder, ema=args.encoder_weights=='ema')
    if args.flow:
        from distill.tile40_flow_pilot import FlowEncoder, FiniteFlowEncoder
        encoder_class = FiniteFlowEncoder if args.finite_flow else FlowEncoder
        encoder = encoder_class(encoder, checkpoint['contract'], steps=args.flow_steps, split=args.flow_split)
    decoder_sha=file_sha(args.decoder)
    if args.original_decoder_control:
        if args.device!='cuda':raise ValueError('Bound original decoder diagnostic toCUDA.')
        from distill.tile40_original_decoder_control import OriginalDecoderControl
        from distill.native_base import teacher_source
        decoder=OriginalDecoderControl(args.device)
        decoder_sha=file_sha(teacher_source().parent/'decoder_model/diffusion_pytorch_model.safetensors')
    else:
        decoder, _ = load_student(args.decoder, ema=args.decoder_weights=='ema')
    generator = Tile40(ExportCoarse(coarse.solver), encoder, decoder, parameters, args.device,
                       whole_decoder=args.whole_decoder)
    rows = []
    # Interleave relief classes;small pilots do not accidentally inspect only mountains.
    cases = sorted((r for r in manifest['cases'] if r['split']=='val'),
                   key=lambda r:(r['seed']//10, list(CLIMATES).index(r['climate']), RELIEFS.index(r['relief'])))[:args.limit]
    for case in cases:
        source = args.dataset/case['file']
        if file_sha(source) != json.loads(source.with_suffix('.json').read_text())['file_sha256']:
            raise ValueError('Validation target changed.')
        with np.load(source, allow_pickle=False) as item:
            elevation = item['sketch_m'].copy()
            reference = item['height_m'].copy()
            target_latents = item['latents'].astype(np.float32)
            target_coarse = item['coarse'].copy()
        actual = generator.generate(elevation, case['climate'], case['seed'], trace=True)
        comparison = compare_height(actual['height_m'], reference,reference_cache=args.reference_metric_cache)
        controls = {}
        if args.oracle_controls:
            native = generator.decode_latents(torch.from_numpy(target_latents).to(args.device), case['seed'])
            guided = resample_tile(guided_reference(native, elevation)['native_height'])
            controls['teacher_latents_decoder_only'] = compare_height(guided, reference,reference_cache=args.reference_metric_cache)
        name = source.stem
        atomic_write(output/f'{name}.npz', lambda handle:np.savez_compressed(handle,
            height_m=actual['height_m'], reference_height_m=reference, sketch_m=elevation,
            metadata=np.array(json.dumps(dict(case=case, side_m=40000., pixels=1334)))))
        pair = Image.new('RGB', (1024, 512))
        pair.paste(shade(reference, 40000./1334, 1.5).resize((512, 512), Image.Resampling.LANCZOS), (0, 0))
        pair.paste(shade(actual['height_m'], 40000./1334, 1.5).resize((512, 512), Image.Resampling.LANCZOS), (512, 0))
        atomic_write(output/f'{name}.png', lambda handle:pair.save(handle, format='PNG'))
        offset = -8-generator.coarse_origin
        comparable_coarse = actual['coarse'][:, offset:offset+24, offset:offset+24]
        row = dict(case=case, metrics=comparison, timing=actual['timing'], controls=controls,
                   coarse_mae_by_channel=np.abs(comparable_coarse-target_coarse).mean(axis=(1, 2)).tolist(),
                   latent_mse_by_channel=np.square(actual['latents']-target_latents).mean(axis=(1, 2)).tolist())
        rows.append(row)
        atomic_json(output/'report.json', dict(status='running', rows=rows, encoder_step=checkpoint['step']))
        print(json.dumps(row), flush=True)
    report = dict(status='complete', rows=rows, encoder_step=checkpoint['step'],
        reference_metric_cache=str(args.reference_metric_cache) if args.reference_metric_cache is not None else None,
        evaluation_producer_sha256=file_sha(Path(__file__)),
        selected_weights=dict(encoder=args.encoder_weights, decoder=args.decoder_weights),
        decoder_layout='finite-root' if args.whole_decoder else 'overlap512',
        encoder_sha256=file_sha(args.encoder), decoder_sha256=decoder_sha,original_decoder_control=args.original_decoder_control,
        coarse_sha256=file_sha(REPO/'distill/weights/coarse2.pt'), dataset_manifest_sha256=file_sha(args.dataset/'manifest.json'),
        parameters=parameters, flow=args.flow, finite_tile_flow=args.finite_flow, sampling_config=dict(intervals=args.flow_steps,
            split=args.flow_split if args.flow_steps == 2 else None, evaluations=2*args.flow_steps) if args.flow else None,
        scope='Original decoder control on generated latents;no teacher latent inputs;not a browser product or timing proof.' if args.original_decoder_control else 'Complete40km numerical-reference generation from sketch/climate/seed;native timings are not browser proof.',
        pixel_comparison='Descriptive only for independent flow samples;assess morphology,band powers and sketch/climate response.' if args.flow else 'Paired-noise reference comparison.',
        boundary_policy='Root denoising on the1856px padded field;final crop7680m inside boundary;not exact upstream48px outer reconstruction.')
    atomic_json(output/'report.json', report)
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(['seed', 'relief', 'climate', 'height_mae_m', 'native_total_seconds', 'relief_p90_p10_m'])
    for row in rows:
        writer.writerow([row['case']['seed'], row['case']['relief'], row['case']['climate'],
            row['metrics']['height_mae_m'], row['timing']['native_total_seconds'], row['metrics']['prediction_stats']['relief_p90_p10_m']])
    atomic_write(output/'measurements.csv', lambda handle:handle.write(text.getvalue().encode()))
    print(f'ARTIFACT full_tile_reference={output/"report.json"}', flush=True)


if __name__ == '__main__':
    main()
