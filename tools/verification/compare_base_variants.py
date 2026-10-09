"""Offline physical comparison of base/decoder inference variants against production.

Variants: reference (production BF16, T=2), fp32base (BF16 noise floor),
onestep (upstream onestep_latent), fp16base, fp16dec. Fresh in-memory worlds per
(variant, site, LOD); no runtime cache is read or written. Window counts are
exact; seconds are indicative when another GPU process runs.

  python tools/verification/compare_base_variants.py run OUTPUT_DIR [--variants ...]
  python tools/verification/compare_base_variants.py sheet OUTPUT_DIR
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from terrain_paths import REPO_ROOT

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

VARIANTS = ['reference', 'fp32base', 'onestep', 'fp16base', 'fp16dec']
LODS = (3, 0)
# Quantization holdout sites (LOD 3 tiles); their centre is used at every LOD.
HOLDOUT = [dict(seed=101, profile='natural', tx=17, ty=-23),
           dict(seed=202, profile='terrestrial-earthlike', tx=-30, ty=11),
           dict(seed=303, profile='natural', tx=2, ty=-6),
           dict(seed=404, profile='terrestrial-archipelago', tx=-7, ty=24)]


def sites():
    shots = json.loads((REPO_ROOT / 'docs/images/screenshots.json').read_text(encoding='utf-8'))
    result = [dict(name=name, seed=int(s['seed']), profile='orogen', x=s['x'], y=s['y'])
              for name, s in shots['sites'].items()]
    span = 256 * 30 * 8
    for i, s in enumerate(HOLDOUT):
        result.append(dict(name=f"holdout{i}-{s['profile']}", seed=s['seed'], profile=s['profile'],
                           x=(s['tx'] + .5) * span, y=(s['ty'] + .5) * span))
    return result


def tile(meters, lod):
    return int(np.floor(meters / (256 * 30 * 2**lod)))


class Cast(nn.Module):
    """Separately loaded network copy in another dtype; returns the caller's dtype."""
    def __init__(self, model, dtype):
        super().__init__()
        self.model, self.dtype = model, dtype

    def forward(self, x, noise_labels=None, conditional_inputs=None, **kw):
        conds = [c.to(self.dtype) for c in (conditional_inputs or [])]
        out = self.model(x.to(self.dtype), noise_labels=noise_labels.to(self.dtype), conditional_inputs=conds, **kw)
        if not torch.isfinite(out).all():
            raise FloatingPointError(f'non-finite output in {self.dtype}')
        return out.to(x.dtype)


class Count(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model, self.windows, self.calls = model, 0, 0

    def forward(self, x, *args, **kw):
        self.windows += x.shape[0]
        self.calls += 1
        return self.model(x, *args, **kw)


def load_copy(folder, dtype):
    from types import MethodType
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    from terrain_diffusion.models.mp_layers import MPConv, MPConvResample
    from terrain_app import resolve_model_source
    import terrain_inference
    with torch.inference_mode(False):  # weight version counters feed the eval cache
        model = EDMUnet2D.from_pretrained(str(Path(resolve_model_source()) / folder)).to('cuda', dtype).eval()
    for module in model.modules():
        if isinstance(module, (MPConv, MPConvResample)):
            module.__dict__['_terrain_original_forward'] = module.forward
            module.forward = MethodType(terrain_inference._mp_forward, module)
    return model


def errors(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(mae=float(d.mean()), rmse=float(np.sqrt((d**2).mean())), p99=float(np.quantile(d, .99)),
                max=float(d.max()), coast=float(np.mean((a > 0) != (b > 0))))


def describe(e, lod):
    gy, gx = np.gradient(e.astype(np.float64), 30 * 2**lod)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    centred = e.astype(np.float64) - e.mean()
    window = np.outer(np.hanning(e.shape[0]), np.hanning(e.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centred * window)))**2
    yy, xx = np.indices(power.shape)
    radius = np.hypot((yy - power.shape[0] / 2) / power.shape[0], (xx - power.shape[1] / 2) / power.shape[1])
    bands = [(0, .03125), (.03125, .0625), (.0625, .125), (.125, .25), (.25, .5)]  # cycles/sample
    return dict(mean=float(e.mean()), std=float(e.std()), land=float((e > 0).mean()),
                slope_mean=float(slope.mean()), slope_p90=float(np.quantile(slope, .9)),
                psd=[float(power[(radius >= lo) & (radius < hi)].mean()) for lo, hi in bands])


@torch.inference_mode()
def run(output, variants):
    from dataclasses import replace
    from terrain_app import load_pipeline
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    from terrain_inference import configure_world
    import terrain_server as server
    server.existing_final_mip = lambda *unused: None
    output.mkdir(parents=True, exist_ok=True)
    site_list = sites()
    loaded = load_pipeline(site_list[0]['seed'])
    config = {k: v for k, v in dict(loaded.config).items() if not k.startswith('_')}
    unwrap = lambda m: m.model if hasattr(m, '_buckets') else m
    raw_base, raw_decoder = unwrap(loaded.base_model), unwrap(loaded.decoder_model)
    report_path, arrays_path = output / 'report.json', output / 'arrays.npz'
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(
        sites=site_list, gpu=torch.cuda.get_device_name(), torch=torch.__version__, variants={},
        note='Fresh in-memory worlds; seconds are indicative when another GPU process runs.')
    arrays = dict(np.load(arrays_path)) if arrays_path.exists() else {}
    for variant in variants:
        base, decoder = raw_base, raw_decoder
        if variant == 'fp32base':
            base = Cast(load_copy('base_model', torch.float32), torch.float32)
        elif variant == 'fp16base':
            base = Cast(load_copy('base_model', torch.float16), torch.float16)
        elif variant == 'fp16dec':
            decoder = Cast(load_copy('decoder_model', torch.float16), torch.float16)
        results = report['variants'][variant] = []
        for site in site_list:
            for lod in LODS:
                tx, ty = tile(site['x'], lod), tile(site['y'], lod)
                item, world = dict(site=site['name'], lod=lod, tx=tx, ty=ty), None
                try:
                    world = WorldPipeline(**(config | dict(seed=site['seed'], dtype='bf16', latents_batch_size=16,
                        cache_limit=1024 * 1024**2, torch_compile=False, log_mode='silent',
                        onestep_latent=variant == 'onestep')))
                    world.coarse_model, world.base_model, world.decoder_model = (
                        loaded.coarse_model, loaded.base_model, loaded.decoder_model)
                    configure_world(world, replace(loaded._terrain_profile, cuda_graphs=False),
                                    world_profile=site['profile'])
                    counter = Count(base)
                    world.base_model, world.decoder_model = counter, decoder
                    world.bind()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    elevation, _, stage = server.sample_physical(world, site['seed'], site['profile'], lod, tx, ty)
                    torch.cuda.synchronize()
                    item.update(stage=stage, seconds=time.perf_counter() - started,
                                base_windows=counter.windows, base_calls=counter.calls, **describe(elevation, lod))
                    arrays[f"{variant}|{site['name']}|{lod}"] = elevation
                    reference = arrays.get(f"reference|{site['name']}|{lod}")
                    if variant != 'reference' and reference is not None:
                        item['vs_reference'] = errors(elevation, reference)
                except Exception as exc:  # keep the sweep going; record failures
                    item['error'] = f'{type(exc).__name__}: {exc}'[:500]
                finally:
                    if world is not None:
                        world.close()
                    gc.collect()
                results.append(item)
                print(json.dumps({k: v for k, v in item.items() if k != 'psd'}), flush=True)
                report_path.write_text(json.dumps(report, indent=1))
                np.savez_compressed(arrays_path, **arrays)
        del base, decoder
        gc.collect()
        torch.cuda.empty_cache()


def sheet(output):
    """One hillshade sheet per site group: rows = site x LOD, columns = variants."""
    from PIL import Image, ImageDraw
    arrays = np.load(output / 'arrays.npz')
    names = [s['name'] for s in sites()]
    def shade(e, spacing, exaggeration):
        gy, gx = np.gradient(e.astype(np.float64) * exaggeration, spacing)
        slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
        light = np.clip(np.sin(np.radians(45)) * np.cos(slope) +
                        np.cos(np.radians(45)) * np.sin(slope) * np.cos(np.radians(315) - aspect), 0, 1)
        colour = np.where((e > 0)[..., None], [200, 185, 150], [120, 150, 190]) / 255.0
        return (colour * (0.25 + 0.75 * light[..., None]) * 255).astype(np.uint8)
    for label, group in (('land', names[:3]), ('holdout', names[3:])):
        size, pad, head = 304, 6, 22
        image = Image.new('RGB', (len(VARIANTS) * (size + pad) + 160, len(group) * len(LODS) * (size + pad) + head),
                          (25, 25, 25))
        draw = ImageDraw.Draw(image)
        for c, variant in enumerate(VARIANTS):
            draw.text((160 + c * (size + pad) + 4, 4), variant, fill=(255, 255, 255))
        for row, (name, lod) in enumerate((n, l) for n in group for l in LODS):
            y = head + row * (size + pad)
            draw.text((4, y + size // 2), f'{name[:20]}\nLOD {lod}', fill=(255, 255, 255))
            for c, variant in enumerate(VARIANTS):
                key = f'{variant}|{name}|{lod}'
                if key in arrays:
                    image.paste(Image.fromarray(shade(arrays[key], 30 * 2**lod, 3 if lod == 3 else 1.5)),
                                (160 + c * (size + pad), y))
        image.save(output / f'sheet-{label}.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['run', 'sheet'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--variants', nargs='+', default=VARIANTS, choices=VARIANTS)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.output, args.variants)
    else:
        sheet(args.output)


if __name__ == '__main__':
    main()
