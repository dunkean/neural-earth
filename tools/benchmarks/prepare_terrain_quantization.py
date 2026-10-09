"""Offline CLI; capture/build require exclusive GPU ownership. Export is CPU.

Nothing in this file is imported by the terrain server. No runtime activation.
"""
from __future__ import annotations

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()


import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

from terrain_quantization import (ROLE, DEFAULT_EXCLUDE, ActivationCollector, FrozenMP,
    freeze_base, sha_file, identity, runtime_identity, write_json, export_qdq,
    build_tensorrt, require_scope)


def load_frozen(path):
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    archive = torch.load(path, map_location='cpu', weights_only=True)
    require_scope(role=archive['role'])
    model = EDMUnet2D.from_config(archive['config'])
    for name, spec in archive['specs'].items():
        weight = archive['state'][name+'.weight']
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, FrozenMP(weight, groups=spec['groups'], no_padding=spec['no_padding']))
    # Preserve the dtype of non-MP embedding/normalization tensors as well.
    model.load_state_dict(archive['state'], strict=True, assign=True)
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def capture(args):
    # GPU imports and initialization deliberately occur ONLY in this subcommand.
    from terrain_app import load_pipeline, resolve_model_source
    from terrain_inference import configure_world
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    output = args.output.resolve()
    sites = json.loads(args.sites.read_text(encoding='utf-8'))
    if not isinstance(sites, list) or not sites:
        raise ValueError('Sites must be a nonempty JSON list')
    for site in sites:
        if set(site) != {'seed', 'profile', 'row', 'col', 'size'}:
            raise ValueError('Each calibration site needs seed/profile/row/col/size exactly')
        if any(type(site[k]) is not int for k in ('seed', 'row', 'col', 'size')):
            raise ValueError('Site coordinates and seed must be integers')
        if not 0 <= site['seed'] < 2**64 or not 1 <= site['size'] <= 256:
            raise ValueError('Invalid seed or bounded latent crop size')
    if args.max_calls < 2 or args.weight_budget_mib < 1 or args.data_budget_mib < 1:
        raise ValueError('Positive budgets and at least two forwards are required')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(role=ROLE, state='running', runtime_enabled=False, sites=sites,
                  runtime=runtime_identity(), packets=[], freeze_checks=[])
    write_json(output/'capture.json', report)
    collector = None
    handle = None
    worlds = []
    try:
        model_source = Path(resolve_model_source()).resolve()
        loaded = load_pipeline(sites[0]['seed'])
        raw = loaded.base_model.model if hasattr(loaded.base_model, '_buckets') else loaded.base_model
        frozen, specs = freeze_base(raw, max_weight_bytes=args.weight_budget_mib*1024**2)
        collector = ActivationCollector(max_calls=args.max_calls).install(raw)
        context, saved_bytes = {}, 0

        def record(module, positional, kwargs):
            nonlocal saved_bytes
            sample = positional[0]
            labels = kwargs['noise_labels']
            conditions = kwargs['conditional_inputs'][0]
            metadata = dict(context, batch=int(sample.shape[0]), noise_labels=labels.float().cpu().tolist())
            if not collector.start(metadata):
                return
            packet = dict(sample=sample.detach().cpu(), noise_labels=labels.detach().cpu(),
                          conditions=conditions.detach().cpu(), metadata=metadata)
            byte_count = sum(packet[k].numel()*packet[k].element_size() for k in ('sample','noise_labels','conditions'))
            if saved_bytes + byte_count > args.data_budget_mib*1024**2:
                raise ValueError('Calibration packet budget exceeded; reduce sites/max-calls')
            saved_bytes += byte_count
            path = output / f'input-{len(report["packets"]):04d}.pt'
            torch.save(packet, path)
            report['packets'].append(dict(file=path.name, sha256=sha_file(path), **metadata))

        handle = raw.register_forward_pre_hook(record, with_kwargs=True)
        config = {k: v for k,v in dict(loaded.config).items() if not k.startswith('_')}
        for site in sites:
            context.clear()
            context.update(site)
            world = WorldPipeline(**(config | dict(seed=site['seed'], dtype='bf16',
                latents_batch_size=16, torch_compile=False, cache_limit=128*1024**2)))
            worlds.append(world)
            world.coarse_model, world.base_model, world.decoder_model = (
                loaded.coarse_model, loaded.base_model, loaded.decoder_model)
            configure_world(world, replace(loaded._terrain_profile, cuda_graphs=False),
                            world_profile=site['profile'])
            context['generation_settings'] = world._terrain_generation_settings
            world.bind()
            _ = world.latents[:, site['row']:site['row']+site['size'],
                                site['col']:site['col']+site['size']]
            world.close()
            worlds.remove(world)
        handle.remove()
        handle = None
        collector.close()
        calibration = collector.result()
        for site in sites:
            matching = [p for p in report['packets'] if all(p[k] == site[k] for k in site)]
            labels = {tuple(p['noise_labels'])[:1] for p in matching}
            if len(labels) < 2:
                raise ValueError('Calibration budget missed a site or one latent pass; increase max-calls')
        # Validate weight freezing on all captured batches before serializing.
        for packet_info in report['packets']:
            packet = torch.load(output/packet_info['file'], weights_only=True)
            device = next(raw.parameters()).device
            inputs = [packet[k].to(device) for k in ('sample','noise_labels','conditions')]
            actual = frozen(inputs[0], noise_labels=inputs[1], conditional_inputs=[inputs[2]])
            expected = raw(inputs[0], noise_labels=inputs[1], conditional_inputs=[inputs[2]])
            exact = torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
            report['freeze_checks'].append(dict(file=packet_info['file'], byte_exact=exact))
            if not exact:
                raise ValueError('Effective weight freezing changed a captured base forward')
        frozen.cpu()
        archive = dict(role=ROLE, config=json.loads(json.dumps(dict(raw.config))), specs=specs,
                       state=frozen.state_dict())
        torch.save(archive, output/'frozen-base.pt')
        source = model_source/'base_model'
        names = ('terrain_app', 'terrain_inference', 'terrain_nn_constants', 'terrain_cuda_kernels', 'terrain_generation',
                 'terrain_conditioning', 'terrain_diffusion.models.edm_unet',
                 'terrain_diffusion.models.unet_block', 'terrain_diffusion.models.mp_layers',
                 'terrain_diffusion.inference.world_pipeline')
        source_paths = {name: str(Path(sys.modules[name].__file__).resolve())
                        for name in names if name in sys.modules}
        source_paths['prepare_terrain_quantization'] = str(Path(__file__).resolve())
        report.update(state='complete', calibration=calibration, effective_weights=specs,
            checkpoint={name:sha_file(source/name) for name in ('config.json','diffusion_pytorch_model.safetensors')},
            frozen_sha=sha_file(output/'frozen-base.pt'), saved_input_bytes=saved_bytes,
            sources={name:dict(path=path, sha256=sha_file(path)) for name,path in source_paths.items()},
            cuda_device=torch.cuda.get_device_name(), source_weight_device=str(next(raw.parameters()).device),
            physical_holdout_passed=False, integer_gpu_execution_proven=False)
        report['artifact_identity'] = identity(report)
    except BaseException as exc:
        report.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if handle is not None:
            handle.remove()
        if collector is not None:
            collector.close()
        for world in worlds:
            world.close()
        write_json(output/'capture.json', report)
    return report


def export(args):
    report = json.loads((args.capture/'capture.json').read_text(encoding='utf-8'))
    require_scope(role=report['role'])
    if report.get('artifact_identity') != identity({k:v for k,v in report.items() if k != 'artifact_identity'}):
        raise ValueError('Capture manifest identity mismatch')
    if report.get('state') != 'complete' or not report.get('freeze_checks') or not all(x['byte_exact'] for x in report['freeze_checks']):
        raise ValueError('Export requires complete byte-exact freezing evidence')
    if sha_file(args.capture/'frozen-base.pt') != report['frozen_sha']:
        raise ValueError('Frozen base SHA mismatch')
    for info in report['packets']:
        if sha_file(args.capture/info['file']) != info['sha256']:
            raise ValueError('Calibration input SHA mismatch')
    model = load_frozen(args.capture/'frozen-base.pt')
    packet = torch.load(args.capture/report['packets'][0]['file'], weights_only=True)
    result = export_qdq(model, tuple(packet[k] for k in ('sample','noise_labels','conditions')),
        report['calibration'], args.output, exclude=tuple(args.exclude) if args.exclude else None)
    result['capture_identity'] = report['artifact_identity']
    result['checkpoint'] = report['checkpoint']
    result['effective_weights_identity'] = identity(report['effective_weights'])
    result['artifact_identity'] = identity({k:v for k,v in result.items() if k != 'artifact_identity'})
    write_json(args.output/'export.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cap = sub.add_parser('capture', help='GPU: exclusive offline real latent forwards')
    cap.add_argument('--sites', type=Path, required=True)
    cap.add_argument('--output', type=Path, required=True)
    cap.add_argument('--max-calls', type=int, default=128)
    cap.add_argument('--weight-budget-mib', type=int, default=1024)
    cap.add_argument('--data-budget-mib', type=int, default=256)
    exp = sub.add_parser('export', help='CPU: frozen base + calibrated Q/DQ ONNX, not validated inference')
    exp.add_argument('--capture', type=Path, required=True)
    exp.add_argument('--output', type=Path, required=True)
    exp.add_argument('--exclude', action='append', help='Replacement sensitive-layer substring list')
    build = sub.add_parser('build', help='GPU: optional TensorRT offline build and layer inspection')
    build.add_argument('--onnx', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--workspace-mib', type=int, default=512)
    build.add_argument('--max-batch', type=int, default=16)
    args = parser.parse_args()
    # Resolve before terrain_app changes the process working directory.
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.command == 'capture':
        result = capture(args)
    elif args.command == 'export':
        result = export(args)
    else:
        result = build_tensorrt(args.onnx, args.output, workspace_mib=args.workspace_mib, max_batch=args.max_batch)
    print(json.dumps(dict(state=result['state'], runtime_enabled=False, output=str(args.output))))


if __name__ == '__main__':
    main()
