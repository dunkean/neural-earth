"""Isolated base-preview experiments; never imported by the Terrain runtime.

TensorRT is loaded only by explicit GPU subcommands. The existing historical
capture/export manifests are read, verified, and never rewritten.
"""
from __future__ import annotations

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()


import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

from terrain_quantization import (ROLE, identity, require_scope, sha_file,
                                  tensorrt_network_flags, write_json)

DLL_HANDLES = []


def trusted_manifest(path, state=None):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    require_scope(role=value['role'])
    if value.get('artifact_identity') != identity({k: v for k, v in value.items() if k != 'artifact_identity'}):
        raise ValueError(f'Manifest identity mismatch: {path}')
    if state is not None and value.get('state') != state:
        raise ValueError(f'Unexpected manifest state: {path}')
    return value


def load_trt(directory):
    directory = Path(directory).resolve()
    if not (directory/'tensorrt_bindings').is_dir():
        raise ValueError('Expected isolated TensorRT pip target')
    sys.path.insert(0, str(directory))
    for candidate in (directory/'tensorrt_libs', Path(torch.__file__).parent/'lib'):
        if os.name == 'nt':
            DLL_HANDLES.append(os.add_dll_directory(str(candidate)))
    import tensorrt as trt
    tensorrt_network_flags(trt)
    return trt


def error(actual, expected):
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite or mismatching outputs')
    delta = np.abs(a-b)
    return dict(max_abs=float(delta.max()), mae=float(delta.mean()),
                rmse=float(np.sqrt(np.mean(delta**2))), p99=float(np.quantile(delta, .99)),
                byte_exact=actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes())


def build(args):
    export = trusted_manifest(args.export/'export.json', 'prepared-not-validated')
    filename = 'base-int8-qdq.onnx' if args.variant == 'qdq' else 'base-fp32-conversion.onnx'
    source = args.export/filename
    if sha_file(source) != export['files'][filename]:
        raise ValueError('ONNX SHA mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(role=ROLE, runtime_enabled=False, state='building', variant=args.variant,
                  export_identity=export['artifact_identity'], onnx_sha=sha_file(source),
                  source_sha=sha_file(__file__), tf32=False, max_batch=16,
                  workspace_mib=2048, integer_gpu_execution_proven=False)
    write_json(args.output/'build.json', report)
    started = time.perf_counter()
    try:
        trt = load_trt(args.trt)
        report.update(tensorrt=trt.__version__, torch=torch.__version__,
                      gpu=torch.cuda.get_device_name(), network_typing='STRONGLY_TYPED')
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(tensorrt_network_flags(trt))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(source.read_bytes()):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2048*1024**2)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            shape = tuple(tensor.shape)
            if shape[0] != -1 or any(d <= 0 for d in shape[1:]):
                raise ValueError(f'Unbounded shape {tensor.name}: {shape}')
            if profile.set_shape(tensor.name, (1, *shape[1:]), (16, *shape[1:]), (16, *shape[1:])) is False:
                raise RuntimeError('Failed to set profile')
            expected_shapes = [(1, *shape[1:]), (16, *shape[1:]), (16, *shape[1:])]
            if [tuple(x) for x in profile.get_shape(tensor.name)] != expected_shapes:
                raise RuntimeError('Profile getter disagrees with requested shapes')
        if not bool(profile):
            raise RuntimeError('Invalid optimization profile')
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('TensorRT build returned no engine')
        plan = args.output/'base-preview.plan'
        plan.write_bytes(bytes(serialized))
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError('Cannot deserialize engine')
        inspector = engine.create_engine_inspector()
        (args.output/'layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON), encoding='utf-8')
        report.update(state='built-not-executed', plan_sha=sha_file(plan),
                      device_memory_bytes=int(engine.device_memory_size),
                      tensor_io=[dict(name=engine.get_tensor_name(i),
                                      dtype=str(engine.get_tensor_dtype(engine.get_tensor_name(i))),
                                      shape=list(engine.get_tensor_shape(engine.get_tensor_name(i))))
                                 for i in range(engine.num_io_tensors)])
    except BaseException as exc:
        report.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['seconds'] = time.perf_counter()-started
        write_json(args.output/'build.json', report)
    return report


class TRTBase(nn.Module):
    """Private experimental pointer wrapper, with no native runtime integration."""
    def __init__(self, directory, trt_directory, config, export_identity):
        super().__init__()
        trt = load_trt(trt_directory)
        report = json.loads((directory/'build.json').read_text(encoding='utf-8'))
        require_scope(role=report['role'])
        if report['state'] != 'built-not-executed' or sha_file(directory/'base-preview.plan') != report['plan_sha']:
            raise ValueError('Unverified engine')
        if report.get('export_identity') != export_identity:
            raise ValueError('Engine/export identity mismatch')
        self.trt, self.report, self.config = trt, report, config
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine((directory/'base-preview.plan').read_bytes())
        self.context = self.engine.create_execution_context()
        self.register_buffer('_device_marker', torch.empty(0, device='cuda'))
        self.calls = 0

    def forward(self, sample, *, noise_labels, conditional_inputs):
        if sample.device != self._device_marker.device or not sample.is_cuda:
            raise ValueError('Experimental TensorRT expects selected CUDA device')
        tensors = dict(sample=sample.float().contiguous(), noise_labels=noise_labels.float().contiguous(),
                       conditions=conditional_inputs[0].float().contiguous())
        stream = torch.cuda.current_stream(sample.device)
        for name, value in tensors.items():
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f'Engine input shape rejected: {name}/{tuple(value.shape)}')
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f'Engine binding rejected: {name}')
        unresolved = self.context.infer_shapes()
        if unresolved:
            raise RuntimeError(f'Engine shapes unresolved: {unresolved}')
        shape = tuple(self.context.get_tensor_shape('prediction'))
        if shape != tuple(sample.shape):
            raise ValueError(f'Unexpected output shape: {shape}')
        output = torch.empty(shape, dtype=torch.float32, device=sample.device)
        if not self.context.set_tensor_address('prediction', output.data_ptr()):
            raise RuntimeError('Engine output binding rejected')
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('Engine execution failed')
        # All allocations and TRT work use this stream; record ownership for the
        # caching allocator as well. Inputs live through asynchronous submission.
        for value in (*tensors.values(), output):
            value.record_stream(stream)
        self.calls += 1
        return output.to(sample.dtype)


def analyse(args):
    capture = checked_capture(args.capture)
    export = trusted_manifest(args.export/'export.json', 'prepared-not-validated')
    if capture['artifact_identity'] != export['capture_identity']:
        raise ValueError('Capture/export relation mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    from prepare_terrain_quantization import load_frozen
    from terrain_quantization import quantize_weight
    model = load_frozen(args.capture/'frozen-base.pt')
    records = []
    for item in export['converted_layers']:
        name = item['layer']
        weight = model.get_submodule(name).weight.float().numpy()
        q, scales = quantize_weight(weight)
        recovered = q.astype(np.float32)*scales[:, None, None, None]
        stats = capture['calibration']['layers'][name]
        records.append(dict(layer=name, parameters=weight.size, weight_error=error(recovered, weight),
            relative_weight_rmse=float(np.sqrt(np.mean((recovered-weight)**2))/np.sqrt(np.mean(weight**2))),
            activation_scale=item['activation_scale'], activation_max=stats['max_abs'],
            activation_rms=float(np.sqrt(stats['sum_squares']/stats['elements'])),
            sampled_p999=stats['diagnostic_p999_abs'],
            minmax_to_sample_p999=stats['max_abs']/max(stats['diagnostic_p999_abs'], 1e-12)))
    report = dict(role=ROLE, state='complete', runtime_enabled=False, source_sha=sha_file(__file__),
                  capture_identity=capture['artifact_identity'], export_identity=export['artifact_identity'],
                  description='CPU weight and historical calibration diagnostics; sample percentiles are not new calibration and do not justify clipping.',
                  layers=records)
    write_json(args.output/'calibration-analysis.json', report)
    return report


class TorchPreviewBase(nn.Module):
    def __init__(self, model, preserve_bf16=False):
        super().__init__()
        self.model, self.config = model, model.config
        self.preserve_bf16 = preserve_bf16
        self.calls = 0

    def forward(self, sample, *, noise_labels, conditional_inputs):
        self.calls += 1
        cast = (lambda value: value) if self.preserve_bf16 else (lambda value: value.float())
        output = self.model(cast(sample), noise_labels=cast(noise_labels),
                            conditional_inputs=[cast(conditional_inputs[0])])
        return output.to(sample.dtype)


def torch_variant(capture, export, variant):
    from prepare_terrain_quantization import load_frozen
    from terrain_quantization import prepare_export_model, quantize_weight
    preserve_bf16 = 'bf16' in variant
    original = load_frozen(capture/'frozen-base.pt')
    model = original if preserve_bf16 else prepare_export_model(original)
    if variant != 'fp32':
        protected = set()
        if variant in ('qdq-protect-res1', 'qdq-bf16-protect-res1'):
            protected = {x['layer'] for x in export['converted_layers'] if x['layer'].endswith('conv_res1')}
        for entry in export['converted_layers']:
            if entry['layer'] in protected:
                continue
            layer = model.get_submodule(entry['layer'])
            q, scales = quantize_weight(layer.weight.float().numpy())
            recovered = q.astype(np.float32)*scales[:, None, None, None]
            layer.weight.copy_(torch.from_numpy(recovered))
            if not variant.startswith('weight-only'):
                scale = entry['activation_scale']
                # Numerical Q/DQ emulation. This does NOT launch integer GEMMs.
                def pre(module, inputs, scale=scale):
                    x = inputs[0]
                    return ((torch.clamp(torch.round(x.float()/scale), -128, 127)*scale).to(x.dtype),)
                layer.register_forward_pre_hook(pre)
    return TorchPreviewBase(model.cuda().eval().requires_grad_(False), preserve_bf16=preserve_bf16)


def checked_capture(directory):
    report = trusted_manifest(directory/'capture.json', 'complete')
    if sha_file(directory/'frozen-base.pt') != report['frozen_sha']:
        raise ValueError('Frozen capture SHA mismatch')
    if not report.get('freeze_checks') or not all(x['byte_exact'] for x in report['freeze_checks']):
        raise ValueError('Capture lacks exact frozen-forward verification')
    return report


@torch.inference_mode()
def packets(args):
    capture = checked_capture(args.capture)
    export = trusted_manifest(args.export/'export.json', 'prepared-not-validated')
    if export['capture_identity'] != capture['artifact_identity']:
        raise ValueError('Source capture/export mismatch')
    holdout = checked_capture(args.holdout)
    if capture['checkpoint'] != holdout['checkpoint']:
        raise ValueError('Holdout checkpoint differs')
    args.output.mkdir(parents=True, exist_ok=False)
    from prepare_terrain_quantization import load_frozen
    raw = load_frozen(args.capture/'frozen-base.pt').cuda().eval()
    inputs, expected = [], []
    for item in holdout['packets']:
        path = args.holdout/item['file']
        if sha_file(path) != item['sha256']:
            raise ValueError('Holdout input SHA mismatch')
        packet = torch.load(path, map_location='cpu', weights_only=True)
        x = [packet[k].cuda() for k in ('sample', 'noise_labels', 'conditions')]
        inputs.append(x)
        expected.append(raw(x[0], noise_labels=x[1], conditional_inputs=[x[2]]).float().cpu().numpy())
        np.save(args.output/f'reference-{len(expected)-1:04d}.npy', expected[-1])
    del raw
    torch.cuda.empty_cache()
    report = dict(role=ROLE, runtime_enabled=False, state='running', source_sha=sha_file(__file__),
                  export_identity=export['artifact_identity'], holdout_identity=holdout['artifact_identity'],
                  variants={}, tf32=False, warning='Torch Q/DQ timings include emulation and do not measure INT8 acceleration.')
    write_json(args.output/'packets.json', report)
    for variant in args.variants:
        model = (TRTBase(args.engine, args.trt, holdout.get('config', {}), export['artifact_identity']) if variant == 'trt'
                 else torch_variant(args.capture, export, variant))
        if variant == 'trt':
            x = inputs[0]
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                model(x[0], noise_labels=x[1], conditional_inputs=[x[2]])
                torch.cuda.synchronize()
            profile.export_chrome_trace(str(args.output/'trt-cuda-trace.json'))
            trace = json.loads((args.output/'trt-cuda-trace.json').read_text(encoding='utf-8'))
            kernels = sorted({e.get('name','') for e in trace['traceEvents'] if e.get('cat') == 'kernel'})
            write_json(args.output/'trt-kernels.json', dict(kernels=kernels,
                description='Actual CUDA kernel names captured during whole engine execution; inspect against engine layer types.'))
            report['engine_build'] = model.report
        results = []
        for index, (x, baseline) in enumerate(zip(inputs, expected)):
            for _ in range(2):
                model(x[0], noise_labels=x[1], conditional_inputs=[x[2]])
            times, output = [], None
            for _ in range(5):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                output = model(x[0], noise_labels=x[1], conditional_inputs=[x[2]])
                end.record()
                end.synchronize()
                times.append(begin.elapsed_time(end))
            actual = output.float().cpu().numpy()
            np.save(args.output/f'{variant}-{index:04d}.npy', actual)
            results.append(dict(packet=holdout['packets'][index], error_vs_bf16=error(actual, baseline),
                                cuda_milliseconds=times, per_channel=[error(actual[:,c], baseline[:,c]) for c in range(5)]))
        report['variants'][variant] = results
        write_json(args.output/'packets.json', report)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report['state'] = 'complete'
    write_json(args.output/'packets.json', report)
    return report


@torch.inference_mode()
def physical(args):
    """Fresh stores for every backend/site. No native or physical cache writes."""
    from dataclasses import replace
    from terrain_app import load_pipeline
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    from terrain_inference import configure_world
    import terrain_server as server
    server.existing_final_mip = lambda *unused: None
    capture = checked_capture(args.capture)
    export = trusted_manifest(args.export/'export.json', 'prepared-not-validated')
    if export['capture_identity'] != capture['artifact_identity']:
        raise ValueError('Source capture/export mismatch')
    sites = json.loads(args.sites.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    loaded = load_pipeline(sites[0]['seed'])
    config = {k:v for k,v in dict(loaded.config).items() if not k.startswith('_')}
    raw = loaded.base_model.model if hasattr(loaded.base_model, '_buckets') else loaded.base_model
    report = dict(role=ROLE, runtime_enabled=False, state='running', source_sha=sha_file(__file__),
                  capture_identity=capture['artifact_identity'], export_identity=export['artifact_identity'],
                  sites=sites, variants={}, tf32=False, gpu=torch.cuda.get_device_name(),
                  gate=dict(elevation_max_m=1., climate_max=[.02,.05,.1,.02,.00001]),
                  warning='Fresh private worlds only; timings for Torch Q/DQ are numerical emulation. Desktop GPU activity uncontrolled.')
    report['gate_scope'] = 'Only finite elevation/climate maximum errors; coastline and gradients are diagnostic, with no visual-fidelity acceptance claim.'
    arrays, reference = {}, {}
    write_json(args.output/'physical.json', report)
    # Reference first; all later variants share immutable coarse/decoder models,
    # but never world stores or generated latents.
    for variant in ['reference', *args.variants]:
        model = raw if variant == 'reference' else (
            TRTBase(args.engine, args.trt, raw.config, export['artifact_identity']) if variant == 'trt'
            else torch_variant(args.capture, export, variant))
        results = []
        report['variants'][variant] = results
        for index, site in enumerate(sites):
            world = None
            try:
                world = WorldPipeline(**(config | dict(seed=site['seed'], dtype='bf16',
                    latents_batch_size=16, cache_limit=512*1024**2, torch_compile=False, log_mode='silent')))
                world.coarse_model, world.base_model, world.decoder_model = loaded.coarse_model, loaded.base_model, loaded.decoder_model
                configure_world(world, replace(loaded._terrain_profile, cuda_graphs=False), world_profile=site['profile'])
                # Replacing AFTER configure prevents eval-weight normalization of
                # already frozen/quantized weights. Loaded native base untouched.
                world.base_model = model
                world.bind()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                elev, climate, stage = server.sample_physical(world, site['seed'], site['profile'], 3, site['tx'], site['ty'])
                torch.cuda.synchronize()
                seconds = time.perf_counter()-started
                arrays[f'{variant}_{index}_elev'], arrays[f'{variant}_{index}_climate'] = elev, climate
                item = dict(site=site, stage=stage, seconds=seconds,
                    peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                    min_elevation=float(elev.min()), max_elevation=float(elev.max()),
                    land_fraction=float(np.mean(elev>0)),
                    qualification='Measured site; geomorphic labels were not preselected by independent ground truth.')
                if variant == 'reference':
                    reference[index] = (elev, climate)
                else:
                    a, b = reference[index]
                    item.update(elevation_error_m=error(elev, a),
                                climate_errors=[error(climate[c], b[c]) for c in range(5)],
                                coastline_sign_disagreement=float(np.mean((elev>0)!=(a>0))),
                                gradient_error=error(np.stack(np.gradient(elev)), np.stack(np.gradient(a))))
                    item['physical_gate_passed'] = item['elevation_error_m']['max_abs'] <= 1 and all(
                        e['max_abs'] <= limit for e, limit in zip(item['climate_errors'], report['gate']['climate_max']))
                results.append(item)
                np.savez_compressed(args.output/'physical.npz', **arrays)
                write_json(args.output/'physical.json', report)
                print(json.dumps(dict(variant=variant,index=index,seconds=seconds,gate=item.get('physical_gate_passed'))), flush=True)
            finally:
                if world is not None:
                    world.close()
        if variant != 'reference':
            del model
            gc.collect()
            torch.cuda.empty_cache()
    report['state'] = 'complete'
    report['physical_holdout_passed'] = all(item.get('physical_gate_passed', True)
                                           for results in report['variants'].values() for item in results)
    write_json(args.output/'physical.json', report)
    return report


@torch.inference_mode()
def matched(args):
    """AB/BA TensorRT candidate vs current cached/graph BF16 base."""
    from terrain_app import load_pipeline
    capture = checked_capture(args.capture)
    export = trusted_manifest(args.export/'export.json', 'prepared-not-validated')
    holdout = checked_capture(args.holdout)
    if export['capture_identity'] != capture['artifact_identity'] or holdout['checkpoint'] != capture['checkpoint']:
        raise ValueError('Matched benchmark provenance mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    loaded = load_pipeline(holdout['sites'][0]['seed'])
    profile=loaded._terrain_profile
    if not (profile.cached_weights and profile.cuda_graphs and profile.exact_kernels
            and profile.attention_backend == 'reference'):
        raise ValueError('Matched reference requires cached weights, graphs, exact kernels and reference attention')
    reference = loaded.base_model
    if not hasattr(reference, '_buckets'):
        raise ValueError('Matched reference requires the current CUDA graph wrapper')
    candidate = TRTBase(args.engine, args.trt, reference.config, export['artifact_identity'])
    from terrain_nn_constants import kernel_status
    def admitted_kernels():
        status=kernel_status()
        if not status['requested'] or not status['effective_by_device'].get(torch.cuda.current_device(),False):
            raise ValueError('Matched reference custom kernels must be prepared and effective on the selected GPU')
        return status
    report = dict(role=ROLE, runtime_enabled=False, state='running', source_sha=sha_file(__file__),
        engine_build=candidate.report, capture_identity=capture['artifact_identity'],
        reference_profile=dict(cached_weights=profile.cached_weights,cuda_graphs=profile.cuda_graphs,
            exact_kernels=profile.exact_kernels,attention_backend=profile.attention_backend),
        reference_kernel_status=admitted_kernels(), selected_cuda_device=torch.cuda.current_device(),
        cuda_stream_handle=int(torch.cuda.current_stream().cuda_stream),
        holdout_identity=holdout['artifact_identity'], export_identity=export['artifact_identity'], tf32=False,
        method='5 alternating AB/BA pairs per real holdout input; both backends warmed, CUDA events on the same nondefault stream. BF16 reference uses current cached weights, exact kernels, CUDA graphs. Desktop GPU activity uncontrolled.',
        packets=[])
    write_json(args.output/'matched.json', report)
    for packet_index, entry in enumerate(holdout['packets']):
        path = args.holdout/entry['file']
        if sha_file(path) != entry['sha256']:
            raise ValueError('Matched input SHA mismatch')
        packet = torch.load(path, map_location='cpu', weights_only=True)
        x = [packet[k].cuda() for k in ('sample', 'noise_labels', 'conditions')]
        outputs = {}
        for backend, model in [('reference', reference), ('trt', candidate)]:
            for _ in range(3):
                outputs[backend] = model(x[0], noise_labels=x[1], conditional_inputs=[x[2]])
        torch.cuda.synchronize()
        pairs = []
        for index in range(5):
            order = ['reference','trt'] if index%2 == 0 else ['trt','reference']
            timings = {}
            for backend in order:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                outputs[backend] = (reference if backend == 'reference' else candidate)(
                    x[0], noise_labels=x[1], conditional_inputs=[x[2]])
                end.record()
                end.synchronize()
                timings[backend] = begin.elapsed_time(end)
            pairs.append(dict(order=order, cuda_milliseconds=timings,
                              speedup=timings['reference']/timings['trt']))
        report['packets'].append(dict(packet=entry, pairs=pairs,
            median_paired_speedup=float(np.median([p['speedup'] for p in pairs])),
            error_vs_current_bf16=error(outputs['trt'].float().cpu().numpy(), outputs['reference'].float().cpu().numpy())))
        write_json(args.output/'matched.json', report)
    if args.sites is not None:
        from terrain_diffusion.inference.world_pipeline import WorldPipeline
        from terrain_inference import configure_world
        import terrain_server as server
        server.existing_final_mip = lambda *unused: None
        sites = json.loads(args.sites.read_text(encoding='utf-8'))
        config = {k:v for k,v in dict(loaded.config).items() if not k.startswith('_')}
        report['physical_pairs'] = []
        def footprint(backend, site):
            world = WorldPipeline(**(config | dict(seed=site['seed'], dtype='bf16',
                latents_batch_size=16, cache_limit=512*1024**2, torch_compile=False, log_mode='silent')))
            try:
                world.coarse_model,world.base_model,world.decoder_model = loaded.coarse_model,loaded.base_model,loaded.decoder_model
                configure_world(world, loaded._terrain_profile, world_profile=site['profile'])
                if backend == 'trt':
                    world.base_model = candidate
                world.bind()
                torch.cuda.synchronize()
                started = time.perf_counter()
                values = server.sample_physical(world, site['seed'], site['profile'], 3, site['tx'], site['ty'])
                torch.cuda.synchronize()
                seconds=time.perf_counter()-started
                repeated=server.sample_physical(world,site['seed'],site['profile'],3,site['tx'],site['ty'])
                torch.cuda.synchronize()
                revisit=dict(elevation=error(repeated[0],values[0]),climate=error(repeated[1],values[1]))
                if not all(value['byte_exact'] for value in revisit.values()):
                    raise ValueError('Same-store revisit must be finite and byte-identical')
                return seconds,values,dict(revisit=revisit,kernel_status=admitted_kernels())
            finally:
                world.close()
        # One full warmup per backend/site captures all physical batch forms.
        # Every timed footprint below still owns a new empty neural store.
        for site in sites:
            footprint('reference', site)
            footprint('trt', site)
            pairs = []
            for index in range(5):
                order = ['reference','trt'] if index%2 == 0 else ['trt','reference']
                times, values,admission = {}, {},{}
                for backend in order:
                    times[backend], values[backend],admission[backend] = footprint(backend, site)
                pairs.append(dict(order=order, seconds=times,
                    admission=admission,
                    speedup=times['reference']/times['trt'],
                    elevation_error_m=error(values['trt'][0],values['reference'][0]),
                    climate_errors=[error(values['trt'][1][c],values['reference'][1][c]) for c in range(5)]))
            report['physical_pairs'].append(dict(site=site,pairs=pairs,
                median_paired_speedup=float(np.median([p['speedup'] for p in pairs]))))
            write_json(args.output/'matched.json',report)
    graph_stats=reference.stats()
    if graph_stats['fallback_calls'] or graph_stats['memory_fallbacks'] or graph_stats['capture_errors']:
        raise ValueError('Matched reference CUDA graphs must not fall back')
    report.update(state='complete',reference_graph_stats=graph_stats,reference_kernel_status_after=admitted_kernels())
    write_json(args.output/'matched.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('analyse')
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    for command in ('packets', 'physical', 'matched'):
        p = sub.add_parser(command)
        p.add_argument('--capture', type=Path, required=True)
        p.add_argument('--export', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--variants', nargs='+', choices=['fp32','qdq','weight-only','qdq-protect-res1','weight-only-bf16','qdq-bf16','qdq-bf16-protect-res1','trt'],
                       default=['fp32','qdq','weight-only','qdq-protect-res1'])
        p.add_argument('--engine', type=Path)
        p.add_argument('--trt', type=Path)
        p.add_argument('--holdout' if command in ('packets','matched') else '--sites', type=Path, required=True)
        if command == 'matched':
            p.add_argument('--sites', type=Path)
    p = sub.add_parser('build')
    p.add_argument('--variant', choices=['qdq', 'fp32'], required=True)
    p.add_argument('--trt', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command in ('packets', 'physical', 'matched') and ('trt' in args.variants or args.command == 'matched') and (args.trt is None or args.engine is None):
        parser.error('trt variant requires --trt and --engine')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.command != 'analyse':
        from terrain_device import select_cuda_device
        selected_device = select_cuda_device()['selected']['index']
    if args.command in ('packets', 'physical', 'matched'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        if args.command in ('packets', 'physical', 'matched'):
            experiment_stream = torch.cuda.Stream(device=f'cuda:{selected_device}')
            with torch.cuda.stream(experiment_stream):
                result = globals()[args.command](args)
            experiment_stream.synchronize()
        else:
            result = globals()[args.command](args)
    except BaseException as exc:
        receipt = args.output/f'{args.command}.json'
        if receipt.exists():
            value = json.loads(receipt.read_text(encoding='utf-8'))
            value.update(state='failed', error=f'{type(exc).__name__}: {exc}')
            write_json(receipt, value)
        raise
    print(json.dumps(dict(state=result['state'], runtime_enabled=False, output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
