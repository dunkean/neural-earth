"""Physical LOD benchmark with fresh NN stores and uncontrolled desktop load.

The first footprint includes runtime first-use/capture; later footprints use
resident models and fresh world stores. Physical disk hits are deliberately
excluded and must be measured separately through HTTP.
"""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

if '--source-dir' in sys.argv:
    sys.path.insert(0,sys.argv[sys.argv.index('--source-dir')+1])

from terrain_app import ROOT, RUNTIME, load_pipeline
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_inference import configure_world, inference_status
import terrain_inference
import terrain_cuda_graphs
import terrain_nn_constants
import terrain_window_scheduler
import terrain_climate
import terrain_server as server
import terrain_app
import terrain_conditioning
import terrain_world
import terrain_manifest
import terrain_diffusion.inference.world_pipeline as world_pipeline
import terrain_diffusion.models.mp_layers as mp_layers


def imported_source_evidence():
    modules = (terrain_inference, terrain_cuda_graphs, terrain_nn_constants,
               terrain_window_scheduler, server, terrain_climate, terrain_app,
               terrain_conditioning, terrain_world, terrain_manifest, world_pipeline,
               mp_layers, sys.modules[__name__])
    paths = {Path(module.__file__).name: str(Path(module.__file__).resolve()) for module in modules}
    return dict(source_paths=paths,
                source_sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in paths.items()})


def reference_evidence(path):
    if path is None:
        return None
    resolved = Path(path).resolve()
    receipt = resolved.with_suffix('.json')
    label = resolved.stem
    evidence = dict(path=str(resolved), sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(), label=label)
    if receipt.exists():
        raw = receipt.read_bytes()
        evidence.update(label=json.loads(raw).get('label', label), receipt_path=str(receipt),
                        receipt_sha256=hashlib.sha256(raw).hexdigest())
    return evidence


class Measurements:
    def __init__(self):
        self.models = defaultdict(list)
        self.batches = defaultdict(list)
        self.pending = {}
        self._stage_wrappers = []

    def install_models(self, world):
        handles = []
        for name in ('coarse', 'base', 'decoder'):
            model = getattr(world, name + '_model')
            def before(module, args, name=name):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.pending[name] = (event, time.perf_counter(), int(args[0].shape[0]))
            def after(module, args, output, name=name):
                start, cpu_start, batch = self.pending.pop(name)
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.models[name].append((start, end, time.perf_counter()-cpu_start, batch))
            handles += [model.register_forward_pre_hook(before), model.register_forward_hook(after)]
        return handles

    def install_stages(self, world):
        seen = set()
        def install(tensor):
            if id(tensor) in seen:
                return
            seen.add(id(tensor))
            for arg in tensor.args:
                install(arg)
            original = tensor._f
            self._stage_wrappers.append((tensor, original))
            def measured(*args):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                cpu_start = time.perf_counter()
                result = original(*args)
                cpu_seconds = time.perf_counter()-cpu_start
                end.record()
                self.batches[tensor.uuid].append((start, end, cpu_seconds, len(args[0])))
                return result
            tensor._f = measured
        install(world.residual)

    def restore_stages(self):
        for tensor, original in reversed(self._stage_wrappers):
            tensor._f = original
        self._stage_wrappers.clear()

    def snapshot(self):
        def summarize(values):
            return dict(calls=len(values), batch_histogram=dict(Counter(v[3] for v in values)),
                        windows=sum(v[3] for v in values),
                        cpu_submission_seconds=sum(v[2] for v in values),
                        cuda_event_seconds=sum(v[0].elapsed_time(v[1])/1000 for v in values))
        return dict(models={k:summarize(v) for k,v in self.models.items()},
                    stage_batches={k:summarize(v) for k,v in self.batches.items()})


def compare(actual, expected):
    delta = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    byte_exact = (actual.shape == expected.shape and actual.dtype == expected.dtype and
                  actual.tobytes(order='C') == expected.tobytes(order='C'))
    return dict(exact=bool(np.array_equal(actual, expected)), byte_exact=byte_exact, max_abs=float(delta.max()),
                mean_abs=float(delta.mean()), p99_abs=float(np.quantile(delta, .99)))


def require_exact_errors(elevation_error, climate_errors):
    errors = [elevation_error, *climate_errors]
    if not all(error['exact'] and error['byte_exact'] for error in errors):
        raise ValueError('Physical arrays must be byte-identical to the reference')


def validate_fidelity(item, require_exact=False):
    limits=(.02,.05,.1,.02,.00001)
    item['fidelity_passed']=(item['elevation_error_m']['max_abs']<=1 and
                            len(item['climate_errors'])==len(limits) and
                            all(error['max_abs']<=limit for error,limit in zip(item['climate_errors'],limits)))
    if not item['fidelity_passed']:
        raise ValueError('Physical height or climate exceeded the numerical limits')
    if require_exact:
        require_exact_errors(item['elevation_error_m'],item['climate_errors'])


def write_receipt(path, report):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(report,indent=2),encoding='utf-8')
    temporary.replace(path)


def run_with_receipt(path, report, run):
    report['status']='running'
    write_receipt(path,report)
    try:
        run()
    except Exception as exc:
        report['status']='failed'
        report['error']=dict(type=type(exc).__name__,message=str(exc))
        if report.get('samples') and report['samples'][-1].get('status')=='running':
            failed=report['samples'][-1]
            failed.update(status='failed',error=report['error'])
            report['failing_sample_key']=failed.get('key')
        write_receipt(path,report)
        raise
    report['status']='complete'
    write_receipt(path,report)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', default=str(RUNTIME/'runtime-lod-optimization'))
    parser.add_argument('--reference')
    parser.add_argument('--require-exact', action='store_true')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--source-dir')
    parser.add_argument('--lods',type=int,nargs='+',default=[3,2])
    args = parser.parse_args()
    if args.require_exact and not args.reference:
        parser.error('--require-exact requires --reference')
    if args.repeats < 1:
        parser.error('--repeats must be at least 1 after the runtime-first-use sample')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Benchmark the actual NN path even if this identity has physical native
    # children on disk; disk/mip serving is a separate HTTP scenario.
    server.existing_final_mip = lambda *unused: None
    report = dict(label=args.label, require_exact=args.require_exact,
                  torch=str(torch.__version__), cuda=torch.version.cuda,
                  method='Terrain server stopped; no concurrent Terrain NN worker. Desktop graphics activity is uncontrolled. Resident unchanged natural BF16 models; fresh in-memory NN window store per footprint. First footprint per LOD includes first-use/capture; subsequent fresh stores reuse runtime initialization. No physical/coarse disk replay, rendering or HTTP.',
                  samples=[])
    receipt_path=output/(args.label+'.json')
    run_with_receipt(receipt_path,report,lambda: run_lod(args,report,output))
    print(json.dumps(dict(report=str(receipt_path),summary=report['summary'])),flush=True)


def run_lod(args,report,output):
    receipt_path=output/(args.label+'.json')
    report.update(**imported_source_evidence(),reference=reference_evidence(args.reference))
    write_receipt(receipt_path,report)
    gpu_query=['nvidia-smi','--query-gpu=temperature.gpu,pstate,power.draw,clocks.current.sm,utilization.gpu,memory.used','--format=csv,noheader']
    def gpu_snapshot():
        return subprocess.run(gpu_query,capture_output=True,text=True,check=True).stdout.strip()
    report['gpu_before']=gpu_snapshot()
    started = time.perf_counter()
    loaded = load_pipeline(42)
    torch.cuda.synchronize()
    report['model_initialization_seconds'] = time.perf_counter()-started
    report['gpu'] = torch.cuda.get_device_name()
    config = {k:v for k,v in dict(loaded.config).items() if not k.startswith('_')}
    reference = np.load(args.reference) if args.reference else None
    arrays = {}
    # A0 terrain sites and negative coordinates; repeated sites use fresh stores
    # and still execute every necessary neural window.
    sites = [(42,-14,10), (0,-1,-1), (43,2,1), (42,-14,10), (0,-1,-1), (43,2,1)]
    for lod in args.lods:
        for index in range(args.repeats+1):
            seed, tx, ty = sites[index % len(sites)]
            key=f'lod{lod}_case{index}'
            item=dict(key=key,lod=lod,seed=seed,tx=tx,ty=ty,status='running',
                      phase='runtime-first-use' if index==0 else 'runtime-warm-new-NN-footprint')
            report['samples'].append(item)
            write_receipt(receipt_path,report)
            world=None
            handles=[]
            measure=Measurements()
            try:
                bind_started = time.perf_counter()
                world = WorldPipeline(**(config | dict(seed=seed, latents_batch_size=16, dtype='bf16',
                                                   cache_limit=512*1024**2, torch_compile=False, log_mode='silent')))
                world.coarse_model,world.base_model,world.decoder_model = loaded.coarse_model,loaded.base_model,loaded.decoder_model
                configure_world(world)
                world.bind()
                report_setup = time.perf_counter()-bind_started
                handles = measure.install_models(world)
                measure.install_stages(world)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = inference_status(world)
                started = time.perf_counter()
                elevation, climate, stage = server.sample_physical(world,seed,'natural',lod,tx,ty)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-started
                arrays[key+'_elev'], arrays[key+'_climate'] = elevation,climate
                item.update(world_bind_seconds=report_setup,seconds=seconds,stage=stage,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      measurements=measure.snapshot(),before_graphs=before['cuda_graphs'],after_inference=inference_status(world))
                item['outputs_finite']=bool(np.isfinite(elevation).all() and np.isfinite(climate).all())
                write_receipt(receipt_path,report)
                if not item['outputs_finite']:
                    raise ValueError('Non-finite physical height or climate')
                if reference is not None:
                    item['elevation_error_m']=compare(elevation,reference[key+'_elev'])
                    item['climate_errors']=[compare(climate[c],reference[key+'_climate'][c]) for c in range(5)]
                    write_receipt(receipt_path,report)
                    validate_fidelity(item,args.require_exact)
                for handle in handles:
                    handle.remove()
                handles=[]
                measure.restore_stages()
                # Live-store hit is distinct from new-footprint generation.
                torch.cuda.synchronize()
                hit_started=time.perf_counter()
                repeated=server.sample_physical(world,seed,'natural',lod,tx,ty)
                torch.cuda.synchronize()
                item['live_window_hit_seconds']=time.perf_counter()-hit_started
                item['live_window_hit_equality']=dict(elevation=compare(elevation,repeated[0]),climate=compare(climate,repeated[1]))
                write_receipt(receipt_path,report)
                if not all(error['exact'] for error in item['live_window_hit_equality'].values()):
                    raise ValueError('Live-store revisit changed physical arrays')
                if args.require_exact:
                    require_exact_errors(item['live_window_hit_equality']['elevation'],[item['live_window_hit_equality']['climate']])
                item['status']='complete'
                write_receipt(receipt_path,report)
                print(json.dumps({k:v for k,v in item.items() if k not in ('after_inference','before_graphs')}),flush=True)
            finally:
                for handle in handles:
                    handle.remove()
                measure.restore_stages()
                if world is not None:
                    world.close()
                del world
            gc.collect()
    report['summary']={str(lod):dict(median_seconds=statistics.median(values),
                                    repetitions=len(values),min_seconds=min(values),max_seconds=max(values))
                       for lod in args.lods for values in [[v['seconds'] for v in report['samples'] if v['lod']==lod and v['phase']=='runtime-warm-new-NN-footprint']]}
    unique = {(item['seed'], item['lod'], item['tx'], item['ty']) for item in report['samples']}
    report['coverage'] = dict(total_footprints=len(report['samples']), unique_site_lod_results=len(unique),
                              repeated_site_lod_results=len(report['samples'])-len(unique))
    report['gpu_after']=gpu_snapshot()
    np.savez_compressed(output/(args.label+'.npz'),**arrays)


if __name__=='__main__':
    main()
