"""Dedicated fresh-store CPU/CUDA operator profiling, separate from serving.

Run with the local server stopped. Trace timings include profiler overhead;
use benchmark_runtime_lod.py for uninstrumented paired latency comparisons.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import argparse
import hashlib
import json
from pathlib import Path
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--lod',type=int,default=2,choices=(0,2,3,4))
parser.add_argument('--tx',type=int,default=-14)
parser.add_argument('--ty',type=int,default=10)
parser.add_argument('--layout',choices=('contiguous','channels-last','channels-last-cached'),default='contiguous')
parser.add_argument('--reference',help='NPZ physical output fidelity gate; experiment only')
args=parser.parse_args()

import numpy as np
import torch
from terrain_app import load_pipeline
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_inference import configure_world
import terrain_server as server
import terrain_profiling as timeline

out=Path(args.output)
out.mkdir(parents=True,exist_ok=True)
timeline.set_enabled(True,cuda=True)
with torch.inference_mode():
    loaded=load_pipeline(args.seed)
    models=[getattr(loaded,k) for k in ('coarse_model','base_model','decoder_model')]
    if args.layout=='channels-last':
        for model in models:
            model.to(memory_format=torch.channels_last)
    if args.layout=='channels-last-cached':
        # Convert the already-normalized eval weights, preserving coefficients;
        # never alter parameter layout or its FP32 normalization reductions.
        import terrain_inference as inference
        original_weight=inference._cached_weight
        converted={}
        def cached_layout(module,x,gain):
            value=original_weight(module,x,gain)
            if value.ndim!=4:return value
            key=id(module);entry=converted.get(key)
            if entry is None or entry[0] is not value:
                converted[key]=(value,value.contiguous(memory_format=torch.channels_last))
            return converted[key][1]
        inference._cached_weight=cached_layout
        for model in models:
            model.register_forward_pre_hook(lambda module,inputs:(inputs[0].contiguous(memory_format=torch.channels_last),*inputs[1:]))
    config={k:v for k,v in dict(loaded.config).items() if not k.startswith('_')}
    world=WorldPipeline(**(config|dict(seed=args.seed,torch_compile=False,dtype='bf16',log_mode='silent')))
    world.coarse_model,world.base_model,world.decoder_model=models
    configure_world(world)
    world.bind()
    server.existing_final_mip=lambda *unused:None
    # Warm the actual forms/graphs, then clear only this transient NN store.
    server.sample_physical(world,args.seed,'natural',args.lod,args.tx,args.ty)
    torch.cuda.synchronize()
    world.empty_cache()
    started=time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True,profile_memory=True) as profiler:
        elevation,climate,stage=server.sample_physical(world,args.seed,'natural',args.lod,args.tx,args.ty)
        torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    profiler.export_chrome_trace(str(out/'operators.json'))
    (out/'operators.txt').write_text(profiler.key_averages().table(sort_by='self_cuda_time_total',row_limit=45),encoding='utf-8')
    (out/'timeline.json').write_text(json.dumps(timeline.snapshot(),indent=2),encoding='utf-8')
    np.savez_compressed(out/'arrays.npz',elev=elevation,climate=climate)
    files=['terrain_inference.py','terrain_nn_constants.py','terrain_cuda_graphs.py','terrain_cuda_kernels.py','terrain_server.py','terrain_profiling.py','tools/benchmarks/profile_terrain.py']
    report=dict(stage=stage,seed=args.seed,lod=args.lod,tx=args.tx,ty=args.ty,layout=args.layout,
                instrumented_seconds=seconds,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
                finite=bool(np.isfinite(elevation).all() and np.isfinite(climate).all()),
                source_sha256={p:hashlib.sha256((server.ROOT/p).read_bytes()).hexdigest() for p in files},
                note='Fresh RAM store; natural profile; CUDA graphs warmed. Profiler overhead prevents direct wall-clock comparisons.')
    if args.reference:
        with np.load(args.reference,allow_pickle=False) as reference:
            def difference(actual,expected):
                return dict(byte_exact=bool(np.array_equal(actual.view(np.uint8),expected.view(np.uint8))),
                            max_abs=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))))
            report['fidelity']=dict(elevation=difference(elevation,reference['elev']),
                                    climate=difference(climate,reference['climate']))
            report['fidelity']['passed']=all(report['fidelity'][key]['byte_exact'] for key in ('elevation','climate'))
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    world.close()
print(json.dumps(report))
