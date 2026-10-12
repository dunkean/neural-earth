"""Matched cold backend viewport timings with a shared coarse and decoder.

Times terrain requests for a 1920x1080 viewport (40 native 256px tiles).
Excludes HTTP, browser rendering, model loading and first kernel warmup.
Base CUDA Graphs are disabled for both variants; this is the Linux native
adapter path, not a browser/WASM benchmark or an optimized graph pipeline.
"""
import argparse
from dataclasses import asdict, replace
import gc
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch

from distill.bench_pipeline import StageCount, require_idle_gpu
from distill.common import DATA, atomic_json, atomic_write, external_path
from distill.inference import install
from distill.native_inference import load_native
from distill.native_weight_quant import file_sha
from distill.student import load_student


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--coarse',type=Path,required=True)
    parser.add_argument('--decoder',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=1920)
    parser.add_argument('--height',type=int,default=1080)
    parser.add_argument('--lods',type=int,nargs='+',choices=(0,3),default=[3,0])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--profile',default='natural')
    args=parser.parse_args()
    if min(args.width,args.height)<1 or args.repeats<3:parser.error('Positive viewport and at least three repeats required.')
    require_idle_gpu()
    output=external_path(args.output)
    if output.exists():raise ValueError('Keep existing benchmark evidence.')
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from terrain_app import load_pipeline
    from terrain_inference import configure_world, prepare_models
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    import terrain_server as server
    server.existing_final_mip=lambda *unused:None
    loaded=load_pipeline(args.seed)
    config={k:v for k,v in dict(loaded.config).items() if not k.startswith('_')}
    native,saved=load_native(args.checkpoint,'cuda')
    shared={stage:load_student(path,'cuda')[0] for stage,path in [('coarse',args.coarse),('decoder',args.decoder)]}
    profile=replace(loaded._terrain_profile,cuda_graphs=False)
    nx,ny=math.ceil(args.width/256),math.ceil(args.height/256)
    report=dict(status='running',gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        checkpoint_sha256=file_sha(args.checkpoint),step=saved['step'],
        coarse_sha256=file_sha(args.coarse),decoder_sha256=file_sha(args.decoder),
        width=args.width,height=args.height,tile_grid=[nx,ny],tile_count=nx*ny,
        seed=args.seed,world_profile=args.profile,inference_profile=asdict(profile),
        whole_system_idle=True,shared_coarse_decoder=True,fresh_world_per_sample=True,
        excludes='model loading, first kernel warmup, HTTP and browser rendering',
        scope='cold backend viewport tile requests; base CUDA Graphs disabled for both',rows=[])
    del saved
    for lod in args.lods:
        # Interleave the same cold-world workloads, including one untimed warmup
        # for each variant. Resident kernels/weights remain warm thereafter.
        for repeat in range(-1,args.repeats):
            for variant in ('original','distilled'):
                world=WorldPipeline(**(config|dict(seed=args.seed,dtype='bf16',
                    latents_batch_size=profile.latent_batch,cache_limit=1024**3,
                    torch_compile=False,log_mode='silent')))
                world.coarse_model,world.base_model,world.decoder_model=(
                    loaded.coarse_model,loaded.base_model,loaded.decoder_model)
                configure_world(world,profile,world_profile=args.profile)
                install(world,shared)
                if variant=='distilled':world.base_model=native
                # Preserve baked kernels: only the unchanged stages are prepared.
                prepare_models(SimpleNamespace(coarse_model=world.coarse_model,
                    base_model=world.coarse_model,decoder_model=world.decoder_model))
                counter=StageCount(world.base_model);world.base_model=counter;world.bind()
                canvas=np.empty((ny*256,nx*256),dtype=np.float32)
                try:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter();stages=[]
                    for y in range(ny):
                        for x in range(nx):
                            field,_,stage=server.sample_physical(world,args.seed,args.profile,lod,x,y)
                            if stage not in ('latent','decoder'):raise ValueError('Unexpected cached or preview stage.')
                            halo=(field.shape[0]-256)//2
                            canvas[y*256:(y+1)*256,x*256:(x+1)*256]=field[halo:halo+256,halo:halo+256]
                            stages.append(stage)
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    if not np.isfinite(canvas).all():raise ValueError('Nonfinite physical terrain.')
                    row=dict(variant=variant,lod=lod,repeat=repeat,warmup=repeat<0,
                        seconds=elapsed,base_calls=counter.calls,base_windows=counter.windows,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),stages=sorted(set(stages)))
                    report['rows'].append(row);atomic_json(output/'report.json',report)
                    if repeat==0:
                        atomic_write(output/f'{variant}-lod{lod}.npz',lambda f:
                            np.savez_compressed(f,height=canvas[:args.height,:args.width]))
                    print(row,flush=True)
                finally:
                    world.close();del world,counter,canvas;gc.collect();torch.cuda.empty_cache()
    report['comparisons']=[]
    for lod in args.lods:
        medians={v:statistics.median(r['seconds'] for r in report['rows']
            if r['variant']==v and r['lod']==lod and not r['warmup']) for v in ('original','distilled')}
        report['comparisons'].append(dict(lod=lod,**medians,speedup=medians['original']/medians['distilled']))
    report['status']='complete';atomic_json(output/'report.json',report)


if __name__=='__main__':main()
