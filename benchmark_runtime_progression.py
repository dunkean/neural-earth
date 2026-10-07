"""Measure shared-store 4→3→2→1→0 refinement separately from cold LOD runs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('--label',required=True)
parser.add_argument('--source-dir')
parser.add_argument('--reference')
parser.add_argument('--require-exact',action='store_true')
args=parser.parse_args()
if args.require_exact and not args.reference:
    parser.error('--require-exact requires --reference')
if args.source_dir:
    sys.path.insert(0,args.source_dir)

import numpy as np
import torch
from terrain_app import RUNTIME, load_pipeline
from terrain_diffusion.inference.world_pipeline import WorldPipeline
import terrain_inference
import terrain_nn_constants
import terrain_cuda_graphs
import terrain_window_scheduler
import terrain_server as server
# Runtime sources may be frozen-before imports, while instrumentation must be
# the harness beside this file rather than the obsolete snapshot harness.
measurement_spec=importlib.util.spec_from_file_location('_runtime_lod_measurements',Path(__file__).with_name('benchmark_runtime_lod.py'))
measurement_module=importlib.util.module_from_spec(measurement_spec)
sys.modules[measurement_spec.name]=measurement_module
measurement_spec.loader.exec_module(measurement_module)
Measurements,compare=measurement_module.Measurements,measurement_module.compare


@torch.inference_mode()
def main():
    output=RUNTIME/'runtime-lod-optimization'
    output.mkdir(parents=True,exist_ok=True)
    report=dict(label=args.label,require_exact=args.require_exact,
                method='Terrain server stopped; no concurrent Terrain NN worker. Desktop graphics activity is uncontrolled. Models resident; a fresh NN store for each seed then physical tiles containing one fixed native camera centre in order LOD4,3,2,1,0. This measures progressive reuse and is distinct from independent cold footprints.',samples=[])
    receipt_path=output/(args.label+'.json')
    measurement_module.run_with_receipt(receipt_path,report,lambda:run_progression(report,output,receipt_path))
    print(json.dumps(dict(report=str(receipt_path))),flush=True)


def run_progression(report,output,receipt_path):
    report.update(**measurement_module.imported_source_evidence(),
                  reference=measurement_module.reference_evidence(args.reference),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report['source_paths'][Path(__file__).name]=str(Path(__file__).resolve())
    report['source_sha256'][Path(__file__).name]=report['harness_sha256']
    measurement_module.write_receipt(receipt_path,report)
    started=time.perf_counter()
    loaded=load_pipeline(42)
    torch.cuda.synchronize()
    report['model_initialization_seconds']=time.perf_counter()-started
    config={k:v for k,v in dict(loaded.config).items() if not k.startswith('_')}
    reference=np.load(args.reference) if args.reference else None
    server.existing_final_mip=lambda *unused:None
    arrays={}
    for seed in (42,0):
        world=None
        try:
            world=WorldPipeline(**(config|dict(seed=seed,latents_batch_size=16,dtype='bf16',cache_limit=512*1024**2,torch_compile=False,log_mode='silent')))
            world.coarse_model,world.base_model,world.decoder_model=loaded.coarse_model,loaded.base_model,loaded.decoder_model
            terrain_inference.configure_world(world)
            world.bind()
            sequence_start=time.perf_counter()
            for lod in (4,3,2,1,0):
                # Fixed centre of native A0 HTTP witness tile (-14,10).
                tx,ty=(-14*256+128)//(256*(1<<lod)),(10*256+128)//(256*(1<<lod))
                key=f'seed{seed}_lod{lod}'
                item=dict(key=key,seed=seed,lod=lod,tx=tx,ty=ty,status='running')
                report['samples'].append(item)
                measurement_module.write_receipt(receipt_path,report)
                measured=Measurements()
                handles=[]
                try:
                    handles=measured.install_models(world)
                    measured.install_stages(world)
                    torch.cuda.synchronize()
                    started=time.perf_counter()
                    elevation,climate,stage=server.sample_physical(world,seed,'natural',lod,tx,ty)
                    torch.cuda.synchronize()
                    seconds=time.perf_counter()-started
                    arrays[key+'_elev'],arrays[key+'_climate']=elevation,climate
                    item.update(stage=stage,seconds=seconds,
                                elapsed_since_sequence_seconds=time.perf_counter()-sequence_start,
                                measurements=measured.snapshot(),
                                outputs_finite=bool(np.isfinite(elevation).all() and np.isfinite(climate).all()))
                    measurement_module.write_receipt(receipt_path,report)
                    if not item['outputs_finite']:
                        raise ValueError('Non-finite physical height or climate')
                    if reference is not None:
                        item['elevation_error_m']=compare(elevation,reference[key+'_elev'])
                        item['climate_errors']=[compare(climate[c],reference[key+'_climate'][c]) for c in range(5)]
                        measurement_module.write_receipt(receipt_path,report)
                        measurement_module.validate_fidelity(item,args.require_exact)
                    item['status']='complete'
                    measurement_module.write_receipt(receipt_path,report)
                    print(json.dumps(item),flush=True)
                finally:
                    for handle in handles:
                        handle.remove()
                    measured.restore_stages()
        finally:
            if world is not None:
                world.close()
    np.savez_compressed(output/(args.label+'.npz'),**arrays)


if __name__=='__main__':
    main()
