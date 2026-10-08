"""Resident-GPU regional SNR validation, separate from serving/latency trials."""
import argparse
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import torch
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True)
args=parser.parse_args()
from terrain_app import load_pipeline,gpu_calls
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_inference import configure_world,inference_status
from terrain_generation import register_generation
import terrain_server as server

report={'cases':[],'notes':['Physical coarse output, fresh RAM store, shared resident weights.',
    'No extra neural forward; regional policy requires CPU reductions and first-use embeddings.']}
with torch.inference_mode():
    loaded=load_pipeline(42)
    config={k:v for k,v in dict(loaded.config).items() if not k.startswith('_')}
    profiles=[('neutral','natural'),
        ('mountain',register_generation('natural',{'snr_altitude_gain':[4,1,1,1,1],'snr_altitude_range_m':[0,1]})),
        ('cold',register_generation('natural',{'snr_driver_gain':[3,1,1,1,1],'snr_driver_range':[50,40]}))]
    server.existing_final_mip=lambda *unused:None
    elevations={}
    for name,profile in profiles:
        world=WorldPipeline(**(config|dict(seed=42,torch_compile=False,dtype='bf16',log_mode='silent')))
        for model in ('coarse_model','base_model','decoder_model'):setattr(world,model,getattr(loaded,model))
        configure_world(world,world_profile=profile)
        world.bind()
        before=dict(gpu_calls)
        started=time.perf_counter()
        height,climate,stage=server.sample_physical(world,42,profile,4,-1,-1)
        torch.cuda.synchronize()
        status=inference_status(world)
        delta={k:gpu_calls[k]-before[k] for k in before}
        assert stage=='coarse' and np.isfinite(height).all() and np.isfinite(climate).all()
        elevations[name]=height
        coarse_seconds=time.perf_counter()-started
        if name!='neutral':
            final,final_climate,_=server.sample_physical(world,42,profile,0,-1,-1)
            neighbor,neighbor_climate,_=server.sample_physical(world,42,profile,0,0,-1)
            assert np.isfinite(final).all() and np.isfinite(final_climate).all()
            assert np.isfinite(neighbor).all() and np.isfinite(neighbor_climate).all()
            overlap_error=float(np.max(np.abs(final[:,256:304]-neighbor[:,:48])))
            assert overlap_error<=1.,f'Native overlap seam {overlap_error} m exceeds existing 1 m gate'
            world.empty_cache()
            replay,replay_climate,_=server.sample_physical(world,42,profile,0,-1,-1)
            assert final.tobytes()==replay.tobytes() and final_climate.tobytes()==replay_climate.tobytes(),'Fresh-store replay drift'
            report.setdefault('native_checks',[]).append(dict(name=name,finite=True,overlap_max_abs_m=overlap_error,
                fresh_store_replay_byte_exact=True,height_sha256=hashlib.sha256(final.tobytes()).hexdigest(),
                climate_sha256=hashlib.sha256(final_climate.tobytes()).hexdigest()))
        report['cases'].append(dict(name=name,profile=profile,seconds=coarse_seconds,
            cuda_calls=delta,regional_snr=status['regional_snr'],
            policies=[list(p) for p in world.__dict__.get('_terrain_snr_cache',{})]))
        world.empty_cache();world.close()
    for case in report['cases'][1:]:
        assert case['cuda_calls']==report['cases'][0]['cuda_calls']
        assert case['regional_snr']['cached_policies']>0
        assert not np.array_equal(elevations[case['name']],elevations['neutral'])
    report['passed']=True
report['source_sha256']={p:hashlib.sha256((server.ROOT/p).read_bytes()).hexdigest() for p in
    ('terrain_inference.py','terrain_snr.py','terrain_generation.py','terrain_cuda_graphs.py','terrain_nn_constants.py','validate_terrain_snr.py')}
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
