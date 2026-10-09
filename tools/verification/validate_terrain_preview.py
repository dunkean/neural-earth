"""CUDA fidelity gates for bounded approximate previews and isolated final stores."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path

import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from terrain_app import load_pipeline
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_inference import configure_world
import terrain_server as server
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
loaded=load_pipeline(42)
config={k:v for k,v in dict(loaded.config).items() if not k.startswith('_')}
models=[getattr(loaded,k) for k in ('coarse_model','base_model','decoder_model')]
server.existing_final_mip=lambda *unused:None

def make(seed):
    world=WorldPipeline(**(config|dict(seed=seed,torch_compile=False,dtype='bf16',cache_limit=128*1024**2,log_mode='silent')))
    world.coarse_model,world.base_model,world.decoder_model=models
    configure_world(world);world.bind();return world

def exact(a,b):return bool(np.array_equal(a.view(np.uint8),b.view(np.uint8)))

report={'cases':[],'notes':['Preview sources are approximate: batch partitions can differ from the full parent.',
                              'Final-store isolation, finite outputs and fresh-store replay are required gates. Timings include capture where a new form is first encountered.']}
with torch.inference_mode():
    warm=make(42);server.sample_physical(warm,42,'natural',0,0,0);warm.empty_cache();warm.close()
    for seed in (42,0,43):
        native=make(seed);begin=time.perf_counter();expected=server.sample_physical(native,seed,'natural',0,-15,10);baseline_seconds=time.perf_counter()-begin
        native.empty_cache();native.close()
        preview=make(seed);final=make(seed)
        assert preview.tile_store is not final.tile_store
        patch=server.sample_latent_preview(preview,2,-4,2)
        assert getattr(final.tile_store,'_bytes',0)==0,'Preview populated final store'
        begin=time.perf_counter();actual=server.sample_physical(final,seed,'natural',0,-15,10);final_seconds=time.perf_counter()-begin
        equal=[exact(a,b) for a,b in zip(expected[:2],actual[:2])]
        final.empty_cache();final.close()
        preview.empty_cache();replay=server.sample_latent_preview(preview,2,-4,2)
        replay_equal=[exact(a,b) for a,b in zip(patch[:2],replay[:2])]
        preview.empty_cache();preview.close()
        parent=make(seed);full=server.sample_physical(parent,seed,'natural',3,-2,1)
        # Patch (-4,2) is the upper-left quadrant of parent (-2,1).
        crop=full[0][24:152,24:152];inner=patch[0][24:-24,24:-24]
        difference=np.abs(inner.astype(np.float64)-crop.astype(np.float64))
        parent.empty_cache();parent.close()
        entry=dict(seed=seed,native_byte_exact=equal,preview_replay_byte_exact=replay_equal,
                   finite=all(np.isfinite(v).all() for v in (*patch[:2],*actual[:2])),
                   parent_preview_height_error_m=dict(max=float(difference.max()),mean=float(difference.mean())),
                   native_first_seconds=baseline_seconds,native_after_isolated_preview_seconds=final_seconds)
        entry['passed']=all(equal+replay_equal) and entry['finite'];report['cases'].append(entry);print(json.dumps(entry),flush=True)
report['passed']=all(c['passed'] for c in report['cases'])
report['source_sha256']={p:hashlib.sha256((source_path(p, root=server.ROOT)).read_bytes()).hexdigest() for p in ('terrain_server.py','terrain_inference.py','terrain_nn_constants.py','tools/verification/validate_terrain_preview.py')}
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
if not report['passed']:raise SystemExit('Preview fidelity gate failed')
