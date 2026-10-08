"""Isolated-port profiling server: frozen source supported, no automatic browser."""
import argparse,sys
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-dir');parser.add_argument('--port',type=int,default=8766);parser.add_argument('--prewarm-base',action='store_true');args=parser.parse_args()
if args.source_dir:sys.path.insert(0,str(Path(args.source_dir).resolve()))
import torch
import terrain_server as server
with server.gpu_lock,torch.inference_mode():
    server.preload_state['state']='loading'
    server.shared_pipeline=server.load_pipeline(42)
    server.preload_state['state']='warming'
    server.sample_elevation(server.shared_pipeline,0,0,0)
    if args.prewarm_base and hasattr(server,'warm_base_forms'):
        server.preload_state['base_forms']=server.warm_base_forms(server.shared_pipeline)
    server.shared_pipeline.empty_cache()
    server.preload_state['state']='ready'
server.app.run(host='127.0.0.1',port=args.port,threaded=True,use_reloader=False)
