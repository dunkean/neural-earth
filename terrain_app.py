"""Local CUDA Terrain Diffusion demo; weights and large artifacts live on E:."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = Path('E:/TerrainDiffusionRuntime')
os.environ['HF_HOME'] = str(RUNTIME / 'huggingface')
os.environ['MPLBACKEND'] = 'Agg'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
import sys
sys.path.insert(0, str(ROOT / 'terrain-diffusion'))
# Upstream resolves its bundled geography data relative to the working directory.
os.chdir(ROOT / 'terrain-diffusion')
import argparse
import json
import threading
import time
import traceback
import numpy as np
import torch
from terrain_device import select_cuda_device
# Set the selected ordinal before the server chooses its numerical/cache profile.
GPU_SELECTION = select_cuda_device()
from PIL import Image
from flask import Flask, jsonify, request, send_from_directory
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_diffusion.inference.relief_map import get_relief_map
from terrain_inference import configure_world

OUTPUT = ROOT / 'generated'
OUTPUT.mkdir(exist_ok=True)
MODEL = 'xandergos/terrain-diffusion-30m'
MODEL_REVISION = '9ef8030cb805b433b98ec25c5dddefbac07a9e26'
# Installed immutable snapshot: avoid Hub HTTP validation on every server boot.
_installed_model = RUNTIME / 'huggingface' / 'hub' / 'models--xandergos--terrain-diffusion-30m' / 'snapshots' / MODEL_REVISION
_required_model_files = ['config.json'] + [f'{stage}/{name}' for stage in
    ('coarse_model', 'base_model', 'decoder_model') for name in
    ('config.json', 'diffusion_pytorch_model.safetensors')]
MODEL_SOURCE = str(_installed_model) if all((_installed_model / name).exists() for name in _required_model_files) else MODEL


def resolve_model_source():
    if MODEL_SOURCE != MODEL:
        return MODEL_SOURCE
    # WorldPipeline.from_pretrained does not propagate `revision` to its three
    # submodels. Resolve the immutable snapshot first, then load its local path.
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=MODEL, revision=MODEL_REVISION,
                             allow_patterns=['config.json', 'coarse_model/*', 'base_model/*', 'decoder_model/*'])
app = Flask(__name__)
lock = threading.Lock()
state = {'busy': False, 'progress': 0, 'message': 'Ready', 'result': None}
pipeline = None
gpu_calls = {'coarse': 0, 'base': 0, 'decoder': 0}
if (OUTPUT / 'terrain.json').exists():
    state['result'] = json.loads((OUTPUT / 'terrain.json').read_text())


def load_pipeline(seed):
    global pipeline
    selected = select_cuda_device()
    cuda_device = f"cuda:{selected['selected']['index']}"
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: this demo requires an NVIDIA GPU.')
    if pipeline is None:
        torch.set_num_threads(8)
        # Weight version counters must remain available for the immutable eval
        # cache, even when the caller already uses torch.inference_mode().
        with torch.inference_mode(False):
            pipeline = WorldPipeline.from_pretrained(
                resolve_model_source(), seed=seed, torch_compile=False, dtype='bf16',
                latents_batch_size=16, log_mode='info', cache_limit=512 * 1024 * 1024,
            ).to(cuda_device)
        configure_world(pipeline)
        pipeline.bind()
        def check_cuda(name):
            def hook(model, args):
                if not args or not isinstance(args[0], torch.Tensor) or not args[0].is_cuda:
                    raise RuntimeError(f'Inference {name} ran outside CUDA')
                gpu_calls[name] += 1
            return hook
        for name, model in zip(gpu_calls, (pipeline.coarse_model, pipeline.base_model, pipeline.decoder_model)):
            model.register_forward_pre_hook(check_cuda(name))
            def note_graph_forwards(x,count,name=name):
                if not x.is_cuda:
                    raise RuntimeError(f'Inference {name} ran outside CUDA')
                gpu_calls[name] += count
            model._terrain_note_graph_forwards=note_graph_forwards
    elif pipeline.seed != seed:
        pipeline.change_seed(seed)
    for model in (pipeline.coarse_model, pipeline.base_model, pipeline.decoder_model):
        assert next(model.parameters()).is_cuda
    return pipeline


@torch.inference_mode()
def generate(seed=42, width=4096, height=2048, x=None, y=None):
    started = time.perf_counter()
    state.update(busy=True, progress=0, message='Loading CUDA model…', error=None)
    try:
        world = load_pipeline(seed)
        if x is None or y is None:
            # Pick a land region from the actual learned coarse map.
            state['message'] = 'Finding a land region…'
            coarse = world.coarse[:, -24:24, -24:24]
            raw = (coarse[0] / (coarse[-1] + 1e-8)).float().cpu().numpy()
            elev = np.sign(raw) * raw**2
            candidates = []
            ch, cw = max(1, height // 256), max(1, width // 256)
            for i in range(1, 47-ch):
                for j in range(1, 47-cw):
                    patch = elev[i:i+ch, j:j+cw]
                    score = np.mean(patch > 0) * 1000 + min(float(patch.std()), 1200) * .2
                    candidates.append((score, i, j))
            _, i, j = max(candidates)
            y, x = (i-24)*256, (j-24)*256
        elevation = np.empty((height, width), np.float32)
        total = ((height+511)//512) * ((width+511)//512)
        done = 0
        torch.cuda.reset_peak_memory_stats()
        first = None
        for row in range(0, height, 512):
            for col in range(0, width, 512):
                h, w = min(512, height-row), min(512, width-col)
                sample = world.get(y+row, x+col, y+row+h, x+col+w, with_climate=False)['elev']
                # InfiniteTensor returns CPU cache tensors; neural forwards are
                # checked on CUDA by the model hooks above.
                array = sample.float().cpu().numpy()
                if not np.isfinite(array).all():
                    raise RuntimeError('The model produced non-finite values.')
                elevation[row:row+h, col:col+w] = array
                if first is None:
                    first = array[:64, :64].copy()
                done += 1
                state.update(progress=round(done/total*95), message=f'GPU generation: {done}/{total} blocks')
                print(f'{done}/{total} blocks, {time.perf_counter()-started:.1f}s', flush=True)
        # Independently query a smaller overlapping region to check consistency.
        repeat = world.get(y, x, y+64, x+64, with_climate=False)['elev'].float().cpu().numpy()
        overlap_error = float(np.max(np.abs(first-repeat)))
        if not np.allclose(first, repeat, atol=1.0, rtol=.001):
            raise RuntimeError(f'Inconsistent overlap: {overlap_error} m')
        state['message'] = 'Creating relief and saving…'
        rgb = get_relief_map(elevation, None, None, None, resolution=world.native_resolution, vmin=0, vmax=4500)
        Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(OUTPUT / 'terrain.tmp.png')
        np.save(OUTPUT / 'elevation.npy', elevation)
        (OUTPUT / 'terrain.tmp.png').replace(OUTPUT / 'terrain.png')
        result = {
            'seed': str(seed), 'width': width, 'height': height, 'x': x, 'y': y,
            'resolution': world.native_resolution, 'model': MODEL,
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'dtype': 'bfloat16', 'seconds': round(time.perf_counter()-started, 2),
            'peak_vram_gb': round(torch.cuda.max_memory_allocated()/1024**3, 2),
            'min_m': round(float(elevation.min()), 2), 'max_m': round(float(elevation.max()), 2),
            'overlap_max_error_m': overlap_error, 'timestamp': time.time(),
            'cuda_forward_calls': dict(gpu_calls),
        }
        (OUTPUT / 'terrain.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        state.update(result=result, progress=100, message='Terrain ready')
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception as exc:
        state.update(error=str(exc), message='Generation error')
        traceback.print_exc()
        raise
    finally:
        state['busy'] = False


@app.get('/')
def index():
    return send_from_directory(ROOT, 'index.html')


@app.get('/generated/<path:name>')
def generated(name):
    return send_from_directory(OUTPUT, name)


@app.get('/api/status')
def status():
    return jsonify(state)


@app.post('/api/generate')
def generate_request():
    data = request.get_json() or {}
    try:
        seed = int(data.get('seed', 42))
        width, height = int(data.get('width', 4096)), int(data.get('height', 2048))
        if not 0 <= seed < 2**64 or width not in (1024, 2048, 4096) or height not in (1024, 2048):
            raise ValueError('Invalid seed or dimensions')
        x = int(data['x']) if data.get('x') not in (None, '') else None
        y = int(data['y']) if data.get('y') not in (None, '') else None
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    with lock:
        if state['busy']:
            return jsonify(error='Generation is already in progress'), 409
        state.update(busy=True, progress=0, error=None, message='Starting…')
        threading.Thread(target=generate, args=(seed, width, height, x, y), daemon=True).start()
    return jsonify(ok=True), 202


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.generate:
        generate()
    else:
        app.run(host='127.0.0.1', port=args.port, threaded=True, debug=False)
