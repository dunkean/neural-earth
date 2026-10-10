"""Natural Terrain Diffusion tiles, camera-aware queue and physical height cache."""

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path
import io
from copy import deepcopy
import hashlib
import json
import math
import os
from collections import OrderedDict
from functools import wraps
from contextlib import nullcontext, contextmanager, ExitStack
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter
from flask import Flask, Response, jsonify, request, send_file, send_from_directory, has_request_context

import terrain_app as terrain_runtime
from terrain_app import ROOT, OUTPUT, MODEL, load_pipeline, gpu_calls
from terrain_diffusion.inference.relief_map import get_relief_map
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_jobs import TerrainJobs, JobCancelled, QueueFull
from terrain_delivery import PhysicalDelivery
from terrain_priority import land_mask, tile_relief_stats
from terrain_disk_cache import TerrainDiskCache
from terrain_climate import CLIMATE_SIZE, MODES, sample_coarse_climate, colorize
from terrain_orogen_layers import MODES as OROGEN_MODES, LEGENDS as OROGEN_LEGENDS, render as render_orogen_layer
from terrain_snr_layer import MODES as SNR_MODES, render as render_snr_layer
from terrain_styles import STYLES, MODES as STYLE_MODES, render_style, apply_contours, parse_contours
from terrain_biomes import colorize_biomes, sample_biome_fields, sample_biome_transport, parse_rock_slope
from terrain_koppen import COLORS as KOPPEN_COLORS, sample_classes
from terrain_soil import sample_soil_transport
from terrain_render import colorize_surface, parse_settings as parse_render_settings, appearance_identity
APPEARANCE_IDENTITY=appearance_identity()
MODES = MODES + STYLE_MODES + OROGEN_MODES + SNR_MODES
from terrain_conditioning import sample_conditioning_preview, WORLD_PROFILES
from terrain_generation import resolve_generation, register_generation, generator_schema
from terrain_bootstrap import RASTER_WIDTH, RASTER_HEIGHT
from terrain_manifest import build_manifest, world_identity, _digest
from terrain_coarse import CoarsePreparation
from terrain_background import CoarseBackground
from terrain_generation_session import GenerationCoordinator, GenerationCancelled, generation_scope, check_generation
from terrain_final_mips import plan_mip, read_mip
from terrain_refinement import MIN_LOD, VERSION as REFINEMENT_VERSION, sample_refined
import terrain_native_coarse as native_coarse
from terrain_profiling import span, trace, measured_lock, snapshot as profiling_snapshot
from functools import lru_cache

app = Flask(__name__)
from terrain_backend_proxy import register_backend_proxy
register_backend_proxy(app)
image_slots = threading.BoundedSemaphore(2)
TILE = 256
HALO = 24
NATIVE = 30
VERSION = 'natural-v1'
GPU_SELECTION = getattr(terrain_runtime, 'GPU_SELECTION', None)
SELECTED_DEVICE = GPU_SELECTION['selected']['index'] if GPU_SELECTION else 0
import terrain_device
GPU_SETTINGS_PATH = OUTPUT / 'gpu-settings.json'
# Startup environment overrides win over the menu and are reported as locked.
GPU_ENVIRONMENT = terrain_device.environment_gpu_settings()


def visible_gpus():
    return GPU_SELECTION['visible_devices'] if GPU_SELECTION else []


def resolve_plan(settings):
    devices = visible_gpus()
    if not devices:
        return dict(requested_mode='single', mode='single', reason='no CUDA inventory',
                    coarse_device=SELECTED_DEVICE, detail_devices=[SELECTED_DEVICE],
                    devices=[SELECTED_DEVICE], settings=terrain_device.default_gpu_settings(), eligible=[])
    return terrain_device.resolve_gpu_plan(devices, SELECTED_DEVICE, dict(settings, **GPU_ENVIRONMENT))


gpu_settings = (terrain_device.load_gpu_settings(GPU_SETTINGS_PATH, visible_gpus()) if visible_gpus()
                else terrain_device.default_gpu_settings())
gpu_plan = resolve_plan(gpu_settings)
# Every learned coarse window of this process comes from this device. It is
# part of the cache identity; changing it requires a server restart.
COARSE_DEVICE = gpu_plan['coarse_device']
COARSE_GPU = next((d for d in visible_gpus() if d['index'] == COARSE_DEVICE),
                  GPU_SELECTION['selected'] if GPU_SELECTION else None)


class DeviceRuntime:
    """Resident models, seed-local worlds and the compute lock of one GPU."""
    def __init__(self, index, lock=None):
        self.index = index
        self.lock = lock or threading.RLock()
        self.shared_pipeline = None
        self.worlds = OrderedDict()
        self.preview_worlds = OrderedDict()
        self.polar_worlds = OrderedDict()
        self.background_world = None
        self.background_world_key = None
        self.preload = dict(state='idle')

    def live_worlds(self):
        return [w for w in (self.shared_pipeline, self.background_world, *self.worlds.values(),
                            *self.preview_worlds.values(), *self.polar_worlds.values()) if w is not None]


runtimes = {COARSE_DEVICE: DeviceRuntime(COARSE_DEVICE)}
runtimes_lock = threading.Lock()
_runtime_local = threading.local()
# Historical names: the coarse device's lock and stores.
gpu_lock = runtimes[COARSE_DEVICE].lock
worlds = runtimes[COARSE_DEVICE].worlds
preview_worlds = runtimes[COARSE_DEVICE].preview_worlds
polar_worlds = runtimes[COARSE_DEVICE].polar_worlds
preload_state = runtimes[COARSE_DEVICE].preload


def runtime_for(index):
    with runtimes_lock:
        runtime = runtimes.get(index)
        if runtime is None:
            runtime = runtimes[index] = DeviceRuntime(index)
        return runtime


def current_runtime():
    return getattr(_runtime_local, 'runtime', None) or runtimes[COARSE_DEVICE]


@contextmanager
def using_runtime(runtime):
    """Run this thread's terrain calls against another device's stores."""
    previous = getattr(_runtime_local, 'runtime', None)
    _runtime_local.runtime = runtime
    try:
        with (terrain_device.using_device(runtime.index) if GPU_SELECTION else nullcontext()):
            yield runtime
    finally:
        _runtime_local.runtime = previous


@contextmanager
def all_gpu_locks():
    """Exclusive use of every GPU. Lock order: detail devices, coarse device last.

    A detail worker may wait for the coarse lock while holding its own (remote
    coarse windows); the coarse device never waits for another device.
    """
    with runtimes_lock:
        ordered = sorted((r for r in runtimes.values() if r.index != COARSE_DEVICE), key=lambda r: r.index)
        ordered.append(runtimes[COARSE_DEVICE])
    with ExitStack() as stack:
        for runtime in ordered:
            stack.enter_context(runtime.lock)
        yield ordered


def synchronize_gpus():
    with runtimes_lock:
        indices = sorted(runtimes)
    for index in indices:
        torch.cuda.synchronize(index)


def gpu_label():
    names = {d['index']: d['name'] for d in visible_gpus()}
    if not names:
        return torch.cuda.get_device_name(SELECTED_DEVICE)
    return ' + '.join(names.get(index, f'GPU {index}') for index in gpu_plan['devices'])


def device_provenance(index):
    """Tile receipt: which GPU ran the latent/decoder networks of this tile."""
    device = next((d for d in visible_gpus() if d['index'] == index), None)
    return dict(device=index, device_name=device['name'] if device else None,
                device_uuid=device['uuid'] if device else None, coarse_device_uuid=COARSE_GPU['uuid'] if COARSE_GPU else None)


def route_job(job):
    """Allowed devices for a queued GPU job, under the live plan."""
    plan = gpu_plan
    if job.role == 'detail':
        return plan['detail_devices']
    # Coarse tiles, native coarse transport, background preparation and any
    # unlabelled job stay on the device that owns learned coarse windows.
    return (plan['coarse_device'],)


def bind_gpu_worker(index):
    if GPU_SELECTION:
        terrain_device.bind_thread_device(index)
    _runtime_local.runtime = runtime_for(index)


try:
    from terrain_inference import choose_profile, VERSION as INFERENCE_VERSION
except ImportError as exc:
    if exc.name != 'terrain_inference':
        raise
    runtime_profile = None
    profile_data = {'version':'legacy-eager-bf16'}
else:
    runtime_profile = choose_profile()
    profile_data = dict(vars(runtime_profile), version=INFERENCE_VERSION)
    profile_data.pop('name', None)  # Free memory is not an identity of numerical output.
    profile_data.pop('coarse_streams', None)  # Batch-one execution count preserves physical output.
coarse_stream_setting = {'coarse': runtime_profile.coarse_streams if runtime_profile else 1}
# Numerical identity follows the coarse device: it decides every learned coarse
# window. In single-GPU mode it is the selected device, as before multi-GPU.
profile_data.update(model_revision=getattr(terrain_runtime, 'MODEL_REVISION', 'unversioned'),
    torch=str(torch.__version__), cuda=torch.version.cuda, cuda_device=COARSE_DEVICE,
    compute_capability=COARSE_GPU['compute_capability'] if COARSE_GPU else None,
    driver=GPU_SELECTION.get('driver') if GPU_SELECTION else None)
if COARSE_GPU:
    profile_data.update(gpu_name=COARSE_GPU['name'],gpu_uuid=COARSE_GPU['uuid'])
from terrain_coastline import VERSION as COASTLINE_VERSION
profile_data.update(physical_lod_version='bandlimit-climate-v3', coastline_version=COASTLINE_VERSION)
profile_data.update(refinement_version=REFINEMENT_VERSION, refinement_min_lod=MIN_LOD)
from terrain_world import WORLD_VERSION, profile_metadata
from terrain_geometry import profile_bounds, geometry_heightmap
from terrain_lighting import parse_lighting, lighting_suffix, render_relief
profile_data.update(world_version=WORLD_VERSION, world_sources=profile_metadata()['source_digest'])
PROFILE = hashlib.sha256(json.dumps(profile_data, sort_keys=True).encode()).hexdigest()[:16]
# The persistent namespace includes the actual generator and source contents,
# rather than just human-readable version labels. Cache it once at startup.
_reference_manifest = build_manifest(0, numerical_profile=profile_data['version'],inference_profile=profile_data)
profile_data['generator_identity'] = world_identity(_reference_manifest)
PROFILE = hashlib.sha256(json.dumps(profile_data, sort_keys=True).encode()).hexdigest()[:16]
CACHE = OUTPUT / VERSION / PROFILE
CACHE.mkdir(parents=True, exist_ok=True)
PHYSICAL_CACHE = CACHE / 'physical-v1'
PHYSICAL_CACHE.mkdir(exist_ok=True)
active_seed = None
jobs = TerrainJobs(cpu_workers=2 if len(visible_gpus()) < 2 else len(visible_gpus())+1,
                   gpu_lanes=max(1, len(visible_gpus())))
jobs.configure_gpus(gpu_plan['devices'], route_job, bind_gpu_worker)
terrain_device.set_multi_gpu_workers(len(gpu_plan['devices']) > 1)
physical_delivery = PhysicalDelivery()
physical_locks = [threading.RLock() for _ in range(64)]
disk_cache = TerrainDiskCache(CACHE, VERSION,
    budget_bytes=int(float(os.environ.get('TERRAIN_DISK_CACHE_GIB', '16'))*1024**3),
    protected_keys=jobs.protected_keys)
metrics = {'generated_tiles': 0, 'cache_hits': 0, 'stage': 'Ready', 'last_seconds': 0}
native_coarse_cache = OrderedDict()
native_coarse_cache_lock = threading.Lock()


@app.errorhandler(RuntimeError)
@app.errorhandler(OSError)
@app.errorhandler(subprocess.SubprocessError)
def generation_failure(exc):
    return jsonify(error=str(exc),stage='generation'),503


def pin_cache_io(key_function):
    """Pin generation/read IO and keep Windows send_file streams pinned."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                key = key_function(*args, **kwargs)
            except (ValueError, TypeError):
                return function(*args, **kwargs)
            with disk_cache.acquire(key):
                response = function(*args, **kwargs)
                if isinstance(response, Response) and response.direct_passthrough:
                    lease = disk_cache.acquire(key)
                    # Flask now wraps the iterator with Response.close(), which
                    # releases this lease after send_file's handle is closed.
                    response.direct_passthrough = False
                    response.call_on_close(lease.release)
                return response
        return wrapped
    return decorate


def generation_profile(value=None):
    value = value or (request.args.get('world_profile','natural') if has_request_context() else 'natural')
    resolve_generation(value)
    return value


def display_mode():
    mode=request.args.get('mode','biomes')
    if mode not in MODES:
        raise ValueError('Invalid map mode')
    if mode in OROGEN_MODES and resolve_generation(generation_profile()).bootstrap_generator != 'orogen':
        raise ValueError('Orogen diagnostic layers require an Orogen pipeline')
    return mode


@lru_cache(maxsize=32)
def world_manifest(seed, world_profile='natural'):
    descriptor = resolve_generation(world_profile)
    ablation = 'A0' if world_profile == 'natural' else world_profile
    runtime = dict(profile_data)
    # Loaded models and modules are fixed for this process. Reuse their startup
    # file/runtime snapshot for every seed; later on-disk edits take effect on
    # restart, never halfway through an active cache namespace.
    manifest=build_manifest(seed, ablation, numerical_profile=profile_data['version'],
                            inference_profile=runtime,file_hashes=False)
    manifest['files']=deepcopy(_reference_manifest['files'] if not descriptor.needs_bootstrap else
                               terrestrial_file_snapshot(descriptor.bootstrap_generator))
    if descriptor.bootstrap_generator=='orogen' and descriptor.settings['height_source'] in ('native','natural-continental'):
        from terrain_bootstrap import implementation_identity as native_identity
        manifest['files']['bootstrap_native']=deepcopy(native_identity())
    manifest['checkpoint']['config']=deepcopy(_reference_manifest['checkpoint']['config'])
    manifest['generation']['runtime_versions']=deepcopy(
        _reference_manifest['generation']['runtime_versions'])
    manifest['complete']=True
    payload={k:v for k,v in manifest.items() if k!='world_hash'}
    manifest['world_hash']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
    return manifest


@lru_cache(maxsize=2)
def terrestrial_file_snapshot(generator='native'):
    """Load native prerequisites only when a terrestrial profile is requested."""
    from terrain_bootstrap import implementation_identity
    files = deepcopy(_reference_manifest['files'])
    if generator == 'orogen':
        from terrain_orogen import implementation_identity as orogen_identity
        files['implementation']['terrain_bootstrap.py'] = _digest(source_path('terrain_bootstrap.py', root=ROOT))
        files['bootstrap_orogen'] = deepcopy(orogen_identity())
        return files
    native_identity=deepcopy(implementation_identity())
    source_digest=_digest(source_path('terrain_bootstrap.py', root=ROOT))
    if source_digest['sha256'] != native_identity['python_source_sha256']:
        raise RuntimeError('Bootstrap source changed since startup: restart the server')
    files['implementation']['terrain_bootstrap.py'] = source_digest
    files['bootstrap_native'] = native_identity
    return files


def conditioning_preview(seed, world_profile, xs_metres, ys_metres, *, polar=False):
    if polar:
        from terrain_polar import conditioning
        from terrain_world import lapse_rate
        fields=conditioning(seed,world_profile).sample(xs_metres,ys_metres)
        data=dict(elev=fields[0],climate=np.concatenate((fields[1:],lapse_rate(fields[3])[None]),axis=0))
    else:
        data = sample_conditioning_preview(seed, world_profile, xs_metres, ys_metres)
    # The transport/shader stores sea-level BIO1, not temperature at terrain
    # height. The factory returns physical BIO1, so undo its lapse exactly once.
    data = dict(data, climate=np.asarray(data['climate'],np.float32).copy())
    data['climate'][0] -= data['climate'][4]*np.maximum(data['elev'],0)
    return data


def clip_world_axes(xs, ys, profile):
    x0, y0, x1, y1 = profile_bounds(profile)
    return np.clip(xs, x0/NATIVE, x1/NATIVE), np.clip(ys, y0/NATIVE, y1/NATIVE)


def _create_world(seed, world_profile, cache_limit, *, polar=False):
    runtime = current_runtime()
    if runtime.shared_pipeline is None:
        runtime.shared_pipeline = load_pipeline(seed, device=runtime.index)
    shared_pipeline = runtime.shared_pipeline
    # Seed-local mutable caches; neural weights remain shared on the GPU.
    config = {k:v for k,v in dict(shared_pipeline.config).items() if not k.startswith('_')}
    config.update(seed=seed, latents_batch_size=16, dtype='bf16',
                  cache_limit=cache_limit, torch_compile=False)
    world = WorldPipeline(**config)
    world.coarse_model = shared_pipeline.coarse_model
    world.base_model = shared_pipeline.base_model
    world.decoder_model = shared_pipeline.decoder_model
    try:
        from terrain_inference import configure_world
        if polar:
            from terrain_polar import conditioning
            world._terrain_polar_conditioning = conditioning(seed,world_profile)
        configure_world(world, runtime_profile, world_profile=world_profile)
        world.bind()
        from terrain_inference import set_coarse_streams
        set_coarse_streams(world, coarse_stream_setting['coarse'])
        world._terrain_manifest = world_manifest(seed,world_profile)
        if polar:
            from terrain_polar import chart_manifest
            world._terrain_manifest = chart_manifest(world._terrain_manifest)
        preparation=CoarsePreparation(OUTPUT/'coarse-worlds',world._terrain_manifest,bounds=profile_bounds(world_profile)).install(world)
        if runtime.index != COARSE_DEVICE:
            preparation.remote = remote_coarse(seed, world_profile, polar=polar)
    except ValueError as exc:
        world.close()
        raise RuntimeError(f'Coarse world admission failed: {exc}') from exc
    world._terrain_device = runtime.index
    return world


def remote_coarse(seed, world_profile, *, polar=False):
    """Missing coarse windows of a detail-device world, computed on the coarse GPU.

    Runs in the detail worker thread, which already owns its own device lock;
    taking the coarse lock second is the global lock order. The coarse world's
    persisted model function loads or computes and queues each window for disk.
    """
    def compute(indices, *args):
        runtime = runtime_for(COARSE_DEVICE)
        with measured_lock(runtime.lock, 'coarse.remote_lock_wait'), using_runtime(runtime), torch.inference_mode():
            world = get_polar_world(seed, world_profile) if polar else get_world(seed, world_profile, foreground=False)
            with span('coarse.remote_windows', windows=len(indices), device=runtime.index):
                # Groups follow the coarse world's own stream count, so a group
                # is never solved as one multi-window network batch.
                size = max(1, world.coarse._batch_size or 1)
                outputs = []
                for start in range(0, len(indices), size):
                    outputs.extend(world.coarse._f(list(indices[start:start+size]), *args))
                # One synchronising host copy; this GPU pair has no peer access.
                return [output.detach().to('cpu', dtype=torch.float32) for output in outputs]
    return compute


preview_worlds=OrderedDict()
polar_worlds=OrderedDict()


def close_world(world):
    preparation=getattr(world,'_terrain_coarse_preparation',None)
    if preparation is not None and hasattr(preparation,'close'):
        preparation.close()
    world.empty_cache()
    world.close()


def get_preview_world(seed,profile):
    # Preview batch composition must never populate the final latent store.
    preview_worlds=current_runtime().preview_worlds
    key=(profile,seed)
    if key not in preview_worlds:
        preview_worlds[key]=_create_world(seed,profile,128*1024*1024)
        while len(preview_worlds)>2:
            _,expired=preview_worlds.popitem(last=False)
            close_world(expired)
    preview_worlds.move_to_end(key)
    return preview_worlds[key]


def get_polar_world(seed,profile,*,preview=False):
    # Approximate latent batching remains isolated from the final chart store.
    polar_worlds=current_runtime().polar_worlds
    key=(profile,seed,preview)
    if key not in polar_worlds:
        polar_worlds[key]=_create_world(seed,profile,128*1024*1024,polar=True)
        while len(polar_worlds)>2:
            _,expired=polar_worlds.popitem(last=False)
            close_world(expired)
    polar_worlds.move_to_end(key)
    return polar_worlds[key]


def warm_base_forms(world):
    from terrain_cuda_graphs import prewarm_base_forms
    before=dict(gpu_calls)
    try:
        result=prewarm_base_forms(world.base_model,world._terrain_profile.latent_batch)
    except Exception as exc:
        # An optional prewarm failure does not discard loaded, usable weights.
        result=dict(enabled=True,state='failed',fully_warmed=False,error=f'{type(exc).__name__}: {exc}'[:400])
    return dict(result,warmup_cuda_forward_calls={key:gpu_calls[key]-before[key] for key in before})


def warm_graphs(world, *, coarse=True, state=None):
    """Capture graphs; ``coarse=False`` on detail-only GPUs that never run it."""
    from terrain_inference import prewarm_coarse_graphs, set_coarse_streams
    from terrain_cuda_graphs import prewarm_decoder_form
    state = preload_state if state is None else state
    set_coarse_streams(world, coarse_stream_setting['coarse'])
    stages = [('coarse', lambda: prewarm_coarse_graphs(world))] if coarse else []
    if os.environ.get('TERRAIN_PREWARM_BASE', '1') == '1':
        stages.append(('base', lambda: warm_base_forms(world)))
    stages.append(('decoder', lambda: prewarm_decoder_form(world.decoder_model, world.decoder_tile_size)))
    results = {}
    for name, prepare in stages:
        state['warming_model'] = name
        try:
            results[name] = prepare()
        except Exception as exc:
            # Optional capture failure must not discard resident usable weights.
            results[name] = dict(enabled=True, fully_warmed=False, state='failed',
                                 error=f'{type(exc).__name__}: {exc}'[:400])
    state.pop('warming_model', None)
    return dict(models=results, fully_warmed=all(
        not result.get('enabled') or result.get('fully_warmed', False) for result in results.values()))


def get_world(seed, world_profile='natural', *, foreground=True):
    global active_seed
    if foreground:
        active_seed = seed
    worlds = current_runtime().worlds
    world_key=(world_profile,seed)
    if world_key in worlds:
        worlds.move_to_end(world_key)
        return worlds[world_key]
    world = _create_world(seed,world_profile,512*1024*1024)
    worlds[world_key] = world
    while len(worlds) > 2:
        _, expired = worlds.popitem(last=False)
        # The lazy stages hold a world/store reference cycle. Drop regenerable
        # GPU windows explicitly instead of waiting for Python's cycle collector.
        close_world(expired)
    return world


def prepare_coarse_quantum(seed, profile, *, budget_windows=1):
    runtime = current_runtime()
    with runtime.lock, torch.inference_mode():
        jobs.check_current_interest()
        world_manifest(seed, profile)
        mask = scheduling_land_mask(seed, profile)
        key=(profile,seed)
        if runtime.background_world_key != key:
            if runtime.background_world is not None:
                close_world(runtime.background_world)
            runtime.background_world = None
            runtime.background_world_key = None
            # One bounded seed-local workspace shares neural weights but never
            # enters the foreground LRU or changes its active seed.
            runtime.background_world = _create_world(seed,profile,64*1024*1024)
            runtime.background_world_key = key
        world=runtime.background_world
        world._terrain_coarse_preparation.prioritize(mask, coarse_background.focus(seed, profile))
        return world._terrain_coarse_preparation.step(world,budget_windows=budget_windows)


def _available_world(seed, profile):
    """Find a prepared world without loading models for untouched previews."""
    runtime=current_runtime()
    world=runtime.worlds.get((profile,seed))
    if world is not None:
        return world
    if runtime.background_world_key==(profile,seed):
        return runtime.background_world
    # After a background world is replaced or a server restart, persisted
    # windows must still be discoverable by a foreground high-LOD request.
    probe=CoarsePreparation(OUTPUT/'coarse-worlds',world_manifest(seed,profile),
                            bounds=profile_bounds(profile))
    if any(probe.windows_dir.glob('*.npy')):
        return get_world(seed,profile)
    return None


coarse_background=CoarseBackground(jobs,prepare_coarse_quantum,None,
                                 window_quantum=lambda:coarse_stream_setting['coarse'])
generation_coordinator=GenerationCoordinator(jobs,coarse_background)


@app.route('/api/inference/streams', methods=['GET', 'POST'])
def inference_stream_settings():
    if request.method == 'POST':
        data = request.get_json(silent=True)
        count = data.get('coarse') if isinstance(data, dict) else None
        if type(count) is not int or count not in (1,2,4,8,16):
            return jsonify(error='Coarse streams must be 1, 2, 4, 8 or 16'),400
        with all_gpu_locks() as locked:
            if count == coarse_stream_setting['coarse']:
                return jsonify(**coarse_stream_setting, options=[1,2,4,8,16])
            from terrain_inference import set_coarse_streams
            live = [world for runtime in locked for world in runtime.live_worlds()]
            owners = set()
            for world in live:
                if world is None:
                    continue
                set_coarse_streams(world, count)
                owner = world.coarse_model
                if id(owner) not in owners:
                    owners.add(id(owner))
                    cached = owner.__dict__.pop('_terrain_stream_pool', None)
                    if cached:
                        cached[1].close()
                    owner.__dict__.pop('_terrain_stream_pool_failed_key', None)
                    owner.__dict__.pop('_terrain_stream_pool_error', None)
            coarse_stream_setting['coarse'] = count
    return jsonify(**coarse_stream_setting, options=[1,2,4,8,16])


@app.before_request
def suspend_terrain_admission():
    if jobs.paused and (request.path.startswith(('/height/','/tiles/','/coarse/','/api/overview/')) or request.path=='/api/view'):
        return jsonify(error='A world is being generated',cancelled=True),409


def finish_generation(token):
    with generation_coordinator.lock:
        if generation_coordinator.current is token:
            worker=app.extensions.get('terrain_reference_worker')
            try:
                if worker is not None:
                    worker.suspend(False)
            finally:
                generation_coordinator.finish(token)


@app.post('/api/terrain/suspend')
def terrain_suspend():
    """Drain a reference worker before the parent uses the shared GPU."""
    if (request.get_json() or {}).get('paused',True):
        generation_coordinator.suspend(True)
        with all_gpu_locks():
            synchronize_gpus()
    else:
        generation_coordinator.suspend(False)
    return jsonify(paused=jobs.paused)


@app.post('/api/coarse/prepare')
def coarse_prepare():
    try:
        data=request.get_json() or {}
        if data.get('action')=='stop':
            return jsonify(coarse_background.stop())
        if data.get('action','start')!='start':
            raise ValueError('Invalid coarse action')
        seed=int(data.get('seed',42))
        if not 0<=seed<2**64:
            raise ValueError('Invalid seed')
        profile=generation_profile(data.get('world_profile','natural'))
        budget = None if data.get('full_world') or 'max_windows' not in data else int(data['max_windows'])
        with generation_coordinator.lock:
            if generation_coordinator.current is not None or jobs.paused:
                return jsonify(error='A world is being generated',state='paused'),409
        with all_gpu_locks():
            if jobs.paused:
                return jsonify(error='A world is being generated',state='paused'),409
            world_manifest(seed, profile)
        with generation_coordinator.lock:
            if generation_coordinator.current is not None or jobs.paused:
                return jsonify(error='A world is being generated',state='paused'),409
            return jsonify(coarse_background.start(seed,profile,budget)),202
    except (ValueError,TypeError) as exc:
        return jsonify(error=str(exc)),400


def _field_block(world, source, x0, y0, x1, y1):
    jobs.check_current_interest()
    from terrain_window_scheduler import read_rect
    data = read_rect(world,source,y0,x0,y1,x1,check=jobs.check_current_interest)
    if source == 'coarse':
        return (data[0] / (data[-1]+1e-8)).float().cpu().numpy()
    return ((data[4] / (data[-1]+1e-8))*38.6-31.4).float().cpu().numpy()


def sample_field(world, xs, ys, source, *, smooth_coarse=False):
    """Aligned samples with bounded host buffers; optional monotone coarse preview.

    Sparse macro views no longer materialize their full intervening rectangle.
    Neural work still grows with the number of new source windows required.
    """
    stride = 256 if source == 'coarse' else world.latent_compression
    cx, cy = xs / stride - .5, ys / stride - .5
    x0, y0 = math.floor(float(cx.min()))-1, math.floor(float(cy.min()))-1
    x1, y1 = math.ceil(float(cx.max()))+2, math.ceil(float(cy.max()))+2
    if (x1-x0)*(y1-y0) <= 512*512:
        field = _field_block(world, source, x0, y0, x1, y1)
        if smooth_coarse and source == 'coarse':
            from terrain_interpolation import monotone_grid
            sqrt_elev = monotone_grid(field,cx-x0,cy-y0)
        else:
            yy, xx = np.meshgrid(cy-y0, cx-x0, indexing='ij')
            sqrt_elev = map_coordinates(field, [yy, xx], order=1, mode='nearest')
    else:
        # Group samples by bounded blocks using the same neighbour halo and
        # interpolation as dense queries.
        sqrt_elev = np.empty((len(cy), len(cx)), dtype=np.float32)
        xgroups, ygroups = np.floor(cx/48).astype(np.int64), np.floor(cy/48).astype(np.int64)
        for gy in np.unique(ygroups):
            yi = np.flatnonzero(ygroups == gy)
            by0, by1 = math.floor(float(cy[yi].min()))-1, math.ceil(float(cy[yi].max()))+2
            for gx in np.unique(xgroups):
                xi = np.flatnonzero(xgroups == gx)
                bx0, bx1 = math.floor(float(cx[xi].min()))-1, math.ceil(float(cx[xi].max()))+2
                field = _field_block(world, source, bx0, by0, bx1, by1)
                if smooth_coarse and source == 'coarse':
                    from terrain_interpolation import monotone_grid
                    sampled = monotone_grid(field,cx[xi]-bx0,cy[yi]-by0)
                else:
                    yy, xx = np.meshgrid(cy[yi]-by0, cx[xi]-bx0, indexing='ij')
                    sampled = map_coordinates(field, [yy, xx], order=1, mode='nearest')
                sqrt_elev[np.ix_(yi, xi)] = sampled
    return np.sign(sqrt_elev)*sqrt_elev**2


def _coarse_area_chunks(preparation, lod, tx, ty, halo=HALO):
    """Global coarse-cell footprints, extended constantly at world edges.

    This deliberately does not use CoarsePreparation.read_mip: that API's
    origin is relative to its finite grid, whereas tile footprints are aligned
    to the global native-pixel lattice at zero.
    """
    scale=(1 << lod)//256
    if scale < 1 or (1 << lod)%256:
        raise ValueError('Area sampling requires native step divisible by 256')
    grid=preparation.grid
    size=TILE+2*halo
    for oy in range(0,size,32):
        yend=min(size,oy+32)
        yi=((ty*TILE+np.arange(oy-halo,yend-halo))[:,None]*scale
            +np.arange(scale)[None,:])
        yi=np.clip(yi,grid.i1,grid.i2-1).reshape(-1)
        for ox in range(0,size,32):
            xend=min(size,ox+32)
            xi=((tx*TILE+np.arange(ox-halo,xend-halo))[:,None]*scale
                +np.arange(scale)[None,:])
            xi=np.clip(xi,grid.j1,grid.j2-1).reshape(-1)
            yield slice(oy,yend),slice(ox,xend),yi,xi


def _coarse_area_ready(preparation,lod,tx,ty):
    checked=set()
    for _,_,yi,xi in _coarse_area_chunks(preparation,lod,tx,ty):
        for index in preparation._required_indices(int(yi[0]),int(xi[0]),
                                                    int(yi[-1])+1,int(xi[-1])+1):
            if index not in checked:
                if not preparation._valid_metadata(index):
                    return False
                checked.add(index)
    return True


def sample_coarse_area(world,lod,tx,ty,halo=HALO):
    """Mean physical metres over each globally aligned display footprint."""
    from terrain_window_scheduler import read_rect
    preparation=world._terrain_coarse_preparation
    size=TILE+2*halo
    result=np.empty((size,size),np.float32)
    for rows,cols,yi,xi in _coarse_area_chunks(preparation,lod,tx,ty,halo):
        jobs.check_current_interest()
        y0,x0=int(yi[0]),int(xi[0])
        fused=read_rect(world,'coarse',y0,x0,int(yi[-1])+1,int(xi[-1])+1,
                        check=jobs.check_current_interest)
        values=fused.float().cpu().numpy()
        weight=values[-1]
        if not np.isfinite(values).all() or not np.all(weight>0):
            raise RuntimeError('Incomplete or invalid learned coarse fusion')
        sqrt_height=values[0]/weight
        metres=np.sign(sqrt_height)*np.square(sqrt_height)
        block=metres[np.ix_(yi-y0,xi-x0)]
        scale=(1<<lod)//256
        result[rows,cols]=block.reshape(rows.stop-rows.start,scale,
                                        cols.stop-cols.start,scale).mean(axis=(1,3))
    return result


def _learned_ready(preparation,xs,ys,lod,tx,ty):
    if preparation is None or not preparation.ready_for_samples(xs,ys,climate_halo=8):
        return False
    return lod<9 or _coarse_area_ready(preparation,lod,tx,ty)


@torch.inference_mode()
def sample_elevation(world, lod, tx, ty, halo=HALO, *, xs=None, ys=None,
                     coarse_interpolation='monotone'):
    step = 2**lod
    size = TILE + 2*halo
    x0, y0 = tx*TILE*step, ty*TILE*step
    if xs is None:
        xs = x0 + (np.arange(-halo, TILE+halo)+.5)*step
    if ys is None:
        ys = y0 + (np.arange(-halo, TILE+halo)+.5)*step
    if lod>=7:
        xs, ys = clip_world_axes(xs, ys, world._terrain_world_profile)
    if lod >= 9:
        elevation,stage=sample_coarse_area(world,lod,tx,ty,halo),'coarse-area-mean'
    elif lod >= 4:
        if lod == 4:
            coarse_interpolation = selected_coarse_interpolation(lod, coarse_interpolation)
            elevation = sample_field(world, xs, ys, 'coarse',
                                     smooth_coarse=coarse_interpolation == 'monotone')
        else:
            elevation = sample_field(world, xs, ys, 'coarse')
        stage = 'coarse'
    elif lod >= 3:
        elevation, stage = sample_field(world, xs, ys, 'latent'), 'latent'
    elif lod < 0:
        i0, j0 = ty*TILE-halo, tx*TILE-halo
        data = sample_refined(world, -lod, i0, j0, i0+size, j0+size,
                              check=jobs.check_current_interest)
        elevation, stage = data.cpu().numpy(), 'decoder-refinement'
    else:
        i0, j0 = y0-halo*step, x0-halo*step
        from terrain_coastline import read_native
        scheduler=getattr(world,'_terrain_window_scheduler',None)
        # Native DEM with coherent land/sea signs at near-zero coasts.
        if scheduler:
            with scheduler.scope(jobs.check_current_interest):
                data = read_native(world, i0, j0, i0+size*step, j0+size*step)
        else:
            data = read_native(world, i0, j0, i0+size*step, j0+size*step)
        # Reduction stays on CUDA; transfer only the display-sized height tile.
        elevation = data.float().reshape(size, step, size, step).mean(dim=(1, 3)).cpu().numpy()
        stage = 'decoder'
    if not np.isfinite(elevation).all():
        raise RuntimeError('Non-finite terrain values')
    return np.ascontiguousarray(elevation, dtype=np.float32), stage


def prepare_preview_view(world,seed,profile,lod,tx,ty,session):
    """Prefetch up to four current, adjacent previews as one bounded read.

    Only the isolated approximate store is touched. Native batching/order is
    unchanged. Repeated requests hit resident windows; cancellation still
    checks interest before each real neural batch.
    """
    if not session:
        return 1
    prefix=f'{VERSION}/{profile}/{seed}/{lod}/'
    with jobs.condition:
        view=jobs.views.get(session)
        wants=list(view['wants']) if view else []
    coordinates={(tx,ty)}
    for key in wants:
        if key.startswith(prefix) and key.endswith('/source3'):
            parts=key[len(prefix):].split('/')
            if len(parts)==3:
                x,y=int(parts[0]),int(parts[1])
                if abs(x-tx)<=1 and abs(y-ty)<=1:
                    coordinates.add((x,y))
    coordinates=sorted(coordinates,key=lambda p:(abs(p[0]-tx)+abs(p[1]-ty),p))[:4]
    if len(coordinates)<2:
        return 1
    inner=TILE//2**(3-lod)
    x0=min(x for x,y in coordinates)*inner-HALO-1
    x1=(max(x for x,y in coordinates)+1)*inner+HALO+1
    y0=min(y for x,y in coordinates)*inner-HALO-1
    y1=(max(y for x,y in coordinates)+1)*inner+HALO+1
    if (x1-x0)*(y1-y0)>384**2:
        return 1
    from terrain_window_scheduler import ensure_rect
    with span('preview.view_dependencies',gpu=True,tiles=len(coordinates),width=x1-x0,height=y1-y0):
        ensure_rect(world,'latent',y0,x0,y1,x1,check=jobs.check_current_interest)
    return len(coordinates)


def sample_latent_preview(world,lod,tx,ty):
    """Same 240 m source grid/halo; calculate a smaller canonical footprint."""
    inner=TILE//2**(3-lod)
    xs=tx*TILE*2**lod+(np.arange(-HALO,inner+HALO)+.5)*8
    ys=ty*TILE*2**lod+(np.arange(-HALO,inner+HALO)+.5)*8
    elevation=gaussian_filter(sample_field(world,xs,ys,'latent'),sigma=.65,mode='reflect').astype(np.float32)
    from terrain_snr import lod_relief
    elevation=lod_relief(world._terrain_generation_settings,elevation,lod)
    cx=np.linspace(xs[0],xs[-1],CLIMATE_SIZE);cy=np.linspace(ys[0],ys[-1],CLIMATE_SIZE)
    climate=sample_coarse_climate(world,cx,cy,check=jobs.check_current_interest)
    if not np.isfinite(elevation).all() or not np.isfinite(climate).all():
        raise RuntimeError('Non-finite latent preview')
    return np.ascontiguousarray(elevation),np.ascontiguousarray(climate),'latent'


def sample_physical(world, seed, world_profile, lod, tx, ty, *, preview_only=False,
                    coarse_interpolation='monotone', polar=False):
    step=2**lod
    xs=(tx*TILE*step+(np.arange(-HALO,TILE+HALO)+.5)*step)
    ys=(ty*TILE*step+(np.arange(-HALO,TILE+HALO)+.5)*step)
    if lod>=7:
        # Only the finite rectangle is displayed. Extend its edge constantly
        # for shading halo samples instead of requiring a planet beyond it.
        xs, ys = clip_world_axes(xs, ys, world_profile)
    if world is None and not preview_only and not polar:
        world=_available_world(seed,world_profile)
    preparation=getattr(world,'_terrain_coarse_preparation',None)
    learned_ready=not preview_only and lod>=7 and _learned_ready(preparation,xs,ys,lod,tx,ty)
    final_mip=None if polar else existing_final_mip(seed,world_profile,lod,tx,ty)
    if final_mip is not None:
        elevation,stage=final_mip,'final-dem-mip'
    elif lod>=7 and not learned_ready:
        macro=conditioning_preview(seed,world_profile,xs*NATIVE,ys*NATIVE,**({'polar':True} if polar else {}))
        elevation=np.asarray(macro['elev'],dtype=np.float32)
        stage='conditioning-preview'
    else:
        if lod == 4:
            elevation,stage=sample_elevation(world,lod,tx,ty,xs=xs,ys=ys,
                                             coarse_interpolation=coarse_interpolation)
        else:
            elevation,stage=sample_elevation(world,lod,tx,ty,xs=xs,ys=ys)
    # The physical height, rather than display color, is filtered. The same
    # halo and globally aligned sample centres prevent tile-edge discontinuity.
    if lod>=3:
        elevation=gaussian_filter(elevation,sigma=.65,mode='reflect').astype(np.float32)
    from terrain_snr import lod_relief
    elevation=lod_relief(resolve_generation(world_profile).settings,elevation,lod)
    cx=np.linspace(xs[0],xs[-1],CLIMATE_SIZE)
    cy=np.linspace(ys[0],ys[-1],CLIMATE_SIZE)
    if stage=='conditioning-preview':
        macro=conditioning_preview(seed,world_profile,cx*NATIVE,cy*NATIVE,**({'polar':True} if polar else {}))
        climate=np.asarray(macro['climate'],dtype=np.float32).copy()
    else:
        climate=sample_coarse_climate(world,cx,cy,check=jobs.check_current_interest)
    if not np.isfinite(elevation).all() or not np.isfinite(climate).all():
        raise RuntimeError('Non-finite terrain or climate')
    return np.ascontiguousarray(elevation),np.ascontiguousarray(climate),stage


def native_mip_paths(seed,profile,lod,tx,ty):
    plan=plan_mip(lod,tx,ty)
    if plan is None:
        return None
    base=CACHE/profile/'physical-v1'/str(seed)/'0'
    paths=[(base/f'{x}_{y}.npy',base/f'{x}_{y}.json') for x,y in plan.children]
    return (plan,paths) if all(a.exists() and b.exists() for a,b in paths) else None


def existing_final_mip(seed,profile,lod,tx,ty):
    dependencies=native_mip_paths(seed,profile,lod,tx,ty)
    if dependencies is None:
        return None
    plan,_=dependencies
    identity=world_identity(world_manifest(seed,profile))
    def child(x,y):
        base=CACHE/profile/'physical-v1'/str(seed)/'0'/f'{x}_{y}'
        key=tile_key(seed,0,x,y,profile)
        with disk_cache.acquire(key), physical_locks[hash(key)%len(physical_locks)]:
            try:
                report=json.loads(base.with_suffix('.json').read_text())
                if report.get('stage')!='decoder' or report.get('world_identity')!=identity:
                    return None
                return np.load(base.with_suffix('.npy'),allow_pickle=False)
            except (OSError,ValueError,EOFError):
                return None
    try:
        return read_mip(plan,child)
    except (OSError,ValueError,EOFError):
        return None


def probe_coarse_preparation(seed, profile, ready):
    """``ready(preparation)`` with any idle device's resident world, else None.

    All devices share one persisted coarse namespace per world. A busy GPU must
    never hold a ready preview hostage, and probing cannot load a world or evict
    an LRU. False means every device was idle and none holds this world.
    """
    with runtimes_lock:
        candidates = [runtimes[COARSE_DEVICE], *(r for i, r in sorted(runtimes.items()) if i != COARSE_DEVICE)]
    unknown = False
    for runtime in candidates:
        if not runtime.lock.acquire(blocking=False):
            unknown = True
            continue
        try:
            world=runtime.worlds.get((profile,seed))
            if world is None and runtime.background_world_key==(profile,seed):
                world=runtime.background_world
            preparation=getattr(world,'_terrain_coarse_preparation',None)
            if preparation is not None:
                return ready(preparation)
        finally:
            runtime.lock.release()
    return None if unknown else False


def learned_tile_state(seed, profile, lod, tx, ty):
    if lod<7:
        return False
    step=2**lod
    xs=tx*TILE*step+(np.arange(-HALO,TILE+HALO)+.5)*step
    ys=ty*TILE*step+(np.arange(-HALO,TILE+HALO)+.5)*step
    xs, ys = clip_world_axes(xs, ys, profile)
    return probe_coarse_preparation(seed, profile, lambda preparation: _learned_ready(preparation,xs,ys,lod,tx,ty))


def learned_tile_ready(seed, profile, lod, tx, ty):
    return learned_tile_state(seed, profile, lod, tx, ty) is True


def valid_physical_report(report,seed,profile,lod,tx,ty,source_lod=None,coarse_interpolation=None):
    if report.get('world_identity') != world_identity(world_manifest(seed,profile)):
        return False
    if lod == 4 and report.get('coarse_interpolation') != selected_coarse_interpolation(lod,coarse_interpolation):
        return False
    if report.get('source_lod') != (source_lod if source_lod is not None else latent_preview_source(lod)):
        return False
    if (not report.get('neural_chart') and report.get('source_lod') is None and 1<=lod<=2 and report.get('stage')!='final-dem-mip' and
            existing_final_mip(seed,profile,lod,tx,ty) is not None):
        return False
    return report.get('stage')!='conditioning-preview' or bool(report.get('neural_chart')) or not learned_tile_ready(seed,profile,lod,tx,ty)


def response_report(report,seed,profile,lod,tx,ty,source):
    # Unknown readiness may serve a labelled provisional input preview. It
    # cannot claim learned terrain or become an immutable HTTP cache entry.
    provisional=(report.get('stage')=='conditioning-preview' and
                 learned_tile_state(seed,profile,lod,tx,ty) is None)
    return dict(report,cache_source=source,provisional=provisional)


def render_elevation(elevation, lod, halo=HALO, climate=None, mode='relief', lighting=None, contours=None, origin=(0, 0), seed=None, world_profile=None, render_settings=None):
    from terrain_styles import MODES as STYLE_MODES, render_style, apply_contours
    if mode in STYLE_MODES:
        rgb=render_style(elevation,NATIVE*2**lod,mode,lighting,origin)
    else:
        rgb = (get_relief_map(elevation, None, None, None, resolution=NATIVE*2**lod, vmin=0, vmax=4500)
               if lighting is None else render_relief(elevation, NATIVE*2**lod, lighting))
    if mode in ('orogen-biomes','orogen-koppen','render','soil','pedology'):
        resolution = NATIVE*2**lod
        xs = origin[0]+(np.arange(elevation.shape[1])+.5)*resolution
        ys = origin[1]+(np.arange(elevation.shape[0])+.5)*resolution
        if mode == 'orogen-koppen':
            rgb = physical_koppen_colors(seed,world_profile,xs,ys,elevation)
        elif mode in ('render','soil','pedology'):
            atlas,settings = biome_atlas(seed,world_profile)
            rgb = colorize_surface(atlas,xs,ys,elevation,resolution,settings,render_settings,
                                   polar=selected_neural_chart()=='polar',mode=mode)
            if mode in ('render','soil'):
                from terrain_lighting import relief_intensity
                rgb *= np.where(elevation<0,1,relief_intensity(elevation,resolution,lighting))[...,None]
        else:
            fields = physical_biome_fields(seed, world_profile, xs, ys, elevation)
            palette = colorize_biomes(elevation,fields,resolution)
            from terrain_lighting import relief_intensity
            light = relief_intensity(elevation,resolution,lighting)
            rgb = palette*np.where(elevation<0,1,light)[...,None]
    elif mode not in STYLE_MODES and mode!='relief' and climate is not None:
        palette=colorize(elevation,climate,mode)
        if mode=='biomes':
            # Preserve relief shading, using luminance rather than its height palette.
            light=np.clip(np.mean(rgb,axis=-1)*1.8,.55,1.)
            if lighting is not None:light=1-lighting['strength']+lighting['strength']*light
            palette*=np.where(elevation<0,1,light)[...,None]
        rgb=palette
    rgb=apply_contours(rgb,elevation,contours,mode)
    return (np.clip(rgb[halo:-halo, halo:-halo], 0, 1)*255).astype(np.uint8)


def biome_atlas(seed, wp):
    from terrain_orogen import get_heightmap
    descriptor = resolve_generation(wp)
    atlas = geometry_heightmap(get_heightmap(seed,descriptor.bootstrap_style,
        options=descriptor.bootstrap_options),descriptor.settings)
    return atlas,descriptor.settings


def physical_biome_fields(seed, wp, xs, ys, elevation=None):
    atlas,settings = biome_atlas(seed,wp)
    return sample_biome_fields(atlas,xs,ys,settings,elevation,
                              polar=selected_neural_chart()=='polar')


def physical_koppen_colors(seed, wp, xs, ys, elevation):
    atlas,settings = biome_atlas(seed,wp)
    return KOPPEN_COLORS[sample_classes(atlas,xs,ys,settings,elevation,
                        polar=selected_neural_chart()=='polar')]


def transport_climate(seed, wp, xs, ys, climate):
    # Keep the five physical conditioning channels and their caches unchanged.
    # Display fields travel with every physical tile so GPU layer switches reuse
    # the DEM and never require another neural inference.
    if resolve_generation(wp).bootstrap_generator != 'orogen':
        return climate
    atlas,settings = biome_atlas(seed,wp)
    # Preserve the initial atlas detail at broad LODs. At finer LODs the
    # continuous lattice can be compact: the GPU classifies at DEM resolution.
    from terrain_climate import expand_climate
    cell_x=(atlas.bounds[2]-atlas.bounds[0])/atlas.width
    cell_y=(atlas.bounds[3]-atlas.bounds[1])/atlas.height
    width=min(304,max(len(xs),int(np.ceil(abs(xs[-1]-xs[0])/cell_x))+1))
    height=min(304,max(len(ys),int(np.ceil(abs(ys[-1]-ys[0])/cell_y))+1))
    if (height,width)!=climate.shape[1:]:
        xs=np.linspace(xs[0],xs[-1],width);ys=np.linspace(ys[0],ys[-1],height)
        climate=expand_climate(climate,(height,width))
    return np.concatenate([climate,sample_biome_transport(atlas,xs,ys,settings,
                           polar=selected_neural_chart()=='polar'),
                           sample_soil_transport(atlas,xs,ys,settings,
                           polar=selected_neural_chart()=='polar')])


def sample_tile(world, seed, lod, tx, ty, halo=HALO):
    """Backward-compatible CPU-render helper, used by older verification scripts."""
    elevation, stage = sample_elevation(world, lod, tx, ty, halo)
    return render_elevation(elevation, lod, halo), stage


@lru_cache(maxsize=8)
def scheduling_land_mask(seed, world_profile):
    x0,y0,x1,y1=profile_bounds(world_profile)
    xs=x0+(np.arange(512)+.5)*(x1-x0)/512
    ys=y0+(np.arange(256)+.5)*(y1-y0)/256
    elevation=sample_conditioning_preview(seed,world_profile,xs,ys)['elev']
    return land_mask(elevation,profile_bounds(world_profile))


def metadata(seed, world_profile=None):
    world_profile=generation_profile(world_profile)
    descriptor=resolve_generation(world_profile)
    extent=profile_bounds(world_profile)
    bounds = [-107520, 76800, 15360, 138240]
    initial_image = None
    # A saved export belongs to its actual seed, never blindly to seed 42.
    try:
        original = json.loads((OUTPUT/'terrain.json').read_text())
        if int(original['seed']) == seed and (OUTPUT/'terrain.png').exists():
            resolution = float(original['resolution'])
            bounds = [original['x']*resolution, original['y']*resolution,
                      (original['x']+original['width'])*resolution,
                      (original['y']+original['height'])*resolution]
            initial_image = '/generated/terrain.png'
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if world_profile!='natural':
        bounds=list(extent)
        initial_image=None
    overview_bounds=list(extent)
    manifest=world_manifest(seed,world_profile)
    return {'version':VERSION, 'cache_profile':PROFILE, 'generation_profile':world_profile,
            'appearance_identity':APPEARANCE_IDENTITY,
            'world_profile':descriptor.base_profile,'generation_settings':descriptor.settings,
            'generation_schema':generator_schema(),
            'conditioning_noise':manifest['conditioning']['cond_snr'],
            'orogen_layers': dict(OROGEN_LEGENDS) if descriptor.bootstrap_generator == 'orogen' else {},
            'world_identity':world_identity(manifest),'world_manifest':manifest,
            'generation_stages':manifest.get('bootstrap',{}).get('stage_state',{}) if manifest.get('bootstrap') else {},
            'scheduling_land_mask':scheduling_land_mask(seed,world_profile),
            'world_topology':descriptor.settings['world_topology'], 'world_diameter_km':descriptor.settings['world_diameter_km'],
            'world_version':manifest['world_profile'], 'world_bounds':overview_bounds,'seed':str(seed), 'model':MODEL, 'tile_size':TILE,
            'native_resolution':NATIVE, 'halo':HALO,
            'min_lod':MIN_LOD, 'refinement_resolutions':[NATIVE*2**(-level) for level in range(1,-MIN_LOD+1)],
            'refinement_experimental':True, 'refinement_disk_cache':False,
            'initial_bounds':bounds, 'initial_image':initial_image,
            'overview_bounds':overview_bounds,
            'overview':f'/api/overview/{VERSION}/{seed}.png?profile={PROFILE}&world_profile={world_profile}&appearance={APPEARANCE_IDENTITY}', 'gpu':gpu_label(),
            'climate_format':'baseline-BIO4-BIO12-BIO15-beta-f32','climate_width':CLIMATE_SIZE,
            'preview_min_lod':7,'preview_label':'Preview of the five network inputs (without NN)',
            'bootstrap_raster_spacing_m': None if not descriptor.needs_bootstrap else
                [(extent[2]-extent[0])/manifest['bootstrap']['raster']['width'],
                 (extent[3]-extent[1])/manifest['bootstrap']['raster']['height']],
             'conditioning_sample_spacing_m':7680,
             'coarse_interpolation_default':'monotone',
             'coarse_interpolation_options':list(COARSE_INTERPOLATIONS),
             'height_format':'float32-le', 'navigation_protocol':1,
            'stages':{'coarse':7680,'latent':240,'decoder':30}}


@app.get('/')
def index():
    return send_from_directory(WEB_ROOT, 'index.html')


@app.get('/terrain_globe.js')
def globe_script():
    return send_from_directory(WEB_ROOT, 'terrain_globe.js')


@app.get('/terrain_styles.js')
def styles_script():
    source='window.TerrainStyles='+json.dumps(STYLES)+';\n'+(source_path('terrain_style_rendering.js', root=ROOT)).read_text(encoding='utf-8')
    return Response(source,mimetype='application/javascript')


@app.get('/terrain_lighting.js')
def lighting_script():
    return send_from_directory(WEB_ROOT, 'terrain_lighting.js')


@app.get('/terrain_renderer.js')
def renderer_script():
    return send_from_directory(WEB_ROOT, 'terrain_renderer.js')


@app.get('/terrain_render_controls.js')
def render_controls_script():
    return send_from_directory(WEB_ROOT, 'terrain_render_controls.js')


@app.get('/terrain_lod.js')
def lod_script():
    return send_from_directory(WEB_ROOT,'terrain_lod.js')


@app.get('/terrain_generation_controls.js')
def generation_controls_script():
    return send_from_directory(WEB_ROOT,'terrain_generation_controls.js')


@app.get('/terrain_toolbar.js')
def toolbar_script():
    return send_from_directory(WEB_ROOT,'terrain_toolbar.js')


@app.get('/terrain_inference_controls.js')
def inference_controls_script():
    return send_from_directory(WEB_ROOT,'terrain_inference_controls.js')


@app.get('/terrain_map_tools.js')
def map_tools_script():
    return send_from_directory(WEB_ROOT, 'terrain_map_tools.js')


@app.get('/api/generation/schema')
def generation_schema():
    return jsonify(generator_schema())


@app.get('/webgpu/<path:name>')
def webgpu_asset(name):
    if os.environ.get('TERRAIN_WEBGPU_PROBE')!='1' or any(part.startswith('.') or part=='node_modules' for part in name.replace('\\','/').split('/')) or not name.endswith(('.html','.js','.mjs','.json','.css','.wasm')):
        return jsonify(error='WebGPU experiment disabled or file not public'),404
    return send_from_directory(ROOT/'webgpu',name)


@app.get('/webgpu-models/<path:name>')
def webgpu_model(name):
    if os.environ.get('TERRAIN_WEBGPU_PROBE')!='1' or '/' in name or '\\' in name or not name.endswith(('.onnx','.data','.bin','.json')):
        return jsonify(error='WebGPU experiment disabled or file not public'),404
    return send_from_directory(terrain_runtime.RUNTIME/'webgpu-models',name,conditional=True)


@app.get('/api/status')
def status():
    inference = None
    try:
        from terrain_inference import inference_status
        if worlds:
            inference = inference_status(next(reversed(worlds.values())))
    except ImportError:
        pass
    return jsonify(version=VERSION, cache_profile=PROFILE, gpu=gpu_label(), cuda=torch.version.cuda,
                   device_selection=GPU_SELECTION, model_preload=dict(preload_state), gpus=gpu_status(),
                   active_seed=str(active_seed), cached_seeds=[str(s[1]) for s in worlds],
                   metrics=dict(metrics), scheduler=jobs.status(), disk_cache=disk_cache.status(),
                   physical_delivery=physical_delivery.status(), inference=inference,
                   inference_streams=dict(coarse_stream_setting),
                   latent_previews=dict(max_worlds=2,world_cache_limit_bytes=128*1024**2,
                       cached_seeds=[str(k[1]) for k in preview_worlds],
                       window_cache_bytes=sum(getattr(w.tile_store,'_bytes',0) for w in preview_worlds.values()),
                       isolated_final_store=True),
                   coarse_preparation=coarse_background.status(),
                   cuda_forward_calls=dict(gpu_calls))


RESTART_COMMAND = None  # Set when this file is the server entry point.
STARTED_AT = time.time()
gpu_settings_lock = threading.Lock()


def gpu_status():
    lanes = {lane['device']: lane for lane in jobs.status()['gpu_lanes']}
    calls = getattr(terrain_runtime, 'gpu_calls_by_device', {})
    pending = resolve_plan(gpu_settings)
    enabled = gpu_settings['devices']
    devices = []
    for device in visible_gpus():
        index = device['index']
        runtime = runtimes.get(index)
        roles = [role for role, member in (('coarse', index == gpu_plan['coarse_device']),
                                           ('detail', index in gpu_plan['detail_devices'])) if member]
        lane = lanes.get(index) or {}
        devices.append(dict(index=index, uuid=device['uuid'], name=device['name'],
            memory_bytes=device['memory_bytes'], compute_capability=device['compute_capability'],
            eligible=terrain_device.multi_gpu_eligible(device), primary=index == SELECTED_DEVICE,
            enabled=enabled == 'auto' or device['uuid'] in enabled, roles=roles,
            preload=dict(runtime.preload) if runtime else None,
            tiles=lane.get('completed', 0), failed=lane.get('failed', 0), cancelled=lane.get('cancelled', 0),
            compute_seconds=lane.get('compute_seconds', 0.0), current_job=lane.get('current'),
            affinity_steals=lane.get('steals', 0), forward_calls=dict(calls.get(index, {}))))
    plan = {k: v for k, v in gpu_plan.items() if k != 'settings'}
    return dict(settings=gpu_settings, environment=sorted(GPU_ENVIRONMENT), plan=plan,
                pending_plan={k: v for k, v in pending.items() if k != 'settings'},
                restart_required=pending['coarse_device'] != COARSE_DEVICE,
                coarse_device=COARSE_DEVICE, modes=list(terrain_device.GPU_MODES), devices=devices,
                restart_available=RESTART_COMMAND is not None, pid=os.getpid(), started_at=STARTED_AT)


def release_runtime(index):
    """Free a GPU removed from the plan once its running tile has finished."""
    runtime = runtimes.get(index)
    if runtime is None or index == COARSE_DEVICE:
        return
    with runtime.lock:
        if index in gpu_plan['devices']:
            return  # Re-enabled meanwhile.
        for world in [runtime.background_world, *runtime.worlds.values(),
                      *runtime.preview_worlds.values(), *runtime.polar_worlds.values()]:
            if world is not None:
                close_world(world)
        runtime.worlds.clear();runtime.preview_worlds.clear();runtime.polar_worlds.clear()
        runtime.background_world = runtime.background_world_key = None
        runtime.shared_pipeline = None
        release = getattr(terrain_runtime, 'release_pipeline', None)
        if release is not None:
            release(index)
        runtime.preload.clear();runtime.preload.update(state='released')


def warm_device(index):
    runtime = runtime_for(index)
    state = runtime.preload
    try:
        with using_runtime(runtime), runtime.lock, torch.inference_mode(), span('startup.load_models', device=index):
            if index not in gpu_plan['devices']:
                return
            state.update(state='loading', error=None)
            if runtime.shared_pipeline is None:
                runtime.shared_pipeline = load_pipeline(42, device=index)
            if os.environ.get('TERRAIN_PREWARM', '1') == '1':
                state.update(state='warming')
                try:
                    state['cuda_graphs'] = warm_graphs(runtime.shared_pipeline, coarse=index == COARSE_DEVICE, state=state)
                finally:
                    runtime.shared_pipeline.empty_cache()
            state.update(state='ready')
    except Exception as exc:
        state.update(state='failed',error=str(exc)[:400])
        # The request path remains able to retry a transient load failure.
        print(f'Model preloading failed on GPU {index}: {exc}', flush=True)


def preload_devices(indices):
    for index in indices:
        threading.Thread(target=warm_device, args=(index,), name=f'terrain-model-preload-gpu{index}', daemon=True).start()


def apply_gpu_plan(plan):
    """Live change of detail devices; the coarse device is fixed per process."""
    global gpu_plan
    if plan['coarse_device'] != COARSE_DEVICE:
        raise ValueError('Changing the coarse GPU requires a server restart')
    previous, gpu_plan = gpu_plan, plan
    jobs.configure_gpus(plan['devices'], route_job, bind_gpu_worker)
    terrain_device.set_multi_gpu_workers(len(plan['devices']) > 1)
    added = [index for index in plan['devices'] if index not in previous['devices']]
    if added and os.environ.get('TERRAIN_PRELOAD_MODELS', '1') == '1':
        preload_devices(added)
    for index in previous['devices']:
        if index not in plan['devices']:
            threading.Thread(target=release_runtime, args=(index,), name=f'terrain-release-gpu{index}', daemon=True).start()


@app.route('/api/gpu', methods=['GET', 'POST'])
def gpu_settings_api():
    global gpu_settings
    if request.method == 'POST':
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error='GPU settings must be a JSON object'), 400
        locked = sorted(set(data) & set(GPU_ENVIRONMENT))
        if locked:
            return jsonify(error='Set by the server environment: ' + ', '.join(locked), locked=locked), 409
        if not visible_gpus():
            return jsonify(error='No CUDA inventory available'), 409
        try:
            with gpu_settings_lock:
                settings = terrain_device.validate_gpu_settings(dict(gpu_settings, **data), visible_gpus())
                plan = resolve_plan(settings)
                terrain_device.save_gpu_settings(GPU_SETTINGS_PATH, settings)
                gpu_settings = settings
                if plan['coarse_device'] == COARSE_DEVICE:
                    apply_gpu_plan(plan)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
    return jsonify(gpu_status())


def _shutdown_for_restart():
    time.sleep(.5)  # Let the HTTP response reach the browser.
    try:
        coarse_background.close()
        with all_gpu_locks() as locked:
            # Drain queued coarse windows to disk before the process exits.
            for runtime in locked:
                for world in runtime.live_worlds():
                    preparation = getattr(world, '_terrain_coarse_preparation', None)
                    if preparation is not None:
                        preparation.flush(timeout=30)
        worker = app.extensions.get('terrain_reference_worker')
        if worker is not None:
            worker.close()
        physical_delivery.flush()
    finally:
        os._exit(0)


@app.post('/api/server/restart')
def server_restart():
    """Replace this server process, e.g. to apply a new coarse GPU."""
    if RESTART_COMMAND is None:
        return jsonify(error='This server was not started as terrain_server.py; restart it manually'), 501
    env = dict(os.environ, TERRAIN_RESTART_AFTER_PID=str(os.getpid()))
    options = ({'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == 'nt' else {'start_new_session': True})
    with open(REPO_ROOT / 'server.log', 'a', encoding='utf-8') as out, \
            open(REPO_ROOT / 'server-error.log', 'a', encoding='utf-8') as err:
        subprocess.Popen(RESTART_COMMAND, cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
                         stdout=out, stderr=err, close_fds=True, **options)
    threading.Thread(target=_shutdown_for_restart, name='terrain-restart', daemon=True).start()
    return jsonify(restarting=True, pid=os.getpid()), 202


@app.get('/api/profile')
def profile_timeline():
    return jsonify(profiling_snapshot())


@app.get('/generated/<path:name>')
def saved_generation(name):
    return send_from_directory(OUTPUT, name)


@app.get('/api/world')
def world_info():
    token=None
    try:
        seed = int(request.args.get('seed', 42))
        if not 0 <= seed < 2**64:
            raise ValueError('Invalid seed')
        profile=generation_profile()
        encoded=request.args.get('generation')
        if encoded is not None:
            if len(encoded)>65536:
                raise ValueError('Generation settings are too large')
            try:
                settings=json.loads(encoded)
            except (ValueError,TypeError) as exc:
                raise ValueError('Invalid generation settings JSON') from exc
            if not isinstance(settings,dict):
                raise ValueError('Generation settings must be a JSON object')
            profile=register_generation(resolve_generation(profile).base_profile,settings)
        if request.args.get('session') and request.args.get('generation_epoch'):
            token=generation_coordinator.begin(request.args['session'],int(request.args['generation_epoch']))
            with generation_scope(token):
                with generation_coordinator.lock:
                    token.check()
                    worker=app.extensions.get('terrain_reference_worker')
                    if worker is not None:
                        worker.suspend(True)
                with all_gpu_locks():
                    token.check()
                    synchronize_gpus()
                    result=metadata(seed,profile)
            return jsonify(result)
        if jobs.paused:
            return jsonify(error='A world is being generated',cancelled=True),409
        return jsonify(metadata(seed,profile))
    except GenerationCancelled as exc:
        return jsonify(error=str(exc),cancelled=True),409
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        return jsonify(error=str(exc), stage='bootstrap-initialization'), 503
    finally:
        if token is not None:
            finish_generation(token)


@app.post('/api/generation/run')
def generation_run():
    token=None
    try:
        data=request.get_json()
        if not isinstance(data,dict):raise ValueError('Generation request must be an object')
        seed=int(data.get('seed',42))
        if not 0<=seed<2**64:raise ValueError('Invalid seed')
        stage=data.get('stage','all')
        if stage not in ('all','relief','erosion','climate','settings','restore'):
            raise ValueError('Unknown generation stage')
        base=resolve_generation(data.get('profile','orogen')).base_profile
        source=data.get('source_profile')
        if stage not in ('all','relief','restore') and source:base=resolve_generation(source).base_profile
        from terrain_orogen_stages import run_generation
        token=generation_coordinator.begin(data.get('session'),int(data.get('generation_epoch',0)))
        with generation_scope(token):
            with generation_coordinator.lock:
                token.check()
                worker=app.extensions.get('terrain_reference_worker')
                if worker is not None:
                    worker.suspend(True)
            with all_gpu_locks():
                token.check()
                # Drain the NN's asynchronous transfers/streams before Orogen.
                synchronize_gpus()
                if stage=='restore':
                    settings,execution=data.get('settings'),{}
                else:
                    settings,execution=run_generation(seed,base,data.get('settings'),stage,source)
                token.check()
                profile=register_generation(base,settings)
                result=metadata(seed,profile)
                result['generation_execution']=execution
                token.check()
        return jsonify(result)
    except GenerationCancelled as exc:
        return jsonify(error=str(exc),cancelled=True),409
    except (ValueError,TypeError) as exc:
        return jsonify(error=str(exc)),400
    except (RuntimeError,OSError,subprocess.SubprocessError) as exc:
        return jsonify(error=str(exc),stage='generation-stage'),503
    finally:
        if token is not None:
            finish_generation(token)


def _session(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{8,80}', value):
        raise ValueError('Invalid session')
    return value


def _tile_coordinates(seed, lod, tx, ty, profile=None):
    seed, lod, tx, ty = int(seed), int(lod), int(tx), int(ty)
    if not 0 <= seed < 2**64 or not MIN_LOD <= lod <= 11 or abs(tx) > 10**7 or abs(ty) > 10**7:
        raise ValueError('Invalid coordinates')
    if has_request_context() and request.args.get('world_identity'):
        expected=world_identity(world_manifest(seed,generation_profile()))
        if request.args['world_identity'] != expected:
            raise ValueError('Outdated world identity: reload the page')
    span=TILE*NATIVE*2**lod
    x0,y0,x1,y1=profile_bounds(profile or generation_profile())
    if (tx*span>=x1 or (tx+1)*span<=x0 or ty*span>=y1 or (ty+1)*span<=y0):
        raise ValueError('Tile outside the world')
    return seed, lod, tx, ty


def latent_preview_source(lod, value=None):
    value = request.args.get('source_lod') if value is None and has_request_context() else value
    if value is None:
        return None
    if str(value) != '3' or lod not in (1, 2):
        raise ValueError('Latent previews require source_lod=3 and geometry LOD 1 or 2')
    return 3


COARSE_INTERPOLATIONS = ('monotone', 'bilinear')
# Disk-cache startup indexing accepts numeric levels. This reserved internal
# level keeps the alternate LOD4 raster separate without changing world identity.
COARSE_BILINEAR_CACHE_LEVEL = 1004


def selected_coarse_interpolation(lod, value=None):
    if value is None:
        value = request.args.get('coarse_interpolation', 'monotone') if has_request_context() else 'monotone'
    if value not in COARSE_INTERPOLATIONS:
        raise ValueError('coarse_interpolation must be monotone or bilinear')
    return value if lod == 4 else None


def _coarse_cache_level(lod, coarse_interpolation=None):
    selected = selected_coarse_interpolation(lod, coarse_interpolation)
    level=COARSE_BILINEAR_CACHE_LEVEL if lod == 4 and selected == 'bilinear' else lod
    return level+2000 if selected_neural_chart() else level


def selected_neural_chart():
    if not has_request_context():
        return None
    value=request.args.get('neural_chart')
    if value is None and request.is_json:
        value=(request.get_json(silent=True) or {}).get('neural_chart')
    if value not in (None,'polar'):
        raise ValueError('Unknown neural chart')
    return value


def request_tile_key(seed,lod,tx,ty,world_profile=None,source_lod=None,coarse_interpolation=None):
    source = latent_preview_source(lod,source_lod)
    key = tile_key(seed,_coarse_cache_level(lod,coarse_interpolation),tx,ty,world_profile)
    return key + '/source3' if source is not None else key


def tile_key(seed, lod, tx, ty, world_profile=None):
    return f'{VERSION}/{generation_profile(world_profile)}/{seed}/{lod}/{tx}/{ty}'


@app.post('/api/view')
def update_view():
    try:
        data = request.get_json()
        session = _session(data['session'])
        epoch = int(data['epoch'])
        seed = int(data['seed'])
        world_profile=generation_profile(data.get('world_profile','natural'))
        if epoch < 0 or not 0 <= seed < 2**64:
            raise ValueError('Invalid epoch or seed')
        coarse_interpolation=data.get('coarse_interpolation','monotone')
        if coarse_interpolation not in COARSE_INTERPOLATIONS:
            raise ValueError('coarse_interpolation must be monotone or bilinear')
        tiles = data['tiles']
        if not isinstance(tiles, list) or len(tiles) > 512:
            raise ValueError('View budget exceeded (512 tiles)')
        wants = {}
        for tile in tiles:
            _, lod, tx, ty = _tile_coordinates(seed, tile['lod'], tile['tx'], tile['ty'], world_profile)
            priority = float(tile.get('priority', 2000))
            if not math.isfinite(priority) or not 0 <= priority < 5000:
                raise ValueError('Invalid priority')
            if tile.get('native_coarse'):
                if lod != native_coarse.GEOMETRY_LOD:
                    raise ValueError('Native coarse geometry must be LOD 7')
                wants[native_coarse_key(seed, tx, ty, world_profile, bool(tile.get('learned')))] = priority
            else:
                source=latent_preview_source(lod,tile.get('source_lod'))
                wants[request_tile_key(seed,lod,tx,ty,world_profile,source,coarse_interpolation)] = priority
        if data.get('overview'):
            light=parse_lighting(json.dumps(data['overview_lighting'])) if data.get('overview_lighting') is not None else None
            contours=parse_contours(json.dumps(data['overview_contours'])) if data.get('overview_contours') is not None else None
            for mode in MODES:
                render_options=parse_render_settings(data.get('overview_render_settings')) if mode=='render' else None
                suffix=(lighting_suffix(light) if mode in ('relief','biomes','orogen-biomes','render')+STYLE_MODES else '')+lighting_suffix(contours)+(lighting_suffix(render_options) if mode=='render' else '')
                if mode in ('render','soil','pedology'):suffix+='.appearance-'+APPEARANCE_IDENTITY
                wants[f'{VERSION}/{world_profile}/{seed}/overview/{mode}{suffix}'] = 0
        focus = data.get('bounds')
        if focus is not None:
            if not isinstance(focus,list) or len(focus)!=4:
                raise ValueError('Invalid view bounds')
            focus = [float(value) for value in focus]
            if not all(math.isfinite(value) for value in focus) or focus[2]<=focus[0] or focus[3]<=focus[1]:
                raise ValueError('Invalid view bounds')
            x0,y0,x1,y1=profile_bounds(world_profile)
            focus = [max(focus[0],x0),max(focus[1],y0),min(focus[2],x1),min(focus[3],y1)]
            if focus[2]<=focus[0] or focus[3]<=focus[1]:
                raise ValueError('View outside the world')
        accepted = jobs.update_view(session, epoch, wants)
        if accepted and focus is not None and not selected_neural_chart():
            coarse_background.set_focus(seed,world_profile,focus)
        return jsonify(accepted=accepted, epoch=epoch, wanted=len(wants))
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    except QueueFull as exc:
        return jsonify(error=str(exc)), 429


@app.post('/api/view/release')
def release_view():
    try:
        session = _session(request.get_json()['session'])
        jobs.release_view(session)
        return jsonify(released=True)
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify(error=str(exc)), 400


def _atomic_bytes(path, data):
    temporary = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _physical_paths(seed, lod, tx, ty, world_profile=None, coarse_interpolation=None):
    source=latent_preview_source(lod)
    level=_coarse_cache_level(lod,coarse_interpolation)
    directory = CACHE/generation_profile(world_profile)/'physical-v1'/str(seed)/(str(level)+'-source3' if source else str(level))
    if lod>=0:
        directory.mkdir(parents=True, exist_ok=True)
    base = directory/f'{tx}_{ty}'
    return base.with_suffix('.npy'), base.with_suffix('.json')


def _record_tile_disk(seed, lod, tx, ty, world_profile=None, source_lod=None, coarse_interpolation=None):
    wp=generation_profile(world_profile)
    source_lod=source_lod if source_lod is not None else latent_preview_source(lod)
    cache_level=_coarse_cache_level(lod,coarse_interpolation)
    level=str(cache_level)+'-source3' if source_lod else str(cache_level)
    base = CACHE/wp/'physical-v1'/str(seed)/level/f'{tx}_{ty}'
    image = CACHE/wp/str(seed)/level/f'{tx}_{ty}'
    paths=[base.with_suffix('.npy'),base.with_suffix('.json'),base.with_suffix('.climate.npy')]
    for mode in MODES:
        paths.extend([image.with_suffix(f'.{mode}.png'),image.with_suffix(f'.{mode}.json')])
    disk_cache.record(request_tile_key(seed,lod,tx,ty,wp,source_lod,coarse_interpolation),paths)


def read_physical_disk(key,path,report_path,climate_path):
    # Pins prevent eviction; this separate striped lock prevents readers from
    # observing height, climate and receipt from different replacements.
    with physical_locks[hash(key)%len(physical_locks)]:
        try:
            report=json.loads(report_path.read_text())
            return report,(np.load(path,allow_pickle=False),np.load(climate_path,allow_pickle=False))
        except (OSError,ValueError,EOFError):
            return None


def physical_tile(seed, lod, tx, ty, *, return_arrays=False):
    if lod<0 and not return_arrays:
        raise ValueError('Refined terrain is memory-only; request arrays')
    wp=generation_profile()
    polar=selected_neural_chart()=='polar'
    if polar and resolve_generation(wp).settings['world_topology']!='sphere':
        raise ValueError('Polar neural chart requires a spherical world')
    requested_profile = request.args.get('profile', PROFILE)
    if requested_profile != PROFILE:
        raise ValueError('Outdated cache profile: reload the page')
    coarse_interpolation=selected_coarse_interpolation(lod)
    path, report_path = _physical_paths(seed, lod, tx, ty,wp,coarse_interpolation)
    climate_path=path.with_suffix('.climate.npy')
    source_lod=latent_preview_source(lod)
    delivery_key=request_tile_key(seed,lod,tx,ty,wp,source_lod,coarse_interpolation)
    entry=physical_delivery.get(delivery_key)
    if entry and valid_physical_report(entry[1],seed,wp,lod,tx,ty,source_lod,coarse_interpolation):
        metrics['cache_hits'] += 1
        report=response_report(entry[1],seed,wp,lod,tx,ty,'ram')
        if return_arrays:
            return entry[0],report,True,entry[2]
        physical_delivery.flush()
        if path.exists() and report_path.exists():
            saved=read_physical_disk(delivery_key,path,report_path,climate_path)
            if (saved and saved[0].get('stage')==report.get('stage') and
                    valid_physical_report(saved[0],seed,wp,lod,tx,ty,source_lod,coarse_interpolation)):
                return path,response_report(saved[0],seed,wp,lod,tx,ty,'disk'),True
    disk=read_physical_disk(delivery_key,path,report_path,climate_path) if lod>=0 else None
    if disk:
        report,arrays=disk
        if valid_physical_report(report,seed,wp,lod,tx,ty,source_lod,coarse_interpolation):
            metrics['cache_hits'] += 1
            _record_tile_disk(seed, lod, tx, ty,wp,source_lod,coarse_interpolation)
            report=response_report(report,seed,wp,lod,tx,ty,'disk')
            return (path, report, True, arrays) if return_arrays else (path, report, True)
    # This may create a new CPU heightmap. Keep it outside the GPU lane/lock.
    world_manifest(seed, wp)
    # Only conditioning previews are admitted to the CPU lane. An already
    # learned distant tile still follows the single CUDA lane and lock.
    preview_only = lod >= 7 and (polar or learned_tile_state(seed, wp, lod, tx, ty) is False)
    queued_at = time.perf_counter()
    def compute():
        waiting = time.perf_counter()
        runtime = current_runtime()
        lock = nullcontext() if preview_only else measured_lock(runtime.lock, 'tile.gpu_lock_wait')
        with lock, torch.inference_mode():
            started = time.perf_counter()
            lock_wait = started-waiting
            jobs.check_current_interest()
            # Another already-completed request may have committed since admission.
            disk=read_physical_disk(delivery_key,path,report_path,climate_path) if lod>=0 else None
            if disk:
                report,_=disk
                if valid_physical_report(report,seed,wp,lod,tx,ty,source_lod,coarse_interpolation):
                    return None,None,report
            before = dict(gpu_calls)
            view_batch_tiles=1
            metrics['stage'] = f'LOD {lod} · {tx}, {ty}'
            world = (None if lod>=7 else get_polar_world(seed,wp,preview=bool(source_lod)) if polar else
                     get_preview_world(seed,wp) if source_lod else get_world(seed,wp))
            try:
                with span('tile.sample', gpu=not preview_only, lod=lod, tx=tx, ty=ty):
                    if preview_only:
                        elevation,climate,stage=sample_physical(None,seed,wp,lod,tx,ty,preview_only=True,**({'polar':True} if polar else {}))
                    else:
                        if source_lod:
                            view_batch_tiles=prepare_preview_view(world,seed,wp,lod,tx,ty,session)
                            elevation,climate,stage=sample_latent_preview(world,lod,tx,ty)
                        elif polar:
                            elevation,climate,stage=sample_physical(world,seed,wp,lod,tx,ty,polar=True,coarse_interpolation=coarse_interpolation)
                        elif lod==4:
                            elevation,climate,stage=sample_physical(
                                world,seed,wp,lod,tx,ty,coarse_interpolation=coarse_interpolation)
                        else:
                            elevation,climate,stage=sample_physical(world,seed,wp,lod,tx,ty)
            finally:
                metrics['stage'] = 'Ready'
            jobs.check_current_interest()
            report = {'stage':stage, 'resolution':NATIVE*2**(source_lod if source_lod else lod),
                      'source_resolution': ((profile_bounds(wp)[2]-profile_bounds(wp)[0])/world_manifest(seed,wp)['bootstrap']['raster']['width']
                          if stage=='conditioning-preview' and (resolve_generation(wp).settings['height_source'] in ('native','orogen') or resolve_generation(wp).settings['relief_pipeline'] in ('orogen','city-gpu')) else
                          {'conditioning-preview':7680,'coarse':7680,'coarse-area-mean':7680,'latent':240,'decoder':30,'final-dem-mip':30}.get(stage,NATIVE*2**lod)),
                      'width':(TILE//2**(3-lod) if source_lod else TILE)+2*HALO, 'halo':HALO,
                      'source_lod':source_lod, 'geometry_lod':lod, 'view_batch_tiles':view_batch_tiles,
                      'coarse_interpolation':coarse_interpolation if lod==4 else None,
                      'compute_seconds':round(time.perf_counter()-started, 4),
                      'queue_seconds':round(waiting-queued_at, 4),
                      'gpu_lock_wait_seconds':round(lock_wait,4),
                      'cuda_calls':{k:0 if preview_only else gpu_calls[k]-before[k] for k in gpu_calls}}
            if not preview_only:
                report.update(device_provenance(runtime.index))
            metrics['stage'] = 'Ready'
            report.update(tile_relief_stats(elevation, HALO))
            report.update(generation_profile=wp,neural_chart='polar' if polar else None,climate_width=CLIMATE_SIZE,climate_height=CLIMATE_SIZE)
            report.update(world_identity=world_identity(world_manifest(seed,wp)),
                          source_kind='experimental-neural-refinement' if stage=='decoder-refinement' else 'conditioning-input' if stage=='conditioning-preview' else 'learned-approximation' if stage in ('coarse','coarse-area-mean','latent') else 'native-dem-reduction',
                           exact_final_mip=stage=='final-dem-mip',height_filter='native-bilinear-coast+interior-parent-mean-preserving-decoder-cascade' if stage=='decoder-refinement' else 'block-mean-native' if stage=='final-dem-mip' else 'coarse-physical-area-mean+gaussian-sigma-0.65px' if stage=='coarse-area-mean' else f'coarse-{"monotone-cubic" if coarse_interpolation=="monotone" else "bilinear"}+gaussian-sigma-0.65px' if stage=='coarse' and lod==4 else 'conditioning-preview+gaussian-sigma-0.65px' if stage=='conditioning-preview' else 'source-approximation+gaussian-sigma-0.65px' if stage in ('coarse','latent') else 'native')
            if stage=='decoder-refinement':
                report.update(refinement_level=-lod, refinement_version=REFINEMENT_VERSION,
                              checkpoint_resolution=NATIVE, detail_amplitude_scale=0.5**(-lod))
            report['climate_source']='conditioning-input' if stage=='conditioning-preview' else 'learned-coarse'
            return elevation,climate,report
    def finalize(value):
        elevation, climate, report = value
        if preview_only and not polar:
            jobs.check_current_interest()
            if learned_tile_state(seed,wp,lod,tx,ty) is True:
                raise JobCancelled('Learned coarse terrain is ready; refresh this overview')
        if elevation is None:
            disk=read_physical_disk(delivery_key,path,report_path,climate_path)
            if disk is None:
                raise JobCancelled('Physical cache evicted; refresh this tile')
            report,arrays=disk
            return path,response_report(report,seed,wp,lod,tx,ty,'disk'),arrays
        if elevation is not None:
            report=dict(report,seconds=report['compute_seconds'],persistence='memory-only' if lod<0 else 'async-cache')
            def persist():
                with disk_cache.acquire(delivery_key), physical_locks[hash(delivery_key)%len(physical_locks)]:
                    started=time.perf_counter()
                    buffer=io.BytesIO()
                    np.save(buffer,elevation,allow_pickle=False)
                    _atomic_bytes(path,buffer.getvalue())
                    buffer=io.BytesIO()
                    np.save(buffer,climate,allow_pickle=False)
                    _atomic_bytes(climate_path,buffer.getvalue())
                    saved=dict(report,cache_write_seconds=round(time.perf_counter()-started,4),persistence='committed')
                    _atomic_bytes(report_path,json.dumps(saved).encode())
                    disk_cache.record(delivery_key,[path,climate_path,report_path])
            current=getattr(jobs.current,'job',None)
            physical_delivery.publish(delivery_key,path,report,(elevation,climate),persist if lod>=0 else None,
                                      current.sequence if current else None)
            metrics.update(generated_tiles=metrics['generated_tiles']+1, last_seconds=report['seconds'])
            print(f"Tile {seed}/{lod}/{tx}/{ty} {report['stage']} {report['seconds']}s", flush=True)
        return path,response_report(report,seed,wp,lod,tx,ty,'generated'),(elevation,climate)
    session = request.args.get('session')
    if session:
        session = _session(session)
    epoch = int(request.args.get('epoch', 0))
    # Learned coarse tiles stay on the coarse GPU. Latent/decoder tiles may use
    # any detail GPU; 2x2-tile blocks keep neighbours (and their shared
    # windows) on one device while it has work.
    detail = lod < 4
    job = jobs.submit(delivery_key, compute, finalize, session=session, epoch=epoch,
                      lane='cpu' if preview_only else 'gpu', role='detail' if detail else 'coarse',
                      affinity=(wp, seed, polar, lod, tx >> 1, ty >> 1) if detail else None)
    with trace(job.sequence), span('tile.wait', key=job.key, lod=lod):
        result_path, report, arrays = job.wait()
    if not return_arrays:
        physical_delivery.flush()
        if not result_path.exists() or not report_path.exists():
            with physical_locks[hash(delivery_key)%len(physical_locks)]:
                for destination,array in ((result_path,arrays[0]),(climate_path,arrays[1])):
                    buffer=io.BytesIO();np.save(buffer,array,allow_pickle=False)
                    _atomic_bytes(destination,buffer.getvalue())
                _atomic_bytes(report_path,json.dumps(report).encode())
                _record_tile_disk(seed,lod,tx,ty,wp,source_lod,coarse_interpolation)
    return (result_path, report, report.get('cache_source')=='disk', arrays) if return_arrays else (result_path, report, False)


def snr_diagnostic_tile(seed, lod, tx, ty, wp, mode):
    if request.args.get('profile', PROFILE) != PROFILE:
        raise ValueError('Outdated cache profile: reload the page')
    manifest = world_manifest(seed, wp)
    identity = world_identity(manifest)
    if request.args.get('world_identity') and request.args['world_identity'] != identity:
        raise ValueError('Outdated world identity: reload the page')
    resolution = NATIVE * 2**lod
    xs = (tx*TILE + np.arange(TILE) + .5) * resolution
    ys = (ty*TILE + np.arange(TILE) + .5) * resolution
    started = time.perf_counter()
    rgb = render_snr_layer(seed, wp, manifest['conditioning']['cond_snr'], mode, xs, ys)
    buffer = io.BytesIO()
    Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(buffer, format='PNG')
    response = Response(buffer.getvalue(), mimetype='image/png')
    response.cache_control.max_age = 31536000
    return _tile_headers(response, dict(stage='snr-diagnostic', resolution=resolution,
        source_resolution=48*7680, world_identity=identity,
        seconds=round(time.perf_counter()-started, 4), geometry_lod=lod), False)


def orogen_diagnostic_tile(seed, lod, tx, ty, wp, mode, *, binary=False):
    if request.args.get('profile', PROFILE) != PROFILE:
        raise ValueError('Outdated cache profile: reload the page')
    manifest = world_manifest(seed, wp)
    identity = world_identity(manifest)
    if request.args.get('world_identity') and request.args['world_identity'] != identity:
        raise ValueError('Outdated world identity: reload the page')
    from terrain_orogen import get_heightmap
    descriptor = resolve_generation(wp)
    atlas = geometry_heightmap(get_heightmap(seed, descriptor.bootstrap_style, options=descriptor.bootstrap_options), descriptor.settings)
    resolution = NATIVE*2**lod
    xs = (tx*TILE+np.arange(-HALO,TILE+HALO)+.5)*resolution
    ys = (ty*TILE+np.arange(-HALO,TILE+HALO)+.5)*resolution
    started = time.perf_counter()
    if binary:
        payload=atlas.sample_height_m(xs, ys).astype('<f4').tobytes()
        pedology_fields=None
        if mode=='pedology':
            # Source-only display: no neural DEM job. Keep its RGB in the same
            # slots as the normal physical transport for the shared GPU shader.
            from terrain_pedology import sample_pedology
            gx=np.linspace(xs[0],xs[-1],65);gy=np.linspace(ys[0],ys[-1],65)
            pedology_fields=np.zeros((50,65,65),np.float32)
            pedology_fields[46:50]=sample_pedology(atlas,gx,gy,descriptor.settings,
                                                  polar=request.args.get('neural_chart')=='polar')
            payload+=pedology_fields.astype('<f4').tobytes()
        response = Response(payload,
                            mimetype='application/octet-stream')
    else:
        rgb = render_orogen_layer(atlas, mode, xs, ys)
        # PNGs cover the tile interior, just like render_elevation; the halo
        # belongs only to physical samples and shading, not the displayed bounds.
        rgb = rgb[HALO:HALO+TILE, HALO:HALO+TILE]
        buffer = io.BytesIO()
        Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8)).save(buffer, format='PNG')
        response = Response(buffer.getvalue(), mimetype='image/png')
    report = dict(stage='orogen-diagnostic', resolution=resolution,
        source_resolution=(profile_bounds(wp)[2]-profile_bounds(wp)[0])/atlas.width,
        world_identity=identity, seconds=round(time.perf_counter()-started,4), geometry_lod=lod)
    report.update(tile_relief_stats(atlas.sample_height_m(xs, ys), HALO))
    if binary and pedology_fields is not None:
        response.headers['X-Terrain-Climate-Width']='65'
        response.headers['X-Terrain-Climate-Height']='65'
        response.headers['X-Terrain-Climate-Layers']='50'
    response.cache_control.max_age=31536000
    return _tile_headers(response, report, False)


def native_coarse_key(seed, tx, ty, profile, learned):
    return f'{VERSION}/{profile}/{seed}/native-coarse-v1/{tx}/{ty}/{"learned" if learned else "available"}'


def native_coarse_ready(seed, profile, xs, ys):
    # A macro preview never waits behind CUDA or generates neural windows.
    # Persisted contributors remain discoverable after a server restart.
    if not gpu_lock.acquire(blocking=False):
        return False
    try:
        world = _available_world(seed, profile)
        preparation = getattr(world, '_terrain_coarse_preparation', None)
        return preparation is not None and preparation.ready_for_samples(xs, ys, climate_halo=8)
    finally:
        gpu_lock.release()


@app.get('/coarse/natural-v1/<int:seed>/<tx>/<ty>.bin')
def native_coarse_tile(seed, tx, ty):
    """Transmit source cells once; browser zoom has no server-side LOD."""
    try:
        seed, _, tx, ty = _tile_coordinates(seed, native_coarse.GEOMETRY_LOD, tx, ty)
        wp = generation_profile()
        if request.args.get('profile', PROFILE) != PROFILE:
            raise ValueError('Outdated cache profile: reload the page')
        manifest = world_manifest(seed, wp)
        identity = world_identity(manifest)
        if request.args.get('world_identity', identity) != identity:
            raise ValueError('Outdated world identity: reload the page')
        learned = request.args.get('learned', '0') == '1'
        key = native_coarse_key(seed, tx, ty, wp, learned)
        cache_key = (wp, seed, tx, ty, identity)
        xs, ys = native_coarse.sample_axes(tx, ty, profile_bounds(wp))
        ready = native_coarse_ready(seed, wp, xs, ys)
        with native_coarse_cache_lock:
            cached = native_coarse_cache.get(cache_key)
            if cached is not None:
                native_coarse_cache.move_to_end(cache_key)
        hit = cached is not None and (cached[2]['stage'] == 'coarse' or
            (not learned and not ready and time.monotonic() - cached[3] < 30))
        if hit:
            root, climate, report, _ = cached
        else:
            preview_only = not learned and not ready
            def compute():
                with (nullcontext() if preview_only else measured_lock(current_runtime().lock, 'coarse.gpu_lock_wait')), torch.inference_mode():
                    started = time.perf_counter()
                    jobs.check_current_interest()
                    cx, cy = native_coarse.climate_axes(tx, ty, profile_bounds(wp))
                    if preview_only:
                        root = native_coarse.encode_height(conditioning_preview(seed, wp, xs*NATIVE, ys*NATIVE)['elev'])
                        climate = conditioning_preview(seed, wp, cx*NATIVE, cy*NATIVE)['climate']
                        stage = 'conditioning-preview'
                    else:
                        from terrain_window_scheduler import read_rect
                        world = get_world(seed, wp)
                        root = native_coarse.read_native(world, xs, ys, read_rect, jobs.check_current_interest)
                        climate = sample_coarse_climate(world, cx, cy, check=jobs.check_current_interest)
                        stage = 'coarse'
                    jobs.check_current_interest()
                    climate = np.ascontiguousarray(climate, dtype=np.float32)
                    if not np.isfinite(root).all() or not np.isfinite(climate).all():
                        raise RuntimeError('Non-finite native coarse')
                    seconds = round(time.perf_counter()-started, 4)
                    raster = (manifest.get('bootstrap') or {}).get('raster') or {}
                    source_resolution = ((profile_bounds(wp)[2]-profile_bounds(wp)[0])/raster['width']
                        if stage == 'conditioning-preview' and raster.get('width') else native_coarse.RESOLUTION)
                    report = dict(stage=stage, resolution=native_coarse.RESOLUTION,
                        source_resolution=source_resolution, geometry_lod=7,
                        source_lod=4 if stage == 'coarse' else 7,
                        width=native_coarse.WIDTH, halo=native_coarse.HALO,
                        seconds=seconds, world_identity=identity, height_filter='native-signed-sqrt-cells')
                    report.update(tile_relief_stats(np.sign(root)*root**2, native_coarse.HALO))
                    return root, climate, report
            def finalize(value):
                jobs.check_current_interest()
                with native_coarse_cache_lock:
                    previous = native_coarse_cache.get(cache_key)
                    if previous is not None and previous[2]['stage'] == 'coarse' and value[2]['stage'] != 'coarse':
                        return previous[:3]
                    native_coarse_cache[cache_key] = (*value, time.monotonic())
                    native_coarse_cache.move_to_end(cache_key)
                    # Bounded host transport cache, independent of neural caches.
                    while sum(v[0].nbytes+v[1].nbytes for v in native_coarse_cache.values()) > 64*1024**2:
                        native_coarse_cache.popitem(last=False)
                return value
            session = request.args.get('session')
            job = jobs.submit(key, compute, finalize, session=_session(session) if session else None,
                epoch=int(request.args.get('epoch', 0)), lane='cpu' if preview_only else 'gpu', role='coarse')
            root, climate, report = job.wait()
        cx, cy = native_coarse.climate_axes(tx, ty, profile_bounds(wp))
        climate = transport_climate(seed,wp,cx*NATIVE,cy*NATIVE,climate)
        payload = root.astype('<f4', copy=False).tobytes() + climate.astype('<f4', copy=False).tobytes()
        response = Response(payload, mimetype='application/octet-stream')
        response.headers['X-Terrain-Encoding'] = 'signed-sqrt'
        response.headers['X-Terrain-Climate-Width'] = str(climate.shape[2])
        response.headers['X-Terrain-Climate-Height'] = str(climate.shape[1])
        response.headers['X-Terrain-Climate-Layers'] = str(climate.shape[0])
        response.cache_control.max_age = 30 if report['stage'] == 'conditioning-preview' else 31536000
        return _tile_headers(response, report, hit)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    except JobCancelled as exc:
        return jsonify(error=str(exc)), 409
    except QueueFull as exc:
        return jsonify(error=str(exc)), 429
    except TimeoutError as exc:
        return jsonify(error=str(exc)), 504


@app.get('/height/natural-v1/<int:seed>/<int(signed=True):lod>/<tx>/<ty>.bin')
@pin_cache_io(lambda seed, lod, tx, ty: request_tile_key(seed, lod, int(tx), int(ty)))
def height_tile(seed, lod, tx, ty):
    try:
        seed, lod, tx, ty = _tile_coordinates(seed, lod, tx, ty)
        mode = display_mode()
        if mode in SNR_MODES:
            raise ValueError('SNR layers use the PNG tile endpoint')
        if mode in OROGEN_MODES and mode not in ('orogen-biomes','orogen-koppen','render','soil'):
            return orogen_diagnostic_tile(seed, lod, tx, ty, generation_profile(), mode, binary=True)
        path, report, cached, arrays = physical_tile(seed, lod, tx, ty, return_arrays=True)
        with span('tile.binary_response', cache_hit=cached, from_memory=arrays is not None):
            elevation = arrays[0] if arrays is not None else np.load(path, allow_pickle=False)
            payload=elevation.astype('<f4',copy=False).tobytes()
            if request.args.get('climate')=='1':
                climate=arrays[1] if arrays is not None else np.load(path.with_suffix('.climate.npy'),allow_pickle=False)
                resolution = report['resolution']
                width = elevation.shape[1]
                halo = report.get('halo',HALO)
                xs = tx*TILE*NATIVE*2**lod+(np.linspace(-halo+.5,width-halo-.5,CLIMATE_SIZE))*resolution
                ys = ty*TILE*NATIVE*2**lod+(np.linspace(-halo+.5,width-halo-.5,CLIMATE_SIZE))*resolution
                climate = transport_climate(seed,generation_profile(),xs,ys,climate)
                payload+=climate.astype('<f4',copy=False).tobytes()
        report = dict(report, **tile_relief_stats(elevation, HALO))
        response = Response(payload, mimetype='application/octet-stream')
        if request.args.get('climate')=='1':
            response.headers['X-Terrain-Climate-Width']=str(climate.shape[2])
            response.headers['X-Terrain-Climate-Height']=str(climate.shape[1])
            response.headers['X-Terrain-Climate-Layers']=str(climate.shape[0])
        response.cache_control.max_age = 31536000
        response.cache_control.public = True
        return _tile_headers(response, report, cached)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    except JobCancelled as exc:
        return jsonify(error=str(exc)), 409
    except QueueFull as exc:
        return jsonify(error=str(exc)), 429
    except TimeoutError as exc:
        return jsonify(error=str(exc)), 504


@app.get('/tiles/natural-v1/<int:seed>/<int(signed=True):lod>/<tx>/<ty>.png')
@pin_cache_io(lambda seed, lod, tx, ty: request_tile_key(seed, lod, int(tx), int(ty)))
def tile(seed, lod, tx, ty):
    try:
        seed, lod, tx, ty = _tile_coordinates(seed, lod, tx, ty)
        wp,mode=generation_profile(),display_mode()
        render_settings = parse_render_settings(request.args.get('render_settings')) if mode=='render' else None
        if mode in SNR_MODES:
            return snr_diagnostic_tile(seed, lod, tx, ty, wp, mode)
        if mode in OROGEN_MODES and mode not in ('orogen-biomes','orogen-koppen','render','soil'):
            return orogen_diagnostic_tile(seed, lod, tx, ty, wp, mode)
        if request.args.get('profile', PROFILE) != PROFILE:
            raise ValueError('Outdated cache profile: reload the page')
        coarse_interpolation=selected_coarse_interpolation(lod)
        lighting = parse_lighting(request.args.get('lighting'))
        from terrain_styles import parse_contours
        contours = parse_contours(request.args.get('contours'))
        if lighting is not None or contours is not None or mode in ('orogen-biomes','orogen-koppen','render','soil'):
            # Reuse physical caches, without accumulating PNGs for slider values.
            physical_path, report, cached, arrays = physical_tile(seed,lod,tx,ty,return_arrays=True)
            with image_slots:
                started=time.perf_counter()
                elevation=arrays[0] if arrays is not None else np.load(physical_path,allow_pickle=False)
                climate=arrays[1] if arrays is not None else np.load(physical_path.with_suffix('.climate.npy'),allow_pickle=False)
                rgb=render_elevation(elevation,latent_preview_source(lod) or lod,climate=climate,mode=mode,seed=seed,world_profile=wp,lighting=lighting,contours=contours,render_settings=render_settings,origin=(tx*256*NATIVE*2**lod-HALO*report['resolution'],ty*256*NATIVE*2**lod-HALO*report['resolution']))
                buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='PNG')
                report=dict(report,**tile_relief_stats(elevation,HALO),render_seconds=round(time.perf_counter()-started,4))
            return _tile_headers(Response(buffer.getvalue(),mimetype='image/png'),report,cached)
        if lod<0:
            _,report,cached,arrays=physical_tile(seed,lod,tx,ty,return_arrays=True)
            report = dict(report, **tile_relief_stats(arrays[0], HALO))
            with image_slots, span('tile.render_refinement', lod=lod):
                started=time.perf_counter()
                rgb=render_elevation(arrays[0],lod,climate=arrays[1],mode=mode,seed=seed,world_profile=wp,origin=((tx*256-HALO)*NATIVE*2**lod,(ty*256-HALO)*NATIVE*2**lod))
                buffer=io.BytesIO()
                Image.fromarray(rgb).save(buffer,format='PNG')
                report=dict(report,render_seconds=round(time.perf_counter()-started,4))
            return _tile_headers(Response(buffer.getvalue(),mimetype='image/png'),report,cached)
        source_lod=latent_preview_source(lod)
        cache_level=_coarse_cache_level(lod,coarse_interpolation)
        directory = CACHE/wp/str(seed)/(str(cache_level)+'-source3' if source_lod else str(cache_level))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/f'{tx}_{ty}.{mode}.png'
        report_path = path.with_suffix('.json')
        if path.exists() and report_path.exists():
            report=json.loads(report_path.read_text())
            if valid_physical_report(report,seed,wp,lod,tx,ty,source_lod,coarse_interpolation):
                if 'elevation_max' not in report:
                    physical_path, _, _, arrays = physical_tile(seed,lod,tx,ty,return_arrays=True)
                    elevation = arrays[0] if arrays is not None else np.load(physical_path,allow_pickle=False)
                    report.update(tile_relief_stats(elevation, HALO))
                    _atomic_bytes(report_path, json.dumps(report).encode())
                metrics['cache_hits'] += 1
                _record_tile_disk(seed, lod, tx, ty,wp,source_lod,coarse_interpolation)
                return _tile_headers(send_file(path, mimetype='image/png', max_age=31536000),response_report(report,seed,wp,lod,tx,ty,'disk'), True)
            path.unlink(missing_ok=True)
            report_path.unlink(missing_ok=True)
        physical_path, report, cached, arrays = physical_tile(seed, lod, tx, ty,return_arrays=True)
        with image_slots:
            if not path.exists() or not report_path.exists():
                started = time.perf_counter()
                elevation = arrays[0] if arrays is not None else np.load(physical_path, allow_pickle=False)
                report = dict(report, **tile_relief_stats(elevation, HALO))
                climate=arrays[1] if arrays is not None else np.load(physical_path.with_suffix('.climate.npy'),allow_pickle=False)
                rgb = render_elevation(elevation,source_lod or lod,climate=climate,mode=mode,seed=seed,world_profile=wp,origin=((tx*256-HALO)*NATIVE*2**lod,(ty*256-HALO)*NATIVE*2**lod))
                buffer = io.BytesIO()
                Image.fromarray(rgb).save(buffer, format='PNG')
                report = dict(report, render_seconds=round(time.perf_counter()-started, 4))
                _atomic_bytes(path, buffer.getvalue())
                _atomic_bytes(report_path, json.dumps(report).encode())
                _record_tile_disk(seed, lod, tx, ty,wp,source_lod,coarse_interpolation)
        return _tile_headers(send_file(path, mimetype='image/png', max_age=31536000), report, cached)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    except JobCancelled as exc:
        return jsonify(error=str(exc)), 409
    except QueueFull as exc:
        return jsonify(error=str(exc)), 429
    except TimeoutError as exc:
        return jsonify(error=str(exc)), 504


def _tile_headers(response, report, cache_hit):
    if report['stage']=='decoder-refinement':
        response.cache_control.no_store=True
        response.cache_control.max_age=None
        response.cache_control.public=False
    if report['stage']=='conditioning-preview':
        response.cache_control.max_age=0 if report.get('provisional') else 2
        response.cache_control.must_revalidate=True
    for key, value in {'Stage':report['stage'], 'Resolution':report['resolution'],
                       'Source-Resolution':report.get('source_resolution',report['resolution']),
                       'Coarse-Interpolation':report.get('coarse_interpolation') or '',
                       'Source-LOD':report.get('source_lod') if report.get('source_lod') is not None else report.get('geometry_lod',''),
                       'Geometry-LOD':report.get('geometry_lod',''),
                       'Elevation-Min':report.get('elevation_min',''),
                       'Elevation-Max':report.get('elevation_max',''),
                       'World-Identity':report.get('world_identity',''),
                       'Source-Kind':report.get('source_kind',report['stage']),
                       'Exact-Final-Mip':str(report.get('exact_final_mip',False)).lower(),
                       'Cache':'hit' if cache_hit else 'miss', 'Seconds':report['seconds'],
                       'Cache-Source':report.get('cache_source','disk' if cache_hit else 'generated'),
                       'Provisional':str(report.get('provisional',False)).lower(),
                       'Compute-Seconds':report.get('compute_seconds', report['seconds']),
                       'Queue-Seconds':report.get('queue_seconds', 0),
                       'GPU-Lock-Wait-Seconds':report.get('gpu_lock_wait_seconds',0),
                       'Device':report.get('device_name') or '',
                       'Render-Seconds':report.get('render_seconds', 0),
                       'Width':report.get('width', TILE+2*HALO), 'Halo':report.get('halo', HALO)}.items():
        response.headers[f'X-Terrain-{key}'] = str(value)
    return response


@lru_cache(maxsize=2)
def physical_overview(seed, world_profile, identity, bounds):
    # Display palettes share one immutable atlas; switching styles never samples
    # the generator or climate again. Identity protects regenerated worlds.
    x0,y0,x1,y1=bounds
    xs=x0+(np.arange(2048)+.5)*(x1-x0)/2048
    ys=y0+(np.arange(1024)+.5)*(y1-y0)/1024
    macro=conditioning_preview(seed,world_profile,xs,ys)
    elevation=np.asarray(macro['elev'],np.float32)
    climate=np.asarray(macro['climate'],np.float32)
    elevation.flags.writeable=False;climate.flags.writeable=False
    return elevation,climate,(x1-x0)/2048


# Display-dependent overviews (lighting, contours, Render settings) are not
# persisted; a small memory LRU still avoids recomputing seconds of CPU colour
# when the client returns to a recent combination or reopens the layer.
_OVERVIEW_BYTES=OrderedDict()
_OVERVIEW_LOCK=threading.Lock()


def _remember_overview(key, value):
    with _OVERVIEW_LOCK:
        _OVERVIEW_BYTES[key]=value;_OVERVIEW_BYTES.move_to_end(key)
        while len(_OVERVIEW_BYTES)>12:_OVERVIEW_BYTES.popitem(last=False)


@app.get('/api/overview/natural-v1/<int:seed>.png')
@pin_cache_io(lambda seed: f'{VERSION}/{generation_profile()}/{seed}/overview')
def overview(seed):
    if seed >= 2**64:
        return jsonify(error='Invalid seed'), 400
    if request.args.get('profile', PROFILE) != PROFILE:
        return jsonify(error='Outdated cache profile: reload the page'), 400
    try:
        wp,mode=generation_profile(),display_mode()
        lighting=parse_lighting(request.args.get("lighting"))
        contours=parse_contours(request.args.get("contours"))
        render_settings=parse_render_settings(request.args.get('render_settings')) if mode=='render' else None
        suffix=lighting_suffix(lighting)+lighting_suffix(contours)+(lighting_suffix(render_settings) if mode=='render' else '')
        if mode in ('render','soil','pedology'):suffix+='.appearance-'+APPEARANCE_IDENTITY
    except ValueError as exc:
        return jsonify(error=str(exc)),400
    directory = CACHE/wp/str(seed)
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/f'overview.{mode}{suffix}.png'
    metadata_snapshot=metadata(seed,wp)
    cache_key=f'{VERSION}/{wp}/{seed}/overview'
    receipt_path=directory/f'overview.{mode}{suffix}.json'
    identity=metadata_snapshot['world_identity']
    if request.args.get('world_identity') and request.args['world_identity'] != identity:
        return jsonify(error='Outdated world identity: reload the page'),400
    cache_paths=[directory/f'overview.{style}.{extension}' for style in MODES
                 for extension in ('png','json')]+[directory/'world.json',path,receipt_path]
    try:
        cached_identity=json.loads(receipt_path.read_text()).get('world_identity')
    except (OSError,ValueError):
        cached_identity=None
    if path.exists() and cached_identity==identity:
        disk_cache.record(cache_key,cache_paths)
        return send_file(path, mimetype='image/png', max_age=31536000)
    memory_key=(identity,wp,seed,mode,suffix)
    with _OVERVIEW_LOCK:
        remembered=_OVERVIEW_BYTES.get(memory_key)
        if remembered is not None:_OVERVIEW_BYTES.move_to_end(memory_key)
    if remembered is not None:
        response=Response(remembered,mimetype='image/png');response.cache_control.no_store=True
        return response
    def compute():
        jobs.check_current_interest()
        x0,y0,x1,y1=metadata_snapshot['overview_bounds']
        width,height=2048,1024
        xs=x0+(np.arange(width)+.5)*(x1-x0)/width
        ys=y0+(np.arange(height)+.5)*(y1-y0)/height
        if mode in SNR_MODES:
            manifest = world_manifest(seed, wp)
            return render_snr_layer(seed, wp, manifest['conditioning']['cond_snr'], mode, xs, ys), None, (x1-x0)/width
        if mode in OROGEN_MODES and mode not in ('orogen-biomes','orogen-koppen','render','soil'):
            from terrain_orogen import get_heightmap
            descriptor = resolve_generation(wp)
            atlas = geometry_heightmap(get_heightmap(seed, descriptor.bootstrap_style, options=descriptor.bootstrap_options), descriptor.settings)
            return render_orogen_layer(atlas, mode, xs, ys), None, (x1-x0)/width
        # A worldwide preview cannot monopolize the compute lane with thousands
        # of learned windows. The incremental coarse worker replaces it later.
        value=physical_overview(seed,wp,identity,tuple(metadata_snapshot['overview_bounds']))
        jobs.check_current_interest()
        return value
    def finalize(value):
        jobs.check_current_interest()
        elevation, climate,resolution = value
        if mode in OROGEN_MODES + SNR_MODES and mode not in ('orogen-biomes','orogen-koppen','render','soil'):
            rgb=elevation
        elif mode in ('orogen-biomes','orogen-koppen','render','soil'):
            x0,y0,x1,y1=metadata_snapshot['overview_bounds']
            xs=x0+(np.arange(2048)+.5)*(x1-x0)/2048
            ys=y0+(np.arange(1024)+.5)*(y1-y0)/1024
            if mode == 'orogen-koppen':
                rgb=physical_koppen_colors(seed,wp,xs,ys,elevation)
            elif mode in ('render','soil'):
                atlas,settings=biome_atlas(seed,wp)
                rgb=colorize_surface(atlas,xs,ys,elevation,((x1-x0)/2048,(y1-y0)/1024),settings,render_settings,mode=mode)
                if mode in ('render','soil'):
                    from terrain_lighting import relief_intensity
                    rgb*=np.where(elevation<0,1,relief_intensity(elevation,((x1-x0)/2048,(y1-y0)/1024),lighting))[...,None]
            else:
                fields=physical_biome_fields(seed,wp,xs,ys,elevation)
                rgb=colorize_biomes(elevation,fields,resolution)
                from terrain_lighting import relief_intensity
                light=relief_intensity(elevation,((x1-x0)/2048,(y1-y0)/1024),lighting)
                rgb*=np.where(elevation<0,1,light)[...,None]
        elif mode in STYLE_MODES:
            x0,y0,x1,y1=metadata_snapshot['overview_bounds']
            rgb=render_style(elevation,((x1-x0)/2048,(y1-y0)/1024),mode,lighting,(x0,y0))
        elif mode!='relief':
            rgb=colorize(elevation,climate,mode)
        elif lighting is None:
            rgb=get_relief_map(elevation,None,None,None,resolution=resolution,vmin=0,vmax=4500)
        else:
            x0,y0,x1,y1=metadata_snapshot['overview_bounds']
            rgb=render_relief(elevation,((x1-x0)/2048,(y1-y0)/1024),lighting)
        if mode not in OROGEN_MODES+SNR_MODES or mode in ('orogen-biomes','orogen-koppen','render','soil'):
            rgb=apply_contours(rgb,elevation,contours,mode)
        buffer = io.BytesIO()
        Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(buffer, format='PNG')
        if lighting is not None or contours is not None or render_settings is not None:
            _remember_overview(memory_key,buffer.getvalue())
            return buffer.getvalue()
        _atomic_bytes(path, buffer.getvalue())
        _atomic_bytes(receipt_path,json.dumps({'world_identity':identity,'mode':mode}).encode())
        _atomic_bytes(directory/'world.json', json.dumps(metadata_snapshot, indent=2).encode())
        disk_cache.record(cache_key,cache_paths)
        return path
    try:
        session = request.args.get('session')
        if session:
            session = _session(session)
        key = f'{VERSION}/{wp}/{seed}/overview/{mode}{suffix}'
        # The client explicitly includes this interest while the overview is useful.
        path = jobs.submit(key, compute, finalize, session=session,
                           epoch=int(request.args.get('epoch', 0)), priority=0, lane='cpu').wait()
        if isinstance(path,bytes):
            response=Response(path,mimetype='image/png');response.cache_control.no_store=True
            return response
        return send_file(path, mimetype='image/png', max_age=31536000)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    except JobCancelled as exc:
        return jsonify(error=str(exc)), 409
    except QueueFull as exc:
        return jsonify(error=str(exc)), 429
    except TimeoutError as exc:
        return jsonify(error=str(exc)), 504


def wait_for_replaced_server(port=8765):
    """A restarted server waits for its predecessor to release its port."""
    previous = os.environ.pop('TERRAIN_RESTART_AFTER_PID', None)
    if not previous:
        return
    import psutil
    import socket
    deadline = time.monotonic()+60
    while time.monotonic() < deadline and psutil.pid_exists(int(previous)):
        time.sleep(.2)
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(('127.0.0.1', port))
                return
            except OSError:
                time.sleep(.2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    arguments = parser.parse_args()
    RESTART_COMMAND = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:]]
    wait_for_replaced_server(arguments.port)
    if os.environ.get('TERRAIN_PRELOAD_MODELS', '1') == '1':
        preload_devices(gpu_plan['devices'])
    app.run(host='127.0.0.1', port=arguments.port, threaded=True, debug=False)
