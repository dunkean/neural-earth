"""Natural Terrain Diffusion tiles, camera-aware queue and physical height cache."""
import io
from copy import deepcopy
import hashlib
import json
import math
import os
from collections import OrderedDict
from functools import wraps
from contextlib import nullcontext
import re
import subprocess
import threading
import time
import uuid

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
from terrain_disk_cache import TerrainDiskCache
from terrain_climate import CLIMATE_SIZE, MODES, sample_coarse_climate, colorize
from terrain_conditioning import sample_conditioning_preview, WORLD_PROFILES
from terrain_generation import resolve_generation, register_generation, generator_schema
from terrain_bootstrap import RASTER_WIDTH, RASTER_HEIGHT
from terrain_manifest import build_manifest, world_identity, _digest
from terrain_coarse import CoarsePreparation
from terrain_background import CoarseBackground
from terrain_final_mips import plan_mip, read_mip
from terrain_refinement import MIN_LOD, VERSION as REFINEMENT_VERSION, sample_refined
from terrain_profiling import span, trace, measured_lock, snapshot as profiling_snapshot
from functools import lru_cache

app = Flask(__name__)
from terrain_backend_proxy import register_backend_proxy
register_backend_proxy(app)
gpu_lock = threading.RLock()
image_slots = threading.BoundedSemaphore(2)
TILE = 256
HALO = 24
NATIVE = 30
VERSION = 'natural-v1'
GPU_SELECTION = getattr(terrain_runtime, 'GPU_SELECTION', None)
SELECTED_DEVICE = GPU_SELECTION['selected']['index'] if GPU_SELECTION else 0
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
profile_data.update(model_revision=getattr(terrain_runtime, 'MODEL_REVISION', 'unversioned'),
    torch=str(torch.__version__), cuda=torch.version.cuda, cuda_device=SELECTED_DEVICE,
    compute_capability=GPU_SELECTION['selected']['compute_capability'] if GPU_SELECTION else None,
    driver=GPU_SELECTION.get('driver') if GPU_SELECTION else None)
if GPU_SELECTION:
    profile_data.update(gpu_name=GPU_SELECTION['selected']['name'],gpu_uuid=GPU_SELECTION['selected']['uuid'])
profile_data.update(physical_lod_version='bandlimit-climate-v3')
profile_data.update(refinement_version=REFINEMENT_VERSION, refinement_min_lod=MIN_LOD)
from terrain_world import WORLD_VERSION, WORLD_BOUNDS, profile_metadata
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
shared_pipeline = None
preload_state = dict(state='idle')
worlds = OrderedDict()
background_world_key = None
background_world = None
jobs = TerrainJobs()
physical_delivery = PhysicalDelivery()
physical_locks = [threading.RLock() for _ in range(64)]
disk_cache = TerrainDiskCache(CACHE, VERSION,
    budget_bytes=int(float(os.environ.get('TERRAIN_DISK_CACHE_GIB', '16'))*1024**3),
    protected_keys=jobs.protected_keys)
metrics = {'generated_tiles': 0, 'cache_hits': 0, 'stage': 'Ready', 'last_seconds': 0}


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
                               terrestrial_file_snapshot())
    manifest['checkpoint']['config']=deepcopy(_reference_manifest['checkpoint']['config'])
    manifest['generation']['runtime_versions']=deepcopy(
        _reference_manifest['generation']['runtime_versions'])
    manifest['complete']=True
    payload={k:v for k,v in manifest.items() if k!='world_hash'}
    manifest['world_hash']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
    return manifest


@lru_cache(maxsize=1)
def terrestrial_file_snapshot():
    """Load native prerequisites only when a terrestrial profile is requested."""
    from terrain_bootstrap import implementation_identity
    files = deepcopy(_reference_manifest['files'])
    native_identity=deepcopy(implementation_identity())
    source_digest=_digest(ROOT/'terrain_bootstrap.py')
    if source_digest['sha256'] != native_identity['python_source_sha256']:
        raise RuntimeError('Bootstrap source changed since startup: restart the server')
    files['implementation']['terrain_bootstrap.py'] = source_digest
    files['bootstrap_native'] = native_identity
    return files


def conditioning_preview(seed, world_profile, xs_metres, ys_metres):
    data = sample_conditioning_preview(seed, world_profile, xs_metres, ys_metres)
    # The transport/shader stores sea-level BIO1, not temperature at terrain
    # height. The factory returns physical BIO1, so undo its lapse exactly once.
    data = dict(data, climate=np.asarray(data['climate'],np.float32).copy())
    data['climate'][0] -= data['climate'][4]*np.maximum(data['elev'],0)
    return data


def _create_world(seed, world_profile, cache_limit):
    global shared_pipeline
    if shared_pipeline is None:
        shared_pipeline = load_pipeline(seed)
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
        configure_world(world, runtime_profile, world_profile=world_profile)
        world.bind()
        world._terrain_manifest = world_manifest(seed,world_profile)
        CoarsePreparation(OUTPUT/'coarse-worlds',world._terrain_manifest,bounds=WORLD_BOUNDS).install(world)
    except ValueError as exc:
        world.close()
        raise RuntimeError(f'Coarse world admission failed: {exc}') from exc
    return world


preview_worlds=OrderedDict()


def close_world(world):
    preparation=getattr(world,'_terrain_coarse_preparation',None)
    if preparation is not None and hasattr(preparation,'close'):
        preparation.close()
    world.empty_cache()
    world.close()


def get_preview_world(seed,profile):
    # Preview batch composition must never populate the final latent store.
    key=(profile,seed)
    if key not in preview_worlds:
        preview_worlds[key]=_create_world(seed,profile,128*1024*1024)
        while len(preview_worlds)>2:
            _,expired=preview_worlds.popitem(last=False)
            close_world(expired)
    preview_worlds.move_to_end(key)
    return preview_worlds[key]


def warm_base_forms(world):
    from terrain_cuda_graphs import prewarm_base_forms
    before=dict(gpu_calls)
    try:
        result=prewarm_base_forms(world.base_model,world._terrain_profile.latent_batch)
    except Exception as exc:
        # An optional prewarm failure does not discard loaded, usable weights.
        result=dict(enabled=True,state='failed',fully_warmed=False,error=f'{type(exc).__name__}: {exc}'[:400])
    return dict(result,warmup_cuda_forward_calls={key:gpu_calls[key]-before[key] for key in before})


def get_world(seed, world_profile='natural'):
    global active_seed
    active_seed = seed
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


def prepare_coarse_quantum(seed, profile):
    global background_world, background_world_key
    # Native CPU initialization happens before the GPU critical section.
    world_manifest(seed, profile)
    with gpu_lock, torch.inference_mode():
        key=(profile,seed)
        if background_world_key != key:
            if background_world is not None:
                close_world(background_world)
            background_world = None
            background_world_key = None
            # One bounded seed-local workspace shares neural weights but never
            # enters the foreground LRU or changes its active seed.
            background_world = _create_world(seed,profile,64*1024*1024)
            background_world_key = key
        world=background_world
        return world._terrain_coarse_preparation.step(world,budget_windows=1)


def _available_world(seed, profile):
    """Find a prepared world without loading models for untouched previews."""
    world=worlds.get((profile,seed))
    if world is not None:
        return world
    if background_world_key==(profile,seed):
        return background_world
    # After a background world is replaced or a server restart, persisted
    # windows must still be discoverable by a foreground high-LOD request.
    probe=CoarsePreparation(OUTPUT/'coarse-worlds',world_manifest(seed,profile),
                            bounds=WORLD_BOUNDS)
    if any(probe.windows_dir.glob('*.npy')):
        return get_world(seed,profile)
    return None


coarse_background=CoarseBackground(jobs,prepare_coarse_quantum,None)


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
        world_manifest(seed, profile)
        return jsonify(coarse_background.start(seed,profile,int(data.get('max_windows',8192)))),202
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
        xs=np.clip(xs,WORLD_BOUNDS[0]/NATIVE,WORLD_BOUNDS[2]/NATIVE)
        ys=np.clip(ys,WORLD_BOUNDS[1]/NATIVE,WORLD_BOUNDS[3]/NATIVE)
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
        scheduler=getattr(world,'_terrain_window_scheduler',None)
        if scheduler:
            with scheduler.scope(jobs.check_current_interest):
                data = world.get(i0, j0, i0+size*step, j0+size*step, with_climate=False)['elev']
        else:
            data = world.get(i0, j0, i0+size*step, j0+size*step, with_climate=False)['elev']
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
    cx=np.linspace(xs[0],xs[-1],CLIMATE_SIZE);cy=np.linspace(ys[0],ys[-1],CLIMATE_SIZE)
    climate=sample_coarse_climate(world,cx,cy,check=jobs.check_current_interest)
    if not np.isfinite(elevation).all() or not np.isfinite(climate).all():
        raise RuntimeError('Non-finite latent preview')
    return np.ascontiguousarray(elevation),np.ascontiguousarray(climate),'latent'


def sample_physical(world, seed, world_profile, lod, tx, ty, *, preview_only=False,
                    coarse_interpolation='monotone'):
    step=2**lod
    xs=(tx*TILE*step+(np.arange(-HALO,TILE+HALO)+.5)*step)
    ys=(ty*TILE*step+(np.arange(-HALO,TILE+HALO)+.5)*step)
    if lod>=7:
        # Only the finite rectangle is displayed. Extend its edge constantly
        # for shading halo samples instead of requiring a planet beyond it.
        xs=np.clip(xs,WORLD_BOUNDS[0]/NATIVE,WORLD_BOUNDS[2]/NATIVE)
        ys=np.clip(ys,WORLD_BOUNDS[1]/NATIVE,WORLD_BOUNDS[3]/NATIVE)
    if world is None and not preview_only:
        world=_available_world(seed,world_profile)
    preparation=getattr(world,'_terrain_coarse_preparation',None)
    learned_ready=not preview_only and lod>=7 and _learned_ready(preparation,xs,ys,lod,tx,ty)
    final_mip=existing_final_mip(seed,world_profile,lod,tx,ty)
    if final_mip is not None:
        elevation,stage=final_mip,'final-dem-mip'
    elif lod>=7 and not learned_ready:
        macro=conditioning_preview(seed,world_profile,xs*NATIVE,ys*NATIVE)
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
    cx=np.linspace(xs[0],xs[-1],CLIMATE_SIZE)
    cy=np.linspace(ys[0],ys[-1],CLIMATE_SIZE)
    if stage=='conditioning-preview':
        macro=conditioning_preview(seed,world_profile,cx*NATIVE,cy*NATIVE)
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


def learned_tile_state(seed, profile, lod, tx, ty):
    if lod<7:
        return False
    # This is also called on HTTP cache hits. A busy GPU must never hold a
    # ready preview hostage, and probing cannot load a world or evict the LRU.
    if not gpu_lock.acquire(blocking=False):
        return None
    try:
        world=worlds.get((profile,seed))
        if world is None and background_world_key==(profile,seed):
            world=background_world
        preparation=getattr(world,'_terrain_coarse_preparation',None)
        if preparation is None:
            return False
        step=2**lod
        xs=tx*TILE*step+(np.arange(-HALO,TILE+HALO)+.5)*step
        ys=ty*TILE*step+(np.arange(-HALO,TILE+HALO)+.5)*step
        xs=np.clip(xs,WORLD_BOUNDS[0]/NATIVE,WORLD_BOUNDS[2]/NATIVE)
        ys=np.clip(ys,WORLD_BOUNDS[1]/NATIVE,WORLD_BOUNDS[3]/NATIVE)
        return _learned_ready(preparation,xs,ys,lod,tx,ty)
    finally:
        gpu_lock.release()


def learned_tile_ready(seed, profile, lod, tx, ty):
    return learned_tile_state(seed, profile, lod, tx, ty) is True


def valid_physical_report(report,seed,profile,lod,tx,ty,source_lod=None,coarse_interpolation=None):
    if report.get('world_identity') != world_identity(world_manifest(seed,profile)):
        return False
    if lod == 4 and report.get('coarse_interpolation') != selected_coarse_interpolation(lod,coarse_interpolation):
        return False
    if report.get('source_lod') != (source_lod if source_lod is not None else latent_preview_source(lod)):
        return False
    if (report.get('source_lod') is None and 1<=lod<=2 and report.get('stage')!='final-dem-mip' and
            existing_final_mip(seed,profile,lod,tx,ty) is not None):
        return False
    return report.get('stage')!='conditioning-preview' or not learned_tile_ready(seed,profile,lod,tx,ty)


def response_report(report,seed,profile,lod,tx,ty,source):
    # Unknown readiness may serve a labelled provisional input preview. It
    # cannot claim learned terrain or become an immutable HTTP cache entry.
    provisional=(report.get('stage')=='conditioning-preview' and
                 learned_tile_state(seed,profile,lod,tx,ty) is None)
    return dict(report,cache_source=source,provisional=provisional)


def render_elevation(elevation, lod, halo=HALO, climate=None, mode='relief'):
    rgb = get_relief_map(elevation, None, None, None, resolution=NATIVE*2**lod, vmin=0, vmax=4500)
    if mode!='relief' and climate is not None:
        palette=colorize(elevation,climate,mode)
        if mode=='biomes':
            # Preserve relief shading, using luminance rather than its height palette.
            light=np.clip(np.mean(rgb,axis=-1)*1.8,.55,1.)
            palette*=np.where(elevation<0,1,light)[...,None]
        rgb=palette
    return (np.clip(rgb[halo:-halo, halo:-halo], 0, 1)*255).astype(np.uint8)


def sample_tile(world, seed, lod, tx, ty, halo=HALO):
    """Backward-compatible CPU-render helper, used by older verification scripts."""
    elevation, stage = sample_elevation(world, lod, tx, ty, halo)
    return render_elevation(elevation, lod, halo), stage


def metadata(seed, world_profile=None):
    world_profile=generation_profile(world_profile)
    descriptor=resolve_generation(world_profile)
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
        bounds=list(WORLD_BOUNDS)
        initial_image=None
    overview_bounds=list(WORLD_BOUNDS)
    manifest=world_manifest(seed,world_profile)
    return {'version':VERSION, 'cache_profile':PROFILE, 'generation_profile':world_profile,
            'world_profile':descriptor.base_profile,'generation_settings':descriptor.settings,
            'generation_schema':generator_schema(),
            'world_identity':world_identity(manifest),'world_manifest':manifest,
            'world_version':manifest['world_profile'], 'world_bounds':overview_bounds,'seed':str(seed), 'model':MODEL, 'tile_size':TILE,
            'native_resolution':NATIVE, 'halo':HALO,
            'min_lod':MIN_LOD, 'refinement_resolutions':[NATIVE*2**(-level) for level in range(1,-MIN_LOD+1)],
            'refinement_experimental':True, 'refinement_disk_cache':False,
            'initial_bounds':bounds, 'initial_image':initial_image,
            'overview_bounds':overview_bounds,
            'overview':f'/api/overview/{VERSION}/{seed}.png?profile={PROFILE}&world_profile={world_profile}', 'gpu':torch.cuda.get_device_name(SELECTED_DEVICE),
            'climate_format':'baseline-BIO4-BIO12-BIO15-beta-f32','climate_width':CLIMATE_SIZE,
            'preview_min_lod':7,'preview_label':'Preview of the five network inputs (without NN)',
            'bootstrap_raster_spacing_m': None if not descriptor.needs_bootstrap else
                [(WORLD_BOUNDS[2]-WORLD_BOUNDS[0])/RASTER_WIDTH,
                 (WORLD_BOUNDS[3]-WORLD_BOUNDS[1])/RASTER_HEIGHT],
             'conditioning_sample_spacing_m':7680,
             'coarse_interpolation_default':'monotone',
             'coarse_interpolation_options':list(COARSE_INTERPOLATIONS),
             'height_format':'float32-le', 'navigation_protocol':1,
            'stages':{'coarse':7680,'latent':240,'decoder':30}}


@app.get('/')
def index():
    return send_from_directory(ROOT, 'index.html')


@app.get('/terrain_renderer.js')
def renderer_script():
    return send_from_directory(ROOT, 'terrain_renderer.js')


@app.get('/terrain_lod.js')
def lod_script():
    return send_from_directory(ROOT,'terrain_lod.js')


@app.get('/terrain_generation_controls.js')
def generation_controls_script():
    return send_from_directory(ROOT,'terrain_generation_controls.js')


@app.get('/terrain_map_tools.js')
def map_tools_script():
    return send_from_directory(ROOT, 'terrain_map_tools.js')


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
    return jsonify(version=VERSION, cache_profile=PROFILE, gpu=torch.cuda.get_device_name(SELECTED_DEVICE), cuda=torch.version.cuda,
                   device_selection=GPU_SELECTION, model_preload=dict(preload_state),
                   active_seed=str(active_seed), cached_seeds=[str(s[1]) for s in worlds],
                   metrics=dict(metrics), scheduler=jobs.status(), disk_cache=disk_cache.status(),
                   physical_delivery=physical_delivery.status(), inference=inference,
                   latent_previews=dict(max_worlds=2,world_cache_limit_bytes=128*1024**2,
                       cached_seeds=[str(k[1]) for k in preview_worlds],
                       window_cache_bytes=sum(getattr(w.tile_store,'_bytes',0) for w in preview_worlds.values()),
                       isolated_final_store=True),
                   coarse_preparation=coarse_background.status(),
                   cuda_forward_calls=dict(gpu_calls))


@app.get('/api/profile')
def profile_timeline():
    return jsonify(profiling_snapshot())


@app.get('/generated/<path:name>')
def saved_generation(name):
    return send_from_directory(OUTPUT, name)


@app.get('/api/world')
def world_info():
    try:
        seed = int(request.args.get('seed', 42))
        if not 0 <= seed < 2**64:
            raise ValueError('Invalid seed')
        profile=generation_profile()
        encoded=request.args.get('generation')
        if encoded is not None:
            if len(encoded)>8192:
                raise ValueError('Generation settings are too large')
            try:
                settings=json.loads(encoded)
            except (ValueError,TypeError) as exc:
                raise ValueError('Invalid generation settings JSON') from exc
            if not isinstance(settings,dict):
                raise ValueError('Generation settings must be a JSON object')
            profile=register_generation(resolve_generation(profile).base_profile,settings)
        return jsonify(metadata(seed,profile))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        return jsonify(error=str(exc), stage='bootstrap-initialization'), 503


def _session(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{8,80}', value):
        raise ValueError('Invalid session')
    return value


def _tile_coordinates(seed, lod, tx, ty):
    seed, lod, tx, ty = int(seed), int(lod), int(tx), int(ty)
    if not 0 <= seed < 2**64 or not MIN_LOD <= lod <= 11 or abs(tx) > 10**7 or abs(ty) > 10**7:
        raise ValueError('Invalid coordinates')
    if has_request_context() and request.args.get('world_identity'):
        expected=world_identity(world_manifest(seed,generation_profile()))
        if request.args['world_identity'] != expected:
            raise ValueError('Outdated world identity: reload the page')
    span=TILE*NATIVE*2**lod
    if (tx*span>=WORLD_BOUNDS[2] or (tx+1)*span<=WORLD_BOUNDS[0] or
            ty*span>=WORLD_BOUNDS[3] or (ty+1)*span<=WORLD_BOUNDS[1]):
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
    return COARSE_BILINEAR_CACHE_LEVEL if lod == 4 and selected == 'bilinear' else lod


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
            _, lod, tx, ty = _tile_coordinates(seed, tile['lod'], tile['tx'], tile['ty'])
            priority = float(tile.get('priority', 2000))
            if not math.isfinite(priority) or not 0 <= priority < 5000:
                raise ValueError('Invalid priority')
            source=latent_preview_source(lod,tile.get('source_lod'))
            wants[request_tile_key(seed,lod,tx,ty,world_profile,source,coarse_interpolation)] = priority
        if data.get('overview'):
            for mode in MODES:
                wants[f'{VERSION}/{world_profile}/{seed}/overview/{mode}'] = 0
        accepted = jobs.update_view(session, epoch, wants)
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
    preview_only = lod >= 7 and learned_tile_state(seed, wp, lod, tx, ty) is False
    queued_at = time.perf_counter()
    def compute():
        waiting = time.perf_counter()
        lock = nullcontext() if preview_only else measured_lock(gpu_lock, 'tile.gpu_lock_wait')
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
            world = None if lod>=7 else get_preview_world(seed,wp) if source_lod else get_world(seed,wp)
            try:
                with span('tile.sample', gpu=not preview_only, lod=lod, tx=tx, ty=ty):
                    if preview_only:
                        elevation,climate,stage=sample_physical(None,seed,wp,lod,tx,ty,preview_only=True)
                    else:
                        if source_lod:
                            view_batch_tiles=prepare_preview_view(world,seed,wp,lod,tx,ty,session)
                            elevation,climate,stage=sample_latent_preview(world,lod,tx,ty)
                        elif lod==4:
                            elevation,climate,stage=sample_physical(
                                world,seed,wp,lod,tx,ty,coarse_interpolation=coarse_interpolation)
                        else:
                            elevation,climate,stage=sample_physical(world,seed,wp,lod,tx,ty)
            finally:
                metrics['stage'] = 'Ready'
            jobs.check_current_interest()
            report = {'stage':stage, 'resolution':NATIVE*2**(source_lod if source_lod else lod),
                      'source_resolution': ((WORLD_BOUNDS[2]-WORLD_BOUNDS[0])/RASTER_WIDTH
                          if stage=='conditioning-preview' and resolve_generation(wp).settings['height_source']=='native' else
                          {'conditioning-preview':7680,'coarse':7680,'coarse-area-mean':7680,'latent':240,'decoder':30,'final-dem-mip':30}.get(stage,NATIVE*2**lod)),
                      'width':(TILE//2**(3-lod) if source_lod else TILE)+2*HALO, 'halo':HALO,
                      'source_lod':source_lod, 'geometry_lod':lod, 'view_batch_tiles':view_batch_tiles,
                      'coarse_interpolation':coarse_interpolation if lod==4 else None,
                      'compute_seconds':round(time.perf_counter()-started, 4),
                      'queue_seconds':round(waiting-queued_at, 4),
                      'gpu_lock_wait_seconds':round(lock_wait,4),
                      'cuda_calls':{k:0 if preview_only else gpu_calls[k]-before[k] for k in gpu_calls}}
            metrics['stage'] = 'Ready'
            report.update(generation_profile=wp,climate_width=CLIMATE_SIZE,climate_height=CLIMATE_SIZE)
            report.update(world_identity=world_identity(world_manifest(seed,wp)),
                          source_kind='experimental-neural-refinement' if stage=='decoder-refinement' else 'conditioning-input' if stage=='conditioning-preview' else 'learned-approximation' if stage in ('coarse','coarse-area-mean','latent') else 'native-dem-reduction',
                           exact_final_mip=stage=='final-dem-mip',height_filter='parent-mean-preserving-decoder-cascade' if stage=='decoder-refinement' else 'block-mean-native' if stage=='final-dem-mip' else 'coarse-physical-area-mean+gaussian-sigma-0.65px' if stage=='coarse-area-mean' else f'coarse-{"monotone-cubic" if coarse_interpolation=="monotone" else "bilinear"}+gaussian-sigma-0.65px' if stage=='coarse' and lod==4 else 'conditioning-preview+gaussian-sigma-0.65px' if stage=='conditioning-preview' else 'source-approximation+gaussian-sigma-0.65px' if stage in ('coarse','latent') else 'native')
            if stage=='decoder-refinement':
                report.update(refinement_level=-lod, refinement_version=REFINEMENT_VERSION,
                              checkpoint_resolution=NATIVE, detail_amplitude_scale=0.5**(-lod))
            report['climate_source']='conditioning-input' if stage=='conditioning-preview' else 'learned-coarse'
            return elevation,climate,report
    def finalize(value):
        elevation, climate, report = value
        if preview_only:
            jobs.check_current_interest()
            if learned_tile_state(seed,wp,lod,tx,ty) is True:
                raise JobCancelled('Le coarse appris est prêt ; actualiser cet aperçu')
        if elevation is None:
            disk=read_physical_disk(delivery_key,path,report_path,climate_path)
            if disk is None:
                raise JobCancelled('Cache physique évincé ; actualiser cette tuile')
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
    job = jobs.submit(delivery_key, compute, finalize, session=session, epoch=epoch,
                      lane='cpu' if preview_only else 'gpu')
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


@app.get('/height/natural-v1/<int:seed>/<int(signed=True):lod>/<tx>/<ty>.bin')
@pin_cache_io(lambda seed, lod, tx, ty: request_tile_key(seed, lod, int(tx), int(ty)))
def height_tile(seed, lod, tx, ty):
    try:
        seed, lod, tx, ty = _tile_coordinates(seed, lod, tx, ty)
        path, report, cached, arrays = physical_tile(seed, lod, tx, ty, return_arrays=True)
        with span('tile.binary_response', cache_hit=cached, from_memory=arrays is not None):
            elevation = arrays[0] if arrays is not None else np.load(path, allow_pickle=False)
            payload=elevation.astype('<f4',copy=False).tobytes()
            if request.args.get('climate')=='1':
                climate=arrays[1] if arrays is not None else np.load(path.with_suffix('.climate.npy'),allow_pickle=False)
                payload+=climate.astype('<f4',copy=False).tobytes()
        response = Response(payload, mimetype='application/octet-stream')
        if request.args.get('climate')=='1':
            response.headers['X-Terrain-Climate-Width']=str(CLIMATE_SIZE)
            response.headers['X-Terrain-Climate-Height']=str(CLIMATE_SIZE)
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
        if request.args.get('profile', PROFILE) != PROFILE:
            raise ValueError('Outdated cache profile: reload the page')
        coarse_interpolation=selected_coarse_interpolation(lod)
        if lod<0:
            _,report,cached,arrays=physical_tile(seed,lod,tx,ty,return_arrays=True)
            with image_slots, span('tile.render_refinement', lod=lod):
                started=time.perf_counter()
                rgb=render_elevation(arrays[0],lod,climate=arrays[1],mode=mode)
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
                climate=arrays[1] if arrays is not None else np.load(physical_path.with_suffix('.climate.npy'),allow_pickle=False)
                rgb = render_elevation(elevation,source_lod or lod,climate=climate,mode=mode)
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
                       'World-Identity':report.get('world_identity',''),
                       'Source-Kind':report.get('source_kind',report['stage']),
                       'Exact-Final-Mip':str(report.get('exact_final_mip',False)).lower(),
                       'Cache':'hit' if cache_hit else 'miss', 'Seconds':report['seconds'],
                       'Cache-Source':report.get('cache_source','disk' if cache_hit else 'generated'),
                       'Provisional':str(report.get('provisional',False)).lower(),
                       'Compute-Seconds':report.get('compute_seconds', report['seconds']),
                       'Queue-Seconds':report.get('queue_seconds', 0),
                       'GPU-Lock-Wait-Seconds':report.get('gpu_lock_wait_seconds',0),
                       'Render-Seconds':report.get('render_seconds', 0),
                       'Width':report.get('width', TILE+2*HALO), 'Halo':report.get('halo', HALO)}.items():
        response.headers[f'X-Terrain-{key}'] = str(value)
    return response


@app.get('/api/overview/natural-v1/<int:seed>.png')
@pin_cache_io(lambda seed: f'{VERSION}/{generation_profile()}/{seed}/overview')
def overview(seed):
    if seed >= 2**64:
        return jsonify(error='Invalid seed'), 400
    if request.args.get('profile', PROFILE) != PROFILE:
        return jsonify(error='Outdated cache profile: reload the page'), 400
    try:
        wp,mode=generation_profile(),display_mode()
    except ValueError as exc:
        return jsonify(error=str(exc)),400
    directory = CACHE/wp/str(seed)
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/f'overview.{mode}.png'
    metadata_snapshot=metadata(seed,wp)
    cache_key=f'{VERSION}/{wp}/{seed}/overview'
    receipt_path=directory/f'overview.{mode}.json'
    identity=metadata_snapshot['world_identity']
    if request.args.get('world_identity') and request.args['world_identity'] != identity:
        return jsonify(error='Outdated world identity: reload the page'),400
    cache_paths=[directory/f'overview.{style}.{extension}' for style in MODES
                 for extension in ('png','json')]+[directory/'world.json']
    try:
        cached_identity=json.loads(receipt_path.read_text()).get('world_identity')
    except (OSError,ValueError):
        cached_identity=None
    if path.exists() and cached_identity==identity:
        disk_cache.record(cache_key,cache_paths)
        return send_file(path, mimetype='image/png', max_age=31536000)
    def compute():
        jobs.check_current_interest()
        x0,y0,x1,y1=metadata_snapshot['overview_bounds']
        width,height=2048,1024
        xs=x0+(np.arange(width)+.5)*(x1-x0)/width
        ys=y0+(np.arange(height)+.5)*(y1-y0)/height
        # A worldwide preview cannot monopolize the compute lane with thousands
        # of learned windows. The incremental coarse worker replaces it later.
        macro=conditioning_preview(seed,wp,xs,ys)
        jobs.check_current_interest()
        climate=macro['climate'].copy()
        return macro['elev'],climate,(x1-x0)/width
    def finalize(value):
        jobs.check_current_interest()
        elevation, climate,resolution = value
        rgb = get_relief_map(elevation, None, None, None, resolution=resolution, vmin=0, vmax=4500)
        if mode!='relief':
            rgb=colorize(elevation,climate,mode)
        buffer = io.BytesIO()
        Image.fromarray((np.clip(rgb, 0, 1)*255).astype(np.uint8)).save(buffer, format='PNG')
        _atomic_bytes(path, buffer.getvalue())
        _atomic_bytes(receipt_path,json.dumps({'world_identity':identity,'mode':mode}).encode())
        _atomic_bytes(directory/'world.json', json.dumps(metadata_snapshot, indent=2).encode())
        disk_cache.record(cache_key,cache_paths)
        return path
    try:
        session = request.args.get('session')
        if session:
            session = _session(session)
        key = f'{VERSION}/{wp}/{seed}/overview/{mode}'
        # The client explicitly includes this interest while the overview is useful.
        path = jobs.submit(key, compute, finalize, session=session,
                           epoch=int(request.args.get('epoch', 0)), priority=0, lane='cpu').wait()
        return send_file(path, mimetype='image/png', max_age=31536000)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400
    except JobCancelled as exc:
        return jsonify(error=str(exc)), 409
    except QueueFull as exc:
        return jsonify(error=str(exc)), 429
    except TimeoutError as exc:
        return jsonify(error=str(exc)), 504


if __name__ == '__main__':
    def warm_models():
        global shared_pipeline
        try:
            with gpu_lock, torch.inference_mode(), span('startup.load_models'):
                preload_state.update(state='loading')
                if shared_pipeline is None:
                    shared_pipeline = load_pipeline(42)
                if os.environ.get('TERRAIN_PREWARM', '1') == '1':
                    preload_state.update(state='warming')
                    try:
                        sample_elevation(shared_pipeline,0,0,0)
                        if os.environ.get('TERRAIN_PREWARM_BASE','0')=='1':
                            preload_state['base_forms']=warm_base_forms(shared_pipeline)
                    finally:
                        shared_pipeline.empty_cache()
                preload_state.update(state='ready')
        except Exception as exc:
            preload_state.update(state='failed',error=str(exc)[:400])
            # The request path remains able to retry a transient load failure.
            print(f'Model preloading failed: {exc}', flush=True)
    if os.environ.get('TERRAIN_PRELOAD_MODELS', '1') == '1':
        threading.Thread(target=warm_models, name='terrain-model-preload', daemon=True).start()
    app.run(host='127.0.0.1', port=8765, threaded=True, debug=False)
