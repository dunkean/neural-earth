"""Persist the original Orogen relief and seasonal climate as a CUDA atlas.

The global graph pipeline runs on CPU; optional City WebGPU erosion precedes
climate on the retained graph. CUDA rasterizes its original triangles and the
neural generator samples the immutable physical fields.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from terrain_generation_session import check_generation, cancellable_process, run_process
import tempfile
import threading
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from terrain_bootstrap import WorldHeightmap, WORLD_BOUNDS, STYLES, CACHE_ROOT, _seed, _sha_file, _json_bytes, _atomic
from terrain_device import select_cuda_device

# Keep the storage namespace readable for saved immutable stage references.
# Source digests invalidate every newly generated atlas and stage automatically.
VERSION = 'orogen-full-pipeline-v7'
HYPSOMETRY_RULE = 'coastal-slope-v1: ocean 10000*e; land 6000*t^4*(5-4*t)+1000*t*(1-t)^4'
WIDTH, HEIGHT = 2048, 1024
ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / 'native' / 'orogen'
_LOCK = threading.RLock()
_SOURCES = [ROOT / 'terrain_orogen.py', ROOT / 'terrain_orogen_cuda.py',
            ROOT / 'terrain_city_erosion.py', ROOT / 'terrain_orogen_gpu.py', ROOT / 'terrain_orogen_stages.py', ROOT / 'terrain_orogen_layers.py', ROOT / 'terrain_soil.py', ROOT / 'terrain_conditioning.py', ROOT / 'terrain_generation.py', ROOT / 'terrain_world.py', ROOT / 'terrain_geometry.py', ROOT / 'terrain_bootstrap.py'] + sorted(
                p for directory in (NATIVE, ROOT / 'native' / 'city_erosion') for p in directory.rglob('*') if p.is_file())
_IMPORTED_DIGESTS = {str(p.relative_to(ROOT)).replace('\\', '/'): _sha_file(p) for p in _SOURCES}


@lru_cache(maxsize=1)
def implementation_identity():
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Orogen sparse plate generation requires Node.js (no npm installation needed)')
    selection = select_cuda_device()
    selected = selection['selected']
    return dict(generator=VERSION, sources=deepcopy(_IMPORTED_DIGESTS),
                node_version=subprocess.run([node, '--version'], check=True, capture_output=True,
                                            text=True).stdout.strip(),
                node_sha256=_sha_file(node), torch_version=str(torch.__version__),
                torch_cuda=torch.version.cuda, cuda_device=selected, cuda_driver=selection['driver'],
                raster_width=WIDTH, raster_height=HEIGHT,
                height_convention='signed-metres; sea iff height<0',
                relief='original-Orogen-superplates-relief-climate; CUDA-spherical-barycentric-rasterization')


def verify_implementation_identity(expected=None):
    identity = implementation_identity() if expected is None else expected
    current = {str(p.relative_to(ROOT)).replace('\\', '/'): _sha_file(p) for p in _SOURCES}
    if current != identity['sources']:
        raise RuntimeError('Orogen sources changed during this process; restart before generating a new world')
    node = shutil.which('node')
    if not node or _sha_file(node) != identity['node_sha256']:
        raise RuntimeError('Orogen Node runtime changed; restart before generating a new world')
    return True


def style_config(style, options=None):
    from terrain_generation import orogen_defaults, OROGEN_PARAMETERS, CITY_EROSION_PARAMETERS, OROGEN_GPU_PARAMETERS, OROGEN_STAGE_PARAMETERS, _number
    config = {k.removeprefix('orogen_'):v for k,v in orogen_defaults(style).items()}
    config.update(initial_source='orogen', relief_pipeline='orogen', initial_settings={})
    config.update({key:spec['default'] for key,spec in CITY_EROSION_PARAMETERS.items()})
    config.update({key:spec['default'] for key,spec in OROGEN_GPU_PARAMETERS.items()})
    config.update({key:spec['default'] for key,spec in OROGEN_STAGE_PARAMETERS.items()})
    if options:
        if set(options)-set(config): raise ValueError('Unknown Orogen parameters')
        for k,v in options.items():
            if k in OROGEN_STAGE_PARAMETERS:
                spec=OROGEN_STAGE_PARAMETERS[k]
                import re
                if (not isinstance(v,str) or ('pattern' in spec and re.fullmatch(spec['pattern'],v) is None)
                        or ('enum' in spec and v not in spec['enum'])):
                    raise ValueError(f'Invalid {k}')
                config[k]=v
                continue
            if k in OROGEN_GPU_PARAMETERS:
                if not isinstance(v,bool):raise ValueError(f'{k} must be a boolean')
                config[k]=v
                continue
            spec=OROGEN_PARAMETERS.get('orogen_'+k) or CITY_EROSION_PARAMETERS.get(k)
            if spec and 'length' in spec:
                if not isinstance(v,(list,tuple)) or len(v)!=spec['length']:
                    raise ValueError(f'orogen_{k} must contain {spec["length"]} values')
                config[k]=[_number(item,'orogen_'+k,spec) for item in v]
            else:
                config[k] = _number(v, 'orogen_'+k, spec) if spec else v
    if config['relief_pipeline'] not in ('original','orogen','city-gpu'):
        raise ValueError('Unknown Orogen relief pipeline')
    if config['relief_pipeline']=='city-gpu' and config['city_erosion_strength']>0:
        from terrain_city_erosion import implementation_identity as city_identity
        config['city_erosion_identity']=city_identity()
    if any(config[key] for key in OROGEN_GPU_PARAMETERS):
        from terrain_orogen_gpu import implementation_identity as gpu_identity
        config['orogen_gpu_identity']=gpu_identity()
    if config['initial_source'] in ('native','natural-continental'):
        from terrain_bootstrap import implementation_identity as native_identity
        config['initial_native_identity']=native_identity()
    if config['initial_source'] in ('natural','natural-continental'):
        from terrain_conditioning import STATS_PATH
        config['initial_statistics_sha256']=_sha_file(STATS_PATH)
    return config


def initial_raster(seed, style, config, width, height):
    source=config['initial_source']
    if source=='orogen': return None
    if source=='native':
        from terrain_bootstrap import get_heightmap
        world=get_heightmap(seed,style)
        xs=WORLD_BOUNDS[0]+(np.arange(width)+.5)*(WORLD_BOUNDS[2]-WORLD_BOUNDS[0])/width
        ys=WORLD_BOUNDS[1]+(np.arange(height)+.5)*(WORLD_BOUNDS[3]-WORLD_BOUNDS[1])/height
        return world.sample_height_m(xs,ys)
    from terrain_generation import _normalize, _descriptor
    from terrain_conditioning import GenerationConditioning
    settings=dict(config['initial_settings'], height_source=source, climate_source='natural', relief_pipeline='original')
    from terrain_generation import OROGEN_GPU_PARAMETERS, OROGEN_STAGE_PARAMETERS
    settings.update({key:False for key in OROGEN_GPU_PARAMETERS})
    settings.update({key:spec['default'] for key,spec in OROGEN_STAGE_PARAMETERS.items()})
    descriptor=_descriptor('orogen-initial', 'natural', _normalize('natural',settings))
    factory=GenerationConditioning(seed,descriptor)
    from terrain_geometry import world_bounds
    x0,y0,x1,y1=world_bounds(descriptor.settings)
    xs=x0+(np.arange(width)+.5)*(x1-x0)/width
    ys=y0+(np.arange(height)+.5)*(y1-y0)/height
    return factory._raw_coordinates(xs,ys)[0]


def sphere_raster(width, height):
    lat = np.pi/2 - (np.arange(height, dtype=np.float64)+.5)*np.pi/height
    lon = (np.arange(width, dtype=np.float64)+.5)*2*np.pi/width - np.pi
    xyz = np.empty((height, width, 3), np.float32)
    xyz[..., 0] = np.cos(lat)[:, None]*np.cos(lon)[None, :]
    xyz[..., 1] = np.sin(lat)[:, None]
    xyz[..., 2] = np.cos(lat)[:, None]*np.sin(lon)[None, :]
    return xyz.reshape(-1, 3)


def _metres(elevation):
    """Map native elevation to metres without a zero-slope coastal plain.

    The bounded coastal term gives a finite 1000 m/unit land slope at zero.
    It adds at most 81.92 m, vanishes at the peak and preserves native relief
    ordering and the exact sea level. No neural details are filtered here.
    """
    elevation=np.asarray(elevation)
    t=np.clip(elevation,0.,1.)
    land=6000*t**4*(5-4*t)+1000*t*(1-t)**4
    return np.where(elevation<=0,elevation*10000,land).astype(np.float32)


def _elevation(metres):
    """Inverse of the calibrated convention; physical imports keep their units."""
    metres=np.asarray(metres)
    lo,hi=np.zeros_like(metres,dtype=np.float64),np.ones_like(metres,dtype=np.float64)
    for _ in range(30):
        mid=(lo+hi)*.5
        below=6000*mid**4*(5-4*mid)+1000*mid*(1-mid)**4<metres
        lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    return np.where(metres<=0,metres/10000,(lo+hi)*.5).astype(np.float32)


def _graph_fields(directory):
    graph=json.loads((directory/'graph.json').read_text())
    fields={k:np.fromfile(directory/(k+'.f32'),dtype='<f4') for k in graph['fields']}
    points=fields.pop('xyz').reshape(-1,3)
    triangles=fields.pop('triangles').astype(np.int64)
    fields={k:v for k,v in fields.items() if len(v)==len(points) and k not in
        {'plate','stress','precipSummer','precipWinter','tempSummer','tempWinter','windSpeedSummer','windSpeedWinter'}}
    return graph,points,triangles,fields


def _atlas_to_elevation(raw, points, original):
    """Sample the GPU atlas back on the retained sphere, using Orogen units."""
    h,w=raw.shape
    x=(np.arctan2(points[:,2],points[:,0])/(2*np.pi)+.5)*w-.5
    y=np.clip((.5-np.arcsin(np.clip(points[:,1],-1,1))/np.pi)*h-.5,0,h-1)
    ix,iy=np.floor(x).astype(np.int64),np.floor(y).astype(np.int64)
    fx,fy=x-ix,y-iy
    low=raw[iy,ix%w]*(1-fx)+raw[iy,(ix+1)%w]*fx
    high=raw[np.minimum(iy+1,h-1),ix%w]*(1-fx)+raw[np.minimum(iy+1,h-1),(ix+1)%w]*fx
    metres=np.maximum(low*(1-fy)+high*fy,0)
    return np.where(original<=0,original,_elevation(metres)).astype('<f4')


def _nearest_regions(points, xyz, config):
    if config.get('orogen_gpu_raster') and config.get('orogen_gpu_identity',{}).get('available'):
        from terrain_orogen_gpu import nearest_regions
        return nearest_regions(points,xyz)
    return cKDTree(points).query(xyz,k=4,workers=4)[1]


def _run_city_pipeline(node, directory, request, width, height, device, base, config):
    from terrain_orogen_cuda import rasterize
    from terrain_city_erosion import erode
    started=time.perf_counter()
    request['externalErosion']=True
    (directory/'request.json').write_bytes(_json_bytes(request))
    with (directory/'node.log').open('w+b') as log, cancellable_process(
        [node,str(NATIVE/'pipeline.mjs'),str(directory/'request.json'),str(directory)],
        stdin=subprocess.PIPE,stdout=log,stderr=log) as process:
        while not (directory/'relief.ready').exists():
            check_generation()
            if process.poll() is not None or time.perf_counter()-started>600:
                log.seek(0)
                raise RuntimeError('Orogen relief stage failed: '+log.read().decode(errors='replace')[-4000:])
            time.sleep(.02)
        native_seconds=time.perf_counter()-started
        _,points,triangles,fields=_graph_fields(directory)
        at=time.perf_counter()
        xyz=sphere_raster(width,height)
        nearest=_nearest_regions(points,xyz,config)
        projection_seconds=time.perf_counter()-at
        at=time.perf_counter()
        if base is None:
            initial,_=rasterize(points,triangles,nearest,xyz,{'elevation':fields['elevation']},device)
            raw=_metres(initial['elevation']).reshape(height,width)
        else:
            raw=np.asarray(base,dtype=np.float32).copy()
        initial_cuda_seconds=time.perf_counter()-at
        eroded,metadata=erode(raw,strength=config['city_erosion_strength'],iterations=config['city_erosion_iterations'],
            talus=config['city_erosion_talus'],motif_km=config['city_erosion_motif_km'])
        at=time.perf_counter()
        elevation=_atlas_to_elevation(eroded,points,fields['elevation'])
        elevation.tofile(directory/'external-elevation.f32')
        metadata['graph_resampling_seconds']=time.perf_counter()-at
        at=time.perf_counter()
        process.communicate(_json_bytes(dict(elevationFile=str(directory/'external-elevation.f32'))),timeout=600)
        native_seconds+=time.perf_counter()-at
        if process.returncode:
            log.seek(0)
            raise RuntimeError('Orogen climate stage failed: '+log.read().decode(errors='replace')[-4000:])
        return eroded,metadata,(points,triangles,nearest,xyz),native_seconds,projection_seconds,initial_cuda_seconds,raw



def generate_atlas(seed, style='earthlike', *, width=WIDTH, height=HEIGHT, include_layers=False, options=None):
    """Uncached generation for benchmarks/QA; production dimensions are fixed."""
    seed = _seed(seed)
    if style not in STYLES:
        raise ValueError(f'Unknown Orogen style {style!r}')
    if width != 2*height or height < 16 or width > 4096:
        raise ValueError('Orogen raster must be 2:1, with height >=16 and width <=4096')
    identity = implementation_identity()
    verify_implementation_identity(identity)
    config = style_config(style, options)
    # Mix every u64 bit before entering upstream's numeric JavaScript RNG.
    digest = hashlib.sha256(f'orogen-tectonic-cuda-v1:{seed}'.encode()).digest()
    sparse_seed = seed if seed <= 0x7fffffff else int.from_bytes(digest[:4], 'little') & 0x7fffffff
    started = time.perf_counter()
    device = torch.device('cuda', identity['cuda_device']['index'])
    before = torch.cuda.memory_allocated(device)
    use_city = config['relief_pipeline']=='city-gpu' and config['city_erosion_strength']>0
    city_metadata = None
    city_geometry = None
    initial_cuda_seconds = 0.
    base = initial_raster(seed, style, config, width, height)
    gpu_metadata = None
    use_gpu = config.get('orogen_gpu_identity',{}).get('available',False) and any(
        config['orogen_gpu_'+stage] for stage in ('relief','propagation','post','climate','erosion'))
    with tempfile.TemporaryDirectory(prefix='orogen-pipeline-') as temporary:
        directory = Path(temporary)
        request=dict(seed=sparse_seed,N=config['detail'],P=config['plate_count'],jitter=.75,
            nMag=config['roughness'],numContinents=config['continent_count'],
            continentSizeVariety=config['continent_variety'],landCoverage=config['land_coverage'],
            smoothing=config['smoothing'],terrainWarp=config['warp'],hydraulicErosion=config['hydraulic'],
            thermalErosion=config['thermal'],glacialErosion=config['glacial'],ridgeSharpening=config['sharpening'],
            temperatureOffset=config['temperature_offset'],precipitationOffset=config['precipitation_offset'],
            threshold=config['convergence_threshold'],motion=config['motion_strength'],spread=config['spread'],
            skipPost=config['relief_pipeline']!='orogen')
        from terrain_generation import OROGEN_CLIMATE_PARAMETERS
        request['climateSettings']={key.removeprefix('orogen_'):config[key.removeprefix('orogen_')]
                                   for key in OROGEN_CLIMATE_PARAMETERS}
        if base is not None:
            base=np.asarray(base,dtype='<f4');base.tofile(directory/'initial.f32')
            request.update(inputFile=str(directory/'initial.f32'),imageWidth=width,imageHeight=height)
        if use_gpu:
            from terrain_orogen_gpu import run as run_gpu
            external = {}
            def city_callback():
                from terrain_orogen_cuda import rasterize
                from terrain_city_erosion import erode
                _,p,t,f=_graph_fields(directory)
                at=time.perf_counter();xyz=sphere_raster(width,height)
                nearest=_nearest_regions(p,xyz,config)
                external['projection_seconds']=time.perf_counter()-at
                at=time.perf_counter()
                if base is None:
                    initial,_=rasterize(p,t,nearest,xyz,{'elevation':f['elevation']},device)
                    initial=_metres(initial['elevation']).reshape(height,width)
                else: initial=np.asarray(base,dtype=np.float32).copy()
                external['initial_cuda_seconds']=time.perf_counter()-at
                eroded,meta=erode(initial,strength=config['city_erosion_strength'],iterations=config['city_erosion_iterations'],
                                  talus=config['city_erosion_talus'],motif_km=config['city_erosion_motif_km'])
                at=time.perf_counter()
                _atlas_to_elevation(eroded,p,f['elevation']).tofile(directory/'external-elevation.f32')
                meta['graph_resampling_seconds']=time.perf_counter()-at
                external.update(raw=eroded,initial=initial,metadata=meta,geometry=(p,t,nearest,xyz))
                return dict(elevationFile=str(directory/'external-elevation.f32')),None
            flags={stage:config['orogen_gpu_'+stage] for stage in ('relief','propagation','post','climate','erosion')}
            gpu_metadata,_=run_gpu(shutil.which('node'),directory,request,flags,city_callback if use_city else None)
            if use_city:
                city_raw,city_initial,city_metadata=external['raw'],external['initial'],external['metadata']
                city_geometry=external['geometry'];projection_seconds=external['projection_seconds']
                initial_cuda_seconds=external['initial_cuda_seconds'];coarse_seconds=gpu_metadata['timings']['seconds']
        elif use_city:
            city_raw,city_metadata,city_geometry,coarse_seconds,projection_seconds,initial_cuda_seconds,city_initial = _run_city_pipeline(
                shutil.which('node'),directory,request,width,height,device,base,config)
        else:
            (directory/'request.json').write_bytes(_json_bytes(request))
            result = run_process([shutil.which('node'), str(NATIVE/'pipeline.mjs'),
                                     str(directory/'request.json'), str(directory)],
                                    check=False, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(f'Orogen pipeline failed: {result.stderr.strip()}')
        graph,points,triangles,fields=_graph_fields(directory)
    if city_geometry is None:
        coarse_seconds = time.perf_counter()-started
        at=time.perf_counter()
        xyz = sphere_raster(width, height)
        indices = _nearest_regions(points,xyz,config)
        projection_seconds = time.perf_counter()-at
    else:
        _,_,indices,xyz=city_geometry
    from terrain_orogen_cuda import rasterize
    at=time.perf_counter()
    fields,fallback = rasterize(points,triangles,indices,xyz,fields,device)
    gpu_seconds = time.perf_counter()-at
    layers={k:v.reshape(height,width) for k,v in fields.items()}
    raw=_metres(layers.pop('elevation'))
    if use_city:
        raw=city_raw
        layers['city_erosion_delta_m']=(raw-city_initial).astype(np.float32)
    elif base is not None and config['relief_pipeline']!='orogen': raw=base.copy()
    layers.setdefault('boundaries',np.zeros_like(raw));layers.setdefault('convergence',np.zeros_like(raw))
    belt=np.maximum(layers.get('tectonic',np.zeros_like(raw)),0).astype(np.float32)
    layers['uplift']=belt
    if not np.isfinite(raw).all() or any(not np.isfinite(v).all() for v in layers.values()):
        raise RuntimeError('Orogen produced non-finite fields')
    ocean=raw<0
    metadata=dict(requested_seed_u64=str(seed),selected_seed_u64=str(seed),selected_attempt=0,
        style=style,generator_version=VERSION,native_config=dict(**config,sparse_seed=sparse_seed,
        backend='original-JS-global; CUDA-barycentric-atlas'+('; City-WebGPU-surface' if use_city else '')),attempts=[],
        raster=dict(width=width,height=height,dtype='<f4',projection='equirectangular',samples='pixel-centres',
            y_axis='south',sea_level_m=0,longitude_axis='atan2(z,x)',reconstruction='spherical-bilinear-at-read'),
        hypsometry=dict(rule=HYPSOMETRY_RULE),
        sign_preserved=True,generation_seconds=time.perf_counter()-started,
        timings=dict(coarse_seconds=coarse_seconds,projection_seconds=projection_seconds,cuda_seconds=gpu_seconds),
        original_pipeline=graph['timing'],original_elevation_timing=graph.get('elevationTiming',[]),
        original_post_timing=graph.get('postTiming',[]),projection_fallback_pixels=fallback,
        cuda_resident_baseline_bytes=before,diagnostics=dict(land_fraction=float(np.mean(raw>0)),
            highland_land_fraction=float(np.sum(raw>2000)/max(1,np.sum(raw>0)))))
    if city_metadata is not None:
        metadata['erosion']=city_metadata
        metadata['timings'].update(city_erosion_seconds=city_metadata['seconds'],
                                  initial_cuda_seconds=initial_cuda_seconds,
                                  graph_resampling_seconds=city_metadata['graph_resampling_seconds'])
    if gpu_metadata is not None:
        metadata['gpu_pipeline']=gpu_metadata
        metadata['native_config']['backend']+='; opt-in CUDA graph stages'
    elif 'orogen_gpu_identity' in config:
        metadata['gpu_pipeline']=dict(identity=config['orogen_gpu_identity'],fallback=not config['orogen_gpu_identity']['available'])
    if config.get('orogen_gpu_raster'):
        metadata['timings']['nearest_backend']='CUDA-grid' if config['orogen_gpu_identity']['available'] else 'CPU-cKDTree'
    if include_layers:
        return raw, ocean, metadata, belt, layers
    return raw, ocean, metadata, belt


class OrogenHeightmap(WorldHeightmap):
    """Reuse native atlas validation, atomic persistence and bilinear sampling."""
    def __init__(self, seed, style='earthlike', *, options=None):
        self.seed = _seed(seed)
        if style not in STYLES:
            raise ValueError(f'Unknown Orogen style {style!r}')
        self.style, self.bounds = style, WORLD_BOUNDS
        self.width, self.height = WIDTH, HEIGHT
        identity = deepcopy(implementation_identity())
        config = style_config(style, options)
        request = dict(seed=self.seed, style=style, width=WIDTH, height=HEIGHT, config=config)
        namespace = dict(implementation=identity, request=request)
        self.cache_key = hashlib.sha256(_json_bytes(namespace)).hexdigest()
        self.cache_path = CACHE_ROOT / VERSION / (self.cache_key+'.npz')
        with _LOCK:
            verify_implementation_identity(identity)
            loaded = self._load(namespace)
            if loaded is None:
                from terrain_orogen_stages import generate_staged_atlas
                raw, ocean, metadata, _, self.layers = generate_staged_atlas(self.seed, style,
                    width=self.width, height=self.height, include_layers=True, options=options)
                metadata.update(namespace=namespace, cache_key=self.cache_key,
                    raw_height_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                    height_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                    physical_ocean_sha256=hashlib.sha256(ocean.tobytes()).hexdigest(),
                    layer_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in self.layers.items()})
                self._save(raw, raw, ocean, metadata)
                loaded = raw, raw, ocean, metadata
            self.raw_height_m, self.height_m, self.physical_ocean, self.metadata = loaded
        for array in (self.raw_height_m, self.height_m, self.physical_ocean):
            array.flags.writeable = False
        for array in self.layers.values():
            array.flags.writeable = False

    def _load(self, namespace):
        loaded = super()._load(namespace)
        if loaded is None:
            return None
        try:
            with np.load(self.cache_path, allow_pickle=False) as data:
                layers = {name:data['layer_'+name] for name in loaded[3]['layer_sha256']}
            for name, array in layers.items():
                if (array.shape != (self.height, self.width) or not np.isfinite(array).all()
                        or hashlib.sha256(array.tobytes()).hexdigest() != loaded[3]['layer_sha256'][name]):
                    return None
            self.layers = layers
            return loaded
        except (OSError, ValueError, KeyError):
            return None

    def _save(self, raw, final, physical_ocean, metadata):
        import io
        stream = io.BytesIO()
        np.savez(stream, raw_height_m=raw, height_m=final,
            physical_ocean=physical_ocean, metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            **{'layer_'+k:v for k,v in self.layers.items()})
        _atomic(self.cache_path, stream.getvalue())


@lru_cache(maxsize=4)
def _get_heightmap(seed, style, options):
    return OrogenHeightmap(seed, style, options=json.loads(options))


def get_heightmap(seed, style='earthlike', *, options=None):
    return _get_heightmap(_seed(seed), style, json.dumps(options or {},sort_keys=True))


def bootstrap_metadata(seed, style='earthlike', *, options=None):
    return deepcopy(get_heightmap(seed, style, options=options).metadata)
