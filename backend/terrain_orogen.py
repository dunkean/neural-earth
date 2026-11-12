"""Persist Orogen v3 geological relief and monthly climate as a CUDA atlas.

The global graph pipeline runs on CPU; optional City WebGPU erosion precedes
climate on the retained graph. CUDA rasterizes its original triangles and the
neural generator samples the immutable physical fields.
"""

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import zipfile

import numpy as np
import torch
from scipy.spatial import cKDTree

from terrain_bootstrap import WorldHeightmap, WORLD_BOUNDS, STYLES, CACHE_ROOT, _seed, _sha_file, _json_bytes
from terrain_device import select_cuda_device
import terrain_array_io as array_io

# Keep the storage namespace readable for saved immutable stage references.
# Source digests invalidate every newly generated atlas and stage automatically.
VERSION = 'orogen-v3-pipeline-v1'
HYPSOMETRY_RULE = 'Orogen-v3: physical kilometres to signed metres; no nonlinear coastal conversion'
WIDTH, HEIGHT = 2048, 1024
ROOT = REPO_ROOT
NATIVE = ROOT / 'native' / 'orogen'
_LOCK = threading.RLock()
_SOURCES = [source_path('terrain_paths.py', root=ROOT), source_path('terrain_orogen.py', root=ROOT), source_path('terrain_orogen_cuda.py', root=ROOT),
            source_path('terrain_city_erosion.py', root=ROOT), source_path('terrain_orogen_gpu.py', root=ROOT), source_path('terrain_orogen_stages.py', root=ROOT), source_path('terrain_orogen_layers.py', root=ROOT), source_path('terrain_soil.py', root=ROOT), source_path('terrain_pedology.py', root=ROOT), source_path('terrain_conditioning.py', root=ROOT), source_path('terrain_generation.py', root=ROOT), source_path('terrain_world.py', root=ROOT), source_path('terrain_geometry.py', root=ROOT), source_path('terrain_bootstrap.py', root=ROOT), source_path('terrain_array_io.py', root=ROOT)] + sorted(
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
                height_convention='signed-metres; explicit connected ocean mask; sea level 0 m',
                relief='Orogen-v3-present-geology-finish-monthly-climate; CUDA-spherical-barycentric-rasterization',
                upstream_revision='ed466f9db71292e912d8d0160e689a67a994f2d9')


_FILE_DIGESTS = {}


def _file_digest(path):
    # Rehash only files whose size or modification time changed.
    stat = Path(path).stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _FILE_DIGESTS:
        _FILE_DIGESTS[key] = _sha_file(path)
    return _FILE_DIGESTS[key]


def verify_implementation_identity(expected=None):
    identity = implementation_identity() if expected is None else expected
    current = {str(p.relative_to(ROOT)).replace('\\', '/'): _file_digest(p) for p in _SOURCES}
    if current != identity['sources']:
        raise RuntimeError('Orogen sources changed during this process; restart before generating a new world')
    node = shutil.which('node')
    if not node or _file_digest(node) != identity['node_sha256']:
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
    """V3 supplies physical kilometres, including coasts and peaks above 6 km."""
    return (np.asarray(elevation, dtype=np.float64) * 1000).astype(np.float32)


def _elevation(metres):
    """Physical imports have no clipping, coastal uplift or inverse curve."""
    return (np.asarray(metres, dtype=np.float64) / 1000).astype(np.float32)


def _graph_fields(directory):
    graph=json.loads((directory/'graph.json').read_text())
    flat=np.fromfile(directory/'fields.f32',dtype='<f4')
    ends=np.cumsum(list(graph['fields'].values()))
    if len(flat)!=(ends[-1] if len(ends) else 0):raise RuntimeError('Orogen graph fields are incomplete')
    fields={k:flat[end-length:end] for (k,length),end in zip(graph['fields'].items(),ends)}
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
    return cKDTree(points).query(xyz,k=4,workers=-1)[1]


def generate_atlas(seed, style='earthlike', *, width=WIDTH, height=HEIGHT, include_layers=False, options=None):
    """Run the v3 relief/finish/monthly-climate chain at a requested resolution."""
    if style not in STYLES:
        raise ValueError(f'Unknown Orogen style {style!r}')
    if width != 2*height or height < 16 or width > 4096:
        raise ValueError('Orogen raster must be 2:1, with height >=16 and width <=4096')
    import terrain_orogen_stages as stages
    verify_implementation_identity()
    seed = _seed(seed)
    config = style_config(style, options)
    begun = time.perf_counter()
    artifacts = {}
    settings = stages._settings(config, style)
    for stage in stages.STAGE_KEYS:
        parent = artifacts.get('relief' if stage == 'erosion' else 'erosion') if stage != 'relief' else None
        arrays, details = stages._execute(stage, seed, style, config, width, height, parent, artifacts.get('relief'))
        namespace = dict(stage=stage, seed=seed, width=width, height=height, parent=parent['id'] if parent else None,
                         settings={k:settings[k] for k in sorted(stages.STAGE_KEYS[stage])})
        artifacts[stage] = dict(id=hashlib.sha256(_json_bytes(namespace)).hexdigest(),
                                arrays=arrays, metadata=dict(namespace=namespace, **details))
    result = stages.compose(seed, style, config, width, height, artifacts, 'erosion')
    result[2]['generation_seconds'] = time.perf_counter() - begun
    return result if include_layers else result[:4]


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
                digests = array_io.sha256_arrays(dict(raw=raw, ocean=ocean, **{'layer_'+k:v for k,v in self.layers.items()}))
                metadata.update(namespace=namespace, cache_key=self.cache_key,
                    raw_height_sha256=digests['raw'], height_sha256=digests['raw'],
                    physical_ocean_sha256=digests['ocean'],
                    layer_sha256={k:digests['layer_'+k] for k in self.layers})
                self._save(raw, raw, ocean, metadata)
                loaded = raw, raw, ocean, metadata
            self.raw_height_m, self.height_m, self.physical_ocean, self.metadata = loaded
        for array in (self.raw_height_m, self.height_m, self.physical_ocean):
            array.flags.writeable = False
        for array in self.layers.values():
            array.flags.writeable = False

    def _load(self, namespace):
        # WorldHeightmap's checks, plus the layers, hashed in parallel.
        array_io.wait(self.cache_path)
        try:
            with np.load(self.cache_path, allow_pickle=False) as data:
                metadata = json.loads(str(data['metadata'].item()))
                if metadata['namespace'] != namespace or metadata['cache_key'] != self.cache_key:
                    return None
                arrays = dict(raw_height=data['raw_height_m'], height=data['height_m'], physical_ocean=data['physical_ocean'],
                              **{'layer_'+name:data['layer_'+name] for name in metadata['layer_sha256']})
            for name, array in arrays.items():
                dtype = np.dtype('bool') if name == 'physical_ocean' else np.dtype('float32')
                if array.shape != (self.height, self.width) or (array.dtype != dtype and not name.startswith('layer_')) or not np.isfinite(array).all():
                    return None
            digests = array_io.sha256_arrays(arrays)
            for name in arrays:
                expected = metadata['layer_sha256'][name[6:]] if name.startswith('layer_') else metadata[name+'_sha256']
                if digests[name] != expected:
                    return None
            if not np.array_equal(arrays['raw_height'] < 0, arrays['height'] < 0):
                return None
            self.layers = {name[6:]:array for name, array in arrays.items() if name.startswith('layer_')}
            return arrays['raw_height'], arrays['height'], arrays['physical_ocean'], metadata
        except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError, zipfile.BadZipFile):
            return None

    def _save(self, raw, final, physical_ocean, metadata):
        array_io.save_npz(self.cache_path, dict(raw_height_m=raw, height_m=final,
            physical_ocean=physical_ocean, metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            **{'layer_'+k:v for k,v in self.layers.items()}))


@lru_cache(maxsize=4)
def _get_heightmap(seed, style, options):
    return OrogenHeightmap(seed, style, options=json.loads(options))


def get_heightmap(seed, style='earthlike', *, options=None):
    return _get_heightmap(_seed(seed), style, json.dumps(options or {},sort_keys=True))


def bootstrap_metadata(seed, style='earthlike', *, options=None):
    return deepcopy(get_heightmap(seed, style, options=options).metadata)
