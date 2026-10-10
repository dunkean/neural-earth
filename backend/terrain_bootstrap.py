"""CPU World Builder atlas, persisted as a finite signed terrestrial heightmap.

Only this module creates the coarse geographic input. The source library owns
geographic arrangement; measured ETOPO land/depth distributions supply metres.
Reads interpolate the same immutable raster at every neural request and LOD.
"""
from __future__ import annotations

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path, RUNTIME_ROOT, configured_path

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from terrain_generation_session import run_process
import tempfile
import threading
import zipfile

import numpy as np

BOOTSTRAP_VERSION = 'world-builder-heightmap-v2'
STYLES = ('gondwana', 'continents', 'earthlike', 'archipelago')
WORLD_BOUNDS = (-20_000_000., -10_000_000., 20_000_000., 10_000_000.)
UPSTREAM_COMMIT = '009efe18c457757da12ffc8a6215806efb87f012'
UPSTREAM_VERSION = 'prototype-0.11.0-hybrid'
ROOT = REPO_ROOT
UPSTREAM = ROOT.parent / 'world-builder-rs'
NATIVE = ROOT / 'native' / 'terrain_bootstrap'
BINARY = NATIVE / 'target' / 'release' / ('terrain-bootstrap.exe' if os.name == 'nt' else 'terrain-bootstrap')
ETOPO = ROOT / 'terrain-diffusion' / 'data' / 'global' / 'etopo_10m.tif'
CACHE_ROOT = configured_path('TERRAIN_BOOTSTRAP_CACHE', RUNTIME_ROOT / 'heightmap-bootstrap')
NATIVE_RESOLUTION = 256
RASTER_WIDTH, RASTER_HEIGHT = 1024, 512
_LOCK = threading.RLock()
_FROZEN_IMPLEMENTATION_IDENTITY = None


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


_IMPORTED_SOURCE_SHA256 = _sha_file(__file__)


def _atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _native_source_identity():
    commit = subprocess.run(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f'World Builder source must be pinned at {UPSTREAM_COMMIT}; found {commit}')
    dirty = subprocess.run(['git', '-C', str(UPSTREAM), 'status', '--porcelain', '--untracked-files=all', '--', 'crates/world-core/src', 'crates/world-core/Cargo.toml', 'crates/world-core/build.rs', 'Cargo.toml'], check=True, capture_output=True, text=True).stdout
    if dirty:
        raise RuntimeError('Pinned World Builder core contains tracked modifications')
    paths = sorted(p for p in (UPSTREAM / 'crates' / 'world-core' / 'src').rglob('*') if p.is_file())
    paths += [UPSTREAM / 'Cargo.toml', UPSTREAM / 'crates' / 'world-core' / 'Cargo.toml']
    if (UPSTREAM / 'crates' / 'world-core' / 'build.rs').exists():
        paths.append(UPSTREAM / 'crates' / 'world-core' / 'build.rs')
    upstream_files = {str(p.relative_to(UPSTREAM)).replace('\\', '/'): _sha_file(p) for p in paths}
    bridge_files = {str(p.relative_to(NATIVE)).replace('\\', '/'): _sha_file(p)
                    for p in [NATIVE / 'Cargo.toml', NATIVE / 'Cargo.lock', NATIVE / 'src' / 'main.rs']}
    identity = {'upstream_commit': commit, 'upstream_generator_version': UPSTREAM_VERSION,
                'upstream_files': upstream_files, 'bridge_files': bridge_files}
    return identity, hashlib.sha256(_json_bytes(identity)).hexdigest()


def _ensure_binary():
    identity, source_hash = _native_source_identity()
    receipt_path = NATIVE / 'build-receipt.json'
    try:
        receipt = json.loads(receipt_path.read_text())
        ready = (receipt['native_source_sha256'] == source_hash and BINARY.is_file()
                 and receipt['binary_sha256'] == _sha_file(BINARY))
    except (OSError, ValueError, KeyError):
        ready = False
    if not ready:
        cargo = shutil.which('cargo')
        if not cargo:
            candidate = Path.home() / '.cargo' / 'bin' / ('cargo.exe' if os.name == 'nt' else 'cargo')
            if candidate.is_file():
                cargo = str(candidate)
        if not cargo:
            raise RuntimeError('Building the CPU heightmap exporter requires cargo')
        try:
            subprocess.run([cargo, 'build', '--release', '--locked', '--manifest-path', str(NATIVE / 'Cargo.toml'), '--target-dir', str(NATIVE / 'target')], cwd=ROOT, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f'CPU heightmap exporter build failed: {error.stderr.strip()}') from error
        receipt = {'native_source_sha256': source_hash, 'binary_sha256': _sha_file(BINARY),
                   'cargo_version': subprocess.run([cargo, '--version'], check=True, capture_output=True, text=True).stdout.strip()}
        _atomic(receipt_path, _json_bytes(receipt))
    return {**identity, **receipt}


@lru_cache(maxsize=1)
def implementation_identity():
    """Pinned source, locked Rust dependencies, executable and Python digests."""
    global _FROZEN_IMPLEMENTATION_IDENTITY
    with _LOCK:
        identity = _ensure_binary()
        identity.update(bootstrap_version=BOOTSTRAP_VERSION,
                        python_source_sha256=_IMPORTED_SOURCE_SHA256,
                        etopo_sha256=_sha_file(ETOPO),
                        native_resolution=NATIVE_RESOLUTION,
                        raster_width=RASTER_WIDTH, raster_height=RASTER_HEIGHT,
                        world_bounds=list(WORLD_BOUNDS),
                        height_convention='signed-metres; sea iff height<0',
                        hypsometry='separate-monotone-spherical-area-land-and-depth-quantiles-v2',
                        numpy_version=np.__version__)
        _FROZEN_IMPLEMENTATION_IDENTITY = deepcopy(identity)
        return identity


def verify_implementation_identity(expected=None):
    """Check current bytes against an initialized identity, without building."""
    identity = expected if expected is not None else _FROZEN_IMPLEMENTATION_IDENTITY
    if identity is None:
        raise RuntimeError('Heightmap identity has not been initialized; call implementation_identity first')
    _, source_hash = _native_source_identity()
    if (source_hash != identity['native_source_sha256'] or
            _sha_file(__file__) != identity['python_source_sha256'] or
            _sha_file(ETOPO) != identity['etopo_sha256']):
        raise RuntimeError('Heightmap sources changed during this process; restart before generating a new world')
    if _sha_file(BINARY) != identity['binary_sha256']:
        raise RuntimeError('Heightmap exporter changed during this process; restart before generating a new world')
    return True


def _generation_binary(identity, directory):
    """Run a private, verified executable, never a mutable shared build target."""
    verify_implementation_identity(identity)
    executable = directory / BINARY.name
    shutil.copyfile(BINARY, executable)
    if _sha_file(executable) != identity['binary_sha256']:
        raise RuntimeError('Heightmap exporter changed during this process; restart before generating a new world')
    if os.name != 'nt':
        executable.chmod(0o700)
    return executable


@lru_cache(maxsize=1)
def _source_hypsometry():
    """Distribution only: geographic coordinates from Earth are never sampled."""
    import rasterio
    with rasterio.open(ETOPO) as dataset:
        data = dataset.read(1, masked=True)
        if not dataset.crs or not dataset.crs.is_geographic or dataset.transform.b or dataset.transform.d:
            raise RuntimeError('ETOPO distribution requires an unrotated geographic raster')
        rows = np.arange(dataset.height + 1)
        latitude_edges = np.clip(dataset.transform.f + rows * dataset.transform.e, -90, 90)
        row_weights = np.abs(np.diff(np.sin(np.deg2rad(latitude_edges))))
        valid = ~np.ma.getmaskarray(data) & np.isfinite(data.data)
        values = np.asarray(data.data[valid], dtype=np.float64)
        weights = np.broadcast_to(row_weights[:, None], data.shape)[valid]
        units = dataset.tags().get('units') or dataset.tags(1).get('units') or 'metres'
    land = values[values > 0]
    depth = -values[values < 0]
    if land.size < 100 or depth.size < 100:
        raise RuntimeError('ETOPO lacks a valid signed land/depth distribution')
    probabilities = np.linspace(0, 1, 257)
    land_quantiles = _weighted_quantile(land, weights[values > 0], probabilities)
    depth_quantiles = _weighted_quantile(depth, weights[values < 0], probabilities)
    land_quantiles[0] = depth_quantiles[0] = 0.
    return probabilities, land_quantiles, depth_quantiles, {'source': 'terrain-diffusion/data/global/etopo_10m.tif', 'sha256': _sha_file(ETOPO),
            'weighting': 'spherical pixel area; independent land/depth conditional distributions',
            'source_units': units, 'probabilities': probabilities.tolist(),
            'land_quantiles_m': land_quantiles.tolist(), 'depth_quantiles_m': depth_quantiles.tolist(),
            'source_land_sample_count': int(land.size), 'source_depth_sample_count': int(depth.size)}


def _weighted_quantile(values, weights, probabilities):
    order = np.argsort(values, kind='stable')
    ordered, mass = values[order], weights[order]
    positions = (np.cumsum(mass) - .5 * mass) / np.sum(mass)
    return np.interp(probabilities, positions, ordered)


def _remap_hypsometry(raw):
    probabilities, land_target, depth_target, source = _source_hypsometry()
    final = np.zeros_like(raw, dtype=np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('Native heightmap must be a finite two-dimensional raster')
    row_weights = np.cos((np.arange(raw.shape[0]) + .5) / raw.shape[0] * np.pi - np.pi / 2)
    weights = np.broadcast_to(row_weights[:, None], raw.shape)
    receipt = {'source': source, 'rule': 'monotone piecewise linear, spherical-area weighted independent positive land/sea depth; 0 stays 0'}
    for name, mask, target in [('land', raw > 0, land_target), ('depth', raw < 0, depth_target)]:
        values = np.abs(raw[mask]).astype(np.float64)
        if values.size < 100:
            raise RuntimeError(f'Native atlas has insufficient {name} samples')
        native = _weighted_quantile(values, weights[mask], probabilities)
        native[0] = 0.
        # Native interpolated heights are continuous. Remove repeated knots if a
        # degenerate distribution contains ties; keep a unique target for each.
        knots, positions = np.unique(native, return_index=True)
        destination = target[positions]
        if knots.size < 10:
            raise RuntimeError(f'Native {name} height distribution is degenerate')
        mapped = np.interp(values, knots, destination).astype(np.float32)
        final[mask] = mapped if name == 'land' else -mapped
        receipt[name] = {'raw_knots_m': knots.tolist(), 'target_knots_m': destination.tolist()}
    return final, receipt


def _seed(seed):
    if isinstance(seed, bool):
        raise ValueError('Seed must be a u64 integer')
    if isinstance(seed, str):
        if not seed.isdigit():
            raise ValueError('Seed string must contain decimal digits')
        seed = int(seed)
    if not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**64:
        raise ValueError('Seed must fit u64 exactly')
    return int(seed)


class WorldHeightmap:
    """Immutable persisted raster with deterministic Cartesian random access."""
    def __init__(self, seed, style='earthlike'):
        self.seed = _seed(seed)
        if style not in STYLES:
            raise ValueError(f'Unknown terrain style {style!r}')
        self.style = style
        self.bounds = WORLD_BOUNDS
        self.width, self.height = RASTER_WIDTH, RASTER_HEIGHT
        identity = deepcopy(implementation_identity())
        request = {'seed': self.seed, 'style': style, 'resolution': NATIVE_RESOLUTION,
                   'width': self.width, 'height': self.height}
        namespace = {'implementation': identity, 'request': request}
        self.cache_key = hashlib.sha256(_json_bytes(namespace)).hexdigest()
        self.cache_path = CACHE_ROOT / BOOTSTRAP_VERSION / (self.cache_key + '.npz')
        with _LOCK:
            loaded = self._load(namespace)
            if loaded is None:
                with tempfile.TemporaryDirectory(prefix='terrain-bootstrap-') as temporary:
                    directory = Path(temporary)
                    (directory / 'request.json').write_bytes(_json_bytes(request))
                    executable = _generation_binary(identity, directory)
                    completed = run_process([str(executable), str(directory / 'request.json'), str(directory / 'output')], cwd=ROOT, capture_output=True, text=True)
                    if completed.returncode:
                        raise RuntimeError(f'Native heightmap generation failed: {completed.stderr.strip()}')
                    raw = np.fromfile(directory / 'output' / 'height.f32', dtype='<f4').reshape(self.height, self.width)
                    physical_ocean = np.fromfile(directory / 'output' / 'ocean.u8', dtype=np.uint8).reshape(self.height, self.width).astype(bool)
                    final, mapping = _remap_hypsometry(raw)
                    metadata = json.loads((directory / 'output' / 'metadata.json').read_text())
                    if (metadata.get('generator_version') != UPSTREAM_VERSION or
                            metadata.get('requested_seed_u64') != str(self.seed) or
                            metadata.get('style') != style):
                        raise RuntimeError('Native heightmap receipt differs from requested world')
                    metadata.update(namespace=namespace, cache_key=self.cache_key, hypsometry=mapping,
                                    raw_height_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                                    height_sha256=hashlib.sha256(final.tobytes()).hexdigest(),
                                    physical_ocean_sha256=hashlib.sha256(physical_ocean.tobytes()).hexdigest(),
                                    sign_preserved=bool(np.array_equal(raw < 0, final < 0)))
                    self._save(raw, final, physical_ocean, metadata)
                    loaded = raw, final, physical_ocean, metadata
            self.raw_height_m, self.height_m, self.physical_ocean, self.metadata = loaded
        for array in (self.raw_height_m, self.height_m, self.physical_ocean):
            array.flags.writeable = False

    def _load(self, namespace):
        try:
            with self.cache_path.open('rb') as stream:
                with np.load(stream, allow_pickle=False) as data:
                    metadata = json.loads(str(data['metadata'].item()))
                    arrays = [data['raw_height_m'], data['height_m'], data['physical_ocean']]
            if metadata['namespace'] != namespace or metadata['cache_key'] != self.cache_key:
                return None
            for array, name, dtype in zip(arrays, ['raw_height', 'height', 'physical_ocean'], [np.dtype('float32'), np.dtype('float32'), np.dtype('bool')]):
                if array.shape != (self.height, self.width) or array.dtype != dtype or not np.isfinite(array).all():
                    return None
                if hashlib.sha256(array.tobytes()).hexdigest() != metadata[name + '_sha256']:
                    return None
            if not np.array_equal(arrays[0] < 0, arrays[1] < 0):
                return None
            return *arrays, metadata
        except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError, zipfile.BadZipFile):
            return None

    def _save(self, raw, final, physical_ocean, metadata):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.cache_path.parent, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, raw_height_m=raw, height_m=final, physical_ocean=physical_ocean,
                                metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
            stream.flush(); os.fsync(stream.fileno())
        try:
            os.replace(temporary, self.cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def sample_height_m(self, xs_metres, ys_metres):
        xs = np.asarray(xs_metres, dtype=np.float64)
        ys = np.asarray(ys_metres, dtype=np.float64)
        if xs.ndim != 1 or ys.ndim != 1 or not np.isfinite(xs).all() or not np.isfinite(ys).all():
            raise ValueError('Heightmap coordinates must be finite one-dimensional axes')
        x0, y0, x1, y1 = self.bounds
        xx = ((xs - x0) % (x1 - x0)) / (x1 - x0) * self.width - .5
        yy = np.clip((ys - y0) / (y1 - y0) * self.height - .5, 0, self.height - 1)
        ix = np.floor(xx).astype(np.int64); iy = np.floor(yy).astype(np.int64)
        tx = (xx - ix)[None, :]; ty = (yy - iy)[:, None]
        j0, j1 = ix % self.width, (ix + 1) % self.width
        i0, i1 = iy, np.minimum(iy + 1, self.height - 1)
        h = self.height_m
        a = h[i0[:, None], j0[None, :]] * (1 - tx) + h[i0[:, None], j1[None, :]] * tx
        b = h[i1[:, None], j0[None, :]] * (1 - tx) + h[i1[:, None], j1[None, :]] * tx
        return np.asarray(a * (1 - ty) + b * ty, dtype=np.float32)


@lru_cache(maxsize=4)
def get_heightmap(seed, style='earthlike'):
    return WorldHeightmap(seed, style)


def bootstrap_metadata(seed, style='earthlike'):
    return deepcopy(get_heightmap(seed, style).metadata)
