"""Real source distributions, units and durable content-addressed statistics.

Geographic generation lives in terrain_bootstrap; the five-channel adapter
lives in terrain_conditioning. This module supplies measured raster statistics.
"""
from __future__ import annotations

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path, RUNTIME_ROOT

from functools import lru_cache
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import rasterio

WORLD_VERSION = 'terrestrial-bootstrap-v1'
WORLD_BOUNDS_METERS = (-20_000_000., -10_000_000., 20_000_000., 10_000_000.)
WORLD_BOUNDS = WORLD_BOUNDS_METERS
COARSE_RESOLUTION = 30 * 256
DATA_ROOT = REPO_ROOT / 'terrain-diffusion' / 'data' / 'global'
SOURCE_FILES = ('etopo_10m.tif', 'wc2.1_10m_bio_1.tif',
                'wc2.1_10m_bio_4.tif', 'wc2.1_10m_bio_12.tif',
                'wc2.1_10m_bio_15.tif')
CHANNEL_NAMES = ('elevation_m', 'temperature_c', 'temperature_std_c_x100',
                 'precipitation_mm_year', 'precipitation_cv_percent')
SOURCE_STATS_CACHE_ROOT = RUNTIME_ROOT / 'source-statistics-cache'
SOURCE_STATS_CACHE_SCHEMA = 2
_STATS_ARRAY_KEYS = ('probabilities', 'elevation', 'land_elevation',
                     'latitudes', 'climate', 'noise_quantiles', 'sea_probability')


def _source_stats_identity():
    fingerprints = {}
    for name in (*SOURCE_FILES, 'synthetic_map_stats.json'):
        path = DATA_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f'Earth conditioning requires {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        fingerprints[name] = digest.hexdigest()
    implementation = inspect.getsource(_compute_source_distributions) + inspect.getsource(lapse_rate)
    identity = {'schema': SOURCE_STATS_CACHE_SCHEMA, 'sources': fingerprints,
                'source_order': list(SOURCE_FILES),
                'implementation': hashlib.sha256(implementation.encode()).hexdigest(),
                'numpy': np.__version__, 'rasterio': rasterio.__version__,
                'gdal': rasterio.__gdal_version__}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity, key


def _stats_array_manifest(arrays):
    return {key: {'shape': list(value.shape), 'dtype': value.dtype.str,
                  'sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}
            for key, value in arrays.items()}


def _read_source_stats_cache(path, identity):
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(_STATS_ARRAY_KEYS) | {'metadata'}:
                return None
            metadata = json.loads(archive['metadata'].item())
            arrays = {key: archive[key] for key in _STATS_ARRAY_KEYS}
        if not isinstance(metadata, dict):
            return None
        checksum = metadata.pop('sha256')
        if checksum != hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest():
            return None
        if metadata['identity'] != identity or metadata['arrays'] != _stats_array_manifest(arrays):
            return None
        shapes = {'probabilities': (65,), 'elevation': (65,), 'land_elevation': (65,),
                  'latitudes': (19,), 'climate': (4, 19, 65), 'sea_probability': ()}
        for key, value in arrays.items():
            expected_dtype = np.dtype(np.float64 if key in ('probabilities', 'sea_probability') else np.float32)
            if value.dtype != expected_dtype or not np.isfinite(value).all():
                return None
            if key in shapes and value.shape != shapes[key]:
                return None
        if arrays['noise_quantiles'].ndim != 2 or arrays['noise_quantiles'].shape[0] < 4:
            return None
        if not 0 <= arrays['sea_probability'].item() <= 1:
            return None
        arrays['sea_probability'] = float(arrays['sea_probability'])
        fingerprints = identity['sources']
        arrays.update(source_hashes=fingerprints,
                      source_digest=hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()[:16])
        return arrays
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return None


def _write_source_stats_cache(path, identity, stats):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray(stats[key]) for key in _STATS_ARRAY_KEYS}
        metadata = {'identity': identity, 'arrays': _stats_array_manifest(arrays)}
        metadata['sha256'] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, metadata=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        # A read-only or unavailable runtime drive must not prevent generation.
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def latitude(y_metres):
    """A flat equirectangular climate convention; no spherical topology."""
    return np.clip(-np.asarray(y_metres, dtype=np.float64) * 90 / 10_000_000, -90, 90)


def lapse_rate(precipitation):
    return np.clip(-6.5 + .0015 * precipitation, -9.8, -4.) / 1000


def _smoothstep(a, b, x):
    t = np.clip((x-a)/(b-a), 0, 1)
    return t*t*(3-2*t)


@lru_cache(maxsize=1)
def source_distributions():
    """Return exact source statistics, using a content-addressed durable cache.

    TERRAIN_SOURCE_STATS_CACHE_DISABLE=1 bypasses disk reads and writes. The
    existing in-process cache remains active; clear it to force a baseline.
    """
    identity, key = _source_stats_identity()
    disabled = os.environ.get('TERRAIN_SOURCE_STATS_CACHE_DISABLE') == '1'
    path = SOURCE_STATS_CACHE_ROOT / f'{key}.npz'
    if not disabled:
        cached = _read_source_stats_cache(path, identity)
        if cached is not None:
            return cached
    stats = _compute_source_distributions(identity['sources'])
    if not disabled:
        _write_source_stats_cache(path, identity, stats)
    return stats


def _compute_source_distributions(fingerprints):
    """Use full rasters, including poles omitted by the upstream default stats.

    BIO4 here is the original monthly temperature standard deviation *100, not
    the temperature-regression residual stored in synthetic_map_stats.json.
    Temperature is lifted to sea level before deriving latitude distributions,
    then brought to the procedural elevation with the upstream lapse formula.
    Longitude is not copied: only quantiles within absolute-latitude bands are
    retained, so worlds are varied and both hemispheres have polar climates.
    """
    rasters = []
    for name in SOURCE_FILES:
        path = DATA_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f'Earth conditioning requires {path}')
        with rasterio.open(path) as dataset:
            data = dataset.read(1).astype(np.float32)
            rows = np.arange(dataset.height)
            row_lat = dataset.transform.f + (rows+.5)*dataset.transform.e
        data[data < -30000] = np.nan
        rasters.append(data)
    elev, temp, temp_std, precip, precip_cv = rasters
    temp -= lapse_rate(precip) * np.maximum(elev, 0)
    probabilities = np.linspace(.0001, .9999, 65)
    # Shared ETOPO ocean/land fraction replaces the water-dropping default: the
    # original default deletes half the ocean in its empirical histogram.
    elevation_quantiles = np.nanquantile(elev, probabilities).astype(np.float32)
    land_quantiles = np.quantile(elev[elev >= 0], probabilities).astype(np.float32)
    sea_probability = float(np.mean(elev[np.isfinite(elev)] < 0))
    lat_nodes = np.arange(0, 91, 5, dtype=np.float32)
    climate_quantiles = np.empty((4, len(lat_nodes), len(probabilities)), np.float32)
    for k, centre in enumerate(lat_nodes):
        selection = np.abs(np.abs(row_lat)-centre) < 5
        for c, data in enumerate((temp, temp_std, precip, precip_cv)):
            values = data[selection]
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError(f'No WorldClim samples for channel {c} at {centre} degrees')
            climate_quantiles[c, k] = np.quantile(values, probabilities)
    stats_path = DATA_ROOT / 'synthetic_map_stats.json'
    stats = json.loads(stats_path.read_text(encoding='utf-8'))
    return {'probabilities': probabilities, 'elevation': elevation_quantiles,
            'land_elevation': land_quantiles, 'sea_probability': sea_probability,
            'latitudes': lat_nodes, 'climate': climate_quantiles,
            'noise_quantiles': np.asarray(stats['noise_quantile_tables'], np.float32),
            'source_hashes': fingerprints,
            'source_digest': hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()[:16]}



def profile_metadata():
    stats = source_distributions()
    return {'version': WORLD_VERSION, 'bounds': WORLD_BOUNDS,
            'source_digest': stats['source_digest'], 'source_hashes': stats['source_hashes'],
            'source_channels': list(CHANNEL_NAMES), 'coarse_resolution': COARSE_RESOLUTION}
