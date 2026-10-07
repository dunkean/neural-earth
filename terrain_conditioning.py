"""Physical conditioning from the native bootstrap, with protected Natural math.

The bootstrap owns all height geometry and metre calibration. Climate uses
fixed source distributions; sampling a different window never renormalizes it.
"""
from __future__ import annotations

from functools import lru_cache
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import torch
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, NoiseType, FractalType

from terrain_world import source_distributions, latitude, lapse_rate
from terrain_generation import resolve_generation, GENERATION_VERSION

BOOTSTRAP_CONDITIONING_VERSION = "native-bootstrap-worldclim-v2"
WORLD_PROFILES = ("natural", "terrestrial-gondwana", "terrestrial-continents",
                  "terrestrial-earthlike", "terrestrial-archipelago")
COARSE_RESOLUTION = 7680
WORLD_BOUNDS_METERS = (-20_000_000., -10_000_000., 20_000_000., 10_000_000.)
WORLD_BOUNDS = WORLD_BOUNDS_METERS
CONDITIONING_SNR = (.05, .5, .5, .5, .5)
CHANNEL_NAMES = ("elevation_m", "temperature_c", "temperature_std_c_x100",
                 "precipitation_mm_year", "precipitation_cv_percent")
STATS_PATH = Path(__file__).resolve().parent / "terrain-diffusion" / "data" / "global" / "synthetic_map_stats.json"


def _noise(seed: int, frequency: float, octaves: int) -> FastNoiseLite:
    n = FastNoiseLite(seed=int(seed) & 0x7fffffff)
    n.noise_type = NoiseType.NoiseType_Perlin
    n.frequency = frequency
    n.fractal_type = FractalType.FractalType_FBm
    n.fractal_octaves = octaves
    n.fractal_lacunarity = 2.
    n.fractal_gain = .5
    return n


@lru_cache(maxsize=1)
def _natural_stats() -> dict:
    data = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    if len(data["noise_quantile_tables"]) != 5 or len(data["data_quantile_tables"]) != 5:
        raise ValueError("Natural statistics require five channels")
    return data


class NaturalConditioning:
    """Protected Natural preview baseline, including the seed-0 fix."""

    def __init__(self, seed: int):
        self.seed = int(seed) & 0xffffffffffffffff
        self.stats = _natural_stats()
        self.noises = [_noise(self.seed + i + 1, .05, 2 if i == 1 else 4) for i in range(5)]

    def sample_raw(self, cj0, ci0, cj1, ci1):
        x, y = np.meshgrid(np.arange(cj0, cj1, dtype=np.float32),
                           np.arange(ci0, ci1, dtype=np.float32))
        coords = np.ascontiguousarray(np.stack((x.ravel(), y.ravel())), dtype=np.float32)
        result = []
        for i, noise in enumerate(self.noises):
            values = noise.gen_from_coords(coords)
            values = np.interp(values, self.stats["noise_quantile_tables"][i],
                               self.stats["data_quantile_tables"][i]).reshape(x.shape)
            result.append(values.astype(np.float32))
        return np.stack(result)

    def sample(self, xs_metres, ys_metres):
        xs = np.asarray(xs_metres, np.float64)
        ys = np.asarray(ys_metres, np.float64)
        if xs.ndim != 1 or ys.ndim != 1:
            raise ValueError("sample expects one-dimensional x/y arrays")
        out = np.empty((5, len(ys), len(xs)), np.float32)
        rows = max(1, 131072//max(1, len(xs)))
        for start in range(0, len(ys), rows):
            x, y = np.meshgrid(xs/COARSE_RESOLUTION-.5,
                               ys[start:start+rows]/COARSE_RESOLUTION-.5)
            coords = np.ascontiguousarray(np.stack((x.ravel(), y.ravel())), dtype=np.float32)
            raw = np.stack([np.interp(n.gen_from_coords(coords),
                                      self.stats["noise_quantile_tables"][i],
                                      self.stats["data_quantile_tables"][i]).reshape(x.shape).astype(np.float32)
                            for i, n in enumerate(self.noises)])
            out[:, start:start+rows] = self.finalize(raw)
        return out

    def finalize(self, raw):
        e, t, tstd, p, pcv = [np.asarray(v, np.float32).copy() for v in raw]
        s = self.stats
        t += lapse_rate(p).astype(np.float32) * np.maximum(e, 0)
        t = np.clip(t, -10, 40)
        t = np.where(t > 20, t, (t-20)*1.25+20)
        frac = (tstd-float(s["temp_std_p1"]))/(float(s["temp_std_p99"])-float(s["temp_std_p1"]))
        baseline = np.maximum(float(s["temp_std_p1"]),
                              -(float(s["a_temp_std"])*t+float(s["b_temp_std"])))
        tstd = np.maximum(frac*(float(s["temp_std_p99"])-baseline)+baseline+
                          float(s["a_temp_std"])*t+float(s["b_temp_std"]), 20)
        pcv *= np.maximum(0, (185-.04111*p)/185)
        return np.stack((e, t, tstd, p, pcv)).astype(np.float32)

    def __call__(self, cj0, ci0, cj1, ci1):
        fields = self.finalize(self.sample_raw(cj0, ci0, cj1, ci1))
        fields[0] = np.sign(fields[0])*np.sqrt(np.abs(fields[0]))
        return torch.from_numpy(fields)


def _coordinates(xs_metres, ys_metres):
    xs = np.asarray(xs_metres, np.float64)
    ys = np.asarray(ys_metres, np.float64)
    if xs.ndim != 1 or ys.ndim != 1:
        raise ValueError("sample expects one-dimensional x/y arrays")
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        raise ValueError("sample coordinates must be finite")
    return xs, ys


def _bootstrap_noise_seed(seed: int, channel: int) -> int:
    """Mix a full u64 seed and climate domain before FastNoiseLite reduction.

    SplitMix64's fixed integer avalanche avoids the systematic high-bit alias
    of truncating world seeds first. Each climate channel has its own domain.
    """
    mask = 0xffffffffffffffff
    value = (int(seed) & mask) ^ (0x434c494d41544500 | int(channel))
    value = (value + 0x9e3779b97f4a7c15) & mask
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
    return (value ^ (value >> 31)) & 0x7fffffff


class BootstrapConditioning:
    """Native physical height plus latitude-calibrated WorldClim channels.

    Climate is a coherent climatological baseline, not a circulation model.
    BIO1 receives the upstream precipitation-dependent lapse adjustment. BIO4
    is the actual source seasonality, not Natural's regression residual.
    """

    def __init__(self, seed: int, style="earthlike", heightmap=None, stats=None):
        if style not in ("gondwana", "continents", "earthlike", "archipelago"):
            raise ValueError(f"Unknown terrestrial style: {style}")
        self.seed = int(seed) & 0xffffffffffffffff
        self.style = style
        if heightmap is None:
            from terrain_bootstrap import get_heightmap
            heightmap = get_heightmap(self.seed, style)
        self.heightmap = heightmap
        self.stats = source_distributions() if stats is None else stats
        self.probabilities = np.asarray(self.stats["probabilities"], np.float64)
        self.latitudes = np.asarray(self.stats["latitudes"], np.float64)
        self.climate_quantiles = np.asarray(self.stats["climate"], np.float64)
        self.noise_quantiles = np.asarray(self.stats["noise_quantiles"], np.float64)
        if (self.probabilities.ndim != 1 or self.latitudes.ndim != 1
                or self.climate_quantiles.shape != (4, len(self.latitudes), len(self.probabilities))
                or self.noise_quantiles.ndim != 2 or self.noise_quantiles.shape[0] != 5
                or self.noise_quantiles.shape[1] < 2
                or len(self.probabilities) < 2 or len(self.latitudes) < 2
                or not np.isfinite(self.probabilities).all()
                or not np.isfinite(self.latitudes).all()
                or not np.isfinite(self.noise_quantiles).all()
                or not (0 <= self.probabilities[0] < self.probabilities[-1] <= 1)
                or not (0 <= self.latitudes[0] < self.latitudes[-1] <= 90)
                or not np.all(np.diff(self.probabilities) > 0)
                or not np.all(np.diff(self.latitudes) > 0)
                or not np.all(np.diff(self.noise_quantiles, axis=1) > 0)
                or not np.all(np.diff(self.climate_quantiles, axis=2) >= 0)
                or not np.isfinite(self.climate_quantiles).all()):
            raise ValueError("Invalid fixed source climate statistics")
        # Separate RNG streams and physical wavelengths, independent of height
        # style, request extent, row batching and raster traversal.
        configs = ((1, .004, 2), (2, .006, 4), (3, .009, 4), (4, .005, 4))
        self.noises = [_noise(_bootstrap_noise_seed(self.seed, i), freq, octaves)
                       for i, freq, octaves in configs]
        self.noise_tables = [self.noise_quantiles[i] for i, _, _ in configs]

    def _quantile(self, channel, absolute_latitude, probability):
        """Bilinearly sample a fixed latitude/probability quantile surface."""
        lat = np.broadcast_to(absolute_latitude, probability.shape)
        hi = np.clip(np.searchsorted(self.latitudes, lat, side="right"), 1, len(self.latitudes) - 1)
        lo = hi - 1
        frac = np.clip((lat - self.latitudes[lo]) / (self.latitudes[hi] - self.latitudes[lo]), 0., 1.)
        table = self.climate_quantiles[channel]
        output = np.empty(probability.shape, np.float64)
        # At most 19 latitude bands; no enormous per-pixel quantile tensor.
        for k in np.unique(lo):
            mask = lo == k
            low_values = np.interp(probability[mask], self.probabilities, table[k])
            high_values = np.interp(probability[mask], self.probabilities, table[k + 1])
            output[mask] = low_values + frac[mask] * (high_values - low_values)
        return output.astype(np.float32)

    def _climate(self, height_m, xs, ys):
        x, y = np.meshgrid(xs / COARSE_RESOLUTION - .5, ys / COARSE_RESOLUTION - .5)
        coords = np.ascontiguousarray(np.stack((x.ravel(), y.ravel())), dtype=np.float32)
        lat = np.abs(latitude(ys))[:, None]
        fields = []
        for channel, (noise, table) in enumerate(zip(self.noises, self.noise_tables)):
            values = noise.gen_from_coords(coords)
            # Fixed empirical noise ranks. No min/max, CDF or quantile is
            # estimated from this seed or this requested patch.
            ranks = np.interp(values, table, np.linspace(.0001, .9999, len(table))).reshape(x.shape)
            fields.append(self._quantile(channel, lat, ranks))
        temperature, seasonality, precipitation, precipitation_cv = fields
        temperature += lapse_rate(precipitation).astype(np.float32) * np.maximum(height_m, 0)
        return np.stack((temperature, seasonality, precipitation, precipitation_cv))

    def sample(self, xs_metres, ys_metres):
        """Return five float32 physical fields on a Cartesian coordinate grid."""
        xs, ys = _coordinates(xs_metres, ys_metres)
        output = np.empty((5, len(ys), len(xs)), np.float32)
        if not len(xs) or not len(ys):
            return output
        rows = max(1, 131072 // len(xs))
        for start in range(0, len(ys), rows):
            row_y = ys[start:start + rows]
            height = np.asarray(self.heightmap.sample_height_m(xs, row_y), np.float32)
            if height.shape != (len(row_y), len(xs)) or not np.isfinite(height).all():
                raise ValueError("Bootstrap must provide finite metre heights with shape (y, x)")
            output[0, start:start + rows] = height
            output[1:, start:start + rows] = self._climate(height, xs, row_y)
        return output

    def sample_raw(self, cj0, ci0, cj1, ci1):
        xs = (np.arange(cj0, cj1, dtype=np.float64) + .5) * COARSE_RESOLUTION
        ys = (np.arange(ci0, ci1, dtype=np.float64) + .5) * COARSE_RESOLUTION
        return self.sample(xs, ys)

    @staticmethod
    def finalize(raw):
        """Bootstrap samples are already calibrated physical channels."""
        return np.asarray(raw, np.float32).copy()

    def __call__(self, cj0, ci0, cj1, ci1):
        fields = self.sample_raw(cj0, ci0, cj1, ci1)
        fields[0] = np.sign(fields[0]) * np.sqrt(np.abs(fields[0]))
        return torch.from_numpy(fields)


def _weighted_height_quantiles(values, probabilities, drop_water_pct):
    """Deterministic empirical ETOPO distribution, with reduced ocean weight.

    Only nondefault Natural water controls use this approximation. The protected
    .5 baseline keeps its original upstream table, including original rounding.
    """
    values = np.asarray(values, np.float64).ravel()
    values = values[np.isfinite(values) & (values > -30000)]
    weights = np.where(values < 0, 1. - drop_water_pct, 1.)
    values, weights = values[weights > 0], weights[weights > 0]
    if not len(values):
        raise ValueError("No weighted valid ETOPO heights")
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    positions = (np.cumsum(weights) - .5 * weights) / weights.sum()
    return np.interp(probabilities, positions, values)


@lru_cache(maxsize=8)
def _drop_height_quantiles(drop, source_digest):
    import rasterio
    path = STATS_PATH.parent / "etopo_10m.tif"
    with rasterio.open(path) as dataset:
        ys = dataset.transform.f + (np.arange(dataset.height) + .5) * dataset.transform.e
        values = dataset.read(1, masked=True).astype(np.float32).filled(np.nan)[np.abs(ys) <= 60]
    result = _weighted_height_quantiles(values, np.linspace(.0001, .9999, 64), drop)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=8)
def _height_source_digest(path, size, mtime_ns):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TunableNaturalConditioning(NaturalConditioning):
    """Original Natural transforms, with explicit noise/water controls."""
    def __init__(self, seed, settings):
        super().__init__(seed)
        self.noises = [_noise(self.seed + i + 1, .05 * settings["frequency_mult"][i],
                              settings["octaves"][i]) for i in range(5)]
        if settings["drop_water_pct"] != .5:
            path = STATS_PATH.parent / "etopo_10m.tif"
            stat = path.stat()
            digest = _height_source_digest(str(path), stat.st_size, stat.st_mtime_ns)
            self.stats = dict(self.stats)
            tables = list(self.stats["data_quantile_tables"])
            tables[0] = _drop_height_quantiles(settings["drop_water_pct"], digest)
            self.stats["data_quantile_tables"] = tables

    def finalize(self, raw):
        # Preserve installed upstream grouping, including its BIO4 float32
        # additions. The protected default preview class above stays untouched.
        e, t, tstd, p, pcv = [np.asarray(v, np.float32) for v in raw]
        s = self.stats
        t = t + lapse_rate(p) * np.maximum(0, e)
        t = np.clip(t, -10, 40)
        t = np.where(t > 20, t, (t - 20) * 1.25 + 20)
        frac = (tstd - float(s["temp_std_p1"])) / (float(s["temp_std_p99"]) - float(s["temp_std_p1"]))
        baseline = np.maximum(float(s["temp_std_p1"]), -(float(s["a_temp_std"]) * t + float(s["b_temp_std"])))
        tstd = frac * (float(s["temp_std_p99"]) - baseline) + baseline
        tstd = tstd + (float(s["a_temp_std"]) * t + float(s["b_temp_std"]))
        tstd = np.maximum(tstd, 20)
        pcv = pcv * np.maximum(0, (185 - .04111 * p) / 185)
        return np.stack((e, t, tstd, p, pcv))

    def sample_raw_coordinates(self, xs, ys):
        x, y = np.meshgrid(xs / COARSE_RESOLUTION - .5, ys / COARSE_RESOLUTION - .5)
        coords = np.ascontiguousarray(np.stack((x.ravel(), y.ravel())), dtype=np.float32)
        return np.stack([np.interp(noise.gen_from_coords(coords),
                                  self.stats["noise_quantile_tables"][i],
                                  self.stats["data_quantile_tables"][i]).reshape(x.shape).astype(np.float32)
                         for i, noise in enumerate(self.noises)])


_MACRO_CACHE = OrderedDict()
_MACRO_LOCK = threading.RLock()
_MACRO_CACHE_SIZE = 4
_MACRO_SHAPE = (512, 1024)


def _sample_fixed_raster(raster, xs, ys):
    """Periodic longitude, clamped latitude; fixed pixel-centre coordinates."""
    x0, y0, x1, y1 = WORLD_BOUNDS
    height, width = raster.shape
    xx = ((xs - x0) % (x1 - x0)) / (x1 - x0) * width - .5
    yy = np.clip((ys - y0) / (y1 - y0) * height - .5, 0, height - 1)
    ix, iy = np.floor(xx).astype(np.int64), np.floor(yy).astype(np.int64)
    tx, ty = (xx - ix)[None, :], (yy - iy)[:, None]
    a = raster[iy[:, None], (ix % width)[None, :]] * (1 - tx) + raster[iy[:, None], ((ix + 1) % width)[None, :]] * tx
    by = np.minimum(iy + 1, height - 1)
    b = raster[by[:, None], (ix % width)[None, :]] * (1 - tx) + raster[by[:, None], ((ix + 1) % width)[None, :]] * tx
    return np.asarray(a * (1 - ty) + b * ty, np.float32)


class GenerationConditioning:
    """Mix source fields before their one final physical/NN transform.

    Continental height replaces only a global Gaussian lowpass in signed-root
    space. Macro scale is sigma, not a cutoff or a guarantee of realistic coast,
    drainage, or learned detail. Original Natural regional residuals remain.
    """
    def __init__(self, seed, descriptor):
        self.seed = int(seed) & 0xffffffffffffffff
        self._terrain_generation_profile = descriptor.profile
        self.generation_settings = descriptor.settings
        self.descriptor = descriptor
        self.natural = (TunableNaturalConditioning(seed, descriptor.settings)
                        if descriptor.settings["height_source"] != "native" or descriptor.settings["climate_source"] == "natural" else None)
        self.native = None
        if descriptor.needs_bootstrap:
            from terrain_bootstrap import get_heightmap
            self.heightmap = get_heightmap(self.seed, descriptor.bootstrap_style)
        if descriptor.settings["climate_source"] == "native":
            self.native = BootstrapConditioning(seed, descriptor.bootstrap_style, heightmap=self.heightmap)
            base_frequencies = (.004, .006, .009, .005)
            s = descriptor.settings
            self.native.noises = [_noise(_bootstrap_noise_seed(self.seed, i + 1),
                                         freq * s["frequency_mult"][i + 1], s["octaves"][i + 1])
                                  for i, freq in enumerate(base_frequencies)]

    def _macro_delta(self):
        receipt = getattr(self.heightmap, "metadata", {})
        # Include actual native height identity, not only a mutable object id.
        key = (GENERATION_VERSION, self.seed, self.descriptor.profile,
               receipt.get("height_sha256", id(self.heightmap)))
        with _MACRO_LOCK:
            if key in _MACRO_CACHE:
                _MACRO_CACHE.move_to_end(key)
                return _MACRO_CACHE[key]
            from scipy.ndimage import gaussian_filter
            h, w = _MACRO_SHAPE
            x0, y0, x1, y1 = WORLD_BOUNDS
            xs = x0 + (np.arange(w) + .5) * (x1 - x0) / w
            ys = y0 + (np.arange(h) + .5) * (y1 - y0) / h
            raw = self.natural.sample_raw_coordinates(xs, ys)[0]
            native = np.asarray(self.heightmap.sample_height_m(xs, ys), np.float32)
            u = np.sign(raw) * np.sqrt(np.abs(raw))
            v = np.sign(native) * np.sqrt(np.abs(native))
            sigma_m = self.descriptor.settings["macro_scale_km"] * 1000.
            sigma = (sigma_m / ((y1 - y0) / h), sigma_m / ((x1 - x0) / w))
            delta = gaussian_filter(v, sigma, mode=("nearest", "wrap")) - gaussian_filter(u, sigma, mode=("nearest", "wrap"))
            delta = np.asarray(delta, np.float32)
            delta.flags.writeable = False
            _MACRO_CACHE[key] = delta
            while len(_MACRO_CACHE) > _MACRO_CACHE_SIZE:
                _MACRO_CACHE.popitem(last=False)
            return delta

    def _raw_coordinates(self, xs, ys):
        raw = (self.natural.sample_raw_coordinates(xs, ys) if self.natural is not None
               else np.empty((5, len(ys), len(xs)), np.float32))
        s = self.descriptor.settings
        if s["height_source"] == "native":
            height = np.asarray(self.heightmap.sample_height_m(xs, ys), np.float32)
            if height.shape != (len(ys), len(xs)):
                raise ValueError("Height source must provide metre heights with shape (y, x)")
            raw[0] = height
        elif s["height_source"] == "natural-continental" and s["continental_strength"] != 0:
            u = np.sign(raw[0]) * np.sqrt(np.abs(raw[0]))
            u += s["continental_strength"] * _sample_fixed_raster(self._macro_delta(), xs, ys)
            raw[0] = np.sign(u) * np.square(u)
        if not np.isfinite(raw[0]).all():
            raise ValueError("Height source must provide finite metre heights")
        if s["climate_source"] == "native":
            # Native climate applies its lapse once at the selected final height.
            raw[1:] = self.native._climate(raw[0], xs, ys)
        return raw

    def finalize(self, raw):
        if self.descriptor.settings["climate_source"] == "natural":
            return self.natural.finalize(raw)
        return np.asarray(raw, np.float32).copy()

    def sample(self, xs_metres, ys_metres):
        xs, ys = _coordinates(xs_metres, ys_metres)
        result = np.empty((5, len(ys), len(xs)), np.float32)
        if not len(xs) or not len(ys):
            return result
        rows = max(1, 131072 // max(1, len(xs)))
        for start in range(0, len(ys), rows):
            result[:, start:start + rows] = self.finalize(self._raw_coordinates(xs, ys[start:start + rows]))
        return result

    def sample_raw(self, cj0, ci0, cj1, ci1):
        xs = (np.arange(cj0, cj1, dtype=np.float64) + .5) * COARSE_RESOLUTION
        ys = (np.arange(ci0, ci1, dtype=np.float64) + .5) * COARSE_RESOLUTION
        return self._raw_coordinates(xs, ys)

    def __call__(self, cj0, ci0, cj1, ci1):
        fields = self.finalize(self.sample_raw(cj0, ci0, cj1, ci1))
        fields[0] = np.sign(fields[0]) * np.sqrt(np.abs(fields[0]))
        return torch.from_numpy(fields)


@lru_cache(maxsize=12)
def _conditioning_factory(seed, world_profile):
    descriptor = resolve_generation(world_profile)
    if not descriptor.is_default:
        return GenerationConditioning(seed, descriptor)
    if world_profile == "natural":
        factory = NaturalConditioning(seed)
    else:
        factory = BootstrapConditioning(seed, style=descriptor.bootstrap_style)
    factory._terrain_generation_profile = world_profile
    factory.generation_settings = descriptor.settings
    return factory


def make_conditioning_factory(seed: int, world_profile="natural"):
    # Resolve even when the expensive field factory is cached: tamper is fatal.
    resolve_generation(world_profile)
    return _conditioning_factory(seed, world_profile)


make_conditioning_factory.cache_clear = _conditioning_factory.cache_clear


def sample_conditioning_preview(seed, profile, xs, ys):
    """Transport physical BIO1 at terrain height; include lapse as climate[4].

    The server reconstructs sea-level BIO1 by subtracting lapse * land height
    before rendering at any displayed elevation. No square-root encoding here.
    """
    fields = make_conditioning_factory(seed, profile).sample(xs, ys)
    climate = np.concatenate((fields[1:], lapse_rate(fields[3])[None]), axis=0).astype(np.float32)
    return {"fields": fields, "elev": fields[0], "climate": climate,
            "stage": "conditioning"}
