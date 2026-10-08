"""Canonical, persistent generation settings; resolving a token never writes.

Macro scale is the Gaussian standard deviation in kilometres, not a promised
landform size or a quality preset. Continental replacement is a soft prototype.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from terrain_geometry import DEFAULT_DIAMETER_KM

GENERATION_VERSION = "terrain-generation-v1"
REGISTRY_ROOT = Path(os.environ.get("TERRAIN_GENERATION_ROOT", "E:/TerrainDiffusionRuntime/generation-settings"))
BASE_PROFILES = ("natural", "orogen", "terrestrial-gondwana", "terrestrial-continents",
                 "terrestrial-earthlike", "terrestrial-archipelago")
STYLES = ("gondwana", "continents", "earthlike", "archipelago")
OROGEN_PARAMETERS = {
    'orogen_plate_count': {'minimum':4, 'maximum':120, 'integer':True, 'default':80},
    'orogen_continent_count': {'minimum':1, 'maximum':10, 'integer':True, 'default':4},
    'orogen_land_coverage': {'minimum':.05, 'maximum':.9, 'default':.30},
    'orogen_continent_variety': {'minimum':0., 'maximum':1., 'default':.35},
    'orogen_motion_strength': {'minimum':0., 'maximum':4., 'default':1.},
    'orogen_convergence_threshold': {'minimum':.05, 'maximum':4., 'default':.75},
    'orogen_detail': {'minimum':20000, 'maximum':1000000, 'integer':True, 'default':204000},
    'orogen_spread': {'minimum':1., 'maximum':12., 'default':5.},
    'orogen_roughness': {'minimum':0., 'maximum':1., 'default':.4},
    'orogen_warp': {'minimum':0., 'maximum':1., 'default':.75},
    'orogen_smoothing': {'minimum':0., 'maximum':1., 'default':.1},
    'orogen_hydraulic': {'minimum':0., 'maximum':1., 'default':.5},
    'orogen_thermal': {'minimum':0., 'maximum':1., 'default':.1},
    'orogen_glacial': {'minimum':0., 'maximum':1., 'default':.5},
    'orogen_sharpening': {'minimum':0., 'maximum':1., 'default':.5},
    'orogen_temperature_offset': {'minimum':-15., 'maximum':15., 'default':0.},
    'orogen_precipitation_offset': {'minimum':-1., 'maximum':1., 'default':0.},
}
OROGEN_CLIMATE_PARAMETERS = json.loads((Path(__file__).parent / 'native' / 'orogen' / 'climate-parameters.json').read_text(encoding='utf-8'))
OROGEN_PARAMETERS.update(OROGEN_CLIMATE_PARAMETERS)
CITY_EROSION_PARAMETERS = {
    'city_erosion_strength': {'minimum': 0., 'maximum': 2., 'default': 1.},
    'city_erosion_iterations': {'minimum': 1, 'maximum': 64, 'integer': True, 'default': 12},
    'city_erosion_talus': {'minimum': .01, 'maximum': 4., 'default': .6},
    'city_erosion_motif_km': {'minimum': 10., 'maximum': 5000., 'default': 300.},
}
OROGEN_GPU_PARAMETERS = {
    'orogen_gpu_relief': {'type': 'boolean', 'default': False},
    'orogen_gpu_propagation': {'type': 'boolean', 'default': False},
    'orogen_gpu_post': {'type': 'boolean', 'default': False},
    'orogen_gpu_climate': {'type': 'boolean', 'default': False},
    'orogen_gpu_erosion': {'type': 'boolean', 'default': False},
    'orogen_gpu_raster': {'type': 'boolean', 'default': False},
}
OROGEN_STAGE_PARAMETERS = {
    **{f'orogen_{stage}_stage': {'type': 'string', 'pattern': r'(?:[0-9a-f]{64})?', 'default': ''}
       for stage in ('relief', 'erosion', 'climate')},
    'orogen_height_stage': {'enum': ['relief', 'erosion'], 'default': 'erosion'},
}


def orogen_defaults(style='earthlike'):
    values = {key:list(spec['default']) if isinstance(spec['default'],list) else spec['default'] for key,spec in OROGEN_PARAMETERS.items()}
    plates, continents, coverage = {'gondwana':(80,1,.3), 'continents':(80,5,.3),
        'earthlike':(80,4,.3), 'archipelago':(110,8,.18)}[style]
    values.update(orogen_plate_count=plates, orogen_continent_count=continents,
                  orogen_land_coverage=coverage)
    return values


def _defaults(base):
    if base not in BASE_PROFILES:
        raise ValueError(f"Unknown base profile: {base}")
    natural = base == "natural"
    settings = dict(world_diameter_km=DEFAULT_DIAMETER_KM, world_topology='sphere',
                height_source="natural" if natural else "orogen" if base == "orogen" else "native",
                climate_source="orogen" if base == "orogen" else "natural" if natural else "native",
                relief_pipeline="orogen" if base == "orogen" else "original",
                continental_style="earthlike" if natural or base == "orogen" else base.removeprefix("terrestrial-"),
                continental_strength=.8, macro_scale_km=600.,
                frequency_mult=[1.] * 5, octaves=[4, 2, 4, 4, 4],
                cond_snr=[.5] * 5 if natural else [.05, .5, .5, .5, .5],
                snr_adaptive_enabled=True,snr_detail_mode='global',snr_altitude_gain=[1.] * 5, snr_altitude_range_m=[500., 3000.],
                snr_driver_channel='temperature', snr_driver_gain=[1.] * 5,
                snr_driver_range=[5., -10.], snr_bins=16,
                snr_latitude_gain=1., snr_latitude_range=[0., 75.],
                snr_lod=[0.] * 15,
                drop_water_pct=.5)
    settings.update(orogen_defaults(settings['continental_style']))
    settings.update({key: spec['default'] for key, spec in CITY_EROSION_PARAMETERS.items()})
    settings.update({key: spec['default'] for key, spec in OROGEN_GPU_PARAMETERS.items()})
    settings.update({key: spec['default'] for key, spec in OROGEN_STAGE_PARAMETERS.items()})
    return settings


def generator_schema():
    """Return JSON-safe control metadata and a fresh defaults dictionary."""
    from terrain_orogen_stages import STAGE_KEYS
    groups={name:sorted(keys) for name,keys in STAGE_KEYS.items()}
    groups['settings']=sorted(set(_defaults('orogen'))-set().union(*STAGE_KEYS.values())-set(OROGEN_STAGE_PARAMETERS))
    return {"version": GENERATION_VERSION, "stage_groups":groups,"defaults_by_profile": {p: _defaults(p) for p in BASE_PROFILES},
            "properties": {
                "world_diameter_km": {"minimum": 10., "maximum": 100000., "default": DEFAULT_DIAMETER_KM},
                "world_topology": {"enum": ["sphere", "plane"], "default": "sphere"},
                "height_source": {"enum": ["natural", "native", "orogen", "natural-continental"]},
                "climate_source": {"enum": ["natural", "native", "orogen"]},
                "relief_pipeline": {"enum": ["original", "orogen", "city-gpu"]},
                "continental_style": {"enum": list(STYLES)},
                "continental_strength": {},
                "macro_scale_km": {"minimum": 0., "description": "Gaussian sigma in km; soft continental constraint"},
                "frequency_mult": {"length": 5},
                "octaves": {"length": 5, "minimum": 1, "integer": True},
                "cond_snr": {"length": 5, "exclusiveMinimum": 0., "description": "Noise/signal amplitude; larger permits more learned correction"},
                "snr_adaptive_enabled": {"type":"boolean","default":True},
                "snr_detail_mode": {"enum":['global','per-lod']},
                "snr_altitude_gain": {"length": 5, "minimum": .03125, "maximum": 32., "description": "Per-channel multiplier at the high altitude threshold; 1 disables"},
                "snr_altitude_range_m": {"length": 2, "description": "Increasing land altitude thresholds in metres"},
                "snr_driver_channel": {"enum": ["temperature", "temperature_variation", "precipitation", "precipitation_variation"]},
                "snr_driver_gain": {"length": 5, "minimum": .03125, "maximum": 32., "description": "Per-channel multiplier at the second driver threshold; 1 disables"},
                "snr_driver_range": {"length": 2, "description": "Driver thresholds; decreasing values allow stronger noise in cold climates"},
                "snr_bins": {"minimum": 2, "maximum": 32, "integer": True},
                "snr_latitude_gain": {"minimum": .03125, "maximum": 32.},
                "snr_latitude_range": {"length": 2, "minimum": 0., "maximum": 90.},
                "snr_lod": {"length": 15, "minimum": 0., "description": "Relief noise by display LOD -3 through 11; 0 inherits conditioning SNR. Scales the local relief residual after NN reconstruction."},
                "drop_water_pct": {"minimum": 0., "maximum": 1., "description": "Natural height distribution: removed ocean weight; 1 means land only"},
                **{k:dict(v) for k,v in OROGEN_PARAMETERS.items()},
                **{k:dict(v) for k,v in CITY_EROSION_PARAMETERS.items()},
                **{k:dict(v) for k,v in OROGEN_GPU_PARAMETERS.items()},
                **{k:dict(v) for k,v in OROGEN_STAGE_PARAMETERS.items()}}}


def _number(value, key, spec):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{key} must be a finite number")
    integer = spec.get("integer", False)
    if integer and int(value) != value:
        raise ValueError(f"{key} must be an integer")
    if "minimum" in spec and value < spec["minimum"]:
        raise ValueError(f"{key} must be at least {spec['minimum']}")
    if "exclusiveMinimum" in spec and value <= spec["exclusiveMinimum"]:
        raise ValueError(f"{key} must be greater than {spec['exclusiveMinimum']}")
    if "maximum" in spec and value > spec["maximum"]:
        raise ValueError(f"{key} must be at most {spec['maximum']}")
    return int(value) if integer else (0. if value == 0 else float(value))


def _normalize(base, overrides):
    settings = _defaults(base)
    if overrides is None:
        return settings
    if not isinstance(overrides, dict):
        raise ValueError("Generation overrides must be an object")
    if set(overrides) - set(settings):
        raise ValueError(f"Unknown generation keys: {sorted(set(overrides) - set(settings), key=str)}")
    if 'continental_style' in overrides and overrides['continental_style'] in STYLES:
        settings.update(orogen_defaults(overrides['continental_style']))
    properties = generator_schema()["properties"]
    for key, value in overrides.items():
        spec = properties[key]
        if spec.get('type') == 'string':
            if not isinstance(value, str) or re.fullmatch(spec['pattern'], value) is None:
                raise ValueError(f'Invalid {key}')
            settings[key] = value
        elif spec.get('type') == 'boolean':
            if not isinstance(value, bool):
                raise ValueError(f'{key} must be a boolean')
            settings[key] = value
        elif "enum" in spec:
            if not isinstance(value, str) or value not in spec["enum"]:
                raise ValueError(f"Invalid {key}: {value!r}")
            settings[key] = value
        elif "length" in spec:
            if not isinstance(value, (list, tuple)) or len(value) != spec["length"]:
                raise ValueError(f"{key} must contain {spec['length']} values")
            settings[key] = [_number(v, key, spec) for v in value]
        else:
            settings[key] = _number(value, key, spec)
    if 'snr_detail_mode' not in overrides and any(settings['snr_lod']):
        settings['snr_detail_mode']='per-lod'
    if settings['snr_altitude_range_m'][0] >= settings['snr_altitude_range_m'][1]:
        raise ValueError('snr_altitude_range_m must be increasing')
    if settings['snr_driver_range'][0] == settings['snr_driver_range'][1]:
        raise ValueError('snr_driver_range thresholds must differ')
    if settings['snr_latitude_range'][0] == settings['snr_latitude_range'][1]:
        raise ValueError('snr_latitude_range thresholds must differ')
    return settings


def _payload(base, settings):
    return {"version": GENERATION_VERSION, "base_profile": base, "settings": settings}


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _token(base, settings):
    if settings == _defaults(base):
        return base
    return base + "--g" + hashlib.sha256(_bytes(_payload(base, settings))).hexdigest()[:24]


@dataclass(frozen=True)
class GenerationDescriptor:
    profile: str
    base_profile: str
    normalized_settings: tuple

    @property
    def settings(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.normalized_settings}

    @property
    def is_default(self):
        return self.profile == self.base_profile

    @property
    def needs_bootstrap(self):
        s = self.settings
        return s["height_source"] in ("native", "orogen", "natural-continental") or s["climate_source"] in ("native", "orogen") or s["relief_pipeline"] in ("orogen", "city-gpu") or any(s[k] for k in OROGEN_GPU_PARAMETERS) or any(s['orogen_'+stage+'_stage'] for stage in ('relief','erosion','climate'))

    @property
    def bootstrap_generator(self):
        s=self.settings
        return 'orogen' if s['height_source']=='orogen' or s['climate_source']=='orogen' or s['relief_pipeline'] in ('orogen', 'city-gpu') or any(s[k] for k in OROGEN_GPU_PARAMETERS) or any(s['orogen_'+stage+'_stage'] for stage in ('relief','erosion','climate')) else 'native'

    @property
    def bootstrap_options(self):
        return (dict({k.removeprefix('orogen_'):self.settings[k] for k in OROGEN_PARAMETERS},
                     **{k:self.settings[k] for k in CITY_EROSION_PARAMETERS},
                     **{k:self.settings[k] for k in OROGEN_GPU_PARAMETERS},
                     **{k:self.settings[k] for k in OROGEN_STAGE_PARAMETERS},
                     initial_source=self.settings['height_source'], relief_pipeline=self.settings['relief_pipeline'], initial_settings=self.settings)
                if self.bootstrap_generator == 'orogen' else {})

    @property
    def bootstrap_style(self):
        return self.settings["continental_style"]


def _descriptor(profile, base, settings):
    return GenerationDescriptor(profile, base, tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in settings.items())))


def resolve_generation(profile):
    if not isinstance(profile, str):
        raise ValueError("Generation profile must be a string")
    if profile in BASE_PROFILES:
        return _descriptor(profile, profile, _defaults(profile))
    match = re.fullmatch(r"(natural|orogen|terrestrial-(?:gondwana|continents|earthlike|archipelago))--g([0-9a-f]{24})", profile)
    if match is None:
        raise ValueError(f"Unknown generation profile: {profile}")
    base = match.group(1)
    try:
        payload = json.loads((REGISTRY_ROOT / (profile + ".json")).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"version", "base_profile", "settings"}:
            raise ValueError("Invalid generation receipt schema")
        settings = _normalize(base, payload["settings"])
        if 'relief_pipeline' not in payload['settings']: settings['relief_pipeline']='original'
        # Existing v1 links contain the complete original key set. Verify
        # their original canonical token, then supply neutral new controls.
        old_keys = set(settings) - {'snr_altitude_gain', 'snr_altitude_range_m', 'snr_driver_channel',
                                    'snr_driver_gain', 'snr_driver_range', 'snr_bins',
                                    'snr_latitude_gain', 'snr_latitude_range', 'snr_lod'}
        saved = payload['settings']
        normalized_saved = {k: settings[k] for k in saved}
        accepted_keys = (set(settings), old_keys, set(settings)-set(OROGEN_PARAMETERS), old_keys-set(OROGEN_PARAMETERS),
                         set(settings)-set(OROGEN_PARAMETERS)-{'relief_pipeline'},
                         old_keys-set(OROGEN_PARAMETERS)-{'relief_pipeline'},
                         set(settings)-set(OROGEN_CLIMATE_PARAMETERS),
                         old_keys-set(OROGEN_CLIMATE_PARAMETERS))
        new_keys = {'snr_latitude_gain', 'snr_latitude_range', 'snr_lod'}
        accepted_keys += tuple(keys-new_keys for keys in accepted_keys)
        accepted_keys += tuple(keys-set(CITY_EROSION_PARAMETERS) for keys in accepted_keys)
        accepted_keys += tuple(keys-set(OROGEN_GPU_PARAMETERS) for keys in accepted_keys)
        accepted_keys += tuple(keys-set(OROGEN_STAGE_PARAMETERS) for keys in accepted_keys)
        accepted_keys += tuple(keys-{'snr_adaptive_enabled'} for keys in accepted_keys)
        accepted_keys += tuple(keys-{'snr_detail_mode'} for keys in accepted_keys)
        accepted_keys += tuple(keys-{'world_diameter_km', 'world_topology'} for keys in accepted_keys)
        if (set(saved) not in accepted_keys or payload != _payload(base, normalized_saved)
                or _token(base, normalized_saved) != profile):
            raise ValueError("Generation receipt does not match its canonical token")
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError(f"Cannot resolve generation profile: {profile}") from exc
    return _descriptor(profile, base, settings)


def register_generation(base, overrides=None):
    """Validate settings, atomically persist a new identity, and return its token."""
    settings = _normalize(base, overrides)
    token = _token(base, settings)
    if token == base:
        return token
    path = REGISTRY_ROOT / (token + ".json")
    if path.exists():
        resolve_generation(token)  # Never repair or overwrite a tampered receipt.
        return token
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_bytes(_payload(base, settings)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return token


def base_profile(profile):
    return resolve_generation(profile).base_profile
