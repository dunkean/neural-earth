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

GENERATION_VERSION = "terrain-generation-v1"
REGISTRY_ROOT = Path(os.environ.get("TERRAIN_GENERATION_ROOT", "E:/TerrainDiffusionRuntime/generation-settings"))
BASE_PROFILES = ("natural", "terrestrial-gondwana", "terrestrial-continents",
                 "terrestrial-earthlike", "terrestrial-archipelago")
STYLES = ("gondwana", "continents", "earthlike", "archipelago")


def _defaults(base):
    if base not in BASE_PROFILES:
        raise ValueError(f"Unknown base profile: {base}")
    natural = base == "natural"
    return dict(height_source="natural" if natural else "native",
                climate_source="natural" if natural else "native",
                continental_style="earthlike" if natural else base.removeprefix("terrestrial-"),
                continental_strength=.8, macro_scale_km=600.,
                frequency_mult=[1.] * 5, octaves=[4, 2, 4, 4, 4],
                cond_snr=[.5] * 5 if natural else [.05, .5, .5, .5, .5],
                drop_water_pct=.5)


def generator_schema():
    """Return JSON-safe control metadata and a fresh defaults dictionary."""
    return {"version": GENERATION_VERSION, "defaults_by_profile": {p: _defaults(p) for p in BASE_PROFILES},
            "properties": {
                "height_source": {"enum": ["natural", "native", "natural-continental"]},
                "climate_source": {"enum": ["natural", "native"]},
                "continental_style": {"enum": list(STYLES)},
                "continental_strength": {},
                "macro_scale_km": {"minimum": 0., "description": "Gaussian sigma in km; soft continental constraint"},
                "frequency_mult": {"length": 5},
                "octaves": {"length": 5, "minimum": 1, "integer": True},
                "cond_snr": {"length": 5, "exclusiveMinimum": 0., "description": "Noise/signal amplitude; larger permits more learned correction"},
                "drop_water_pct": {"minimum": 0., "maximum": 1., "description": "Natural height distribution: removed ocean weight; 1 means land only"}}}


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
    properties = generator_schema()["properties"]
    for key, value in overrides.items():
        spec = properties[key]
        if "enum" in spec:
            if not isinstance(value, str) or value not in spec["enum"]:
                raise ValueError(f"Invalid {key}: {value!r}")
            settings[key] = value
        elif "length" in spec:
            if not isinstance(value, (list, tuple)) or len(value) != spec["length"]:
                raise ValueError(f"{key} must contain five values")
            settings[key] = [_number(v, key, spec) for v in value]
        else:
            settings[key] = _number(value, key, spec)
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
        return s["height_source"] in ("native", "natural-continental") or s["climate_source"] == "native"

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
    match = re.fullmatch(r"(natural|terrestrial-(?:gondwana|continents|earthlike|archipelago))--g([0-9a-f]{24})", profile)
    if match is None:
        raise ValueError(f"Unknown generation profile: {profile}")
    base = match.group(1)
    try:
        payload = json.loads((REGISTRY_ROOT / (profile + ".json")).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"version", "base_profile", "settings"}:
            raise ValueError("Invalid generation receipt schema")
        settings = _normalize(base, payload["settings"])
        if payload != _payload(base, settings) or set(payload["settings"]) != set(settings) or _token(base, settings) != profile:
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
