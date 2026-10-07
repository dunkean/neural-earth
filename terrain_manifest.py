"""Canonical, content-addressed identity for reference and experimental worlds."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import infinite_tensor
import torch

from terrain_bootstrap import (BOOTSTRAP_VERSION, implementation_identity,
                               verify_implementation_identity, bootstrap_metadata)
from terrain_conditioning import (STATS_PATH, WORLD_PROFILES,
                                  BOOTSTRAP_CONDITIONING_VERSION, CONDITIONING_SNR)
from terrain_world import COARSE_RESOLUTION, SOURCE_FILES, DATA_ROOT, WORLD_BOUNDS
from terrain_generation import resolve_generation

ROOT = Path(__file__).resolve().parent
MODEL_REVISION = "9ef8030cb805b433b98ec25c5dddefbac07a9e26"
MODEL_ROOT = (Path("E:/TerrainDiffusionRuntime/huggingface/hub") /
              "models--xandergos--terrain-diffusion-30m/snapshots" / MODEL_REVISION)
MANIFEST_SCHEMA = "terrain-world-manifest-v2"


def _digest(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {"size": stat.st_size,
            "sha256": _cached_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)}


@lru_cache(maxsize=256)
def _cached_digest(resolved_path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    path = Path(resolved_path)
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _files(model_root: Path, world_profile: str = 'natural'):
    needs_bootstrap = resolve_generation(world_profile).needs_bootstrap
    model_names = ["config.json"] + [f"{stage}/{name}" for stage in
        ("coarse_model", "base_model", "decoder_model") for name in
        ("config.json", "diffusion_pytorch_model.safetensors")]
    package_root = Path(infinite_tensor.__file__).resolve().parent
    package_sources = sorted(p for p in package_root.rglob("*.py") if p.is_file())
    upstream_root = ROOT / "terrain-diffusion" / "terrain_diffusion"
    upstream_sources = sorted(p for p in upstream_root.rglob("*.py") if p.is_file())
    implementation_names = (
        "terrain_manifest.py", "terrain_generation.py", "terrain_bootstrap.py", "terrain_conditioning.py", "terrain_world.py", "terrain_inference.py",
        "terrain_server.py", "terrain_app.py", "terrain_final_mips.py",
        "terrain_background.py", "terrain_climate.py", "terrain_window_scheduler.py",
        "terrain_coarse.py", "terrain_device.py", "terrain_jobs.py",
        "terrain_disk_cache.py", "terrain_nn_constants.py", "terrain_cuda_graphs.py")
    implementation = {name: _digest(ROOT / name) for name in implementation_names
                      if name != 'terrain_bootstrap.py' or needs_bootstrap}
    # Hash the entire vendored package: model blocks, utilities and package
    # initializers participate in checkpoint execution, including future imports.
    implementation.update({p.relative_to(ROOT).as_posix(): _digest(p)
                           for p in upstream_sources})
    return {
        "model": {name: _digest(model_root / name) for name in model_names},
        "sources": {name: _digest(DATA_ROOT / name) for name in SOURCE_FILES},
        "statistics": _digest(STATS_PATH),
        "infinite_tensor": {p.relative_to(package_root).as_posix(): _digest(p)
                            for p in package_sources},
        "implementation": implementation,
        "bootstrap_native": deepcopy(implementation_identity()) if needs_bootstrap else None,
    }


def _runtime_versions() -> dict:
    packages = ("infinite-tensor", "diffusers", "numpy", "pyfastnoiselite",
                "rasterio", "scipy", "torch")
    return {"packages": {name: importlib.metadata.version(name) for name in packages},
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}


def build_manifest(seed: int, ablation: str = "A0", *, model_root: Path = MODEL_ROOT,
                   checkpoint_source: Path | None = None,
                   numerical_profile: str = "cuda-resident-v1", precision: str = "bf16",
                   backend: str = "torch-cuda", inference_profile: dict | None = None,
                   file_hashes: bool = True) -> dict:
    """Build a self-contained manifest; file hashing is required for durable identity.

    ``file_hashes=False`` is useful for unit tests and previews but marks the
    result explicitly incomplete and must not be used for persistent caches.
    """
    world_profile = "natural" if ablation == "A0" else ablation
    descriptor = resolve_generation(world_profile)
    settings = descriptor.settings
    seed = int(seed)
    if not 0 <= seed <= 0xffffffffffffffff:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    bootstrap = None
    if descriptor.needs_bootstrap:
        receipt = bootstrap_metadata(seed, descriptor.bootstrap_style)
        # Runtime timings never participate in a reproducible world identity.
        bootstrap = {name: receipt[name] for name in (
            "requested_seed_u64", "selected_seed_u64", "selected_attempt", "style",
            "native_config", "attempts", "raster", "hypsometry",
            "raw_height_sha256", "height_sha256", "sign_preserved")}
    config_path = Path(checkpoint_source or model_root) / "config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8")) if file_hashes else None
    conditioning_snr = (inference_profile or {}).get('checkpoint_kwargs', {}).get(
        'cond_snr', settings['cond_snr'])
    payload = {
        "schema": MANIFEST_SCHEMA,
        "seed_u64": str(seed),
        "world_profile": world_profile,
        "base_world_profile": descriptor.base_profile,
        "ablation": ablation,
        "bootstrap": bootstrap,
        "checkpoint": {"repo": "xandergos/terrain-diffusion-30m", "revision": MODEL_REVISION,
                       "config": model_config},
        "geography": {"bounds_m": list(WORLD_BOUNDS), "coordinate_system": "flat-metre, y-down",
                      "periodic_longitude": False,
                      "bootstrap_periodic_longitude": settings['height_source']=='native', "coarse_cell_m": COARSE_RESOLUTION,
                      "sea_level_m": 0, "bootstrap_version": BOOTSTRAP_VERSION if descriptor.needs_bootstrap else None},
        "conditioning": {"channels": ["elevation_m", "BIO1_c", "BIO4_c_x100",
                                      "BIO12_mm_year", "BIO15_percent"],
                         "elevation_transform": "sign(h)*sqrt(abs(h))",
                         "normalization": "checkpoint coarse_means/coarse_stds",
                         "cond_snr": list(conditioning_snr),
                         "frequency_mult": settings['frequency_mult'],
                         "drop_water_pct": settings['drop_water_pct'],
                         "generation_settings": settings,
                         "physical_climate_version": BOOTSTRAP_CONDITIONING_VERSION if settings['climate_source']=='native' else None},
        "generation": {"backend": backend, "precision": precision,
                       "numerical_profile": numerical_profile,
                       "inference_profile": inference_profile or {},
                       "runtime_versions": _runtime_versions(),
                       "coarse_solver_steps": 20, "coarse_window": 64, "coarse_stride": 48,
                       "base_window": 64, "base_stride": 32,
                       "decoder_window": 512, "decoder_stride": 384,
                       "base_cell_m": 240, "decoder_cell_m": 30},
        "files": _files(Path(checkpoint_source or model_root), world_profile) if file_hashes else None,
        "complete": bool(file_hashes),
    }
    payload["world_hash"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def world_identity(manifest: dict) -> str:
    """Refuse partial or tampered identities before using persistent caches."""
    if not manifest.get("complete") or not manifest.get("files"):
        raise ValueError("Persistent world identity requires full file hashes")
    claimed = manifest.get("world_hash")
    payload = {k: v for k, v in manifest.items() if k != "world_hash"}
    actual = hashlib.sha256(_canonical(payload)).hexdigest()
    if claimed != actual:
        raise ValueError("World manifest hash mismatch")
    return actual


def verify_manifest_files(manifest: dict, checkpoint_source: Path = MODEL_ROOT) -> str:
    """Verify current local bytes still match a complete generation identity."""
    identity = world_identity(manifest)
    if manifest['files']['bootstrap_native'] is not None:
        verify_implementation_identity(manifest['files']['bootstrap_native'])
    if _files(Path(checkpoint_source), manifest['world_profile']) != manifest["files"]:
        raise ValueError("Source, model, or runtime file changed since manifest creation")
    return identity


def write_manifest(path: Path, manifest: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
