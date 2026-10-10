"""Repository and configurable storage paths shared on Windows and Linux."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "backend"
WEB_ROOT = REPO_ROOT / "web"


def runtime_root(environ=None, platform=None):
    """Keep the Windows storage layout and follow XDG cache storage on Linux."""
    environ = os.environ if environ is None else environ
    platform = os.name if platform is None else platform
    if environ.get("TERRAIN_RUNTIME_ROOT"):
        return Path(environ["TERRAIN_RUNTIME_ROOT"]).expanduser().resolve()
    if platform == "nt":
        return Path("E:/TerrainDiffusionRuntime")
    cache = Path(environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
    return (cache / "neural-earth").resolve()


def configured_path(variable, default):
    return Path(os.environ.get(variable) or default).expanduser().resolve()


RUNTIME_ROOT = runtime_root()
HF_HOME = configured_path("HF_HOME", RUNTIME_ROOT / "huggingface")
HF_HUB_CACHE = configured_path("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE") or HF_HOME / "hub")
OUTPUT_ROOT = configured_path("TERRAIN_OUTPUT_ROOT", REPO_ROOT / "generated")


def model_snapshot(revision):
    return HF_HUB_CACHE / "models--xandergos--terrain-diffusion-30m" / "snapshots" / revision

def source_path(name, root=REPO_ROOT):
    """Resolve source files; explicit flat fixture directories remain supported."""
    root = Path(root)
    name = Path(name)
    candidate = root / name
    if candidate.exists() or len(name.parts) != 1:
        return candidate
    if name.name.startswith("terrain_") and name.suffix == ".py":
        return root / "backend" / name if (root / "backend").is_dir() else candidate
    if name.name == "index.html" or (name.name.startswith("terrain_") and name.suffix in (".js", ".json")):
        return root / "web" / name if (root / "web").is_dir() else candidate
    return candidate
