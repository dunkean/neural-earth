"""Paths shared by runtime modules after separating backend and viewer assets."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "backend"
WEB_ROOT = REPO_ROOT / "web"

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
