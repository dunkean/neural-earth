"""Locate the shared Magna Urbis illustration client without copying it."""
import os
from pathlib import Path


def illustration_root(repo_root=None):
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    candidates = [Path(os.environ["MAGNA_ILLUSTRATION_ROOT"])] if os.environ.get("MAGNA_ILLUSTRATION_ROOT") else []
    candidates += [root.parent / "magna-urbis", root / "native" / "magna-urbis"]
    return next((path for path in candidates if (path / "illustration" / "web" / "index.mjs").is_file()), None)
