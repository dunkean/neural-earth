"""Resolve repository imports and IO for standalone verification tools."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent.parent

def activate():
    for directory in (ROOT, ROOT / "backend", ROOT / "tests/python", ROOT / "tools/benchmarks", ROOT / "tools/verification"):
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)
    inherited = os.environ.get("PYTHONPATH", "")
    paths = [str(ROOT / "backend"), str(ROOT)]
    if inherited:
        paths.append(inherited)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    os.chdir(ROOT)
