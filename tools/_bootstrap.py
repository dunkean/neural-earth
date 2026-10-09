"""Resolve repository imports and IO for standalone verification tools."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent.parent

def activate():
    for directory in (ROOT, ROOT / "tests/python", ROOT / "tools/benchmarks", ROOT / "tools/verification"):
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(ROOT)
