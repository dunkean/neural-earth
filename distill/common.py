"""Paths, provenance and atomic persistence, without loading CUDA models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('DISTILL_ROOT', Path.home() / 'data/distill')).expanduser().resolve()
REVISION = '9ef8030cb805b433b98ec25c5dddefbac07a9e26'
HOLDOUT_SEEDS = frozenset((42, 101, 202, 303, 404))
SCHEMA = 1


def bootstrap():
    for path in (REPO, REPO / 'backend', REPO / 'terrain-diffusion'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ.setdefault('TERRAIN_RUNTIME_ROOT', str(Path.home() / 'data/runtime'))
    os.environ.setdefault('TERRAIN_OUTPUT_ROOT', str(DATA / 'runtime-output'))
    os.environ.setdefault('HF_HOME', str(Path.home() / '.cache/neural-earth/huggingface'))
    os.environ.setdefault('TERRAIN_CUDA_DEVICE', '0')
    os.environ.setdefault('TERRAIN_GPU_MODE', 'single')
    os.environ.setdefault('TERRAIN_CUDA_DEVICES', '0')


def external_path(path):
    path = Path(path).expanduser().resolve()
    if path == REPO or REPO in path.parents or str(path).startswith('/mnt/'):
        raise ValueError(f'Data/checkpoints must be outside Git and /mnt: {path}')
    return path


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_write(path, lambda handle: handle.write(
        (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()))


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')
        handle.flush()


def source_digest():
    files = ['terrain-diffusion/terrain_diffusion/inference/world_pipeline.py',
             'backend/terrain_inference.py', 'backend/terrain_conditioning.py',
             'backend/terrain_generation.py', 'distill/features.py', 'distill/teacher.py']
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in files}


bootstrap()
