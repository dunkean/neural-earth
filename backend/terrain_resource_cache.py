"""Atomic resource caches: lossless arrays, fast level-1 DEFLATE, compact JSON.

The existing NPZ/JSON names remain readable by older compilers. No pickle and
no new dependency. Publication completes with JSON so incomplete pairs miss.
"""
import json
from pathlib import Path
import uuid
import zipfile
import time
import numpy as np


def _replace(source, target):
    for attempt in range(40):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.025)


def save_world(root, key, fields, deposits):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    arrays = root / f'{key}-{nonce}.tmp.npz'
    metadata = root / f'{key}-{nonce}.tmp.json'
    try:
        with zipfile.ZipFile(arrays, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, value in fields.items():
                if not isinstance(value, np.ndarray) or name.startswith(('stock_', 'recoverable_', '_')):
                    continue
                with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
                    np.lib.format.write_array(stream, np.ascontiguousarray(value), allow_pickle=False)
        metadata.write_text(json.dumps(deposits, separators=(',', ':')), encoding='utf-8')
        _replace(arrays, root / f'{key}.npz')
        _replace(metadata, root / f'{key}.json')
    finally:
        arrays.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
