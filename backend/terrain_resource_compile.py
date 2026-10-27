"""Compile one world's resource authority in its own process.

Started by terrain_resource_layers for a world whose cache is missing. Running
apart from the server keeps the compile off the server's GIL (tile requests
are served at full speed), bounds its memory to this process and isolates a
native crash. The server writes the atlas layers the compiler reads to
``<key>-atlas.npz``; this process compiles stage by stage and publishes, in
the cache directory:

- ``<key>-status.json``: stages done with their times, or the error;
- ``<key>-stage.npz``: all fields computed so far (uncompressed, rewritten
  after each stage) so that layers needing only them can be served;
- at the end, ``<key>-deposits.json``, then the compressed caches
  ``<key>.npz``/``<key>.json`` (and the Render v2 environment), after which
  the stage file is removed.

usage: python terrain_resource_compile.py KEY V2KEY CACHE_DIR [FANTASY]
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _replace(temp, path, tries=100):
    """os.replace with retries: on Windows it fails while the server reads the target."""
    for n in range(tries):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if n == tries - 1:
                raise
            time.sleep(.05)


def _write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value), encoding='utf-8')
    _replace(temp, path)


def main(key, v2key, cache, fantasy):
    sys.path[:0] = [str(ROOT), str(HERE)]
    try:
        from tools._bootstrap import activate
        activate()
    except ImportError:
        pass
    import numpy as np
    import terrain_resources as R
    import terrain_resource_workers as W
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    status_path = cache / f'{key}-status.json'
    started = time.perf_counter()
    status = dict(state='compiling', stages=[], error=None, pid=None)
    _write_json(status_path, status)

    def slim(fields):
        out = {}
        for k, v in fields.items():
            if k.startswith('_') or k in TRANSIENT or not isinstance(v, np.ndarray):
                continue
            if v.dtype == np.float64 and k not in R.EXTENSIVE and k != 'area_m2':
                v = v.astype(np.float32)
            out[k] = v
        return out

    def publish(stage, fields):
        temp = cache / f'{key}-stage.tmp.npz'
        np.savez(temp, **slim(fields))
        _replace(temp, cache / f'{key}-stage.npz')
        status['stages'].append([stage, round(time.perf_counter() - started, 2)])
        _write_json(status_path, status)

    try:
        atlas_path = cache / f'{key}-atlas.npz'
        with np.load(atlas_path, allow_pickle=False) as data:
            layers = {k[6:]: data[k] for k in data.files if k.startswith('layer_')}
            height, seed, bounds = data['height_m'], int(data['seed']), tuple(data['bounds'])
        atlas_path.unlink(missing_ok=True)
        status['atlas_s'] = round(time.perf_counter() - started, 2)
        v2 = None
        try:
            with np.load(cache / f'{v2key}-v2.npz', allow_pickle=False) as data:
                v2 = {k: data[k] for k in data.files}
        except (OSError, ValueError):
            v2 = None
        status['v2_cached'] = v2 is not None
        grid, fields, deposits = R.compile_world(layers, height, seed, fantasy=fantasy,
                                                 progress=publish, parallel=W.parallel_map, v2_fields=v2, bounds=bounds)
        _write_json(cache / f'{key}-deposits.json', deposits)
        status['state'] = 'ready'
        status['compile_s'] = round(time.perf_counter() - started, 2)
        _write_json(status_path, status)
        # Durable caches (compressed, ~0.15 GB), then the stage file goes.
        arrays = {k: v for k, v in slim(fields).items() if not k.startswith(('stock_', 'recoverable_'))}
        if v2 is None:
            temp = cache / f'{v2key}-v2.tmp.npz'
            np.savez_compressed(temp, **{k: np.asarray(fields[k]) for k in R.V2_FIELDS if k in fields})
            _replace(temp, cache / f'{v2key}-v2.npz')
        temp = cache / f'{key}.tmp.npz'
        np.savez_compressed(temp, **arrays)
        _replace(temp, cache / f'{key}.npz')
        _replace(cache / f'{key}-deposits.json', cache / f'{key}.json')
        status['saved_s'] = round(time.perf_counter() - started, 2)
        _write_json(status_path, status)
        for _ in range(100):
            try:
                (cache / f'{key}-stage.npz').unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(.05)
    except Exception as exc:
        traceback.print_exc()
        status['state'], status['error'] = 'error', f'{type(exc).__name__}: {exc}'
        _write_json(status_path, status)
        return 1
    return 0


TRANSIENT = ('pet_mm', 'aet_mm', 'precipitation_mm', 'temperature_c', 'snow_mm', 'v2_moisture_month',
             'v2_leaf_month', 'v2_grass_month', 'v2_ice_month', 'forage_month_kg_ha')

if __name__ == '__main__':
    args = sys.argv[1:]
    sys.exit(main(args[0], args[1], args[2], len(args) > 3 and args[3] == '1'))
