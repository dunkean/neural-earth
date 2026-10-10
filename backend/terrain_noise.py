"""Tile cache for the portable upstream Gaussian noise field.

``gaussian_noise_patch`` regenerates every whole noise tile a window touches.
Decoder windows (512 px, stride 384) touch up to four 512 px tiles, latent
windows (64 px, stride 32) four 64 px tiles, so neighbouring windows draw the
same tiles again and again on the CPU while the GPU waits. Tiles are pure
functions of (seed, tile index, shape): caching them returns the same bytes.
"""
from collections import OrderedDict
import os
import threading

import numpy as np
import torch

from terrain_diffusion.inference.portable_rng import fill_standard_normal
from terrain_diffusion.inference.world_pipeline import _tile_seed

_tiles = OrderedDict()
_bytes = 0
_lock = threading.Lock()
_budget = int(float(os.environ.get('TERRAIN_NOISE_CACHE_MIB', '256')) * 1024**2)
stats = dict(hits=0, misses=0)


def _tile(base_seed, ty, tx, channels, tile_h, tile_w):
    global _bytes
    key = (int(base_seed), int(ty), int(tx), channels, tile_h, tile_w)
    with _lock:
        tile = _tiles.get(key)
        if tile is not None:
            _tiles.move_to_end(key)
            stats['hits'] += 1
            return tile
    # Generate outside the lock; a concurrent duplicate yields the same bytes.
    tile = np.empty((channels, tile_h, tile_w), dtype=np.float32)
    fill_standard_normal(_tile_seed(base_seed, ty, tx), tile.ravel())
    tile.flags.writeable = False
    with _lock:
        stats['misses'] += 1
        if key not in _tiles and tile.nbytes <= _budget:
            _tiles[key] = tile
            _bytes += tile.nbytes
            while _bytes > _budget:
                _, evicted = _tiles.popitem(last=False)
                _bytes -= evicted.nbytes
    return tile


def noise_patch(base_seed, y0, x0, h, w, channels=1, tile_h=256, tile_w=256):
    """Byte-identical to ``gaussian_noise_patch`` (float32), from cached tiles."""
    out = np.empty((channels, h, w), dtype=np.float32)
    for ty in range(y0 // tile_h, (y0 + h - 1) // tile_h + 1):
        oy0, oy1 = max(y0, ty * tile_h), min(y0 + h, (ty + 1) * tile_h)
        for tx in range(x0 // tile_w, (x0 + w - 1) // tile_w + 1):
            ox0, ox1 = max(x0, tx * tile_w), min(x0 + w, (tx + 1) * tile_w)
            tile = _tile(base_seed, ty, tx, channels, tile_h, tile_w)
            out[:, oy0-y0:oy1-y0, ox0-x0:ox1-x0] = \
                tile[:, oy0-ty*tile_h:oy1-ty*tile_h, ox0-tx*tile_w:ox1-tx*tile_w]
    return out


def upload(array, device, dtype):
    """Copy host noise without waiting for queued GPU work.

    A pageable copy synchronises the stream, so the CPU could not prepare the
    next window while the GPU computes. Pinned staging is reused by PyTorch's
    host allocator only after the copy completed. The cast then runs on the
    GPU, with the same round-to-nearest-even as on the host.
    """
    value = torch.from_numpy(array)
    if torch.device(device).type != 'cuda':
        return value.to(device, dtype=dtype)
    return value.pin_memory().to(device, non_blocking=True).to(dtype)


def clear():
    global _bytes
    with _lock:
        _tiles.clear()
        _bytes = 0
