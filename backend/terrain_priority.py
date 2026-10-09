"""Physical relief classification for tile scheduling."""
import numpy as np


def land_mask(elevation, bounds, block=4):
    elevation = np.asarray(elevation)
    height, width = elevation.shape
    # Unknown samples are treated as land; retain any sampled island or coast.
    land = (~np.isfinite(elevation)) | (elevation > -10)
    water = np.isfinite(elevation) & (elevation <= 0)
    water = water.reshape(height // block, block, width // block, block).any(axis=(1, 3))
    land = land.reshape(height // block, block, width // block, block).any(axis=(1, 3))
    return dict(bounds=list(bounds), width=width // block, height=height // block,
                rows=[''.join('1' if cell else '0' for cell in row) for row in land],
                sea_rows=[''.join('1' if cell else '0' for cell in row) for row in water])


def tile_relief_stats(elevation, halo=24):
    """Classify the displayed interior, excluding neighbouring halo samples."""
    interior = np.asarray(elevation)
    if halo:
        interior = interior[halo:-halo, halo:-halo]
    if not interior.size or not np.isfinite(interior).all():
        return {}
    return dict(elevation_min=float(interior.min()), elevation_max=float(interior.max()))
