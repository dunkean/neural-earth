"""CPU mirror of the WebGPU physical water appearance (``hydrologyShader``).

A hydrology tile is an ``(H, W, 4)`` float raster on exactly the same sample
centres as the physical elevation ``(H, W)``, halo included:

- R: water surface level, metres above sea level;
- G: wet coverage of the sample, 0..1;
- B: class, 0 dry, 1 ocean, 2 lake, 3 river;
- A: 1 where hydrology is authoritative for the sample, 0 otherwise.

Ocean and unauthoritative samples keep the legacy elevation-below-zero
appearance. Server use, around the existing CPU renderer::

    rgb = render_elevation(ground_for_shading(elevation, water), lod, ...)
    rgb = composite_water(rgb, elevation, water, crop=HALO)

Method note: docs/methods/hydrology-render.md.
"""
import numpy as np

DRY, OCEAN, LAKE, RIVER = 0, 1, 2, 3
DRY_GROUND_M = .5  # valid dry land below sea level shades as low ground
DEPTH_RANGE_M = (.15, 12.)  # shallow -> deep colour ramp of water depth
SHALLOW = np.array([.1019608, .3019608, .3607843], np.float32)
DEEP = np.array([.0431373, .1607843, .2705882], np.float32)


def _checked(elevation, hydrology):
    elevation = np.asarray(elevation, np.float32)
    hydrology = np.asarray(hydrology, np.float32)
    if elevation.ndim != 2 or hydrology.shape != elevation.shape+(4,):
        raise ValueError(f'hydrology {hydrology.shape} must be elevation {elevation.shape} + (4,)')
    # Non-finite channels carry no authority (the GPU result would be undefined).
    valid = np.isfinite(hydrology).all(-1) & (hydrology[..., 3] > .5)
    return elevation, hydrology, valid


def ground_for_shading(elevation, hydrology=None):
    """Height to shade as terrain: valid dry samples are at least ``DRY_GROUND_M``.

    An endorheic basin or dry depression below sea level is land, not ocean.
    The returned array only drives the legacy shading; physical heights and
    water depths still use the raw elevation.
    """
    if hydrology is None:
        return elevation
    elevation, hydrology, valid = _checked(elevation, hydrology)
    dry = valid & (hydrology[..., 2] < DRY+.5)
    return np.where(dry, np.maximum(elevation, DRY_GROUND_M), elevation)


def water_colour(depth_m):
    """Lake/river colour from water depth in metres (WGSL ``smoothstep``)."""
    lo, hi = DEPTH_RANGE_M
    t = np.clip((np.asarray(depth_m, np.float32)-lo)/(hi-lo), 0, 1)
    t = (t*t*(3-2*t))[..., None]
    return SHALLOW+(DEEP-SHALLOW)*t


def composite_water(rgb, elevation, hydrology=None, *, crop=0):
    """Blend lake and river water over an already shaded tile.

    ``elevation`` and ``hydrology`` are the full rasters given to the renderer.
    ``rgb`` covers them minus ``crop`` samples on every side, as returned by
    ``render_elevation(..., halo=crop)``; any other shape is rejected rather
    than resampled. uint8 colours stay uint8, float colours are 0..1. Samples
    that are not lake or river keep their exact input colour.
    """
    if hydrology is None:
        return rgb
    elevation, hydrology, valid = _checked(elevation, hydrology)
    h, w = elevation.shape
    crop = int(crop)
    if crop < 0 or np.shape(rgb) != (h-2*crop, w-2*crop, 3):
        raise ValueError(f'rgb {np.shape(rgb)} is not elevation {elevation.shape} cropped by {crop}')
    inner = (slice(crop, h-crop), slice(crop, w-crop))
    water, valid, height = hydrology[inner], valid[inner], elevation[inner]
    cover = np.clip(water[..., 1], 0, 1)
    wet = valid & (water[..., 2] >= LAKE-.5) & (cover > 0)
    if not wet.any():
        return rgb
    integer = np.asarray(rgb).dtype == np.uint8
    base = np.asarray(rgb, np.float32)[wet]/(255 if integer else 1)
    # fmax: a missing bed height reads as zero depth, never as a NaN colour.
    colour = water_colour(np.fmax(water[..., 0][wet]-height[wet], 0))
    blended = base+(colour-base)*cover[wet][:, None]
    out = np.array(rgb, copy=True)
    out[wet] = np.rint(np.clip(blended, 0, 1)*255) if integer else blended
    return out
