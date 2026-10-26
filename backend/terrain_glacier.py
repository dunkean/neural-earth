"""Canonical coastal-glacier provenance for Render v2 iceberg appearance.

This is a source-atlas proxy, not a calving or glacier-flow simulation. A
perennially cold, precipitation-fed land cell adjoining source ocean supplies
the proxy. Chord distances on the sphere avoid longitude/pole seams; at the
80 km support radius their difference from surface distances is negligible.
No tile-local distance transform or neural terrain requests are involved.
"""
import numpy as np
from scipy.spatial import cKDTree

REACH_METRES = 80000.


def _points(xs, ys, bounds, spherical, polar=False):
    x, y = np.meshgrid(np.asarray(xs, np.float64), np.asarray(ys, np.float64))
    if not spherical:
        return np.stack((x, y, np.zeros_like(x)), -1)
    x0, y0, x1, y1 = bounds
    lon = (x-x0)/(x1-x0)*2*np.pi-np.pi
    lat = np.pi/2-(y-y0)/(y1-y0)*np.pi
    direction = np.stack((np.cos(lat)*np.cos(lon), np.sin(lat), np.cos(lat)*np.sin(lon)), -1)
    if polar:
        direction = np.stack((direction[..., 0], -direction[..., 2], direction[..., 1]), -1)
    return direction*((x1-x0)/(2*np.pi))


def _coastal_sources(world, settings):
    source = getattr(world, 'source', world)
    spherical = settings.get('world_topology', 'sphere') != 'plane'
    key = (tuple(world.bounds), spherical)
    cache = getattr(source, '_terrain_v2_glacier_sources', {})
    if key in cache:
        return cache[key]
    height = getattr(source, 'height_m', None)
    layers = getattr(source, 'layers', {})
    names = ('temperature_summer', 'temperature_winter', 'precip_summer', 'precip_winter')
    tree = None
    if height is not None and all(name in layers for name in names):
        height = np.asarray(height)
        ts, tw, ps, pw = (np.asarray(layers[name]) for name in names)
        if all(field.shape == height.shape for field in (ts, tw, ps, pw)):
            ocean = np.isfinite(height) & (height < 0)
            # Adjoining source cells; no y wrapping across the poles.
            padded = np.pad(ocean, ((1, 1), (0, 0)), mode='edge')
            adjacent = np.zeros_like(ocean)
            for dy in (-1, 0, 1):
                rows = padded[1+dy:1+dy+height.shape[0]]
                for dx in (-1, 0, 1):
                    if dx and spherical:
                        nearby = np.roll(rows, dx, axis=1)
                    elif dx:
                        nearby = np.pad(rows, ((0, 0), (1, 1)), mode='edge')[:, 1+dx:1+dx+height.shape[1]]
                    else:
                        nearby = rows
                    adjacent |= nearby
            finite = np.isfinite(ts) & np.isfinite(tw) & np.isfinite(ps) & np.isfinite(pw)
            glaciated = finite & (height > 20) & (np.maximum(ts, tw)*90-45 < -1) & ((ps+pw)*1000 > 80)
            eligible = glaciated & adjacent
            if eligible.any():
                h, w = height.shape
                x0, y0, x1, y1 = world.bounds
                xs = x0+(np.arange(w)+.5)*(x1-x0)/w
                ys = y0+(np.arange(h)+.5)*(y1-y0)/h
                points = _points(xs, ys, world.bounds, spherical)[eligible]
                tree = cKDTree(points)
    cache[key] = tree
    source._terrain_v2_glacier_sources = cache
    return tree


def sample_glacier_proximity(world, xs, ys, settings, *, polar=False):
    """0..1 source influence, stable under crop, tile, resolution and chart."""
    tree = _coastal_sources(world, settings)
    shape = (len(ys), len(xs))
    if tree is None:
        return np.zeros(shape, np.float32)
    points = _points(xs, ys, world.bounds, settings.get('world_topology', 'sphere') != 'plane', polar)
    distances = tree.query(points.reshape(-1, 3), k=1, distance_upper_bound=REACH_METRES)[0].reshape(shape)
    t = np.clip(distances/REACH_METRES, 0, 1)
    return np.asarray(1-t*t*(3-2*t), np.float32)
