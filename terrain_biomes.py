"""Orogen biome appearance evaluated on the current physical terrain.

Transport appends sixteen planes to the five existing climate channels:
base RGB, alpine/snow lines (km), then eleven appearance constants.
Classification stays global; coastlines, altitude and slope follow each DEM.
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from terrain_climate import expand_climate
from terrain_orogen_layers import sample

MODE = 'orogen-biomes'
CLIMATE_LAYERS = 21
DEFAULT_ROCK_SLOPE = 40.0
_SPECS = json.loads((Path(__file__).parent/'native/orogen/climate-parameters.json').read_text())
_CODES = ('af am aw bwh bwk bsh bsk cfa cfb cfc csa csb csc cwa cwb cwc '
          'dfa dfb dfc dfd dsa dsb dsc dsd dwa dwb dwc dwd et ef').split()


def parse_rock_slope(value=None):
    angle = DEFAULT_ROCK_SLOPE if value is None else float(value)
    if not np.isfinite(angle) or not 1 <= angle <= 89:
        raise ValueError('Biome rock slope must be between 1 and 89 degrees')
    return angle


def _land_classes(world):
    # Extend the nearest land climate under the source ocean. New neural islands
    # and moved shorelines inherit a land biome rather than an ocean color.
    source = getattr(world, 'source', world)
    periodic = getattr(world, 'periodic_longitude', True)
    cached = getattr(source, '_terrain_biome_land_classes', {})
    if periodic in cached:
        return cached[periodic]
    classes = np.asarray(source.layers['koppen'], np.uint8)
    if not np.any(classes):
        result = np.full_like(classes, 9)  # Cfb for an entirely oceanic atlas.
    elif np.all(classes):
        result = classes
    else:
        extended = np.tile(classes, (1, 3)) if periodic else classes
        indices = distance_transform_edt(extended == 0, return_distances=False,
                                         return_indices=True)
        filled = extended[tuple(indices)]
        result = filled[:, classes.shape[1]:2*classes.shape[1]].copy() if periodic else filled
    cached[periodic] = result
    source._terrain_biome_land_classes = cached
    return result


def sample_biome_fields(world, xs, ys, settings, *, polar=False):
    def setting(name):
        key = 'orogen_biome_' + name
        return settings.get(key, _SPECS[key]['default'])
    classes = _land_classes(world)
    if polar:
        from terrain_polar import from_chart
        x, y = from_chart(*np.meshgrid(xs, ys), settings)
        x0,y0,x1,y1 = world.bounds
        ix = np.floor((x-x0)/(x1-x0)*world.width).astype(np.int64) % world.width
        iy = np.clip(np.floor((y-y0)/(y1-y0)*world.height).astype(np.int64),0,world.height-1)
        ids = classes[iy,ix].astype(np.int64)
    else:
        ids = sample(world, classes, xs, ys, categorical=True).astype(np.int64)
    palette = np.asarray([[.12, .38, .10]] + [setting('color_'+code) for code in _CODES], np.float32)
    thresholds = np.zeros((31, 2), np.float32)
    for i in range(1, 31):
        group = ('tropical' if i <= 3 else 'arid' if i <= 7 else
                 'temperate' if i <= 16 else 'continental' if i in (17,18,21,22,25,26) else
                 'subarctic' if i <= 28 else 'tundra' if i == 29 else 'ice')
        thresholds[i] = setting(group+'_alpine'), setting(group+'_snow')
    constants = [setting(name) for name in ('low_height', 'low_darkening', 'mid_darkening',
                                           'rock_transition', 'snow_transition')]
    constants += list(setting('rock_color')) + list(setting('snow_color'))
    shape = ids.shape
    return np.concatenate([palette[ids].transpose(2,0,1), thresholds[ids].transpose(2,0,1),
                           np.broadcast_to(np.asarray(constants,np.float32)[:,None,None],
                                           (len(constants),*shape))]).astype(np.float32)


def colorize_biomes(elevation, fields, resolution, rock_slope=None):
    c = fields if fields.shape[1:] == elevation.shape else expand_climate(fields, elevation.shape)
    base = c[:3].transpose(1,2,0).copy()
    alpine, snow = c[3], c[4]
    low, low_dark, mid_dark, rock_span, snow_span = c[5:10]
    rock, snow_color = c[10:13].transpose(1,2,0), c[13:16].transpose(1,2,0)
    h = np.maximum(elevation, 0)/1000
    dark = np.where(h < low, 1-low_dark+low_dark*h/np.maximum(low,1e-6), 1)
    dark *= np.where((alpine>0)&(h>low)&(h<alpine),
                     1-(h-low)/np.maximum(alpine-low,1e-6)*mid_dark, 1)
    rgb = base*dark[...,None]
    rock_t = np.where(alpine>0, np.clip((h-alpine)/np.maximum(
        np.where(snow>alpine,snow-alpine,rock_span),1e-6),0,1)**2, 0)[...,None]
    rgb = rgb*(1-rock_t)+rock*rock_t
    snow_t = np.where(snow>0,np.clip((h-snow)/np.maximum(snow_span,1e-6),0,1)**2,0)[...,None]
    rgb = rgb*(1-snow_t)+snow_color*snow_t
    # Central differences in physical metres, with the tile halo on both sides.
    # This override comes after snow and vegetation, without hillshade scaling.
    if resolution <= 30:
        dy, dx = np.gradient(np.asarray(elevation,np.float32), resolution)
        steep = np.hypot(dx,dy) > np.tan(np.deg2rad(parse_rock_slope(rock_slope)))
        rgb = np.where(steep[...,None],rock,rgb)
    e = elevation/10000
    deep_t = np.clip((e+.5)/.4,0,1)[...,None]
    shallow_t = np.clip((e+.1)/.1,0,1)[...,None]
    ocean = np.where((e<-.1)[...,None],np.array([.04,.06,.30])+deep_t*np.array([.07,.14,.18]),
                     np.array([.11,.20,.48])+shallow_t*np.array([.19,.22,.12]))
    return np.clip(np.where((elevation<0)[...,None],ocean,rgb),0,1)
