"""Classify interpolated seasonal climate at the displayed DEM resolution.

Class IDs are never interpolated. Temperatures are transported at sea level
and reconstructed at each terrain sample, for both CPU and GPU rendering.
"""

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path
import json
from pathlib import Path
import re

import numpy as np
from scipy.ndimage import map_coordinates

from terrain_orogen_layers import sample

ROOT = REPO_ROOT
SPECS = json.loads((ROOT/'native/orogen/climate-parameters.json').read_text())
CODES = ('af am aw bwh bwk bsh bsk cfa cfb cfc csa csb csc cwa cwb cwc '
         'dfa dfb dfc dfd dsa dsb dsc dsd dwa dwb dwc dwd et ef').split()
# Match the native classifier's palette without maintaining a second palette.
COLORS = np.asarray([json.loads('['+match+']') for match in re.findall(
    r'color:\s*\[([^]]+)\]', (ROOT/'native/orogen/vendor/koppen.js').read_text(encoding='utf-8'))], np.float32)
PARAMETERS = ('ice', 'tundra', 'tropical', 'temperate', 'hot_summer', 'shoulder',
              'extreme_cold', 'hot_arid', 'aridity_temperature', 'aridity_summer',
              'aridity_mixed', 'summer_fraction', 'winter_fraction', 'desert_fraction',
              'dry_summer_mm', 'dry_summer_ratio', 'dry_winter_ratio',
              'rainforest_mm', 'monsoon_factor', 'monsoon_mm', 'shoulder_fraction')


def setting(settings, name):
    return settings.get('orogen_'+name, SPECS['orogen_'+name]['default'])


def sample_field(world, field, xs, ys, settings, *, polar=False):
    if not polar:
        return sample(world, field, xs, ys)
    from terrain_polar import from_chart
    x, y = from_chart(*np.meshgrid(xs, ys), settings)
    x0, y0, x1, y1 = world.bounds
    ix = (x-x0)/(x1-x0)*world.width-.5
    iy = np.clip((y-y0)/(y1-y0)*world.height-.5, 0, world.height-1)
    # A padded longitude seam permits bilinear interpolation through poles.
    padded = np.pad(field, ((0,0),(1,1)), mode='wrap')
    return map_coordinates(padded, [iy, ix % world.width+1], order=1, mode='nearest')


def seasonal_fields(world, xs, ys, settings, *, polar=False):
    source = getattr(world,'source',world)
    key = tuple(setting(settings,name) for name in ('temperature_wet_lapse','temperature_dry_lapse_extra'))
    cached = getattr(source,'_terrain_koppen_seasons',None)
    if cached is None or cached[0]!=key:
        fields = []
        reference = np.maximum(world.layers.get('climate_height_m', world.height_m), 0)
        for season in ('summer', 'winter'):
            rain = np.maximum(world.layers['precip_'+season], 0)
            lapse = (key[0]+key[1]*(1-np.clip(rain,0,1)))/1000
            # Remove the source altitude before interpolation, then apply the DEM.
            sea_level = -45+np.clip(world.layers['temperature_'+season],0,1)*90+lapse*reference
            fields.extend((sea_level,rain*1000,lapse))
        source._terrain_koppen_seasons = cached = (key,fields)
    return np.asarray([sample_field(world,field,xs,ys,settings,polar=polar)
                       for field in cached[1]],np.float32)


def classify(elevation, fields, settings):
    def k(name):
        return setting(settings, 'koppen_'+name)
    h = np.maximum(elevation,0)
    ts, ps, ls, tw, pw, lw = fields
    ts, tw = np.clip(ts-ls*h,-45,45), np.clip(tw-lw*h,-45,45)
    hot, cold = np.maximum(ts,tw), np.minimum(ts,tw)
    annual = (ts+tw)*.5
    shoulder = hot-(hot-cold)*k('shoulder_fraction')
    summer, winter = np.where(ts>=tw,ps,pw), np.where(ts>=tw,pw,ps)
    total = summer+winter
    fraction = np.divide(summer,total,out=np.full_like(total,.5),where=total>0)
    threshold = np.maximum(0, k('aridity_temperature')*annual+np.where(
        fraction>=k('summer_fraction'), k('aridity_summer'), np.where(
        fraction<=k('winter_fraction'),0,k('aridity_mixed'))))
    pattern = np.where((summer<winter)&(summer/6<k('dry_summer_mm'))&
                       (summer<winter/k('dry_summer_ratio')),1,np.where(
                       (summer>=winter)&(winter<summer/k('dry_winter_ratio')),2,0))
    letter = np.where(hot>=k('hot_summer'),0,np.where(shoulder>=k('shoulder'),1,
                     np.where(cold>=k('extreme_cold'),2,3)))
    # C has three subletters, D has four; absent C*d falls back to Cfb.
    ids = np.where(cold>=k('temperate'),np.where(letter<3,8+pattern*3+letter,9),
                   17+pattern*4+letter)
    tropical = np.where(np.minimum(summer,winter)/6>=k('rainforest_mm'),1,np.where(
        total>=k('monsoon_factor')*(k('monsoon_mm')-np.minimum(summer,winter)/6),2,3))
    ids = np.where(cold>=k('tropical'),tropical,ids)
    arid = np.where(total<threshold*k('desert_fraction'),4,6)+(annual<k('hot_arid'))
    ids = np.where(total<threshold,arid,ids)
    ids = np.where(hot<k('tundra'),29,ids)
    ids = np.where(hot<k('ice'),30,ids)
    return np.where(elevation<0,0,ids).astype(np.uint8)


def sample_classes(world, xs, ys, settings, elevation=None, *, polar=False):
    fields = seasonal_fields(world,xs,ys,settings,polar=polar)
    if elevation is None:
        elevation = sample_field(world,world.height_m,xs,ys,settings,polar=polar)
    return classify(elevation,fields,settings)
