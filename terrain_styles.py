"""City display palettes and physical isolines; independent of terrain generation."""
import json
import math
from pathlib import Path


# Palette tokens adapted from city_generator (GPL-3.0); see README rendering section.
import numpy as np
from scipy.ndimage import gaussian_filter

from terrain_lighting import DEFAULT_LIGHTING

STYLES = json.loads(Path(__file__).with_name('terrain_styles.json').read_text(encoding='utf-8'))
MODES = tuple(STYLES)


def parse_contours(value):
    if value is None:
        return None
    data = json.loads(value)
    if not isinstance(data, dict) or set(data) - {'enabled', 'interval', 'automatic', 'width', 'density'}:
        raise ValueError('Invalid contour settings')
    if 'automatic' in data and not isinstance(data['automatic'], bool):
        raise ValueError('Invalid automatic contour setting')
    enabled, interval = data.get('enabled', False), data.get('interval', 200)
    if not isinstance(enabled, bool) or isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or not 1 <= interval <= 10000:
        raise ValueError('Invalid contour interval (1–10000 m)')
    width, density = data.get('width', .9), data.get('density', 25)
    for name, number, lower, upper in (('width', width, .3, 4), ('density', density, 3.125, 200)):
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f'Invalid contour {name} ({lower}–{upper})')
    return dict(enabled=enabled, interval=interval, width=width)


def _rgb(hex_value):
    return np.array([int(hex_value[i:i+2], 16)/255 for i in (1, 3, 5)])


def _paper_noise(x, y):
    """World-cell grain, matching WGSL integer wrapping and exact 24-bit output."""
    x = (np.asarray(x, dtype=np.int64) & 0xffffffff).astype(np.uint32)
    y = (np.asarray(y, dtype=np.int64) & 0xffffffff).astype(np.uint32)
    with np.errstate(over='ignore'):
        h = (x*np.uint32(0x9e3779b9)) ^ (y*np.uint32(0x85ebca6b))
        h = (h ^ (h >> 16))*np.uint32(0x7feb352d)
        h = (h ^ (h >> 15))*np.uint32(0x846ca68b)
        h ^= h >> 16
    return (h >> 8).astype(np.float64)/16777216


def render_style(elevation, resolution, mode, lighting=None, origin=(0, 0)):
    """Fixed 0–4500 m palette range keeps adjoining tiles and LODs consistent."""
    pal = STYLES[mode]
    s = dict(DEFAULT_LIGHTING, **(lighting or {}))
    sx, sy = resolution if isinstance(resolution, tuple) else (resolution, resolution)
    az, alt = np.radians([s['azimuth'], s['altitude']])
    light = (np.sin(az)*np.cos(alt), -np.cos(az)*np.cos(alt), np.sin(alt))
    def shade(sigma):
        dy, dx = np.gradient(gaussian_filter(elevation, sigma), sy, sx)
        dx, dy = dx*s['exaggeration'], dy*s['exaggeration']
        return (-dx*light[0]-dy*light[1]+light[2])/np.sqrt(dx*dx+dy*dy+1)
    dd = .75*shade(6)+.25*shade(1.2)-light[2]
    factor = np.clip(1+pal['shade']*np.where(dd>0, 1, .8)*dd*1.15, .58, 1.25)
    factor = 1+s['strength']*(factor-1)*(1-s['ambient'])/.65
    factor = np.maximum(factor, .01)**s['contrast']
    t = np.clip(elevation/4500, 0, 1)**(.85 if mode!='copernicus' else 1)
    stops = pal['hypso']
    rgb = np.stack([np.interp(t, [v[0] for v in stops], [_rgb(v[1])[c] for v in stops]) for c in range(3)], axis=-1)
    yy, xx = np.indices(elevation.shape)
    # Keep grain/hatching at sample scale; a 60 m minimum made 16 px blocks at LOD -3.
    px = np.floor(origin[0]/sx+xx+.5)
    py = np.floor(origin[1]/sy+yy+.5)
    noise = _paper_noise(px, py)
    grain = 1+(noise-.5)*2*pal['grain']
    if pal['hatch']:
        darkness = np.clip((1-factor)/.2, 0, 1)
        hatch = (np.mod(px+py, 6)<1.5)*darkness*pal['hatch']*.65*s['strength']
        rgb = rgb*((1+.3*(factor-1))*grain)[..., None]
        rgb = rgb*(1-hatch[..., None])+_rgb(pal['ink'])*hatch[..., None]
    else:
        rgb *= (factor*grain)[..., None]
    ocean = _rgb(pal['seaFill'])
    if mode=='copernicus':
        depth = np.clip(-elevation/10000, 0, 1)[..., None]
        ocean = (1-depth)*np.array([35,125,175])/255+depth*np.array([8,40,96])/255
    return np.clip(np.where((elevation<0)[..., None], ocean, rgb), 0, 1)


def apply_contours(rgb, elevation, settings, mode='relief'):
    if not settings or not settings['enabled']:
        return rgb
    import contourpy
    from PIL import Image, ImageDraw
    interval = settings['interval']
    pal = STYLES.get(mode, STYLES['topographic'])
    height, width = elevation.shape
    scale = 3
    mask = Image.new('L', (width*scale, height*scale))
    draw = ImageDraw.Draw(mask)
    generator = contourpy.contour_generator(z=elevation, line_type='Separate', corner_mask=False)
    maximum = float(np.nanmax(elevation))
    minimum = float(np.nanmin(elevation))
    for k in range(max(1, math.floor(minimum/interval)+1), math.ceil(maximum/interval)):
        major = k%5 == 0
        line_width = settings.get('width', .9)*(pal['contourIndexW'] if major else 1)
        alpha = round(255*pal['contourOpacity']*(1 if major else .7))
        for line in generator.lines(k*interval):
            if len(line) < 2:
                continue
            points = [((x+.5)*scale, (y+.5)*scale) for x, y in line]
            draw.line(points, fill=alpha, width=max(1, round(line_width*scale)), joint='curve')
    # Supersampling gives continuous antialiased strokes, even on steep terrain.
    alpha = np.asarray(mask.resize((width, height), Image.Resampling.LANCZOS), dtype=np.float32)/255
    alpha *= elevation>0
    return rgb*(1-alpha[..., None])+_rgb(pal['contour'])*alpha[..., None]
