"""Display-only relief lighting; physical tiles and NN caches stay unchanged."""
import hashlib
import json
import math

DEFAULT_LIGHTING = dict(strength=1., ambient=.35, contrast=1., exaggeration=6., azimuth=315., altitude=45.)
LIMITS = dict(strength=(0., 1.), ambient=(0., 1.), contrast=(.1, 3.),
              exaggeration=(0., 20.), azimuth=(0., 360.), altitude=(5., 90.))


def parse_lighting(value):
    if value is None:
        return None
    supplied = json.loads(value)
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_LIGHTING):
        raise ValueError('Invalid lighting settings')
    settings = dict(DEFAULT_LIGHTING, **supplied)
    for key, (lo, hi) in LIMITS.items():
        v = settings[key]
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError('Invalid lighting ' + key)
    return settings


def lighting_suffix(settings):
    if settings is None:
        return ''
    return '.light-' + hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def relief_intensity(elevation, resolution, settings=None):
    """Terrain illumination, independent of altitude palette or biome color."""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    s = dict(DEFAULT_LIGHTING, **(settings or {}))
    az, alt = math.radians(s['azimuth']), math.radians(s['altitude'])
    light = (math.sin(az)*math.cos(alt), -math.cos(az)*math.cos(alt), math.sin(alt))
    sx, sy = resolution if isinstance(resolution, tuple) else (resolution, resolution)
    def shade(sigma):
        dy, dx = np.gradient(gaussian_filter(elevation, sigma), sy, sx)
        dx, dy = dx*s['exaggeration'], dy*s['exaggeration']
        return np.clip((dx*light[0]+dy*light[1]+light[2])/np.sqrt(dx*dx+dy*dy+1), 0, 1)
    hs = np.power(.75*shade(6.)+.25*shade(1.2), .85*s['contrast'])
    return (1-s['strength']) + s['strength']*(s['ambient']+(1-s['ambient'])*hs)


def render_relief(elevation, resolution, settings):
    import numpy as np
    from matplotlib import colormaps
    intensity = relief_intensity(elevation, resolution, settings)
    rgb = colormaps['terrain'](.25+.75*np.power(np.clip(elevation/4500, 0, 1), .7))[..., :3]
    rgb *= intensity[..., None]
    depth = np.power(np.clip(-elevation/10000, 0, 1), .7)[..., None]
    ocean = (1-depth)*np.array([.68,.88,1.])+depth*np.array([0.,.1,.45])
    return np.asarray(np.where((elevation<0)[..., None], ocean, rgb), np.float32)
