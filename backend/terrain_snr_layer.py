"""Display the actual conditioning-noise policy of each coarse NN window.

Each cell belongs to the nearest 64 x 64 window centre (48-sample stride).
The displayed number is the window policy, not a measured output-image SNR.
"""
from functools import lru_cache
import numpy as np
from terrain_snr import active, window_snr

CHANNELS = ('elevation', 'temperature', 'temperature-variation',
            'precipitation', 'precipitation-variation')
MODES = tuple('snr-' + channel for channel in CHANNELS)
RESOLUTION = 7680
STRIDE = 48


def scale_bounds(settings, baseline, channel):
    base = baseline[channel]
    if not settings.get('snr_adaptive_enabled',True):return base*.25,base*4
    a, b = settings['snr_altitude_gain'][channel], settings['snr_driver_gain'][channel]
    c = settings.get('snr_latitude_gain', 1.) if channel == 0 else 1.
    return (base * min(.25, min(1, a) * min(1, b) * min(1, c)),
            base * max(4, max(1, a) * max(1, b) * max(1, c)))


@lru_cache(maxsize=16384)
def _window(seed, profile, baseline, row, column):
    from terrain_conditioning import make_conditioning_factory
    from terrain_generation import resolve_generation
    settings = resolve_generation(profile).settings
    x, y = column * STRIDE, row * STRIDE
    inputs = make_conditioning_factory(seed, profile)(x, y, x + 64, y + 64)
    from terrain_geometry import world_latitude
    return window_snr(settings, baseline, inputs, latitude=float(world_latitude((y+32)*RESOLUTION, settings)))[0]


def sample(seed, profile, baseline, xs, ys, channel):
    from terrain_generation import resolve_generation
    settings = resolve_generation(profile).settings
    baseline = tuple(baseline)
    if not active(settings):
        return np.full((len(ys), len(xs)), baseline[channel], np.float32)
    # The centre of window k is (48*k + 32) coarse samples from the origin.
    columns = np.floor((np.asarray(xs) / RESOLUTION - 8) / STRIDE).astype(np.int64)
    rows = np.floor((np.asarray(ys) / RESOLUTION - 8) / STRIDE).astype(np.int64)
    unique_x, ix = np.unique(columns, return_inverse=True)
    unique_y, iy = np.unique(rows, return_inverse=True)
    values = np.empty((len(unique_y), len(unique_x)), np.float32)
    for i, row in enumerate(unique_y):
        for j, column in enumerate(unique_x):
            values[i, j] = _window(seed, profile, baseline, int(row), int(column))[channel]
    return values[iy[:, None], ix[None, :]]


def render(seed, profile, baseline, mode, xs, ys):
    from terrain_generation import resolve_generation
    channel = MODES.index(mode)
    values = sample(seed, profile, baseline, xs, ys, channel)
    low, high = scale_bounds(resolve_generation(profile).settings, baseline, channel)
    t = np.clip(np.log(values / low) / np.log(high / low), 0, 1)
    stops = np.array([[.08, .18, .38], [.12, .68, .72], [1., .78, .18], [.88, .2, .13]])
    return np.stack([np.interp(t, np.linspace(0, 1, len(stops)), stops[:, c])
                     for c in range(3)], axis=-1)
