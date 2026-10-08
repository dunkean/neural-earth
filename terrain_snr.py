"""Regional conditioning noise from the five already sampled coarse inputs.

This is noise/signal amplitude (the historical UI name is SNR), not its inverse.
Quantized ramps allow reusable embeddings; neutral gains skip all extra maths.
"""
import numpy as np

DRIVER_INDEX = {'temperature': 1, 'temperature_variation': 2,
                'precipitation': 3, 'precipitation_variation': 4}


def active(settings):
    return any(v != 1 for key in ('snr_altitude_gain', 'snr_driver_gain') for v in settings[key])


def _ramp(value, limits, bins):
    a, b = limits
    value = min(1., max(0., (float(value)-a)/(b-a)))
    return round(value*(bins-1))/(bins-1)


def window_snr(settings, baseline, inputs):
    """Return five amplitudes and two stable bucket coordinates, or neutral None.

    Inputs are CPU float32: signed sqrt elevation, BIO1, BIO4, BIO12, BIO15.
    Land-only means prevent nearby deep ocean from diluting mountain/cold rules.
    No additional raster access, latent read or neural forward is performed.
    """
    if not active(settings):
        return None
    fields = inputs.numpy() if hasattr(inputs, 'numpy') else np.asarray(inputs)
    z = fields[0]
    land = np.isfinite(z) & (z > 0)
    altitude = float(np.mean(np.square(z[land]), dtype=np.float64)) if land.any() else 0.
    driver = fields[DRIVER_INDEX[settings['snr_driver_channel']]]
    selected = driver[land & np.isfinite(driver)]
    if not len(selected):
        selected = driver[np.isfinite(driver)]
    other = float(np.mean(selected, dtype=np.float64)) if len(selected) else settings['snr_driver_range'][0]
    a = _ramp(altitude, settings['snr_altitude_range_m'], settings['snr_bins'])
    b = _ramp(other, settings['snr_driver_range'], settings['snr_bins'])
    gains = (1+(np.asarray(settings['snr_altitude_gain'])-1)*a) * (1+(np.asarray(settings['snr_driver_gain'])-1)*b)
    return tuple((np.asarray(baseline)*gains).tolist()), (a, b)
