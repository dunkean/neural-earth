"""Remove pixel-scale land/sea specks from the native 30 m DEM.

The decoder works in signed square-root space: ``elev = sign(s) * s**2``. Along
low coasts ``s`` hovers around zero with the decoder's high-frequency noise,
so isolated pixels flip between, say, -0.02 m and +0.03 m. Every renderer
classifies sea as ``height < 0``, which scatters single land/sea pixels.

Only *specks* change: 4-connected land or sea components of at most
MAX_PIXELS pixels whose heights all lie within +/-EPSILON. Their sign flips to
the surrounding one and their magnitude is kept, so a height changes by less
than 2 * EPSILON. Coastlines, islets and lagoons larger than a speck, and any
speck with real relief, are untouched: occasional discontinuities remain,
pixel-scale noise does not.

Reads carry a MAX_PIXELS halo. A speck touching the cropped area then lies
entirely inside the read, and a larger component crossing the read border
shows more than MAX_PIXELS pixels, so overlapping tile reads agree exactly.
"""
import numpy as np
from scipy import ndimage
import torch

VERSION = 'coast-specks-v1'
EPSILON = 1.0
MAX_PIXELS = 4
HALO = MAX_PIXELS


def remove_coast_specks(elev):
    """Return ``elev`` (2-D, metres) with near-zero land/sea specks flipped.

    Components touching the array border are never changed: their full extent
    is unknown. Callers needing agreement between reads use ``read_native``.
    """
    tensor = torch.as_tensor(elev)
    values = tensor.detach().to('cpu', torch.float32).numpy()
    land = values >= 0
    result = None
    for mask in (land, ~land):
        labels, count = ndimage.label(mask)
        if not count:
            continue
        sizes = np.bincount(labels.ravel(), minlength=count+1)
        small = sizes <= MAX_PIXELS
        small[0] = False
        # A component's largest |height| must stay within the noise band.
        peak = ndimage.maximum(np.abs(values), labels, index=np.arange(count+1))
        small &= np.asarray(peak) < EPSILON
        edge = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        small[edge] = False
        speck = small[labels]
        if speck.any():
            if result is None:
                result = values.copy()
            # Land specks become shallow sea, sea specks become low land.
            magnitude = np.maximum(np.abs(values[speck]), 1e-3)
            result[speck] = np.where(land[speck], -magnitude, magnitude)
    if result is None:
        return tensor.to(torch.float32)
    return torch.from_numpy(result).to(tensor.device)


def read_native(world, i1, j1, i2, j2):
    """Native DEM rectangle without coast specks, identical across reads."""
    data = world.get(i1-HALO, j1-HALO, i2+HALO, j2+HALO, with_climate=False)['elev']
    return remove_coast_specks(data)[HALO:-HALO, HALO:-HALO]
