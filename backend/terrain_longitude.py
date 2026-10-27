"""C1 longitude strip for finite neural DEMs, in physical metres.

The network is not periodic and the circumference need not be an integer
number of cells. Mirror both finite end strips into one canonical surface.
This is a display/source reconstruction; it does not alter trained weights.
"""
import numpy as np

VERSION = 'longitude-strip-v1'
STRIP_M = 8*7680.


def stitch(elevation, xs_m, bounds, lod, tx, read_tile, *, tile=256, halo=24, offset=.5):
    """Join a regular tile to its opposite end strip before derivatives/filtering.

    read_tile(column) supplies the same raw LOD and row, including its halo.
    Longitude remains in metres until sampling, with no rounded cell period.
    """
    width = bounds[2]-bounds[0]
    x = (np.asarray(xs_m, np.float64)-bounds[0]) % width + bounds[0]
    distance = np.minimum(x-bounds[0], bounds[2]-x)
    strip = min(STRIP_M, width*.05)
    near = distance < strip
    if not near.any():
        return elevation
    step = 30.*2.**lod
    cache = {tx: np.asarray(elevation, np.float32)}

    def sample(points):
        q = points/step-offset
        columns = np.floor(q/tile).astype(np.int64)
        out = np.empty((elevation.shape[0], len(points)), np.float64)
        for col in np.unique(columns):
            col = int(col)
            if col not in cache:
                cache[col] = np.asarray(read_tile(col), np.float32)
            at = np.flatnonzero(columns == col)
            local = q[at]-col*tile+halo
            i = np.floor(local).astype(int)
            t = local-i
            out[:, at] = cache[col][:, i]*(1-t)+cache[col][:, i+1]*t
        return out

    d = distance[near]
    warped = d*d*(2*strip-d)/(strip*strip)
    left = sample(bounds[0]+warped)
    right = sample(np.minimum(bounds[2]-warped, np.nextafter(bounds[2], bounds[0])))
    t = np.clip(d/strip, 0, 1)
    fade = .5*(1-t*t*(3-2*t))
    a = np.where(x[near] < (bounds[0]+bounds[2])*.5, left, right)
    b = np.where(x[near] < (bounds[0]+bounds[2])*.5, right, left)
    out = np.array(elevation, np.float32, copy=True)
    out[:, near] = a*(1-fade)+b*fade
    return out
