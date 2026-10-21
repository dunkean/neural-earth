"""One geographic neural parent field for map and globe hydrology.

The trained ordinary and rotated charts are different realizations. Choose
and blend them by world position, never by camera. Near the poles and the
ordinary chart's longitude seam, the rotated chart is authoritative. The
blend is in physical metres, before coarse/base residual composition.
"""
import numpy as np

SAMPLING_VERSION = 'hydro-geographic-charts-v1'
SEAM_STRIP_M = 8*7680.


def _smooth(a, b, x):
    t = np.clip((x-a)/(b-a), 0., 1.)
    return t*t*(3.-2.*t)


def _periodic_chart(chart, x, y, bounds, sample_chart):
    """C1 two-sided strip for a nonperiodic learned realization.

    Both ordinary and rotated charts have a longitude seam. Mirror their
    end strips into one shared blend. The cubic distance warp has derivative
    zero at the seam and one at the far edge; it never samples outside the
    finite trained chart. This defines a new canonical strip, not a claim
    that the raw network is periodic.
    """
    width = bounds[2]-bounds[0]
    x = (np.asarray(x, np.float64)-bounds[0]) % width+bounds[0]
    y = np.asarray(y, np.float64)
    distance = np.minimum(x-bounds[0], bounds[2]-x)
    strip = min(SEAM_STRIP_M, width*.05)
    near = distance < strip
    out = np.empty(x.shape, np.float64)
    if (~near).any():
        out[~near] = sample_chart(chart, x[~near], y[~near])
    if near.any():
        d = distance[near]
        warped = d*d*(2*strip-d)/(strip*strip)
        left, right = bounds[0]+warped, bounds[2]-warped
        # Keep the exact endpoint inside the same finite chart convention.
        right = np.minimum(right, np.nextafter(bounds[2], bounds[0]))
        a = sample_chart(chart, left, y[near])
        b = sample_chart(chart, right, y[near])
        fade = .5*(1.-_smooth(0., strip, d))
        out[near] = np.where(x[near] < (bounds[0]+bounds[2])*.5,
                             (1.-fade)*a+fade*b, (1.-fade)*b+fade*a)
    return out


def canonical_field(xs, ys, settings, sample_chart):
    """Regular geographic grid; callback(chart, x_points, y_points) -> metres.

    The callback owns neural scheduling. Only the required points of each
    chart are read. All points and blending stay float64 until the result.
    """
    x, y = np.meshgrid(np.asarray(xs, np.float64), np.asarray(ys, np.float64))
    return canonical_points(x, y, settings, sample_chart)


def canonical_points(x, y, settings, sample_chart):
    """The same geographic parent field on arbitrary broadcast point arrays."""
    from terrain_geometry import world_bounds
    from terrain_polar import to_chart
    bounds = world_bounds(settings)
    x, y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    shape = x.shape
    x, y = x.ravel(), y.ravel()
    if settings.get('world_topology', 'sphere') != 'sphere':
        return np.asarray(sample_chart(None, x, y), np.float32).reshape(shape)
    period = bounds[2]-bounds[0]
    x = (x-bounds[0]) % period+bounds[0]
    y = np.clip(y, bounds[1], bounds[3])
    latitude = np.abs(y)/(bounds[3]-bounds[1])*180.
    longitude = np.abs(x-(bounds[0]+bounds[2])*.5)/period
    weight = np.maximum(_smooth(54., 64., latitude), _smooth(.45, .49, longitude))
    out = np.zeros(x.size, np.float64)
    ordinary = weight < 1.
    rotated = weight > 0.
    if ordinary.any():
        out[ordinary] += (1.-weight[ordinary])*_periodic_chart(None, x[ordinary], y[ordinary], bounds, sample_chart)
    if rotated.any():
        px, py = to_chart(x[rotated], y[rotated], settings)
        out[rotated] += weight[rotated]*_periodic_chart('polar', px, py, bounds, sample_chart)
    if not np.isfinite(out).all():
        raise ValueError('Non-finite canonical neural parent field')
    return out.astype(np.float32).reshape(shape)


def sample_points(world, x_m, y_m, source, read_block):
    """Bilinear NN samples in bounded source windows, supplied under a GPU job.

    ``read_block(world, source, x0, y0, x1, y1)`` returns physical metres.
    Integer coordinates address neural cells (7680 m coarse, 240 m base).
    Grouping limits host buffers; its order never affects the samples.
    """
    from scipy.ndimage import map_coordinates
    spacing = 7680. if source == 'coarse' else 240.
    cx = np.asarray(x_m, np.float64)/spacing-.5
    cy = np.asarray(y_m, np.float64)/spacing-.5
    group = np.stack((np.floor(cy/128).astype(np.int64),
                      np.floor(cx/128).astype(np.int64)), axis=-1)
    groups, inverse = np.unique(group, axis=0, return_inverse=True)
    out = np.empty(cx.shape, np.float64)
    for i in range(len(groups)):
        at = np.flatnonzero(inverse == i)
        x0, y0 = int(np.floor(cx[at].min()))-1, int(np.floor(cy[at].min()))-1
        x1, y1 = int(np.ceil(cx[at].max()))+2, int(np.ceil(cy[at].max()))+2
        field = read_block(world, source, x0, y0, x1, y1)
        out[at] = map_coordinates(field, [cy[at]-y0, cx[at]-x0], order=1, mode='nearest')
    return out
