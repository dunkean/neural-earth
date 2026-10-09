"""World extents and Cartesian/spherical coordinates, independent of UI views."""
import math

DEFAULT_DIAMETER_KM = 40_000 / math.pi
REFERENCE_BOUNDS = (-20_000_000., -10_000_000., 20_000_000., 10_000_000.)


def world_bounds(settings):
    if settings.get('world_topology', 'sphere') == 'sphere' and settings.get('world_diameter_km', DEFAULT_DIAMETER_KM) == DEFAULT_DIAMETER_KM:
        return REFERENCE_BOUNDS
    diameter = settings.get('world_diameter_km', DEFAULT_DIAMETER_KM) * 1000
    width = diameter if settings.get('world_topology', 'sphere') == 'plane' else diameter * math.pi
    height = diameter if settings.get('world_topology', 'sphere') == 'plane' else width / 2
    return (-width / 2, -height / 2, width / 2, height / 2)


def profile_bounds(profile):
    from terrain_generation import resolve_generation
    return world_bounds(resolve_generation(profile).settings)


def world_latitude(y, settings):
    import numpy as np
    if settings.get('world_topology', 'sphere') == 'plane':
        return np.zeros_like(np.asarray(y, dtype=np.float64))
    return np.clip(-np.asarray(y, dtype=np.float64) * 180 / (world_bounds(settings)[3] * 2), -90, 90)


def reference_coordinates(xs, ys, settings):
    """Stretch the immutable source atlas; keep NN cells in physical metres.

    A plane has finite, constant edges and no longitude wrap. Its source atlas
    may still come from the selected tectonic generator.
    """
    import numpy as np
    b = world_bounds(settings)
    xs = np.asarray(xs, np.float64) * (REFERENCE_BOUNDS[2] / b[2])
    ys = np.asarray(ys, np.float64) * (REFERENCE_BOUNDS[3] / b[3])
    if settings.get('world_topology', 'sphere') == 'plane':
        # Stay at source pixel centres so interpolation cannot cross the seam.
        xs = np.clip(xs, REFERENCE_BOUNDS[0] + 40_000_000 / 2048,
                     REFERENCE_BOUNDS[2] - 40_000_000 / 2048)
        ys = np.clip(ys, REFERENCE_BOUNDS[1], REFERENCE_BOUNDS[3])
    return xs, ys


class GeometryHeightmap:
    """Read-only view of a cached source atlas at the requested world size."""
    def __init__(self, source, settings):
        self.source, self.settings = source, settings
        self.bounds = world_bounds(settings)
        self.periodic_longitude = settings.get('world_topology', 'sphere') == 'sphere'

    def __getattr__(self, name):
        return getattr(self.source, name)

    def sample_height_m(self, xs, ys):
        from terrain_orogen_layers import sample
        return sample(self, self.height_m, xs, ys)


def geometry_heightmap(source, settings):
    return GeometryHeightmap(source, settings)
