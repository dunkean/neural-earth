"""LOD-independent coarse transport: global cell centres, signed sqrt metres.

128 coarse cells cover exactly one existing geometry-LOD7 tile. The 16-cell
halo is aligned for three GPU mip reductions and is shared by neighbours.
"""
import numpy as np

CELLS = 128
HALO = 16
WIDTH = CELLS + 2 * HALO
STRIDE = 256
RESOLUTION = 30 * STRIDE
GEOMETRY_LOD = 7
CLIMATE_STRIDE = 4
CLIMATE_SIZE = WIDTH // CLIMATE_STRIDE + 1


def sample_axes(tx, ty, bounds):
    offsets = np.arange(-HALO, CELLS + HALO, dtype=np.float64) + .5
    # Clamp to the nearest *cell centre*, including partial world-edge cells.
    def axis(tile, low, high):
        indices = tile * CELLS + offsets
        first = np.floor(low / RESOLUTION) + .5
        last = np.ceil(high / RESOLUTION) - .5
        return np.clip(indices, first, last) * STRIDE
    return axis(tx, bounds[0], bounds[2]), axis(ty, bounds[1], bounds[3])


def climate_axes(tx, ty, bounds):
    # Include one endpoint beyond the height array. This keeps the compact
    # climate lattice globally aligned across blocks, including their halos.
    offsets = np.arange(CLIMATE_SIZE) * CLIMATE_STRIDE - HALO + .5
    def axis(tile, low, high):
        return np.clip(tile * CELLS + offsets, np.floor(low/RESOLUTION)+.5,
                       np.ceil(high/RESOLUTION)-.5) * STRIDE
    return axis(tx, bounds[0], bounds[2]), axis(ty, bounds[1], bounds[3])


def read_native(world, xs, ys, read_rect, check):
    """Gather actual fused coarse cells; never interpolate or filter heights."""
    xi = np.rint(xs / STRIDE - .5).astype(np.int64)
    yi = np.rint(ys / STRIDE - .5).astype(np.int64)
    x0, y0 = int(xi.min()), int(yi.min())
    data = read_rect(world, 'coarse', y0, x0, int(yi.max()) + 1,
                     int(xi.max()) + 1, check=check)
    root = (data[0] / data[-1].clamp_min(1e-8)).float().cpu().numpy()
    return np.ascontiguousarray(root[np.ix_(yi - y0, xi - x0)], dtype=np.float32)


def encode_height(height):
    height = np.asarray(height, dtype=np.float32)
    return np.ascontiguousarray(np.sign(height) * np.sqrt(np.abs(height)))
