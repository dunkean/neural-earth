"""Natural resources on Neural Earth's LOD lattice: provinces, stocks, flows, budgets.

Pure CPU NumPy/SciPy: no Flask, torch, network or clock. One spatial address
for every scale: cell (lod, i, j) covers [i, i+1) x [j, j+1) * 30 * 2**lod map
metres, origin at the map centre, y southward, exactly like display tiles.
The world authority is evaluated once at LOD 10 (30.72 km) from the retained
Orogen atlas. Finer LODs only redistribute committed integer budgets; coarser
LODs and regions only sum them. Every random draw is a pure function of
(seed, layer, lod, cell) using the Chronos key derivation (FNV-1a + splitmix64).

Units: area m2 (cells), ha (yields), km2 (densities); mass t (kg for gold and
silver); wood m3; temperature degC; month 0 = January. All outputs are a
procedural prior ("derived" from Orogen inputs or "synthetic" when keyed noise
decides), not a survey. Design and limits: docs/RESOURCE_LAYERS.md.
"""
from __future__ import annotations

import functools
import math
import os

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter, minimum_filter
from scipy.special import ndtri

VERSION = 'resources-v1-prototype'
NATIVE_M = 30.0
# Cache identity of the compiled authority (terrain_resource_layers): bump
# COMPILER_VERSION whenever a canonical output of compile_world changes
# (values, fields, deposits, keys); display-only edits (layers, WGSL, mirror
# colours) must not bump it. V2_INPUTS_VERSION names the inputs given to the
# Render v2 evaluator (sampling of the atlas); its cached environment also
# depends on the source of terrain_render_v2.py, an external module.
COMPILER_VERSION = 'resources-compiler-3'
V2_INPUTS_VERSION = 'v2-inputs-1'
WORLD_LOD = 10           # authority grid, 30.72 km
BUDGET_LOD = 3           # 240 m: last level allocated from world information only
MIN_LOD = -6             # addressing limit (47 cm), see cell_id
BOUNDS = (-20_000_000., -10_000_000., 20_000_000., 10_000_000.)
RADIUS_M = (BOUNDS[2] - BOUNDS[0]) / (2 * math.pi)
MONTHS = 12

CONTEXTS = ('arc', 'collision', 'rift', 'hotspot', 'backarc', 'ophiolite',
            'craton', 'passive', 'platform', 'oceanic')
CONTEXT_LABELS = ('Arc de subduction', 'Collision continentale', 'Rift continental', 'Point chaud',
                  'Bassin arrière-arc', 'Ophiolite / croûte océanique obductée', 'Craton',
                  'Marge passive', 'Plate-forme sédimentaire', 'Domaine océanique')
ROCKS = ('granite', 'limestone', 'sandstone', 'marl', 'basalt', 'metamorphic', 'ultramafic', 'evaporite')
ROCK_LABELS = ('Granite et roches plutoniques acides', 'Calcaire', 'Grès', 'Argilo-marneux',
               'Basalte et volcanites', 'Métamorphique (schiste, gneiss, ardoise)', 'Ultrabasique',
               'Évaporites')
# Proportions of each rock class in each tectonic context (rows sum to 1).
# Proposed associations, not calibrated against a real lithological map.
LITHOLOGY = np.array([
    # granite limest sandst marl  basalt meta  ultra  evap
    [.25, .05, .15, .10, .40, .05, .00, .00],   # arc: plutons and andesitic volcanics
    [.25, .12, .13, .10, .02, .38, .00, .00],   # collision: metamorphic belt, S-type granites
    [.05, .08, .32, .20, .30, .03, .00, .02],   # rift: red beds, flood basalt, evaporites
    [.02, .03, .05, .05, .85, .00, .00, .00],   # hotspot
    [.05, .15, .25, .30, .22, .03, .00, .00],   # back-arc basin
    [.02, .05, .05, .13, .40, .05, .30, .00],   # ophiolite
    [.45, .03, .10, .04, .08, .30, .00, .00],   # craton: granite-gneiss, greenstone
    [.03, .35, .28, .30, .02, .02, .00, .00],   # passive margin
    [.04, .30, .32, .30, .02, .02, .00, .00],   # platform
    [.00, .10, .05, .15, .70, .00, .00, .00],   # oceanic floor
], np.float64)
assert np.allclose(LITHOLOGY.sum(axis=1), 1)
# Realized geology (see realize_rocks): fixed noise layer of the rock classes
# and the classes that form belts along the structural trend.
ROCK_LAYER = 4001
ROCK_ALIGNED = (False, True, True, True, False, True, True, True)   # granite, basalt isotropic
# Province groups: plutonic, volcanic, metamorphic (+ ultramafic sutures), sedimentary.
ROCK_GROUPS = (0, 3, 3, 3, 1, 2, 2, 3)
ROCK_GROUP_ALIGNED = (False, False, True, False)                   # metamorphic belts

# Medieval-relevant deposit families. contexts: weight per CONTEXTS entry;
# hosts: weight per ROCKS entry; metals: contained mass fraction of ore (t/t);
# density: expected deposits per 10 000 km2 at prospectivity 1; tonnage: median
# ore t and log10 standard deviation (lognormal); depth: median top depth m.
# Context and host associations follow USGS descriptive models (Cox & Singer
# 1986, Bulletin 1693). Median tonnage/grade of porphyry Cu (17), Cyprus VMS
# (24a), low-sulfide Au-quartz veins (36a), placer Au-PGE (39a), Sn veins
# (15b) and Superior/Algoma Fe are read from its grade-tonnage appendix; the
# other medians, every log10 spread and every density are provisional.
FAMILIES = {
    'porphyry_cu':      dict(contexts=dict(arc=1., backarc=.2, collision=.15), hosts=dict(granite=1., basalt=.7),
                             metals=dict(copper=5.4e-3, gold=2e-7, silver=2e-6), density=.15, tonnage=(1.44e8, .65), depth=60, sulfide=True),
    'epithermal_au_ag': dict(contexts=dict(arc=.8, backarc=.6, rift=.4, hotspot=.3), hosts=dict(basalt=1., sandstone=.3),
                             metals=dict(gold=5e-6, silver=1e-4), density=.6, tonnage=(8e5, .7), depth=15, sulfide=True),
    'sn_w_granite':     dict(contexts=dict(collision=1., craton=.3, arc=.1), hosts=dict(granite=1., metamorphic=.4),
                             metals=dict(tin=1.3e-2, tungsten=3e-3), density=.5, tonnage=(2.4e5, .7), depth=20, sulfide=False),
    'polymetallic_vein': dict(contexts=dict(arc=.5, collision=.7, rift=.3, craton=.15), hosts=dict(granite=.8, metamorphic=1., sandstone=.6, limestone=.4),
                             metals=dict(lead=6e-2, zinc=4e-2, silver=3e-4, copper=2e-3), density=.8, tonnage=(2e5, .7), depth=10, sulfide=True),
    'vms':              dict(contexts=dict(ophiolite=1., backarc=.8, arc=.3), hosts=dict(basalt=1., ultramafic=.6, marl=.3),
                             metals=dict(copper=1.6e-2, zinc=2e-2, lead=3e-3, gold=1e-6, silver=3e-5), density=.4, tonnage=(1.3e6, .75), depth=25, sulfide=True),
    'orogenic_au':      dict(contexts=dict(collision=.8, craton=.6), hosts=dict(metamorphic=1., basalt=.6, granite=.3),
                             metals=dict(gold=1.6e-5, silver=2e-6), density=.5, tonnage=(3e4, .8), depth=10, sulfide=True),
    'mvt_pb_zn':        dict(contexts=dict(platform=.7, passive=.6, backarc=.2), hosts=dict(limestone=1.),
                             metals=dict(lead=2e-2, zinc=6e-2, silver=1e-5), density=.15, tonnage=(5e6, .7), depth=40, sulfide=True),
    'sediment_cu':      dict(contexts=dict(rift=1., platform=.3), hosts=dict(sandstone=1., marl=.8),
                             metals=dict(copper=2e-2, silver=3e-5), density=.1, tonnage=(2e7, .7), depth=50, sulfide=True),
    'skarn_fe_cu':      dict(contexts=dict(arc=.5, collision=.4), hosts=dict(limestone=1., granite=.5),
                             metals=dict(iron=.45, copper=5e-3), density=.3, tonnage=(3e6, .7), depth=15, sulfide=False),
    'bif_iron':         dict(contexts=dict(craton=1.), hosts=dict(metamorphic=1., basalt=.3),
                             metals=dict(iron=.50), density=.08, tonnage=(1.65e8, .8), depth=5, sulfide=False),
    'ironstone':        dict(contexts=dict(platform=.8, passive=.8, backarc=.3), hosts=dict(marl=1., sandstone=.7, limestone=.6),
                             metals=dict(iron=.30), density=.25, tonnage=(2e7, .8), depth=20, sulfide=False),
    'laterite_fe':      dict(contexts=dict(craton=.6, hotspot=.6, ophiolite=.6, platform=.2), hosts=dict(basalt=1., ultramafic=1., granite=.4),
                             metals=dict(iron=.45), density=.3, tonnage=(5e6, .7), depth=1, sulfide=False, climate='tropical'),
    'evaporite_salt':   dict(contexts=dict(rift=.8, backarc=.5, passive=.4, platform=.5), hosts=dict(evaporite=1., marl=.15),
                             metals=dict(salt=.9, gypsum=.05), density=.2, tonnage=(5e7, .8), depth=80, sulfide=False, requires='evaporite'),
    'coal':             dict(contexts=dict(platform=.6, passive=.4, backarc=.4, collision=.2), hosts=dict(sandstone=1., marl=1.),
                             metals=dict(coal=.9), density=.1, tonnage=(5e7, .8), depth=60, sulfide=False, climate='humid'),
}
# Placers are fed by eroded primary sources upstream, never by a new province.
PLACERS = {
    'placer_au': dict(source=('epithermal_au_ag', 'orogenic_au', 'porphyry_cu'), metals=dict(gold=2e-7), density=.8, tonnage=(1.1e6, .7), depth=2),
    'placer_sn': dict(source=('sn_w_granite',), metals=dict(tin=3e-3), density=.6, tonnage=(3e5, .7), depth=3),
}
FANTASY = {
    # Isolated optional extension: own key domain, never alters a non-fantasy layer.
    'fantasy_star_metal': dict(contexts=dict(hotspot=1., collision=.3), hosts=dict(basalt=1., ultramafic=1.),
                               metals=dict(star_metal=2e-5), density=.02, tonnage=(2e5, .6), depth=40, sulfide=False),
}
GRAM_METALS = {'gold', 'silver', 'star_metal'}   # integers in kg, others in t
MEDIEVAL_RECOVERY = dict(gold=.8, silver=.7, copper=.6, lead=.75, zinc=0., tin=.7, tungsten=0., iron=.5,
                         salt=.9, gypsum=.9, coal=.8, star_metal=.5)  # zinc/tungsten: no medieval metal

FNV_OFFSET, FNV_PRIME, MASK = 0xcbf29ce484222325, 0x100000001b3, (1 << 64) - 1


# ---------------------------------------------------------------------------
# Stable keys: identical to chronos-core keys.rs; cells are mixed by splitmix.

def splitmix(z):
    z = (z + 0x9e3779b97f4a7c15) & MASK
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & MASK
    return z ^ (z >> 31)


def key(seed, parts):
    """Chronos `key(seed, parts)`: FNV-1a over parts with 0xff separators."""
    return _key(int(seed), tuple(str(part) for part in parts))


@functools.lru_cache(maxsize=65536)
def _key(seed, parts):
    h = FNV_OFFSET ^ splitmix(int(seed) & MASK)
    for part in parts:
        for b in str(part).encode():
            h = ((h ^ b) * FNV_PRIME) & MASK
        h = ((h ^ 0xff) * FNV_PRIME) & MASK
    return splitmix(h)


def splitmix_np(z):
    z = np.asarray(z, np.uint64)
    with np.errstate(over='ignore'):
        z = z + np.uint64(0x9e3779b97f4a7c15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return z ^ (z >> np.uint64(31))


def cell_keys(seed, parts, ids, draw=0):
    """key_cell = splitmix(key(seed, parts) ^ cell_id), then ^ draw index."""
    prefix = np.uint64(key(seed, parts))
    k = splitmix_np(prefix ^ np.asarray(ids, np.uint64))
    return splitmix_np(k ^ np.uint64(draw)) if draw else k


def to_unit(k):
    """Uniform [0, 1) from 53 high bits, as Chronos `unit`."""
    return (np.asarray(k, np.uint64) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


# ---------------------------------------------------------------------------
# LOD lattice

def cell_size(lod):
    return NATIVE_M * 2.0 ** lod


def cell_id(lod, i, j):
    """64-bit id: 5 bits lod+16, 30 bits i+2**29, 29 bits j+2**28 (LOD >= -6)."""
    if lod < MIN_LOD or lod > 15:
        raise ValueError('resource addressing covers LOD -6..15')
    i, j = np.asarray(i, np.int64), np.asarray(j, np.int64)
    return ((np.uint64(lod + 16) << np.uint64(59)) | ((i + (1 << 29)).astype(np.uint64) << np.uint64(29))
            | (j + (1 << 28)).astype(np.uint64))


def decode_id(ids):
    ids = np.asarray(ids, np.uint64)
    lod = (ids >> np.uint64(59)).astype(np.int64) - 16
    i = ((ids >> np.uint64(29)) & np.uint64((1 << 30) - 1)).astype(np.int64) - (1 << 29)
    j = (ids & np.uint64((1 << 29) - 1)).astype(np.int64) - (1 << 28)
    return lod, i, j


def parent_index(i, j, up):
    """Arithmetic shift: floor division by 2**up for negative indices too."""
    return np.asarray(i, np.int64) >> up, np.asarray(j, np.int64) >> up


def latitude(y):
    return 90 - (np.asarray(y, np.float64) - BOUNDS[1]) / (BOUNDS[3] - BOUNDS[1]) * 180


class Lattice:
    """Cells of one LOD intersecting a map window, with exact spherical areas.

    Edge cells are clipped to the world bounds (seam and poles); their area and
    sample point are those of the clipped part, so areas sum to 4 pi R^2.
    """

    def __init__(self, lod, window=None):
        s = cell_size(lod)
        x0, y0, x1, y1 = BOUNDS if window is None else window
        self.lod, self.size = lod, s
        self.i = np.arange(math.floor(x0 / s), math.ceil(x1 / s), dtype=np.int64)
        self.j = np.arange(math.floor(y0 / s), math.ceil(y1 / s), dtype=np.int64)
        ax0 = np.maximum(self.i * s, max(x0, BOUNDS[0])); ax1 = np.minimum((self.i + 1) * s, min(x1, BOUNDS[2]))
        ay0 = np.maximum(self.j * s, max(y0, BOUNDS[1])); ay1 = np.minimum((self.j + 1) * s, min(y1, BOUNDS[3]))
        self.x, self.y = (ax0 + ax1) / 2, (ay0 + ay1) / 2
        north, south = np.radians(latitude(ay0)), np.radians(latitude(ay1))
        # Exact equirectangular cell area on the sphere: dx * R * (sin(north) - sin(south)).
        self.area_m2 = (np.maximum(ax1 - ax0, 0)[None, :] * RADIUS_M
                        * np.maximum(np.sin(north) - np.sin(south), 0)[:, None])
        self.lat = np.broadcast_to(latitude(self.y)[:, None], self.area_m2.shape)
        self.ids = cell_id(lod, self.i[None, :], self.j[:, None])

    @property
    def shape(self):
        return self.area_m2.shape


# ---------------------------------------------------------------------------
# Exact integer apportionment (largest remainder, ties by ascending cell id)

def quantize(weights, bits=20):
    """Non-negative float weights -> integers <= 2**bits, preserving zeros."""
    w = np.nan_to_num(np.maximum(np.asarray(weights, np.float64), 0))
    top = w.max(axis=-1, keepdims=True) if w.ndim else w
    q = np.floor(np.divide(w, top, out=np.zeros_like(w), where=top > 0) * (1 << bits))
    return np.where(w > 0, np.maximum(q, 1), 0).astype(np.int64)


def apportion(total, weights, ids):
    """Split integer ``total`` by integer ``weights``; sum is exact.

    Pure integer arithmetic (totals < 2**40, weights <= 2**20) so Rust, NumPy
    and any request order produce the same result. A zero-weight set receives
    the total uniformly, so a committed budget is never lost.
    """
    total = int(total)
    w = np.asarray(weights, np.int64)
    if total < 0 or (w < 0).any():
        raise ValueError('apportion needs non-negative integers')
    if total == 0 or w.size == 0:
        return np.zeros(w.shape, np.int64)
    if w.sum() == 0:
        w = np.ones_like(w)
    if total >= 1 << 40 or w.max() > 1 << 20 or w.size > 1 << 22:
        raise ValueError('apportion bounds exceeded')
    s = int(w.sum())
    share, rest = np.divmod(total * w, s)
    left = total - int(share.sum())
    order = np.lexsort((np.asarray(ids, np.uint64), -rest))
    share[order[:left]] += 1
    return share


# ---------------------------------------------------------------------------
# Sampling the Orogen atlas (equirectangular pixel centres, periodic longitude)

def sample_grid(field, xs, ys, *, categorical=False):
    field = np.asarray(field)
    h, w = field.shape
    x0, y0, x1, y1 = BOUNDS
    xx = ((np.asarray(xs) - x0) % (x1 - x0)) / (x1 - x0) * w - .5
    yy = np.clip((np.asarray(ys) - y0) / (y1 - y0) * h - .5, 0, h - 1)
    if categorical:
        return field[np.rint(yy).astype(np.int64)[:, None], (np.floor(xx + .5).astype(np.int64) % w)[None, :]]
    ix, iy = np.floor(xx).astype(np.int64), np.floor(yy).astype(np.int64)
    tx, ty = (xx - ix)[None, :], (yy - iy)[:, None]
    by = np.minimum(iy + 1, h - 1)
    a = field[iy][:, ix % w] * (1 - tx) + field[iy][:, (ix + 1) % w] * tx
    b = field[by][:, ix % w] * (1 - tx) + field[by][:, (ix + 1) % w] * tx
    return (a * (1 - ty) + b * ty).astype(np.float32)


def _wrap_edt(mask, spacing_km, pad=128):
    """Distance (km) to True cells, wrapping longitude. Planar metric (see doc)."""
    if not mask.any():
        return np.full(mask.shape, 1e9, np.float32)
    padded = np.concatenate((mask[:, -pad:], mask, mask[:, :pad]), axis=1)
    d = distance_transform_edt(~padded, sampling=spacing_km)
    return d[:, pad:-pad].astype(np.float32)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def value_noise(seed, name, lod, x, y):
    """Keyed value noise on the LOD lattice, bilinear: the same at every scale."""
    s = cell_size(lod)
    gx, gy = np.asarray(x, np.float64) / s - .5, np.asarray(y, np.float64) / s - .5
    i0, j0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = gx - i0, gy - j0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    period = int(round((BOUNDS[2] - BOUNDS[0]) / s))
    out = 0
    for di, wx in ((0, 1 - fx), (1, fx)):
        for dj, wy in ((0, 1 - fy), (1, fy)):
            ii = (i0 + di + period // 2) % period - period // 2
            out = out + wx * wy * (2 * to_unit(cell_keys(seed, ('resources', 'noise', name, f'lod{lod}'),
                                                         cell_id(lod, ii, j0 + dj))) - 1)
    return out


# ---------------------------------------------------------------------------
# World authority (LOD 10)

def tectonic_contexts(layers, height):
    """Soft context fractions at atlas resolution from Orogen plate fields."""
    h, w = height.shape
    spacing = (BOUNDS[2] - BOUNDS[0]) / w / 1000
    crust = np.asarray(layers['crust']) > .5
    kind = np.rint(np.asarray(layers['boundaries'])).astype(np.int8)
    land = height > 0
    cmax = maximum_filter(crust.astype(np.uint8), 5, mode=('nearest', 'wrap')) > 0
    cmin = minimum_filter(crust.astype(np.uint8), 5, mode=('nearest', 'wrap')) > 0
    subduction = (kind == 1) & cmax & ~cmin
    island_arc = (kind == 1) & cmin
    collision = (kind == 1) & ~cmax
    rift = (kind == 2) & ~cmax
    d = {name: _wrap_edt(mask, spacing) for name, mask in
         dict(sub=subduction, io=island_arc, col=collision, rift=rift, conv=kind == 1, sea=~land).items()}
    ocean_plate = crust.astype(np.float64)
    stress = np.asarray(layers['convergence'], np.float64)
    # Orogen boundary zones are several pixels wide: buffers are kept narrow.
    arc = np.exp(-(d['sub'] / 300) ** 2) * (1 - ocean_plate) + np.exp(-(d['io'] / 200) ** 2) * ocean_plate
    col = np.exp(-(d['col'] / 250) ** 2)
    rif = np.exp(-(d['rift'] / 200) ** 2)
    hot = np.clip(np.asarray(layers['hotspot']) / .25, 0, 1)
    back = np.clip(-np.asarray(layers['backArc']) / .06, 0, 1) ** 2
    # Oceanic plate above sea level (obducted or accreted) and suture cores.
    ophi = ocean_plate * land * .7 + np.exp(-(d['col'] / 80) ** 2) * .3
    # Cratons: continental interiors far from any convergent zone, low stress
    # and no young uplift (Orogen's tecActivity saturates on busy worlds).
    craton = ((1 - ocean_plate) * smoothstep(350, 900, d['conv']) * (1 - smoothstep(.3, .9, stress))
              * (1 - smoothstep(0, .08, layers['uplift'])))
    passive = (1 - ocean_plate) * np.exp(-(d['sea'] / 250) ** 2) * (1 - np.maximum(arc, col))
    platform = (1 - ocean_plate) * .5
    oceanic = ocean_plate * (1 - land) + .02
    weights = np.stack((3 * arc, 3 * col, 3 * rif, 3 * hot, 1.5 * back, 2.5 * ophi,
                        3 * craton, 1.5 * passive, platform, oceanic))
    return weights / weights.sum(axis=0), d['sea']


def _atlas_xy(shape):
    h, w = shape
    x0, y0, x1, y1 = BOUNDS
    return x0 + (np.arange(w) + .5) * (x1 - x0) / w, y0 + (np.arange(h) + .5) * (y1 - y0) / h


def lithology(contexts, aridity, lowland, seed, xs, ys):
    """Expected rock-class shares: context table and aridity-dependent evaporites.
    The canonical shares are realized from them by realize_rocks."""
    rock = np.einsum('kyx,kr->ryx', contexts, LITHOLOGY)
    basin = contexts[CONTEXTS.index('rift')] + contexts[CONTEXTS.index('backarc')] + \
        contexts[CONTEXTS.index('passive')] + contexts[CONTEXTS.index('platform')]
    X, Y = np.meshgrid(xs, ys)
    # Evaporites record past basins: present aridity is only one proxy, a keyed
    # province field supplies "paleo-arid" basins independent of today's climate.
    paleo = np.clip(value_noise(seed, 'paleo-evaporite', 13, X, Y), 0, 1)
    rock[ROCKS.index('evaporite')] += .10 * basin * lowland * np.maximum(aridity, .6 * paleo)
    return rock / rock.sum(axis=0)


def structural_strike(f):
    """Structural azimuth (radians, grid frame) parallel to orogenic belts:
    normal to the gradient of the smoothed collision, arc, rift and back-arc contexts."""
    belt = sum(np.asarray(f.get('context_' + c, 0), np.float64) for c in ('collision', 'arc', 'rift', 'backarc'))
    belt = gaussian_filter(belt, 1.5, mode=('nearest', 'wrap'))
    gy, gx = np.gradient(belt)
    return np.arctan2(gy, gx) + np.pi / 2


def realized_rock_field(expected, strike, bias):
    """(K + 2, H, W) field of realized_class: biased log expected shares, cos 2θ, sin 2θ."""
    logs = np.log(np.maximum(expected, 1e-4)) + np.asarray(bias, np.float64)[:, None, None]
    return np.concatenate((logs, np.cos(2 * strike)[None], np.sin(2 * strike)[None])).astype(np.float32)


def realized_classes(field, i0, j0, cells, side, lod, seed32):
    """Realized rock class of side x side samples in each LOD-10 cell (rows, cols):
    (n, side * side). Pure function of its arguments (worker processes)."""
    import terrain_resource_gpu as G
    pkg = dict(seed32=seed32, layer=ROCK_LAYER, classes=len(ROCKS), aligned=ROCK_ALIGNED, groups=ROCK_GROUPS,
               group_aligned=ROCK_GROUP_ALIGNED, field=G.Field(field, i0, j0))
    sub = np.arange(side)
    gi = (i0 + cells[:, 1])[:, None, None] * side + sub[None, None, :] + np.zeros((1, side, 1), np.int64)
    gj = (j0 + cells[:, 0])[:, None, None] * side + sub[None, :, None] + np.zeros((1, 1, side), np.int64)
    return G.realized_class(pkg, gi, gj, lod).reshape(len(cells), side * side)


def _classes(field, grid, cells, side, lod, seed32, parallel, chunks=16):
    if parallel is None or len(cells) < 20000:
        return realized_classes(field, grid.i[0], grid.j[0], cells, side, lod, seed32)
    bounds = np.linspace(0, len(cells), chunks + 1).astype(int)
    tasks = [('terrain_resources', 'realized_classes',
              dict(field=field, i0=int(grid.i[0]), j0=int(grid.j[0]), cells=cells[a:b], side=side, lod=lod, seed32=seed32))
             for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    return np.concatenate(parallel(tasks))


def calibrate_rock_bias(expected, strike, land, grid, seed, parallel=None, iterations=6, stride=8):
    """Per-class bias of the realized geology so that world shares follow the
    expected ones: log-ratio iterations on one sample (LOD 10) in every
    `stride`-th land cell. Cheap: the display needs only this, not the shares."""
    kcls = len(ROCKS)
    bias = np.zeros(kcls)
    if grid.lod != WORLD_LOD:
        return bias
    seed32 = key(seed, ['resources', 'gpu']) & 0xffffffff
    cells = np.argwhere(land)[::stride]
    area = grid.area_m2[cells[:, 0], cells[:, 1]]
    target = (expected[:, cells[:, 0], cells[:, 1]] * area).sum(axis=1) / area.sum()
    for _ in range(iterations):
        # Serial: a few tens of thousands of samples, cheaper than starting workers.
        cls = realized_classes(realized_rock_field(expected, strike, bias), grid.i[0], grid.j[0], cells, 1, WORLD_LOD, seed32)[:, 0]
        got = np.bincount(cls, weights=area, minlength=kcls) / area.sum()
        bias += np.log(np.maximum(target, 1e-6) / np.maximum(got, 1e-6)) * (target > 0)
    return bias


def realize_rocks(expected, strike, land, grid, seed, bias=None, side=2, parallel=None):
    """Canonical rock shares of each land cell: area shares of the realized
    geology (terrain_resource_gpu.realized_class) on side x side samples per
    LOD-10 cell (LOD 9 for side 2: 15 km, a quarter of the shortest noise
    wavelength, 61 km). The display evaluates the same function at every LOD,
    so it agrees with these shares (tolerance measured in RESOURCE_LAYERS §16.3).
    Returns (shares (K, H, W), bias (K,))."""
    kcls = len(ROCKS)
    if grid.lod != WORLD_LOD:          # the realization is defined on the LOD-10 authority
        return expected.copy(), np.zeros(kcls)
    if bias is None:
        bias = calibrate_rock_bias(expected, strike, land, grid, seed, parallel)
    seed32 = key(seed, ['resources', 'gpu']) & 0xffffffff
    cells = np.argwhere(land)
    lod = WORLD_LOD - int(round(math.log2(side)))
    cls = _classes(realized_rock_field(expected, strike, bias), grid, cells, side, lod, seed32, parallel)
    shares = expected.copy()
    for k in range(kcls):
        shares[k][cells[:, 0], cells[:, 1]] = (cls == k).mean(axis=1)
    return shares, bias


def _months(temp_summer, temp_winter, precip_summer, precip_winter, lat):
    """Monthly temperature and precipitation, Orogen northern half-year convention."""
    from terrain_hydrology_balance import NORTH_SUMMER_HALF, PEAK_MONTH
    m = np.arange(MONTHS)[:, None, None]
    mean, amp = (temp_summer + temp_winter) / 2, (temp_summer - temp_winter) / 2
    temp = mean[None] + amp[None] * np.cos(2 * np.pi * (m - PEAK_MONTH) / 12)
    precip = np.where(NORTH_SUMMER_HALF[:, None, None], precip_summer[None] / 6, precip_winter[None] / 6)
    return temp, precip


def trapezoid(x, a, b, c, d):
    """Ecocrop-style suitability: 0 outside [a, d], 1 in [b, c]."""
    x = np.asarray(x, np.float64)
    return np.clip(np.minimum((x - a) / max(b - a, 1e-9), (d - x) / max(d - c, 1e-9)), 0, 1)


PARAMS = dict(
    seed_rate_t_ha=.15,          # sown grain per ha (provisional)
    yield_ratio_max=5.,          # grain harvested per grain sown at full suitability;
                                 # England 1250-1499 averaged 4.7 (Slicher van Bath 1963)
    cropped_share=.6,            # three-field rotation: two of three fields sown
    proper_use=.25,              # harvest efficiency, native range grazed all season (NDSU R1810)
    lsu_kg_dm_month=228.,        # 250 kg tropical livestock unit x ~3 % body weight x 30.4 days
    wood_density_t_m3=.55,       # basic density, oak/beech/ash 0.57-0.58, birch 0.51 (IPCC/FAO)
    woody_npp_share=.40,
    usable_wood_share=.5,
    old_growth_years=60.,        # standing stock = MAI x years (cap of the initial stock)
    game_mass_kg=80.,            # reference wild herbivore
    game_harvest_rate=.15,       # sustainable annual take of standing number
    floodplain_fish_kg_ha=42.,   # 4.23 t/km2 floodplain (Crul 1992, after Welcomme 1985)
    shelf_fish_t_km2=2.,         # coastal shelf, per km2 per year (provisional, not verified)
    salt_pan_t_m2=.05,           # solar salt per m2 of pan per dry warm month (provisional)
    peat_m=1.5,                  # mean accumulated peat thickness (provisional)
    peat_growth_m_yr=.0004,      # ~0.4 mm/yr derived from Loisel et al. 2014 C rate and density
    bog_iron_t_km2_yr=.4,        # bog iron renewal in iron-rich wetlands (provisional)
)


def compile_world(layers, height, seed, *, lod=WORLD_LOD, fantasy=False, params=None, balance=True,
                  progress=None, parallel=None, v2_fields=None):
    """World authority on the LOD lattice. Returns (lattice, fields, deposits).

    ``layers``/``height`` are the retained Orogen atlas arrays. Monthly fields
    have a leading axis of 12. Extensive values are totals per cell; intensive
    values are per unit area, and ``fields['units']`` names each unit.

    Stages, in order (``progress(stage, fields)`` is called after each, with
    the fields computed so far, so that layers needing only them can be shown):
    ``tectonics`` (contexts, expected rocks, structural strike, rock bias,
    monthly climate, water balance), ``environment`` (Render v2 proxies,
    biology), ``geology`` (realized rock shares, materials, bog iron),
    ``deposits`` (deposits and stocks). ``parallel`` maps tasks over worker
    processes (terrain_resource_workers); ``v2_fields`` reuses a cached Render
    v2 environment of the same world. Results do not depend on either.
    """
    p = dict(PARAMS, **(params or {}))
    report = progress or (lambda stage, fields: None)
    height = np.asarray(height, np.float32)
    grid = Lattice(lod)
    ax, ay = _atlas_xy(height.shape)
    contexts_atlas, sea_km = tectonic_contexts(layers, height)
    ts = -45 + 90 * np.asarray(layers['temperature_summer'], np.float64)
    tw = -45 + 90 * np.asarray(layers['temperature_winter'], np.float64)
    ps = 1000 * np.maximum(np.asarray(layers['precip_summer'], np.float64), 0)
    pw = 1000 * np.maximum(np.asarray(layers['precip_winter'], np.float64), 0)
    pet_proxy = 150 + 55 * np.maximum((ts + tw) / 2 + 5, 0)       # pedology moisture index convention
    aridity_atlas = 1 - smoothstep(.12, .55, (ps + pw) / pet_proxy)
    mean_h = gaussian_filter(height.astype(np.float64), 1.5, mode=('nearest', 'wrap'))
    rugged_atlas = np.sqrt(np.maximum(gaussian_filter(height.astype(np.float64) ** 2, 1.5, mode=('nearest', 'wrap')) - mean_h ** 2, 0))
    lowland_atlas = (1 - smoothstep(150, 900, mean_h)) * (1 - smoothstep(150, 900, rugged_atlas))
    rocks_atlas = lithology(contexts_atlas, aridity_atlas, lowland_atlas, seed, ax, ay)

    S = lambda a: sample_grid(a, grid.x, grid.y).astype(np.float64)
    f = dict(area_m2=grid.area_m2, latitude=np.array(grid.lat), units=UNITS)
    f['height_m'] = S(height)
    land = f['height_m'] > 0
    f['land'] = land
    f['rugged_m'] = S(rugged_atlas)
    f['sea_km'] = S(sea_km)
    for k, name in enumerate(CONTEXTS):
        f['context_' + name] = S(contexts_atlas[k])
    expected = np.stack([S(rocks_atlas[k]) for k in range(len(ROCKS))])
    expected /= np.maximum(expected.sum(axis=0), 1e-12)
    f['structural_strike'] = structural_strike(f)
    f['rock_bias'] = calibrate_rock_bias(expected, f['structural_strike'], land, grid, seed, parallel)
    for k, name in enumerate(ROCKS):
        f['rock_expected_' + name] = expected[k]
    for name in ('sandy', 'calcareous', 'clayey', 'ferrallitic', 'organic', 'podzolic', 'mineral'):
        f['pedology_' + name] = S(layers['pedology_' + name]) if 'pedology_' + name in layers else np.full(grid.shape, 1 / 7)
    T, P = _months(S(ts), S(tw), S(ps), S(pw), f['latitude'])
    f['temperature_c'], f['precipitation_mm'] = T.astype(np.float32), P.astype(np.float32)
    _water(f, balance)
    report('tectonics', f)
    sand = S(layers['soil_sand']) if 'soil_sand' in layers else np.full(grid.shape, .4)
    if v2_fields is not None:
        f.update(v2_fields)
    else:
        render_v2_fields(f, grid, seed, S(ts), S(tw), S(ps), S(pw), sand, parallel)
    _biology(f, p)
    report('environment', f)
    realized, _ = realize_rocks(expected, f['structural_strike'], land, grid, seed, bias=f['rock_bias'], parallel=parallel)
    for k, name in enumerate(ROCKS):
        f['rock_' + name] = realized[k]
    _bog_iron(f, p)
    _materials(f)
    report('geology', f)
    deposits = world_deposits(grid, f, seed, fantasy=fantasy, parallel=parallel)
    stock_layers(grid, f, deposits)
    report('deposits', f)
    return grid, f, deposits


V2_FIELDS = ('v2_snow_month', 'v2_grass_month', 'v2_leaf_month', 'v2_ice_month', 'v2_moisture_month',
             'v2_woody', 'v2_canopy_m', 'v2_needle', 'v2_perennial_snow')


def _water(f, balance):
    """Monthly AET, runoff and snow from the shared hydrology water balance."""
    land = f['land']
    T, P = f['temperature_c'], f['precipitation_mm']
    shape = (MONTHS,) + land.shape
    for name in ('pet_mm', 'aet_mm', 'runoff_mm', 'snow_mm'):
        f[name] = np.zeros(shape, np.float32)
    if not land.any():
        return
    from terrain_hydrology_balance import monthly_water_balance, oudin_pet_mm
    lat = f['latitude'][land]
    if balance:
        b = monthly_water_balance(None, None, latitude_deg=lat,
                                  temp_summer_c=T[6][land], temp_winter_c=T[0][land],
                                  precip_summer_mm=P[6][land] * 6, precip_winter_mm=P[0][land] * 6)
        for out, name in (('pet_mm', 'potential_evaporation_mm'), ('aet_mm', 'evapotranspiration_mm'),
                          ('snow_mm', 'snow_mm')):
            f[out][:, land] = b[name]
        f['runoff_mm'][:, land] = b['runoff_mm'] + b['baseflow_mm']
    else:  # Fast bucket-free proxy for tests: AET = min(P, PET).
        for m in range(MONTHS):
            pet = oudin_pet_mm(T[m][land], lat, m)
            f['pet_mm'][m, land] = pet
            f['aet_mm'][m, land] = np.minimum(P[m][land], pet)
            f['runoff_mm'][m, land] = np.maximum(P[m][land] - pet, 0)
            f['snow_mm'][m, land] = np.where(T[m][land] < 0, P[m][land], 0)


def _v2_inputs(grid, ground):
    """Geometry inputs of the Render v2 evaluator at the cell centres."""
    from terrain_render import terrain_derivatives
    from terrain_soil import coordinates
    point = coordinates(grid.x, grid.y, BOUNDS, True)
    xx, yy = np.meshgrid(grid.x, grid.y)
    lon = np.remainder((xx - BOUNDS[0]) / (BOUNDS[2] - BOUNDS[0]), 1) * 2 * np.pi - np.pi
    land_point = np.stack((RADIUS_M * np.cos(lon), RADIUS_M * np.sin(lon), yy - BOUNDS[1]), -1).astype(np.float32)
    gradient, tpi = terrain_derivatives(ground, grid.size)
    return point, land_point, gradient, tpi


def v2_months(lod, ground, ts, tw, ps, pw, sand, v2seed, months, mask, stand=False):
    """Render v2 environment for some months at the masked cells (1-D results;
    the evaluator is per sample). Pure; run in worker processes, which rebuild
    the geometry inputs instead of receiving them."""
    from terrain_render_v2 import evaluate
    grid = Lattice(lod)
    ground = np.asarray(ground, np.float64)
    point, land_point, gradient, tpi = _v2_inputs(grid, ground)
    sub = [a[mask] for a in (point, ground.astype(np.float32), gradient, tpi, ts, tw, ps, pw, sand)]
    keep = ('snow', 'grass', 'leaf', 'ice', 'moisture') + (('woody', 'height', 'needle') if stand else ())
    out = {}
    for m in months:
        e = evaluate(*sub, 0., (m + .5) / MONTHS, v2seed, grid.size, land_point[mask])
        out[m] = {k: np.asarray(e[k], np.float32) for k in keep}
    return out


def render_v2_fields(f, grid, seed, ts, tw, ps, pw, sand, parallel=None):
    """Monthly Render v2 environment at the cell centres (consumed, not copied).

    Calls ``terrain_render_v2.evaluate`` with the cell size as footprint, so
    its 2.4 km/670 m mosaics are filtered out and only climate, relief and
    province terms remain. Display settings (forest slider, legacy crowns) are
    deliberately not read: they change the picture, never the resource.
    Outputs are appearance proxies, stored with a ``v2_`` prefix.
    """
    from terrain_render_v2 import perennial_snow
    from terrain_soil import seed_value
    ground = np.maximum(f['height_m'], 0)
    v2seed = seed_value(seed)
    args = [np.asarray(a, np.float32) for a in (ground, ts, tw, ps, pw, sand)]
    shape = (MONTHS,) + grid.shape
    for name in ('snow', 'grass', 'leaf', 'ice', 'moisture'):
        f['v2_%s_month' % name] = np.zeros(shape, np.float32)
    # Land and shelf (> -200 m) only: deep-sea values are never read (zero).
    mask = f['land'] | (f['height_m'] > -200)
    common = dict(lod=grid.lod, ground=ground, ts=args[1], tw=args[2], ps=args[3], pw=args[4], sand=args[5], v2seed=v2seed,
                  mask=mask)
    groups = [[m, m + 1] for m in range(0, MONTHS, 2)]
    tasks = [dict(common, months=g, stand=g[-1] == MONTHS - 1) for g in groups]
    results = [v2_months(**t) for t in tasks] if parallel is None else         parallel([('terrain_resources', 'v2_months', t) for t in tasks])
    for g, r in zip(groups, results):
        for m in g:
            for name in ('snow', 'grass', 'leaf', 'ice', 'moisture'):
                f['v2_%s_month' % name][m][mask] = r[m][name]
    e = {}
    for name in ('woody', 'height', 'needle'):
        e[name] = np.zeros(grid.shape, np.float32)
        e[name][mask] = results[-1][MONTHS - 1][name]
    # Date-independent stand properties (identical for every month).
    f['v2_woody'] = np.asarray(e['woody'], np.float32) * f['land']
    f['v2_canopy_m'] = np.asarray(e['height'], np.float32) * f['land']
    f['v2_needle'] = np.asarray(e['needle'], np.float32) * f['land']
    _, _, gradient, _ = _v2_inputs(grid, ground)
    slope = np.linalg.norm(gradient, axis=-1)
    supply = np.clip((ps + pw) / np.maximum(120, 20 * (ts + tw) * .5 + 280), 0, 5)
    f['v2_perennial_snow'] = (perennial_snow(np.maximum(ts, tw), supply, slope) * f['land']).astype(np.float32)


def _biology(f, p):
    land, area = f['land'], f['area_m2']
    T, P, aet, pet = f['temperature_c'], f['precipitation_mm'], f['aet_mm'], f['pet_mm']
    t_mean, p_year = T.mean(axis=0), P.sum(axis=0)
    # Miami model (Lieth 1975): NPP limited by temperature or precipitation, g DM m-2 yr-1.
    npp_t = 3000 / (1 + np.exp(1.315 - .119 * t_mean))
    npp_p = 3000 * (1 - np.exp(-.000664 * p_year))
    npp = np.where(land, np.minimum(npp_t, npp_p), 0)
    growth = np.where(T >= 5, aet, 0) + 1e-6 * land          # growing months carry production
    share = growth / np.maximum(growth.sum(axis=0), 1e-12)
    f['npp_g_m2'] = npp.astype(np.float32)
    f['npp_month_g_m2'] = (share * npp).astype(np.float32)
    # GAEZ-like length of growing period: T >= 5 degC and AET >= 0.5 PET.
    growing = (T >= 5) & (aet >= .5 * np.maximum(pet, 1e-6))
    f['growing_months'] = growing.sum(axis=0).astype(np.uint8)
    f['tree_line_ok'] = T.max(axis=0) >= 10
    # Potential tree cover: water surplus, season length, warmest month >= 10 degC.
    # Potential natural woody cover is Render v2's settings-free evaluator
    # (render_v2_fields), minus perennial snow: an appearance proxy used as a
    # physical prior, never the display sliders or the drawn crowns.
    tree = land * f['v2_woody'] * (1 - f['v2_perennial_snow'])
    f['tree_cover'] = tree.astype(np.float32)

    # Cereals: Ecocrop-like trapezoids for a temperate (wheat/barley/rye) and a
    # warm (millet/sorghum) profile; the better profile is kept per cell.
    season_t = np.where(growing, T, np.nan)
    count = growing.sum(axis=0)
    gs_t = np.where(count > 0, np.nansum(season_t, axis=0) / np.maximum(count, 1), -50)
    temperate = trapezoid(gs_t, 5, 12, 20, 28) * trapezoid(p_year, 250, 500, 1100, 2000) * trapezoid(T.min(axis=0), -30, -12, 15, 25)
    warm = trapezoid(gs_t, 16, 22, 30, 36) * trapezoid(p_year, 250, 450, 900, 1600)
    climate = np.maximum(temperate, warm) * smoothstep(2, 4, f['growing_months'])
    soil = (.55 * f['pedology_sandy'] + .9 * f['pedology_calcareous'] + 1. * f['pedology_clayey']
            + .45 * f['pedology_ferrallitic'] + .5 * f['pedology_organic'] + .45 * f['pedology_podzolic']
            + .6 * f['pedology_mineral'])
    terrain = 1 - smoothstep(80, 700, f['rugged_m'])
    suit = land * climate * soil * terrain
    f['cereal_suitability'] = suit.astype(np.float32)
    f['cereal_profile'] = np.where(warm > temperate, 2, 1).astype(np.uint8) * (suit > 0)
    # Arable share of the cell: terrain and soil, before any clearing (natural state).
    arable_share = (land * terrain * smoothstep(.05, .3, climate) * (1 - .6 * f['pedology_organic'])
                    * (1 - f['v2_perennial_snow']))
    f['arable_area_m2'] = (arable_share * area).astype(np.float64)
    ratio = 1 + (p['yield_ratio_max'] - 1) * suit
    yield_t_ha = p['seed_rate_t_ha'] * ratio * (suit > 0)
    f['cereal_yield_ratio'] = (ratio * (suit > 0)).astype(np.float32)
    f['cereal_yield_t_ha'] = yield_t_ha.astype(np.float32)
    # GAEZ v4 classes on attainable share: VS>=.8 S>=.6 MS>=.4 mS>=.2 vmS>0 NS.
    f['cereal_class'] = np.select([suit >= .8, suit >= .6, suit >= .4, suit >= .2, suit > 0], [5, 4, 3, 2, 1], 0).astype(np.uint8)
    net = np.maximum(yield_t_ha - p['seed_rate_t_ha'] * (suit > 0), 0)
    f['cereal_net_t_year'] = net * f['arable_area_m2'] / 1e4 * p['cropped_share']
    # Harvest month: last month of the longest growing run (cyclic).
    f['harvest_month'] = _last_month_of_longest_run(growing)

    # Pasture: herbaceous production under tree-free conditions.
    # Herbaceous share per month from Render v2 grass (already reduced by its
    # seasonal snow), under the potential canopy: snow-covered pasture is not grazed.
    herb = np.clip(f['v2_grass_month'] * (1 - .8 * tree), 0, 1)
    forage = f['npp_month_g_m2'] * 10 * herb * p['proper_use']          # kg DM ha-1 month-1
    f['forage_month_kg_ha'] = forage.astype(np.float32)
    f['lsu_month_km2'] = (forage * 100 / p['lsu_kg_dm_month']).astype(np.float32)
    f['lsu_year_km2'] = (forage.sum(axis=0) * 100 / (12 * p['lsu_kg_dm_month'])).astype(np.float32)
    f['lsu_lean_km2'] = f['lsu_month_km2'].min(axis=0)
    # Transhumance: summer upland forage reachable within ~150 km of a lowland
    # whose lean season is short of forage.
    upland = (f['height_m'] > 900) & land
    summer = forage.max(axis=0) * upland
    reach = maximum_filter(summer, size=11, mode=('nearest', 'wrap'))
    winter_low = np.where(land & ~upland, forage.min(axis=0), 0)
    f['transhumance'] = (smoothstep(0, 150, reach) * (1 - smoothstep(0, 80, winter_low)) * land * ~upland).astype(np.float32)

    # Wood: mean annual increment from woody NPP; initial standing stock.
    mai = npp / 100 * p['woody_npp_share'] * p['usable_wood_share'] / p['wood_density_t_m3']   # m3 ha-1 yr-1
    f['wood_increment_m3_ha'] = (mai * tree).astype(np.float32)
    f['wood_stock_m3'] = mai * p['old_growth_years'] * tree * area / 1e4
    f['wood_flux_m3_year'] = mai * tree * area / 1e4
    # Render v2 canopy height and needle share split the same stock (no double count):
    # tall stands give building timber, the rest fuel/coppice; conifers vs broadleaf.
    f['timber_share'] = (smoothstep(10, 25, f['v2_canopy_m']) * (tree > 0)).astype(np.float32)
    f['conifer_share'] = (f['v2_needle'] * (tree > 0)).astype(np.float32)
    f['felling_months'] = (f['v2_leaf_month'] < .35) & (tree > 0)          # leafless or dormant stands
    f['sledge_months'] = (f['v2_snow_month'] > .5) & land                  # snow haulage season
    # Game: Damuth-type density at a reference mass, scaled by NPP.
    # Damuth 1981: log10 D = -0.75 log10 W(g) + 4.23 for primary consumers.
    density = 10 ** 4.23 * (p['game_mass_kg'] * 1000) ** -.75
    f['game_per_km2'] = (land * density * np.clip(npp / 1000, 0, 2)).astype(np.float32)
    f['game_harvest_kg_year'] = f['game_per_km2'] * area / 1e6 * p['game_harvest_rate'] * p['game_mass_kg'] * .5
    f['fur_potential'] = (land * tree * smoothstep(-5, -20, T.min(axis=0))).astype(np.float32)

    # Water and fisheries.
    runoff = f['runoff_mm']
    f['water_yield_m3'] = runoff.sum(axis=0) / 1000 * area * land
    floodplain = land * smoothstep(100, 800, runoff.sum(axis=0)) * (1 - smoothstep(100, 600, f['rugged_m'])) * .03
    river_fish = floodplain * area / 1e4 * p['floodplain_fish_kg_ha'] / 1000
    pulse = np.roll(runoff, 1, axis=0) + 1e-6 * land          # catch peaks after the flood month
    f['fish_inland_month_t'] = (river_fish * pulse / np.maximum(pulse.sum(axis=0), 1e-12)).astype(np.float32)
    shelf = (~land) & (f['height_m'] > -200)
    warmth = smoothstep(-2, 12, t_mean)
    f['fish_marine_t_year'] = shelf * area / 1e6 * p['shelf_fish_t_km2'] * (.5 + warmth)
    # Months with ice (Render v2 freezing potential): halved access, never a new stock.
    access = 1 - .5 * f['v2_ice_month'] * shelf
    f['fish_marine_month_t'] = (f['fish_marine_t_year'] * access / np.maximum(access.sum(axis=0), 1e-12)).astype(np.float32)
    # Solar salt: low coastal land, warm dry months (P < PET, T >= 15).
    coast = land * np.exp(-f['sea_km'] / 15) * (1 - smoothstep(30, 300, f['rugged_m']))
    dry = (T >= 15) & (P < pet)
    pan_m2 = coast * area * 1e-4
    f['sea_salt_month_t'] = (pan_m2 * dry * p['salt_pan_t_m2']).astype(np.float32)
    # Peat: cold-humid waterlogged lowlands (organic pedology), slow regeneration.
    wet = smoothstep(.8, 1.2, P.sum(axis=0) / np.maximum(pet.sum(axis=0), 1))
    peat_share = land * wet * f['pedology_organic'] * (1 - smoothstep(50, 400, f['rugged_m'])) * (1 - smoothstep(6, 14, t_mean)) * smoothstep(5, 10, T.max(axis=0))
    f['peat_area_m2'] = peat_share * area
    f['peat_stock_m3'] = f['peat_area_m2'] * p['peat_m']
    f['_wet_index'] = wet


def _bog_iron(f, p):
    """Bog iron in wet lowlands over iron-bearing rocks (needs the realized geology)."""
    wet = f.pop('_wet_index')
    iron_rocks = f['rock_basalt'] + f['rock_granite'] * .5 + f['rock_ultramafic'] + f['rock_metamorphic'] * .5
    f['bog_iron_t_year'] = f['land'] * wet * (1 - smoothstep(100, 500, f['rugged_m'])) * iron_rocks * f['area_m2'] / 1e6 * p['bog_iron_t_km2_yr']


def _last_month_of_longest_run(mask):
    m = mask.astype(np.int32)
    run = np.zeros_like(m)
    best, best_end = np.zeros(m.shape[1:], np.int32), np.full(m.shape[1:], 255, np.int32)
    for k in range(2 * MONTHS):   # two passes make runs cyclic
        cur = m[k % MONTHS]
        prev = run[(k - 1) % MONTHS] if k else np.zeros_like(cur)
        run[k % MONTHS] = np.minimum(np.where(cur > 0, prev + 1, 0), MONTHS)
        better = run[k % MONTHS] > best
        best = np.where(better, run[k % MONTHS], best)
        best_end = np.where(better, k % MONTHS, best_end)
    return best_end.astype(np.uint8)


def _materials(f):
    """Construction materials: potentials only (medieval stocks are not limiting)."""
    r = lambda n: f['rock_' + n]
    f['stone_ashlar'] = (.9 * r('limestone') + .8 * r('sandstone') + .7 * r('granite') + .4 * r('basalt') + .5 * r('metamorphic')).astype(np.float32)
    f['lime'] = r('limestone').astype(np.float32)
    f['slate'] = r('metamorphic').astype(np.float32) * .6
    f['clay'] = (np.clip(.6 * r('marl') + .5 * f['pedology_clayey'], 0, 1) * f['land']).astype(np.float32)
    f['sand_gravel'] = (np.clip(.5 * r('sandstone') + .4 * f['pedology_sandy'] + .3 * np.exp(-f['sea_km'] / 20), 0, 1) * f['land']).astype(np.float32)


# ---------------------------------------------------------------------------
# Discrete deposits: count, tonnage, grade and position from keyed draws

def prospectivity(f, spec):
    ctx = sum(w * f['context_' + c] for c, w in spec['contexts'].items())
    host = sum(w * f['rock_' + r] for r, w in spec['hosts'].items())
    p = ctx * np.minimum(1, 1.5 * host)
    if 'requires' in spec:   # a necessary host: no salt deposit without evaporites
        p = p * smoothstep(0, .01, f['rock_' + spec['requires']])
    climate = spec.get('climate')
    if climate == 'tropical':
        p = p * smoothstep(18, 24, f['temperature_c'].mean(axis=0)) * smoothstep(1000, 1600, f['precipitation_mm'].sum(axis=0))
    elif climate == 'humid':   # coal-forming swamps: a keyed paleoclimate would be better (doc §5.3)
        p = p * (.3 + .7 * smoothstep(500, 1200, f['precipitation_mm'].sum(axis=0)))
    return p


def _poisson(lam, u, kmax=40):
    """Inverse CDF of Poisson(lam) at u, vectorised, float64 recurrence."""
    lam = np.asarray(lam, np.float64)
    out = np.zeros(lam.shape, np.int64)
    live = lam > 0                         # lam = 0 gives 0 (cdf starts at 1): same result
    lam, u = lam[live], np.broadcast_to(u, out.shape)[live]
    term = np.exp(-lam)
    cdf = term.copy()
    n = np.zeros(lam.shape, np.int64)
    for k in range(1, kmax + 1):
        more = u > cdf
        n += more
        term = term * lam / k
        cdf = cdf + term
    out[live] = n
    return out


def world_deposits(grid, f, seed, *, fantasy=False, parallel=None):
    """Discrete deposits owned by their LOD-10 cell. List of dicts, id-sorted."""
    families = dict(FAMILIES, **(FANTASY if fantasy else {}))
    prosp = {name: prospectivity(f, spec) * f['land'] for name, spec in families.items()}
    for name, spec in PLACERS.items():
        src = sum(prosp[s] for s in spec['source'])
        spread = gaussian_filter(src, 2, mode=('nearest', 'wrap'))     # downstream reach ~60 km
        runoff = f['runoff_mm'].sum(axis=0)
        prosp[name] = spread * smoothstep(100, 600, runoff) * (1 - smoothstep(200, 900, f['rugged_m'])) * f['land']
    f.update({'prospect_' + k: v.astype(np.float32) for k, v in prosp.items()})
    jobs = []
    for name, p in prosp.items():
        spec = families.get(name) or PLACERS[name]
        domain = ('resources', 'fantasy' if name in FANTASY else 'deposit', name)
        lam = spec['density'] * np.clip(p, 0, 1) * grid.area_m2 / 1e10
        u = to_unit(cell_keys(seed, domain + ('count', f'lod{grid.lod}'), grid.ids))
        count = _poisson(lam, u)
        jj, ii = np.nonzero(count)
        for j, i in zip(jj, ii):
            for k in range(int(count[j, i])):
                jobs.append((domain, name, spec, j, i, k))
    # Positions per family in vectorised batches (same values as _place_in_cell).
    by_name = {}
    for n, job in enumerate(jobs):
        by_name.setdefault(job[1], []).append(n)
    tasks, owners = [], []
    for name, idx in by_name.items():
        for a in range(0, len(idx), 512):
            part = idx[a:a + 512]
            tasks.append(('terrain_resources', 'place_positions',
                          dict(seed=seed, domain=jobs[part[0]][0], lod=grid.lod, field=f['prospect_' + name],
                               jj=np.array([jobs[n][3] for n in part]), ii=np.array([jobs[n][4] for n in part]),
                               kk=np.array([jobs[n][5] for n in part]))))
            owners.append(part)
    results = [place_positions(**t[2]) for t in tasks] if parallel is None else parallel(tasks)
    positions = {n: pos for part, res in zip(owners, results) for n, pos in zip(part, res)}
    out = [_deposit(seed, job[0], job[1], job[2], grid, f, *job[3:], position=positions[n]) for n, job in enumerate(jobs)]
    out.sort(key=lambda d: d['id'])
    return out


def _deposit(seed, domain, name, spec, grid, f, j, i, k, position=None):
    cid = int(grid.ids[j, i])
    draw = lambda what: float(to_unit(cell_keys(seed, domain + (what, f'lod{grid.lod}'), np.uint64(cid), draw=k + 1)))
    median, sigma = spec['tonnage']
    ore = median * 10 ** (sigma * float(ndtri(min(max(draw('tonnage'), 1e-9), 1 - 1e-9))))
    grade = 10 ** (.25 * float(ndtri(min(max(draw('grade'), 1e-9), 1 - 1e-9))))   # shared grade factor
    depth = spec['depth'] * 10 ** (.4 * float(ndtri(min(max(draw('depth'), 1e-9), 1 - 1e-9))))
    contained = {}
    for metal, frac in spec['metals'].items():
        scale = 1000 if metal in GRAM_METALS else 1
        contained[metal] = int(round(ore * frac * grade * scale))
    # Medieval accessible share: ore within reach of drainage adits or shallow
    # shafts. Vertical extent scales with the cube root of the body volume.
    extent = max(20., (ore / 2.7) ** (1 / 3))
    reach = 40 + .5 * f['rugged_m'][j, i]
    accessible = float(np.clip((reach - depth) / extent, 0, 1))
    outcrop = depth < 5
    if position is None:
        position = _place_in_cell(seed, domain, grid, f, name, j, i, k)
    lod_b, ib, jb = decode_id(np.uint64(position))
    return dict(id=f"{name}:{cid:016x}:{k}", family=name, owner_cell=cid, index=k,
                position_cell=int(position), position_lod=int(lod_b), position_i=int(ib), position_j=int(jb),
                ore_t=int(round(ore)), grade_factor=round(grade, 4), depth_top_m=round(depth, 1),
                contained=contained, contained_units={m: ('kg' if m in GRAM_METALS else 't') for m in contained},
                medieval_accessible_share=round(accessible, 4),
                medieval_recoverable={m: int(v * accessible * MEDIEVAL_RECOVERY.get(m, .5)) for m, v in contained.items()},
                outcrop=bool(outcrop), gossan=bool(outcrop and spec.get('sulfide') and f['temperature_c'][:, j, i].max() > 0),
                status='synthetic' if name in FANTASY else 'derived')


def _place_in_cell(seed, domain, grid, f, name, j, i, k):
    """LOD-10 -> LOD-3 position: keyed weighted choice on interpolated prospectivity."""
    n = 1 << (grid.lod - placement_lod(grid.lod))
    ci, cj = int(grid.i[i]), int(grid.j[j])
    sub_i = ci * n + np.arange(n)
    sub_j = cj * n + np.arange(n)
    plod = placement_lod(grid.lod)
    weights = descend_weights(grid, f['prospect_' + name], j, i, sub_i, sub_j)
    ids = cell_id(plod, sub_i[None, :], sub_j[:, None])
    keys = to_unit(cell_keys(seed, domain + ('position', f'lod{plod}'), ids, draw=k + 1))
    # Exponential race: argmin(-ln u / w) is a weighted choice independent of order.
    if not (weights > 0).any():
        weights = np.ones_like(weights)
    with np.errstate(divide='ignore'):
        race = np.where(weights > 0, -np.log(np.maximum(keys, 1e-300)) / np.maximum(weights, 1e-300), np.inf)
    flat = int(np.argmin(race))
    return ids.ravel()[flat]


def place_positions(seed, domain, lod, field, jj, ii, kk):
    """_place_in_cell for deposits of one family (pure; run in worker processes)."""
    grid = Lattice(lod)
    f = {'prospect_': field}
    return [int(_place_in_cell(seed, domain, grid, f, '', int(j), int(i), int(k))) for j, i, k in zip(jj, ii, kk)]


def placement_lod(lod):
    """Deposits are positioned at LOD 3 (240 m) from LOD 10; descents never
    exceed 2**7 children per axis, so coarse test worlds stay bounded."""
    return max(BUDGET_LOD, lod - 7)


def descend_weights(grid, field, j, i, sub_i, sub_j):
    """Bilinear interpolation of a world intensity at child centres (parent + 8 neighbours)."""
    s = grid.size
    child = s / len(sub_i)
    xs = (sub_i + .5) * child
    ys = (sub_j + .5) * child
    gx = (xs - grid.x[0]) / s
    gy = (ys - grid.y[0]) / s
    i0 = np.clip(np.floor(gx).astype(int), 0, field.shape[1] - 2)
    j0 = np.clip(np.floor(gy).astype(int), 0, field.shape[0] - 2)
    tx = np.clip(gx - i0, 0, 1)[None, :]
    ty = np.clip(gy - j0, 0, 1)[:, None]
    a = field[j0][:, i0] * (1 - tx) + field[j0][:, i0 + 1] * tx
    b = field[j0 + 1][:, i0] * (1 - tx) + field[j0 + 1][:, i0 + 1] * tx
    return np.maximum(a * (1 - ty) + b * ty, 0)


def stock_layers(grid, f, deposits):
    """Integer stock per LOD-10 cell and metal: exactly the sum of its deposits."""
    metals = sorted({m for d in deposits for m in d['contained']})
    index = {int(v): k for k, v in enumerate(grid.ids.ravel())}
    for metal in metals:
        total = np.zeros(grid.shape[0] * grid.shape[1], np.int64)
        recov = np.zeros_like(total)
        for d in deposits:
            if metal in d['contained']:
                total[index[d['owner_cell']]] += d['contained'][metal]
                recov[index[d['owner_cell']]] += d['medieval_recoverable'][metal]
        f['stock_' + metal] = total.reshape(grid.shape)
        f['recoverable_' + metal] = recov.reshape(grid.shape)


# ---------------------------------------------------------------------------
# Descent and aggregation of areal budgets

EXTENSIVE = ('arable_area_m2', 'cereal_net_t_year', 'wood_stock_m3', 'wood_flux_m3_year',
             'game_harvest_kg_year', 'water_yield_m3', 'fish_marine_t_year', 'peat_area_m2',
             'peat_stock_m3', 'bog_iron_t_year')
QUANTUM = dict(arable_area_m2=1., cereal_net_t_year=.001, wood_stock_m3=1., wood_flux_m3_year=.01,
               game_harvest_kg_year=1., water_yield_m3=1000., fish_marine_t_year=.001,
               peat_area_m2=1., peat_stock_m3=1., bog_iron_t_year=.001)


def committed(f, name):
    """Integer budget per LOD-10 cell in QUANTUM units: the canonical total."""
    return np.rint(np.asarray(f[name], np.float64) / QUANTUM[name]).astype(np.int64)


def descend_budget(grid, f, name, j, i, *, lod=BUDGET_LOD, seed=0, relief=None):
    """Exact allocation of one LOD-10 cell budget to its LOD ``lod`` children.

    Weights use world information only (interpolated intensity, child area and
    an optional finer canonical factor ``relief`` such as 240 m slope), so the
    result is a pure function of the parent and never of the requested tile.
    Returns (child ids [n, n], integer amounts [n, n]).
    """
    n = 1 << (grid.lod - lod)
    ci, cj = int(grid.i[i]), int(grid.j[j])
    sub_i, sub_j = ci * n + np.arange(n), cj * n + np.arange(n)
    intensity = np.asarray(f[name], np.float64) / np.maximum(grid.area_m2, 1)
    weights = descend_weights(grid, intensity, j, i, sub_i, sub_j)
    child = Lattice(lod, (ci * grid.size, cj * grid.size, (ci + 1) * grid.size, (cj + 1) * grid.size))
    weights = weights * child.area_m2
    if relief is not None:
        weights = weights * relief
    ids = cell_id(lod, sub_i[None, :], sub_j[:, None])
    total = committed(f, name)[j, i]
    amounts = apportion(total, quantize(weights.ravel()), ids.ravel()).reshape(n, n)
    return ids, amounts


def aggregate(values, groups):
    """Sum integer values by group label (regions or coarser LOD parents)."""
    labels, inverse = np.unique(np.asarray(groups).ravel(), return_inverse=True)
    out = np.zeros(labels.shape, np.int64)
    np.add.at(out, inverse, np.asarray(values, np.int64).ravel())
    return dict(zip(labels.tolist(), out.tolist()))


# ---------------------------------------------------------------------------
# Natural state + historical delta (owned by Chronos)

def deposit_remaining(deposit, extracted):
    """Remaining contained mass after Chronos' cumulative extraction (never < 0)."""
    return {m: max(0, v - int(extracted.get(m, 0))) for m, v in deposit['contained'].items()}


def extraction_cost(base_days_per_t, extracted, accessible):
    """Labour per tonne rises as the accessible ore is worked out."""
    used = np.clip(np.asarray(extracted, np.float64) / max(accessible, 1), 0, None)
    return base_days_per_t * (1 + used) ** 2


def clearing_order(seed, ids, suitability, parent_id):
    """Canonical order in which cells of one parent are cleared (most suitable first).

    Ties broken by keyed jitter then id; the same order serves every date, so
    the cleared set at date t is a prefix and is monotone in cumulative area.
    """
    jitter = to_unit(cell_keys(seed, ('resources', 'history', 'clearing', str(int(parent_id))), np.asarray(ids).ravel()))
    score = np.asarray(suitability, np.float64).ravel() * (.9 + .2 * jitter)
    return np.lexsort((np.asarray(ids, np.uint64).ravel(), -score))


def cleared_cells(order, areas, cleared_area):
    """Fraction of each cell cleared for a cumulative cleared area (prefix rule)."""
    a = np.asarray(areas, np.float64).ravel()[order]
    before = np.concatenate(([0], np.cumsum(a)[:-1]))
    frac = np.clip((cleared_area - before) / np.maximum(a, 1e-12), 0, 1)
    out = np.zeros_like(frac)
    out[order] = frac
    return out


def forest_stock(stock0, capacity, growth_rate, removals):
    """Logistic regrowth with removals per year, the Chronos stock rule."""
    s, path = float(stock0), []
    for e in removals:
        s = max(0., s + growth_rate * s * (1 - s / max(capacity, 1e-9)) - e)
        path.append(s)
    return np.array(path)


# ---------------------------------------------------------------------------
# Export contract (neutral, JSON-serialisable)

UNITS = dict(area_m2='m2', arable_area_m2='m2', cereal_yield_t_ha='t/ha/an (brut)', cereal_net_t_year='t/an',
             npp_g_m2='g MS/m2/an', npp_month_g_m2='g MS/m2/mois', forage_month_kg_ha='kg MS/ha/mois',
             lsu_month_km2='UGB/km2', lsu_year_km2='UGB/km2', lsu_lean_km2='UGB/km2',
             wood_increment_m3_ha='m3/ha/an', wood_stock_m3='m3', wood_flux_m3_year='m3/an',
             game_per_km2='ind./km2', game_harvest_kg_year='kg/an', water_yield_m3='m3/an',
             runoff_mm='mm/mois', fish_inland_month_t='t/mois', fish_marine_t_year='t/an',
             sea_salt_month_t='t/mois', peat_area_m2='m2', peat_stock_m3='m3', bog_iron_t_year='t Fe/an',
             temperature_c='degC', precipitation_mm='mm/mois', growing_months='mois', harvest_month='mois (0=janvier)')


def region_summary(grid, f, deposits, region_of_cell):
    """Chronos view: integer natural state per region, deposits by owner region."""
    regions = np.asarray(region_of_cell)
    out = {}
    for name in EXTENSIVE:
        for reg, v in aggregate(committed(f, name), regions).items():
            out.setdefault(reg, dict(natural={}, deposits=[]))['natural'][name] = dict(value=v, quantum=QUANTUM[name], unit=UNITS.get(name, ''))
    lookup = {int(c): int(r) for c, r in zip(grid.ids.ravel(), regions.ravel())}
    for d in deposits:
        out.setdefault(lookup[d['owner_cell']], dict(natural={}, deposits=[]))['deposits'].append(d['id'])
    return out
