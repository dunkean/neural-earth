"""Multi-scale descent of resource fields to any displayed LOD, with conservation.

A field at LOD L is defined on the canonical display tile grid (256 samples
plus a 24-sample halo, spacing 30*2**L m). It is derived from its parent
LOD P = min(authority, L+3) (L >= 0) as

    raw   = parent(bilinear) * exp(local modulation + keyed multi-octave noise)
    value = raw corrected per parent cell so that the area-weighted mean of the
            children equals the parent value exactly (dasymetric / Tobler
            pycnophylactic constraint; logit-space shift for bounded indices).

The local modulation reads only inputs canonical at LOD L (neural DEM and its
derivatives, canonical source rivers), never display settings. Noise octaves
of wavelength 2*30*2**L .. 30*2**P (8 x 8 children per parent) are keyed by (seed, layer, octave LOD,
lattice cell); coarser octaves live in the parents, so every LOD is the
low-pass of the finer ones. Parent values are always read from the parent
tile that owns them in its interior: tile halos are never reused, so adjacent
tiles agree. Negative LODs add noise only (no amplified relief is read).
"""
from collections import OrderedDict
import math
import threading

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

import terrain_resources as R

TILE, HALO = 256, 24
N = TILE + 2 * HALO
WORLD_W = R.BOUNDS[2] - R.BOUNDS[0]

# Log-linear sensitivity of each profile to local terrain features, and the
# standard deviation of its keyed log-noise. Proposed calibration (doc §16).
PROFILES = {
    'arable':  (dict(slope=-1 / 5, valley=1.0, ridge=-.8), .15),
    'pasture': (dict(slope=-1 / 12, valley=.6, ridge=-.3), .20),
    'forest':  (dict(valley=.6, ridge=-.5, steep=-.15, north=.3), .25),
    'wet':     (dict(valley=2.5, slope=-1 / 3, ridge=-1., river=2.), .30),
    'rock':    (dict(slope=1 / 6, ridge=1.2, valley=-1.), .20),
    'lowflat': (dict(slope=-1 / 4, valley=1.4, river=.8), .25),
    'cold':    (dict(dz=.012, north=.5, slope=1 / 30), .10),   # ~6.5 degC/km lapse in logit units
    'season':  (dict(dz=-.006, north=-.2), .08),
    'runoff':  (dict(slope=1 / 15, valley=.5), .15),
    'mineral': (dict(ridge=.6, slope=1 / 25), .60),
    'neutral': (dict(valley=.3, ridge=-.2), .20),
    'marine':  ({}, .30),
}
PASSES = 1   # one ring of blocks; with filter radii <= 8 samples every block is tile-independent
SEA_MODES = ('fish-marine',)


def profile_of(mode):
    m = mode.split('orogen-resources-', 1)[-1]
    rules = [(('cereal', 'arable'), 'arable'), (('lsu', 'forage', 'transhumance'), 'pasture'),
             (('tree', 'canopy', 'conifer', 'timber', 'wood', 'felling'), 'forest'),
             (('peat', 'bog-iron', 'fish-inland'), 'wet'), (('fish-marine',), 'marine'),
             (('season-fish',), 'wet'), (('stone', 'slate', 'lime'), 'rock'),
             (('clay', 'sand', 'salt'), 'lowflat'), (('snow', 'fur'), 'cold'), (('growing',), 'season'),
             (('water-yield', 'runoff'), 'runoff'), (('prospect',), 'mineral')]
    for keys, name in rules:
        if any(k in m for k in keys):
            return name
    return 'neutral'


def _bincount(index, weights, minlength):
    """np.bincount that always returns float64 (empty inputs give int64 in NumPy)."""
    return np.bincount(index, weights, minlength=minlength).astype(np.float64)


def smooth(a, b, x):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def tile_axes(lod, tx, ty):
    r = R.cell_size(lod)
    i0, j0 = tx * TILE - HALO, ty * TILE - HALO
    return (i0 + np.arange(N) + .5) * r, (j0 + np.arange(N) + .5) * r, i0, j0


STEP = 3   # one descent step = 8 x 8 children: a full ring of blocks fits in the 24-sample halo


def parent_lod_of(lod, authority):
    if lod >= 0:
        return min(authority, lod + STEP)
    return min(0, lod + STEP)


def octave_noise(seed, name, lod, xs, ys):
    """Keyed value noise on the LOD lattice (quintic), wrapping in longitude."""
    lam = R.cell_size(lod)
    period = max(1, int(round(WORLD_W / lam)))
    lam_x = WORLD_W / period
    gx = (np.asarray(xs, np.float64) - R.BOUNDS[0]) / lam_x - .5
    gy = np.asarray(ys, np.float64) / lam - .5
    i0, j0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = gx - i0, gy - j0
    fx = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    fy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    parts = ('resources', 'noise', name)
    # Hash each lattice node once (separable axes), then gather the 4 corners.
    ui = np.unique(np.concatenate((i0, i0 + 1)))
    uj = np.unique(np.concatenate((j0, j0 + 1)))
    ids = R.cell_id(lod, (ui % period)[None, :], uj[:, None])
    node = 2 * R.to_unit(R.cell_keys(seed, parts, ids)) - 1
    a0, a1 = np.searchsorted(ui, i0), np.searchsorted(ui, i0 + 1)
    b0, b1 = np.searchsorted(uj, j0), np.searchsorted(uj, j0 + 1)
    top = node[b0][:, a0] * (1 - fx) + node[b0][:, a1] * fx
    bottom = node[b1][:, a0] * (1 - fx) + node[b1][:, a1] * fx
    v = top * (1 - fy)[:, None] + bottom * fy[:, None]
    return v / .45   # value noise std ~0.45 after smoothing -> unit variance


def multi_noise(seed, name, lod, parent, xs, ys, hurst=.6):
    """Sum of octaves lod+1 .. parent (wavelengths 2 samples .. parent cell)."""
    octaves = list(range(lod + 1, parent + 1))
    if not octaves:
        return np.zeros((len(ys), len(xs)))
    # The octave at two samples per wavelength is damped to avoid speckle.
    amps = np.array([2. ** (hurst * (o - parent)) * (.25 if o == lod + 1 else 1.) for o in octaves])
    amps /= np.sqrt((amps ** 2).sum())
    return sum(a * octave_noise(seed, f'{name}-o{o}', o, xs, ys) for a, o in zip(amps, octaves))


def features(inputs, r):
    """Terrain features at the tile samples, from canonical inputs only.

    Every kernel reaches at most 8 samples, so features are identical in all
    tiles for samples at least 8 inside the array (the 24-sample halo then
    leaves one ring of 8-sample blocks plus the high-pass support).
    """
    e = np.asarray(inputs['elevation'], np.float64)
    ground = np.maximum(e, 0)
    gy, gx = np.gradient(ground, r)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    north = -gy / np.maximum(np.hypot(gx, gy), 1e-6) * smooth(1, 6, slope)
    # Topographic position: small minus large Gaussian (radii 3 and 7.5 samples), slope units.
    tpi = (gaussian_filter(ground, 1., truncate=3.) - gaussian_filter(ground, 2.5, truncate=3.)) / (2.5 * r)
    dz = e - gaussian_filter(e, 2.5, truncate=3.)
    river = inputs.get('river')
    if river is not None and np.any(river):
        dist = distance_transform_edt(~np.asarray(river, bool))
        near = np.exp(-dist / 3) * (dist <= 8)
    else:
        near = np.zeros_like(e)
    return dict(slope=slope, valley=smooth(0, -.05, tpi), ridge=smooth(0, .05, tpi), north=north,
                steep=np.maximum(slope - 35, 0), dz=dz, river=near, land=e > 0,
                shelf=(e <= 0) & (e > -200), elevation=e)


class Detail:
    """Per-world descent engine with LRU caches (thread-safe)."""

    def __init__(self, seed, grid, fields, deposits, authority_value, max_bytes=768 << 20, max_features=48):
        self.seed, self.grid, self.fields, self.deposits = seed, grid, fields, deposits
        self.authority_value = authority_value          # mode -> LOD-authority array (cached by caller)
        self.lod = grid.lod
        self.tiles, self.max_bytes, self.tile_bytes = OrderedDict(), max_bytes, 0
        self.feats, self.max_features = OrderedDict(), max_features
        self.lock = threading.RLock()
        self._strike = None

    # -- caches ---------------------------------------------------------------
    def _get(self, cache, key):
        with self.lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
        return None

    def _put(self, cache, key, value, limit):
        with self.lock:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > limit:
                cache.popitem(last=False)

    def features(self, lod, tx, ty, inputs):
        if lod < 0 or inputs is None:
            return None
        key = (lod, tx, ty)
        f = self._get(self.feats, key)
        if f is None:
            data = inputs(lod, tx, ty)
            if data is None:
                return None
            f = {k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in features(data, R.cell_size(lod)).items()}
            self._put(self.feats, key, f, self.max_features)
        return f

    # -- authority --------------------------------------------------------------
    def _authority_cells(self, value, gi, gj):
        g = self.grid
        period = len(g.i)
        ii = (np.asarray(gi) - g.i[0]) % period
        jj = np.clip(np.asarray(gj) - g.j[0], 0, len(g.j) - 1)
        if value.ndim == 3:
            return value[:, jj[:, None], ii[None, :]]
        return value[jj[:, None], ii[None, :]]

    def authority_tile(self, mode, lod, tx, ty):
        """Values of the authority LOD (or coarser: mean of authority cells) on a tile."""
        value = self.authority_value(mode)
        d = lod - self.lod
        base_i, base_j = tx * TILE - HALO, ty * TILE - HALO
        gi, gj = base_i + np.arange(N), base_j + np.arange(N)
        if d <= 0:
            return self._authority_cells(value, gi, gj)
        k = 1 << d
        sub = np.arange(k)
        acc = 0.
        for a in sub:
            for b in sub:
                acc = acc + self._authority_cells(value, gi * k + a, gj * k + b)
        return acc / (k * k)

    # -- descent ------------------------------------------------------------------
    def values(self, mode, lod, tx, ty, inputs, kind, bounded):
        """Full (N, N) or (K, N, N) array; only the interior is exact."""
        key = (mode, lod, tx, ty)
        cached = self._get(self.tiles, key)
        if cached is None:
            if lod >= self.lod:
                full = self.authority_tile(mode, lod, tx, ty)
            else:
                full = self._descend(mode, lod, tx, ty, inputs, kind, bounded)
            # Only the interior is canonical: keep it compact (float32) under a byte budget.
            cached = np.ascontiguousarray(full[..., HALO:HALO + TILE, HALO:HALO + TILE], np.float32)
            with self.lock:
                if key not in self.tiles:
                    self.tiles[key] = cached
                    self.tile_bytes += cached.nbytes
                while self.tile_bytes > self.max_bytes and len(self.tiles) > 1:
                    _, old = self.tiles.popitem(last=False)
                    self.tile_bytes -= old.nbytes
        out = np.zeros(cached.shape[:-2] + (N, N))
        out[..., HALO:HALO + TILE, HALO:HALO + TILE] = cached
        return out

    def _parent_grid(self, mode, lod, tx, ty, inputs, kind, bounded):
        """Parent values over the cells covering this tile, read from owning tiles' interiors."""
        plod = parent_lod_of(lod, self.lod)
        d = plod - lod
        i0, j0 = tx * TILE - HALO, ty * TILE - HALO
        lo_i, hi_i = ((i0) >> d) - 1, ((i0 + N - 1) >> d) + 1
        lo_j, hi_j = ((j0) >> d) - 1, ((j0 + N - 1) >> d) + 1
        shape = (hi_j - lo_j + 1, hi_i - lo_i + 1)
        grid = None
        for ptx in range(lo_i >> 8, (hi_i >> 8) + 1):
            for pty in range(lo_j >> 8, (hi_j >> 8) + 1):
                parent = self.values(mode, plod, ptx, pty, inputs, kind, bounded)
                if grid is None:
                    grid = np.zeros(parent.shape[:-2] + shape)
                # Interior cells of this parent tile intersecting the needed range.
                ci0, cj0 = ptx * TILE, pty * TILE
                a0, a1 = max(lo_i, ci0), min(hi_i, ci0 + TILE - 1)
                b0, b1 = max(lo_j, cj0), min(hi_j, cj0 + TILE - 1)
                if a0 > a1 or b0 > b1:
                    continue
                grid[..., b0 - lo_j:b1 - lo_j + 1, a0 - lo_i:a1 - lo_i + 1] = \
                    parent[..., HALO + b0 - cj0:HALO + b1 - cj0 + 1, HALO + a0 - ci0:HALO + a1 - ci0 + 1]
        return plod, d, grid, lo_i, lo_j

    def _descend(self, mode, lod, tx, ty, inputs, kind, bounded):
        plod, d, pgrid, lo_i, lo_j = self._parent_grid(mode, lod, tx, ty, inputs, kind, bounded)
        xs, ys, i0, j0 = tile_axes(lod, tx, ty)
        k = 1 << d
        # One pycnophylactic iteration on the parent lattice: the bilinear surface
        # of the adjusted lattice has cell means close to the parent values, so
        # the per-block corrections below stay near one (no visible blocks).
        exact = pgrid                      # parent values: the conservation targets
        pgrid = _mean_preserving(pgrid)
        # Bilinear parent surface at child samples, in parent-index space.
        u = (i0 + np.arange(N) + .5) / k - .5 - lo_i
        v = (j0 + np.arange(N) + .5) / k - .5 - lo_j
        iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = (u - iu)[None, :], (v - iv)[:, None]
        p = pgrid
        base = (p[..., iv[:, None], iu[None, :]] * (1 - fu) + p[..., iv[:, None], iu[None, :] + 1] * fu) * (1 - fv) + \
               (p[..., iv[:, None] + 1, iu[None, :]] * (1 - fu) + p[..., iv[:, None] + 1, iu[None, :] + 1] * fu) * fv
        f = self.features(lod, tx, ty, inputs)
        name = mode.split('orogen-resources-', 1)[-1]
        coef, sigma = PROFILES[profile_of(mode)]
        logw = sigma * multi_noise(self.seed, name, lod, plod, xs, ys)
        mask = np.ones((N, N), bool)
        if f is not None:
            terms = np.clip(sum(c * f[k_] for k_, c in coef.items()), -3, 3) if coef else np.zeros((N, N))
            # Remove the parent-scale mean of the terms (kept by the parent); the
            # filter support (3 x 2.5 samples) stays inside the halo with two rings.
            terms = terms - gaussian_filter(terms, min(2.5, max(1., k / 2)), truncate=3.)
            logw = logw + terms
            mask = f['shelf'] if any(m in mode for m in SEA_MODES) else f['land']
        # Work region: interior blocks plus one complete ring of blocks, all
        # inside the halo, so every block's data is identical in every tile.
        e0, e1 = HALO - PASSES * k, HALO + TILE + PASSES * k
        ext = (slice(e0, e1), slice(e0, e1))
        nb = (e1 - e0) // k
        blocks = (np.arange(e1 - e0) // k)[:, None] * nb + (np.arange(e1 - e0) // k)[None, :]
        pb_i = ((i0 + e0) >> d) - lo_i + np.arange(nb)
        pb_j = ((j0 + e0) >> d) - lo_j + np.arange(nb)
        target = exact[..., pb_j[:, None], pb_i[None, :]].reshape(exact.shape[:-2] + (nb * nb,))
        weight = (np.cos(np.radians(np.clip(R.latitude(ys), -90, 90)))[:, None] * np.ones((1, N)))[ext]
        out = np.zeros(base.shape)
        if kind == 'mix':
            out[(slice(None),) + ext] = self._mix(name, lod, plod, xs, ys, base, ext, blocks, nb, k, target, weight, f)
            return out
        valid = mask[ext]
        if bounded:
            out[ext] = _bounded_fit(base[ext], logw[ext], target, blocks, nb, k, weight, valid, bounded)
        else:
            raw = np.maximum(base[ext], 0) * np.exp(logw[ext])
            for _ in range(PASSES):   # smoothed corrections (Tobler-like), each block's data is tile-independent
                c = _block_ratio(raw, target, blocks, nb, weight, valid)
                raw = raw * np.exp(_smooth_blocks(np.log(np.clip(c, 1e-6, 1e6)), nb, k))
            raw = _bump_exact(raw, target, blocks, nb, k, weight, valid)   # exact, continuous at block edges
            raw[~valid] = 0
            out[ext] = raw
        return out

    # -- mixtures (rock classes, tectonic contexts) ------------------------------
    def strike(self):
        """Structural azimuth (radians) at authority cells: parallel to orogenic belts."""
        if self._strike is None:
            belt = sum(self.fields.get('context_' + c, 0) for c in ('collision', 'arc', 'rift', 'backarc'))
            belt = gaussian_filter(np.asarray(belt, np.float64), 1.5, mode=('nearest', 'wrap'))
            gy, gx = np.gradient(belt)
            self._strike = np.arctan2(gy, gx) + np.pi / 2
        return self._strike

    def _mix(self, name, lod, plod, xs, ys, base, ext, blocks, nb, k, target, weight, f):
        kcls = base.shape[0]
        X, Y = np.meshgrid(xs[ext[1]], ys[ext[0]])
        g = self.grid
        si = (np.floor(X / g.size).astype(int) - g.i[0]) % len(g.i)
        sj = np.clip(np.floor(Y / g.size).astype(int) - g.j[0], 0, len(g.j) - 1)
        theta = self.strike()[sj, si]
        r = R.cell_size(lod)
        warp = (r * 4 * multi_noise(self.seed, name + '-warp', lod, plod, xs, ys))[ext]
        ucoord = (X + warp) * np.cos(theta) + (Y + warp) * np.sin(theta)
        rocks = name.startswith('geo-rock')
        logs = np.zeros((kcls,) + X.shape)
        for c in range(kcls):
            n = multi_noise(self.seed, f'{name}-c{c}', lod, plod, xs, ys, hurst=.8)[ext]
            if rocks and R.ROCKS[c] in ('limestone', 'sandstone', 'marl', 'evaporite', 'metamorphic'):
                # Strata parallel to the structure, at the wavelengths of this LOD step.
                band = sum(np.cos(2 * np.pi * ucoord / (R.cell_size(o) * 3) + 1.7 * c + 2.3 * o)
                           for o in range(lod + 1, plod + 1)) / max(1, plod - lod) ** .5
                logs[c] = 1.3 * band + .6 * n
            else:
                logs[c] = 1.5 * n                       # intrusions, flows, blobs
        if f is not None and rocks:
            # Hard rocks crop out on ridges and steep slopes, soft rocks in valleys.
            hard = np.array([R.ROCKS[c] in ('granite', 'basalt', 'metamorphic', 'ultramafic') for c in range(kcls)])
            relief = (f['ridge'] - f['valley'] + f['slope'] / 30)[ext]
            relief = relief - gaussian_filter(relief, min(2.5, max(1., k / 2)), truncate=3.)
            logs += np.where(hard, .6, -.4)[:, None, None] * relief[None]
        frac = np.maximum(base[(slice(None),) + ext], 1e-9) * np.exp(logs)
        frac /= frac.sum(axis=0)
        valid = np.ones(X.shape, bool)
        for _ in range(PASSES):   # smoothed corrections per class
            for c in range(kcls):
                frac[c] *= np.exp(_smooth_blocks(np.log(np.maximum(_block_ratio(frac[c], target[c], blocks, nb, weight, valid), 1e-9)), nb, k))
            frac /= frac.sum(axis=0)
        for _ in range(3):      # fitting with edge-free bumps: class means per block
            for c in range(kcls):
                frac[c] = _bump_exact(frac[c], target[c], blocks, nb, k, weight, valid)
            frac /= np.maximum(frac.sum(axis=0), 1e-12)
        for c in range(kcls):   # final exact class means; sums stay within ~1e-3 of one
            frac[c] = _bump_exact(frac[c], target[c], blocks, nb, k, weight, valid)
        return frac


def _mean_preserving(p):
    """p + (p - S p), with S the cell mean of the bilinear surface: [1, 6, 1] / 8 per axis."""
    q = np.pad(p, [(0, 0)] * (p.ndim - 2) + [(1, 1), (1, 1)], mode='edge')
    rows = (q[..., :-2, :] + 6 * q[..., 1:-1, :] + q[..., 2:, :]) / 8
    smooth_p = (rows[..., :, :-2] + 6 * rows[..., :, 1:-1] + rows[..., :, 2:]) / 8
    return np.maximum(2 * p - smooth_p, 0)


def _bump(nb, k):
    """Per-block bump, zero on block edges, unit mean: (pi/2)^2 sin(pi u) sin(pi v)."""
    u = (np.arange(nb * k) % k + .5) / k
    b = np.pi / 2 * np.sin(np.pi * u)
    return b[:, None] * b[None, :]


def _bump_exact(values, target, blocks, nb, k, weight, valid):
    """values * (1 + eps_b * bump): exact block means, continuous across block edges.

    Where the needed eps would make values negative (eps < -0.4), the block
    falls back to the plain ratio (exact, possibly a visible step).
    """
    phi = _bump(nb, k)
    num = _bincount(blocks[valid], (values * weight)[valid], minlength=nb * nb)
    nphi = _bincount(blocks[valid], (values * phi * weight)[valid], minlength=nb * nb)
    den = _bincount(blocks[valid], weight[valid], minlength=nb * nb)
    mean = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    mphi = np.divide(nphi, den, out=np.zeros_like(nphi), where=den > 0)
    eps = np.divide(target - mean, mphi, out=np.zeros_like(mean), where=mphi > 0)
    ok = (eps >= -.4) & (mphi > 0)
    out = values * (1 + np.where(ok, eps, 0)[blocks] * phi)
    ratio = np.where(mean > 0, target / np.where(mean > 0, mean, 1), 0.)
    return np.where(ok[blocks], out, values * ratio[blocks])


def _block_ratio(values, target, blocks, nb, weight, valid):
    """target / area-weighted block mean (1 where undefined), as an (nb, nb) grid."""
    num = _bincount(blocks[valid], (values * weight)[valid], minlength=nb * nb)
    den = _bincount(blocks[valid], weight[valid], minlength=nb * nb)
    mean = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    ratio = np.where(mean > 0, target / np.where(mean > 0, mean, 1), np.where(target > 0, 1., 0.))
    return np.where(den > 0, ratio, 1.).reshape(nb, nb)


def _smooth_blocks(grid, nb, k):
    """Bilinear interpolation of per-block values between block centres (edges clamp)."""
    s = (np.arange(nb * k) + .5) / k - .5
    i = np.clip(np.floor(s).astype(int), 0, nb - 2)
    t = np.clip(s - i, 0, 1)
    rows = grid[i] * (1 - t)[:, None] + grid[i + 1] * t[:, None]
    return rows[:, i] * (1 - t)[None, :] + rows[:, i + 1] * t[None, :]


def _bounded_fit(base, logw, target, blocks, nb, k, weight, valid, rng):
    """Logit-space shift per block, smoothed then exact: block means of [lo, hi] values."""
    lo, hi = rng
    eps = 1e-6
    p = np.clip((base - lo) / (hi - lo), eps, 1 - eps)
    z = np.log(p) - np.log1p(-p) + logw
    tgt = np.clip((target - lo) / (hi - lo), 0, 1)
    w, b_valid = weight[valid], blocks[valid]
    den = _bincount(b_valid, w, minlength=nb * nb)

    def shift(zz):
        lo_b, hi_b = np.full(nb * nb, -40.), np.full(nb * nb, 40.)
        zv = zz[valid]
        for _ in range(50):   # bisection on the monotone block mean
            mid = (lo_b + hi_b) / 2
            mean = _bincount(b_valid, w / (1 + np.exp(-(zv + mid[b_valid]))), minlength=nb * nb) / np.maximum(den, 1e-300)
            high = mean > tgt
            hi_b, lo_b = np.where(high, mid, hi_b), np.where(high, lo_b, mid)
        return np.where(den > 0, (lo_b + hi_b) / 2, 0.)

    for _ in range(PASSES):
        z = z + _smooth_blocks(np.clip(shift(z), -20, 20).reshape(nb, nb), nb, k)
    phi = _bump(nb, k)

    def shift_phi(zz):   # bisection with the edge-free bump: exact and continuous
        lo_b, hi_b = np.full(nb * nb, -40.), np.full(nb * nb, 40.)
        zv, pv = zz[valid], phi[valid]
        for _ in range(50):
            mid = (lo_b + hi_b) / 2
            mean = _bincount(b_valid, w / (1 + np.exp(-(zv + mid[b_valid] * pv))), minlength=nb * nb) / np.maximum(den, 1e-300)
            high = mean > tgt
            hi_b, lo_b = np.where(high, mid, hi_b), np.where(high, lo_b, mid)
        return np.where(den > 0, (lo_b + hi_b) / 2, 0.)

    out = 1 / (1 + np.exp(-(z + shift_phi(z)[blocks] * phi)))
    out = np.where(tgt[blocks] <= 0, 0, np.where(tgt[blocks] >= 1, 1, out))
    out = lo + (hi - lo) * out
    out[~valid] = 0
    return out


def block_means(values, lod, tx, ty, parent_lod, weight_mask=None):
    """Area-weighted means of the interior per parent cell (for tests and checks)."""
    d = parent_lod - lod
    xs, ys, i0, j0 = tile_axes(lod, tx, ty)
    w = np.cos(np.radians(np.clip(R.latitude(ys), -90, 90)))[:, None] * np.ones((1, N))
    if weight_mask is not None:
        w = w * weight_mask
    inner = (slice(HALO, HALO + TILE), slice(HALO, HALO + TILE))
    k = 1 << d
    v = values[..., inner[0], inner[1]]
    ww = w[inner]
    shape = v.shape[:-2] + (TILE // k, k, TILE // k, k)
    num = (v * ww).reshape(shape).sum(axis=(-3, -1))
    den = ww.reshape(TILE // k, k, TILE // k, k).sum(axis=(1, 3))
    return num / np.maximum(den, 1e-300)
