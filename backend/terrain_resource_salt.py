"""Medieval coastal salt. Provisional calibration and sources: RESOURCE_LAYERS §24.

Native-atlas zero crossings retain the coast inside coarse authority sea cells.
All choices are addressed; no query window, mutable RNG or inventory withdrawal.
"""
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

FAMILIES = ('sea_salt_solar', 'sea_salt_boiled')
FLOW_EQUIVALENCE = 100.  # display only: 1 t/year has the area of 100 stock tonnes


def coast_candidates(grid, f, height):
    import terrain_resources as R
    h, w = height.shape
    west, north, east, south = grid.bounds
    sx, sy = (east-west)/w, (south-north)/h
    parts = []
    f['coastal_salt_kg_year'] = np.zeros(grid.shape, np.int64)
    for axis, step in ((1, sx), (0, sy)):
        other = np.roll(height, -1, axis=axis)
        mask = ((height > 0) != (other > 0))
        if axis == 0:
            mask[-1] = False  # no north/south wrapping
        row, col = np.nonzero(mask)
        a, b = height[row, col], other[row, col]
        # Slope on landward half of the crossing: a low-flat proxy, not a tide model.
        land_h = np.maximum(a, b)
        metric = step * (np.maximum(.15, np.cos((north+(row+.5)*sy)/grid.radius_m)) if axis else 1.)
        slope = np.abs(a-b)/metric
        flat = (1-R.smoothstep(.015, .08, slope))*(1-R.smoothstep(100, 350, land_h))
        frac = a/(a-b)
        # Stay 90 m landward of the atlas shoreline, quantized to the LOD-0 address.
        frac = np.clip(frac + np.where(a > 0, -90/step, 90/step), 0, 1)
        x = west+(col+.5+(frac if axis else 0))*sx
        y = north+(row+.5+(frac if not axis else 0))*sy
        x = (x-west) % (east-west)+west
        live = flat > .05
        parts.append((np.floor(x[live]/30).astype(np.int64), np.floor(y[live]/30).astype(np.int64), flat[live]))
    for k, name in enumerate(('i', 'j', 'flat')):
        f['salt_coast_'+name] = np.concatenate([p[k] for p in parts]).astype(np.float32 if k == 2 else np.int64)


def append_sites(table, grid, f, seed):
    import terrain_resources as R
    from terrain_resource_deposits import DepositTable
    if not isinstance(table, DepositTable) or 'salt_coast_i' not in f:
        return table
    a = dict(table.arrays)
    # Extend every column, including caches and shared-memory transport, uniformly.
    a['annual_kg'] = np.zeros(len(table), np.int64)
    a['season_mask'] = np.zeros(len(table), np.uint16)
    a['pan_m2'] = np.zeros(len(table), np.int64)
    pi, pj, flat = (f['salt_coast_'+n] for n in ('i', 'j', 'flat'))
    address = R.cell_id(0, pi, pj)
    priority = R.cell_keys(seed, ('resources', 'coastal-salt', 'spacing'), address)
    x, y = (pi+.5)*30, (pj+.5)*30
    # One candidate per 12-km physical coastal square, then a 6-km hard core.
    # Chord distances handle longitude wrapping and the polar metric.
    lat, lon = y/grid.radius_m, x/grid.radius_m
    xyz = grid.radius_m*np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=1)
    boxes = np.floor(xyz/12000).astype(np.int64)
    _, inverse = np.unique(boxes, axis=0, return_inverse=True)
    order = np.lexsort((address, priority, inverse))
    idx = order[np.r_[True, np.diff(inverse[order]) != 0]] if len(order) else order
    pairs = cKDTree(xyz[idx]).query_pairs(6000, output_type='ndarray')
    keep = np.ones(len(idx), bool)
    if len(pairs):
        left, right = pairs.T
        lose = np.where(priority[idx[left]] < priority[idx[right]], right, left)
        keep[lose] = False
    idx = idx[keep]
    pi, pj, flat, address = pi[idx], pj[idx], flat[idx], address[idx]
    ci = (pi//(1 << grid.lod)-grid.i[0]) % grid.shape[1]
    cj = np.clip(pj//(1 << grid.lod)-grid.j[0], 0, grid.shape[0]-1)
    owner = cj*grid.shape[1]+ci
    # Nearest coastal authority climate; it is meaningful on either side of the coast.
    t, rain, pet = (f[n][:, cj, ci] for n in ('temperature_c', 'precipitation_mm', 'pet_mm'))
    solar_months = (t >= 15) & (rain < pet)
    # Tidal-flat / lagoon suitability proxy: native flat relief + sediment-retaining soil.
    basin = flat*(.5+.5*np.maximum(f['pedology_clayey'][cj, ci], f['pedology_sandy'][cj, ci]))
    solar = (solar_months.sum(axis=0) >= 2) & (basin >= .25)
    # Ignigenous salt can be made in wet/cold climates, but ice and agricultural
    # peak labour restrict the season. Months are 1-based in the exported record.
    harvest = f['harvest_month'][cj, ci]
    months = np.where(solar[None], solar_months,
        (t > 0) & (f['v2_ice_month'][:, cj, ci] < .5) & (np.arange(12)[:, None] != harvest))
    counts = np.bincount(owner, minlength=grid.ids.size)[owner]
    # Same 1e-4 pan-area basis as the continuous solar diagnostic; total assigned
    # area per owner cannot exceed that fraction. Native flat proxy replaces EDT.
    pan = np.rint(grid.area_m2.ravel()[owner]*1e-4*basin/counts).astype(np.int64)
    solar_kg = np.rint(pan*solar_months.sum(axis=0)*f.get('_salt_pan_t_m2',R.PARAMS['salt_pan_t_m2'])*1000).astype(np.int64)
    # 30-km nearby fuel density, including sea-owned coastal segments. Separate
    # initial fuel-limited capacity from renewable supply; peat is depletable.
    area = np.maximum(grid.area_m2, 1)
    wood = uniform_filter(f['wood_stock_m3']/area, size=3, mode=('nearest','wrap'))
    peat = uniform_filter(f['peat_stock_m3']/area, size=3, mode=('nearest','wrap'))
    fuel_t = (wood[cj, ci]*.002/12 + peat[cj, ci]*.0002/20)*area[cj, ci]/counts
    u = R.to_unit(R.cell_keys(seed, ('resources','coastal-salt','labour'), address))
    boiled_t = np.minimum(fuel_t, months.sum(axis=0)*3*(.5+u))
    annual = np.where(solar, solar_kg, np.floor(boiled_t*1000).astype(np.int64))
    live = (annual >= 1000) & months.any(axis=0)
    pi, pj, owner, pan, annual, solar = (v[live] for v in (pi,pj,owner,pan,annual,solar))
    months = months[:, live]
    size = len(pi)
    names = tuple(sorted(table.families+FAMILIES))
    a['family'] = np.array([names.index(n) for n in table.families], np.int16)[a['family']]
    present = np.zeros((len(names), len(table.metals)), bool)
    for k, n in enumerate(table.families):
        present[names.index(n)] = a['present'][k]
    extra = {k:np.zeros((size,)+v.shape[1:], v.dtype) for k,v in a.items() if k != 'present'}
    extra.update(family=np.where(solar, names.index(FAMILIES[0]), names.index(FAMILIES[1])).astype(np.int16),
        source=R.cell_id(0,pi,pj), owner=grid.ids.ravel()[owner], position=R.cell_id(0,pi,pj),
        i=pi,j=pj,annual_kg=annual,season_mask=np.sum(months.astype(np.uint16)*(1 << np.arange(12, dtype=np.uint16))[:,None],axis=0,dtype=np.uint16),
        pan_m2=np.where(solar,pan,2000),access=np.ones(size),outcrop=np.ones(size,bool))
    arrays = {k:np.concatenate((v,extra[k])) for k,v in a.items() if k != 'present'}
    arrays['present'] = present
    # Integer annual budget (kg/year) equals the site sum exactly in each owner.
    total = np.zeros(grid.ids.size, np.int64)
    np.add.at(total, owner, annual)
    f['coastal_salt_kg_year'] = total.reshape(grid.shape)
    return DepositTable(arrays, names, table.metals)
