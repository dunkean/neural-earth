// The fixed climate grid: a regular lat-lon grid, identical at every terrain resolution and on every device.
// Cell (i, j): longitude −180 + (i + 0.5)·step (west → east), latitude 90 − (j + 0.5)·step (north → south),
// so a field is stored row-major from the north-west corner, like an equirectangular image.
// DOM-free: shared by the new climate core, the climate lab and the Node tuning scripts.

export const DEG = Math.PI / 180;

export class LatLonGrid {
    /** @param {number} stepDeg  cell size in degrees (1 → 360 × 180 cells) */
    constructor(stepDeg = 1) {
        this.step = stepDeg;
        this.w = Math.round(360 / stepDeg);
        this.h = Math.round(180 / stepDeg);
        this.size = this.w * this.h;
        this.lat = new Float64Array(this.h);             // degrees, row centres
        this.lon = new Float64Array(this.w);             // degrees, column centres
        for (let j = 0; j < this.h; j++) this.lat[j] = 90 - (j + 0.5) * stepDeg;
        for (let i = 0; i < this.w; i++) this.lon[i] = -180 + (i + 0.5) * stepDeg;
        // Relative cell area ∝ cos(lat); normalised so the whole sphere sums to 1.
        this.rowArea = new Float64Array(this.h);
        let total = 0;
        for (let j = 0; j < this.h; j++) { this.rowArea[j] = Math.cos(this.lat[j] * DEG); total += this.rowArea[j] * this.w; }
        for (let j = 0; j < this.h; j++) this.rowArea[j] /= total;
    }

    /** Fractional column/row of a lat/lon (degrees), for sampling. */
    colOf(lonDeg) { return ((lonDeg + 180) / this.step) - 0.5; }
    rowOf(latDeg) { return ((90 - latDeg) / this.step) - 0.5; }

    /** Cell index containing lat/lon (degrees). */
    cellAt(latDeg, lonDeg) {
        let i = Math.floor((lonDeg + 180) / this.step);
        let j = Math.floor((90 - latDeg) / this.step);
        i = ((i % this.w) + this.w) % this.w;
        j = Math.max(0, Math.min(this.h - 1, j));
        return j * this.w + i;
    }

    /**
     * Bilinear sample of a field (one layer of `size` values) at lat/lon (degrees). Longitude wraps; latitude clamps
     * at the pole rows. NaN cells are skipped (weights renormalised); all four NaN → NaN.
     */
    sample(field, latDeg, lonDeg, offset = 0) {
        const x = this.colOf(lonDeg), y = Math.max(0, Math.min(this.h - 1, this.rowOf(latDeg)));
        const x0 = Math.floor(x), y0 = Math.floor(y);
        const fx = x - x0, fy = y - y0;
        const i0 = ((x0 % this.w) + this.w) % this.w, i1 = i0 + 1 === this.w ? 0 : i0 + 1;
        const j0 = y0, j1 = y0 + 1 < this.h ? y0 + 1 : y0;
        const w = this.w;
        const v00 = field[offset + j0 * w + i0], v10 = field[offset + j0 * w + i1];
        const v01 = field[offset + j1 * w + i0], v11 = field[offset + j1 * w + i1];
        const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
        if (v00 === v00 && v10 === v10 && v01 === v01 && v11 === v11) {       // common case: no gaps
            return v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11;
        }
        let sum = 0, wsum = 0;
        if (v00 === v00 && w00 > 0) { sum += v00 * w00; wsum += w00; }
        if (v10 === v10 && w10 > 0) { sum += v10 * w10; wsum += w10; }
        if (v01 === v01 && w01 > 0) { sum += v01 * w01; wsum += w01; }
        if (v11 === v11 && w11 > 0) { sum += v11 * w11; wsum += w11; }
        return wsum > 0 ? sum / wsum : NaN;
    }

    /** Area-weighted mean of a field over cells where mask[c] is truthy and the value is finite. */
    areaMean(field, mask = null, offset = 0) {
        let s = 0, a = 0;
        for (let j = 0; j < this.h; j++) {
            const wa = this.rowArea[j];
            for (let i = 0; i < this.w; i++) {
                const c = j * this.w + i;
                if (mask && !mask[c]) continue;
                const v = field[offset + c];
                if (v === v) { s += v * wa; a += wa; }
            }
        }
        return a > 0 ? s / a : NaN;
    }
}

/**
 * Block average of a fine-grid field onto a coarser grid of the same family (ratio × ratio cells, area-weighted;
 * NaN skipped, all-NaN → 0). ratio 1 copies.
 */
export function coarsenField(fine, coarse, ratio, src) {
    if (ratio === 1) { const out = new Float64Array(src.length); for (let c = 0; c < src.length; c++) { const v = src[c]; out[c] = v === v ? v : 0; } return out; }
    const out = new Float64Array(coarse.size);
    for (let J = 0; J < coarse.h; J++) for (let I = 0; I < coarse.w; I++) {
        let s = 0, ws = 0;
        for (let dj = 0; dj < ratio; dj++) {
            const j = J * ratio + dj, wt = fine.rowArea[j];
            for (let di = 0; di < ratio; di++) { const v = src[j * fine.w + I * ratio + di]; if (v === v) { s += v * wt; ws += wt; } }
        }
        out[J * coarse.w + I] = ws > 0 ? s / ws : 0;
    }
    return out;
}

/** Bilinear interpolation of a coarse-grid field onto the fine grid, written at out[offset …]. ratio 1 copies. */
export function refineField(coarse, fine, ratio, src, out, offset = 0) {
    if (ratio === 1) { for (let c = 0; c < fine.size; c++) out[offset + c] = src[c]; return; }
    const map = refineMap(coarse, fine);
    for (let c = 0, k = 0; c < fine.size; c++, k += 4) {
        out[offset + c] = map.w[k] * src[map.i[k]] + map.w[k + 1] * src[map.i[k + 1]] + map.w[k + 2] * src[map.i[k + 2]] + map.w[k + 3] * src[map.i[k + 3]];
    }
}

/** The four coarse cells and weights behind every fine cell (built once per grid pair). */
const refineMaps = new Map();
function refineMap(g, core) {
    const key = `${g.step}|${core.step}`;
    let m = refineMaps.get(key);
    if (m) return m;
    m = { i: new Int32Array(4 * core.size), w: new Float64Array(4 * core.size) };
    for (let j = 0; j < core.h; j++) for (let ii = 0; ii < core.w; ii++) {
        const c = j * core.w + ii, k = 4 * c;
        const x = g.colOf(core.lon[ii]), y = Math.max(0, Math.min(g.h - 1, g.rowOf(core.lat[j])));
        const x0 = Math.floor(x), y0 = Math.min(g.h - 1, Math.floor(y)), fx = x - x0, fy = y - y0;
        const i0 = ((x0 % g.w) + g.w) % g.w, i1 = (i0 + 1) % g.w, j1 = Math.min(g.h - 1, y0 + 1);
        m.i[k] = y0 * g.w + i0; m.w[k] = (1 - fx) * (1 - fy);
        m.i[k + 1] = y0 * g.w + i1; m.w[k + 1] = fx * (1 - fy);
        m.i[k + 2] = j1 * g.w + i0; m.w[k + 2] = (1 - fx) * fy;
        m.i[k + 3] = j1 * g.w + i1; m.w[k + 3] = fx * fy;
    }
    refineMaps.set(key, m);
    return m;
}

// ── For drawing: a coarse field to the core grid without the coarse grid showing through ──
// A field that only means something on one side of the coast (the sea's currents and ice, the land's ice sheets) is zero
// or arbitrary on the other side's coarse cells; refined linearly, every coast then shows the coarse cells' straight
// edges, and near the poles — where a row's cells are much narrower than they are tall — wedges. So: carry the field
// across the coast from its own side (as far as it takes), even out the polar rows, and refine with a smooth curve.

/** How far a field is carried across the coast (km): enough to close any coastal rim; farther — an island far out at sea,
 *  an inland sea — a cell keeps its own value (all of them mean something there: the energy balance's ice sheet and
 *  temperature are worked out everywhere). */
export const FILL_REACH_KM = 600;
/**
 * Carry a field across from the cells where `valid` is set to the others, ring by ring (each taking its set neighbours'
 * mean; longitude wraps), up to `reachKm` from them. In place; returns the field.
 */
export function fillAcross(grid, field, valid, reachKm = FILL_REACH_KM) {
    const { w, h, size } = grid;
    const have = new Uint8Array(size);                                // (a plain loop: TypedArray.from with a callback is slow)
    for (let c = 0; c < size; c++) have[c] = valid[c] ? 1 : 0;
    let left = 0;
    for (let c = 0; c < size; c++) if (!have[c]) left++;
    if (left === size) return field;                                  // (nothing to carry)
    const rings = Math.max(1, Math.round(reachKm / (6371 * grid.step * DEG)));
    // (ring by ring, only the cells beside the last ring's — the first ring's beside any set cell — are looked at: the
    // same cells, the same means of their set neighbours in the same order, without scanning the grid each ring)
    const queued = new Uint8Array(size), set = [];
    let front = [];
    const near = (c, out) => {
        const j = (c / w) | 0, i = c - j * w;
        const W = j * w + (i === 0 ? w - 1 : i - 1), E = j * w + (i === w - 1 ? 0 : i + 1);
        if (!have[W] && !queued[W]) { queued[W] = 1; out.push(W); }
        if (!have[E] && !queued[E]) { queued[E] = 1; out.push(E); }
        if (j > 0 && !have[c - w] && !queued[c - w]) { queued[c - w] = 1; out.push(c - w); }
        if (j < h - 1 && !have[c + w] && !queued[c + w]) { queued[c + w] = 1; out.push(c + w); }
    };
    for (let c = 0; c < size; c++) if (have[c]) near(c, front);
    for (let ring = 0; ring < rings && left > 0 && front.length; ring++) {
        set.length = 0;
        for (const c of front) {
            const j = (c / w) | 0, i = c - j * w;
            const W = j * w + (i === 0 ? w - 1 : i - 1), E = j * w + (i === w - 1 ? 0 : i + 1);
            let s = 0, k = 0;
            if (have[W]) { s += field[W]; k++; }
            if (have[E]) { s += field[E]; k++; }
            if (j > 0 && have[c - w]) { s += field[c - w]; k++; }
            if (j < h - 1 && have[c + w]) { s += field[c + w]; k++; }
            if (k) set.push(c, s / k);
        }
        if (!set.length) break;
        for (let q = 0; q < set.length; q += 2) { field[set[q]] = set[q + 1]; have[set[q]] = 1; }
        left -= set.length / 2;
        const next = [];
        for (const c of front) queued[c] = 0;
        for (let q = 0; q < set.length; q += 2) near(set[q], next);
        front = next;
    }
    return field;
}

/**
 * Even out the rows near the poles: each row averaged along its length over about as many cells as one cell is tall
 * (a box of ≈ 1 / cos latitude cells) — nothing changes where the cells are square. In place; returns the field.
 */
export function evenPolarRows(grid, field) {
    const { w, h } = grid, row = new Float64Array(w);
    for (let j = 0; j < h; j++) {
        const r = Math.round((1 / Math.max(1e-3, Math.cos(grid.lat[j] * DEG)) - 1) / 2);
        if (r <= 0) continue;
        if (2 * r + 1 >= w) { let s = 0; for (let i = 0; i < w; i++) s += field[j * w + i]; s /= w; for (let i = 0; i < w; i++) field[j * w + i] = s; continue; }
        for (let i = 0; i < w; i++) row[i] = field[j * w + i];
        let s = 0;
        for (let d = -r; d <= r; d++) s += row[(d + w) % w];
        for (let i = 0; i < w; i++) {
            field[j * w + i] = s / (2 * r + 1);
            s += row[(i + r + 1) % w] - row[(i - r + w) % w];
        }
    }
    return field;
}

/** Catmull–Rom weights for a point t (0…1) between the 2nd and 3rd of four samples. */
function cubicWeights(t, out) {
    const t2 = t * t, t3 = t2 * t;
    out[0] = 0.5 * (-t3 + 2 * t2 - t); out[1] = 0.5 * (3 * t3 - 5 * t2 + 2); out[2] = 0.5 * (-3 * t3 + 4 * t2 + t); out[3] = 0.5 * (t3 - t2);
}
/** A coarse-grid field onto the fine grid by a smooth (Catmull–Rom, 4 × 4) curve, written at out[offset …]; longitude
 *  wraps, latitude holds at the poles; the result kept within [lo, hi]. ratio 1 copies. */
export function refineSmooth(coarse, fine, ratio, src, out, offset = 0, lo = -Infinity, hi = Infinity) {
    if (ratio === 1) { for (let c = 0; c < fine.size; c++) out[offset + c] = src[c]; return; }
    const W = coarse.w, H = coarse.h, wx = new Float64Array(4), wy = new Float64Array(4), ix = new Int32Array(4);
    for (let j = 0; j < fine.h; j++) {
        const y = Math.max(0, Math.min(H - 1, coarse.rowOf(fine.lat[j]))), y0 = Math.floor(y);
        cubicWeights(y - y0, wy);
        for (let i = 0; i < fine.w; i++) {
            const x = coarse.colOf(fine.lon[i]), x0 = Math.floor(x);
            cubicWeights(x - x0, wx);
            for (let a = 0; a < 4; a++) ix[a] = (((x0 - 1 + a) % W) + W) % W;
            let v = 0;
            for (let b = 0; b < 4; b++) {
                const row = Math.max(0, Math.min(H - 1, y0 - 1 + b)) * W;
                v += wy[b] * (wx[0] * src[row + ix[0]] + wx[1] * src[row + ix[1]] + wx[2] * src[row + ix[2]] + wx[3] * src[row + ix[3]]);
            }
            out[offset + j * fine.w + i] = v < lo ? lo : v > hi ? hi : v;
        }
    }
}

/**
 * A coarse field for drawing: carried across the coast from where `valid` is set (or not, with null; FILL_REACH_KM), its polar rows
 * evened, refined smoothly onto the core grid at out[offset …] (within [lo, hi]). `src` is left as it is.
 */
export function refineForDrawing(coarse, fine, ratio, src, out, offset = 0, valid = null, lo = -Infinity, hi = Infinity) {
    const f = Float64Array.from(src);
    if (valid) fillAcross(coarse, f, valid);
    evenPolarRows(coarse, f);
    refineSmooth(coarse, fine, ratio, f, out, offset, lo, hi);
}

/**
 * Regrid a field given on any regular source lat/lon grid (1-D coordinate arrays in degrees, either orientation,
 * longitudes 0…360 or −180…180) onto `grid` by bilinear interpolation. `src` holds `layers` stacked layers of
 * srcLat.length × srcLon.length values. Returns Float32Array(layers × grid.size), NaN where no source data.
 */
export function regridBilinear(src, srcLat, srcLon, layers, grid) {
    const nLat = srcLat.length, nLon = srcLon.length;
    const lat0 = srcLat[0], dLat = (srcLat[nLat - 1] - srcLat[0]) / (nLat - 1);
    const lon0 = srcLon[0], dLon = (srcLon[nLon - 1] - srcLon[0]) / (nLon - 1);
    const lonsWrap = Math.abs(dLon * nLon - 360) < 1e-6;
    const out = new Float32Array(layers * grid.size);
    for (let j = 0; j < grid.h; j++) {
        let y = (grid.lat[j] - lat0) / dLat;
        y = Math.max(0, Math.min(nLat - 1, y));
        const y0 = Math.min(nLat - 2, Math.floor(y)), fy = y - y0;
        for (let i = 0; i < grid.w; i++) {
            let x = (grid.lon[i] - lon0) / dLon;
            if (lonsWrap) x = ((x % nLon) + nLon) % nLon;
            else x = Math.max(0, Math.min(nLon - 1, x));
            const x0 = Math.floor(x), fx = x - x0;
            const x1 = lonsWrap ? (x0 + 1) % nLon : Math.min(nLon - 1, x0 + 1);
            const corners = [
                [y0 * nLon + x0, (1 - fx) * (1 - fy)], [y0 * nLon + x1, fx * (1 - fy)],
                [(y0 + 1) * nLon + x0, (1 - fx) * fy], [(y0 + 1) * nLon + x1, fx * fy],
            ];
            for (let k = 0; k < layers; k++) {
                const base = k * nLat * nLon;
                let s = 0, ws = 0;
                for (const [idx, wt] of corners) {
                    const v = src[base + idx];
                    if (wt > 0 && v === v) { s += v * wt; ws += wt; }
                }
                out[k * grid.size + j * grid.w + i] = ws > 1e-9 ? s / ws : NaN;
            }
        }
    }
    return out;
}
