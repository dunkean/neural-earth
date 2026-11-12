// Colouring a height field: hypsometric tints (deep sea → shelf, lowland green → brown → rock → snow) and
// hillshading lit from the north-west, per cell of whatever mesh the heights live on. DOM-free.

const SEA_STOPS = [[-9, [8, 16, 48]], [-6, [18, 38, 92]], [-4, [26, 60, 128]], [-2, [40, 96, 164]], [-0.5, [70, 140, 196]], [0, [118, 182, 214]]];
const LAND_STOPS = [[0, [74, 122, 70]], [0.3, [104, 144, 82]], [0.8, [158, 164, 104]], [1.6, [166, 138, 92]], [2.6, [138, 110, 86]], [3.8, [128, 118, 112]], [5, [206, 206, 210]], [7, [255, 255, 255]]];

function ramp(stops, v, out, o) {
    if (v <= stops[0][0]) { const c = stops[0][1]; out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2]; return; }
    for (let i = 1; i < stops.length; i++) {
        if (v > stops[i][0] && i < stops.length - 1) continue;
        const a = stops[i - 1][0], ca = stops[i - 1][1], b = stops[i][0], cb = stops[i][1], t = Math.min(1, Math.max(0, (v - a) / (b - a)));
        out[o] = ca[0] + (cb[0] - ca[0]) * t; out[o + 1] = ca[1] + (cb[1] - ca[1]) * t; out[o + 2] = ca[2] + (cb[2] - ca[2]) * t;
        return;
    }
}

/** RGB of a height (km above sea level) on the hypsometric ramp. */
export function tintOf(km, out = [0, 0, 0], o = 0) { ramp(km < 0 ? SEA_STOPS : LAND_STOPS, km, out, o); return out; }

/**
 * Hillshade per cell: 0 (facing away from the light) … ~1.4 (facing it); 1 on flat ground. Slopes are exaggerated
 * so relief reads at planet scale.
 * @returns {Float32Array}
 */
export function hillshade(mesh, heightKm, { exaggeration = 20, seaLevelKm = 0, seaExaggeration = 0.35 } = {}) {
    const { xyz, adjOffset, adjList, numCells } = mesh, R = 6371, out = new Float32Array(numCells);
    const Le = -0.5, Ln = 0.5, Lu = Math.SQRT1_2;
    for (let c = 0; c < numCells; c++) {
        const x = xyz[3 * c], y = xyz[3 * c + 1], z = xyz[3 * c + 2];
        let ex = -y, ey = x;
        const el = Math.hypot(ex, ey) || 1; ex /= el; ey /= el;
        const nx = -z * ey, ny = z * ex, nz = x * ey - y * ex;                       // north = p × east
        let a11 = 0, a12 = 0, a22 = 0, b1 = 0, b2 = 0;
        const h = heightKm[c];
        for (let k = adjOffset[c]; k < adjOffset[c + 1]; k++) {
            const nb = adjList[k], dx = (xyz[3 * nb] - x) * R, dy = (xyz[3 * nb + 1] - y) * R, dz = (xyz[3 * nb + 2] - z) * R;
            const u = dx * ex + dy * ey, v = dx * nx + dy * ny + dz * nz, dh = heightKm[nb] - h;
            a11 += u * u; a12 += u * v; a22 += v * v; b1 += u * dh; b2 += v * dh;
        }
        const det = a11 * a22 - a12 * a12;
        let ge = 0, gn = 0;
        if (Math.abs(det) > 1e-12) { ge = (a22 * b1 - a12 * b2) / det; gn = (a11 * b2 - a12 * b1) / det; }
        const ex2 = h < seaLevelKm ? exaggeration * seaExaggeration : exaggeration;
        const px = -ex2 * ge, py = -ex2 * gn, len = Math.sqrt(px * px + py * py + 1);
        out[c] = Math.max(0, (px * Le + py * Ln + Lu) / len) / Lu;
    }
    return out;
}

/** Tinted, shaded colours per cell into `out` (3 bytes per cell). `mode`: 'terrain' | 'shade' (grey) | 'tint' (no shading). */
export function paintRelief(heightKm, shade, out, { seaLevelKm = 0, mode = 'terrain' } = {}) {
    const rgb = [0, 0, 0];
    for (let c = 0; c < heightKm.length; c++) {
        const s = shade ? shade[c] : 1;
        if (mode === 'shade') { const g = Math.min(255, 150 * s); out[3 * c] = out[3 * c + 1] = out[3 * c + 2] = g; continue; }
        tintOf(heightKm[c] - seaLevelKm, rgb);
        const k = mode === 'tint' ? 1 : 0.35 + 0.65 * s;
        out[3 * c] = Math.min(255, rgb[0] * k); out[3 * c + 1] = Math.min(255, rgb[1] * k); out[3 * c + 2] = Math.min(255, rgb[2] * k);
    }
}
