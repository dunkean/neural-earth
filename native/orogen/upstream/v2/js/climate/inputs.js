// Stage inputs used while the new climate is being built stage by stage (plans/CLIMATE_REBUILD_PLAN.md):
//  • on Earth, a stage not built yet is replaced by the observations (so each new stage is scored on its own);
//  • on other planets, by a simple zonal first guess — placeholders, replaced as the real stages arrive.
// All fields are monthly (12 layers, month m centred on t = m + 0.5), month-major on the climate grid. DOM-free.

/** Observed monthly winds (m/s) → a winds stage. */
export function observedWinds(u10, v10) {
    return () => ({ u: u10, v: v10, source: 'observed' });
}

/**
 * Placeholder winds until the circulation stage (Phase 3): textbook belts — trade winds, westerlies, polar
 * easterlies — shifted a few degrees toward the summer hemisphere.
 */
export function zonalWinds({ tiltDeg = 23.44 } = {}) {
    return ({ grid }) => {
        const G = grid.size, u = new Float32Array(12 * G), v = new Float32Array(12 * G);
        const g = (x, m, s) => Math.exp(-(((x - m) / s) ** 2));
        for (let m = 0; m < 12; m++) {
            const shift = 5 * (tiltDeg / 23.44) * Math.sin(2 * Math.PI * (m + 0.5 - 3.63) / 12);
            for (let j = 0; j < grid.h; j++) {
                const p = grid.lat[j] - shift, a = Math.abs(p), sgn = p >= 0 ? 1 : -1;
                const uu = -7 * g(a, 12, 10) + 9 * g(a, 47, 12) - 3 * g(a, 78, 8);
                const vv = sgn * (-2.5 * g(a, 12, 9) + 1.0 * g(a, 45, 12));
                for (let i = 0; i < grid.w; i++) { u[m * G + j * grid.w + i] = uu; v[m * G + j * grid.w + i] = vv; }
            }
        }
        return { u, v, source: 'zonal first guess' };
    };
}

/** Values of 12 monthly layers at (possibly fractional) anchor months → K layers. */
export function atAnchors(monthly, G, anchorMonths) {
    const K = anchorMonths.length, out = new Float32Array(K * G);
    anchorMonths.forEach((mf, k) => {
        const m0 = Math.floor(mf), f = mf - m0, a = ((m0 % 12) + 12) % 12, b = (a + 1) % 12;
        for (let c = 0; c < G; c++) out[k * G + c] = f ? (1 - f) * monthly[a * G + c] + f * monthly[b * G + c] : monthly[a * G + c];
    });
    return out;
}

/** Observed monthly temperature (°C, land) → a temperature stage (the oracle). */
export function observedTemperature(t2m) {
    return ({ grid, anchorMonths }) => ({ T: atAnchors(t2m, grid.size, anchorMonths), monthly: t2m, source: 'observed' });
}

/** Observed monthly rain (mm/month, land) → a rain stage. */
export function observedRain(precip) {
    return ({ grid, anchorMonths }) => ({ P: atAnchors(precip, grid.size, anchorMonths), monthly: precip, source: 'observed' });
}

/**
 * Placeholder rain until the rain stage (Phase 5): a tropical rain belt that follows the sun, dry subtropics,
 * wetter mid-latitudes, drier interiors. Only there so other planets get a Köppen map at all; judge temperature,
 * not rain, on them for now.
 */
export function zonalRain({ tiltDeg = 23.44 } = {}) {
    return ({ grid, surface, anchorMonths }) => {
        const G = grid.size, monthly = new Float32Array(12 * G);
        const g = (x, m, s) => Math.exp(-(((x - m) / s) ** 2));
        for (let m = 0; m < 12; m++) {
            const itcz = 8 * (tiltDeg / 23.44) * Math.sin(2 * Math.PI * (m + 0.5 - 3.63) / 12);
            for (let j = 0; j < grid.h; j++) {
                const lat = grid.lat[j], a = Math.abs(lat);
                const base = 15 + 230 * g(lat, itcz, 9) + 75 * g(a, 50, 13);
                for (let i = 0; i < grid.w; i++) {
                    const c = j * grid.w + i;
                    monthly[m * G + c] = base * (1 - 0.35 * surface.landFrac[c]);
                }
            }
        }
        return { P: atAnchors(monthly, G, anchorMonths), monthly, source: 'zonal placeholder' };
    };
}
