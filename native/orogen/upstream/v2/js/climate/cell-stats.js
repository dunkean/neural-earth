// What a place's twelve months say about its climate, beyond Köppen — for the Inspect layers of Generate, Import and
// Evolution (plans/V1_NEW_CLIMATE_PLAN.md, "Inspect layers"). Real-world measures, each from T12 (°C) and P12 (mm):
//   tRange      the warmest month − the coldest (°C): continentality's effect
//   aridity     the UNEP aridity index, P / PET, PET by Thornthwaite with the day-length correction
//   growing     months warmer than 10 °C
//   rainSeason  Walsh & Lawler's seasonality index, Σ|month − year/12| / year (0 even … ≥ 1.2 in a month or two)
// Kept either as Float32 (exact) or as one byte each (Q: compact enough for a 10M-cell finished moment).

const DEG = Math.PI / 180;

/** Hours of daylight at a latitude (radians) in the middle of month m. */
function dayHours(latRad, m) {
    const doy = 15 + 30.44 * m, decl = 23.44 * DEG * Math.sin(2 * Math.PI * (doy - 81) / 365);
    const x = -Math.tan(latRad) * Math.tan(decl);
    return x <= -1 ? 24 : x >= 1 ? 0 : 24 * Math.acos(x) / Math.PI;
}

/** The four measures of one place (`out`: an object reused between calls). */
export function monthlyStats(T12, P12, latRad, out = {}) {
    let tMin = Infinity, tMax = -Infinity, pSum = 0, grow = 0, I = 0;
    for (let m = 0; m < 12; m++) {
        const t = T12[m];
        if (t < tMin) tMin = t;
        if (t > tMax) tMax = t;
        pSum += P12[m];
        if (t > 10) grow++;
        if (t > 0) I += Math.pow(t / 5, 1.514);
    }
    let si = 0;
    if (pSum > 0) { for (let m = 0; m < 12; m++) si += Math.abs(P12[m] - pSum / 12); si /= pSum; }
    const a = 6.75e-7 * I * I * I - 7.71e-5 * I * I + 1.792e-2 * I + 0.49239;
    let pet = 0;
    for (let m = 0; m < 12; m++) {
        const t = T12[m];
        if (t <= 0) continue;
        const raw = t >= 26.5 ? -415.85 + 32.24 * t - 0.43 * t * t : I > 0 ? 16 * Math.pow(10 * t / I, a) : 0;
        pet += raw * (dayHours(latRad, m) / 12) * (30.44 / 30);
    }
    out.tRange = tMax - tMin;
    out.growing = grow;
    out.rainSeason = si;
    out.aridity = pet > 0 ? pSum / pet : (pSum > 0 ? 9.99 : 0);
    return out;
}

/** One-byte encodings (0 = no data: the sea): tRange ¼ °C, aridity on a log scale (0.001 … 50), growing ×20, rainSeason ×100. */
export const Q = {
    tRange: { enc: v => Math.max(1, Math.min(255, Math.round(4 * v) + 1)), dec: q => (q - 1) / 4 },
    aridity: { enc: v => Math.max(1, Math.min(255, Math.round(54 * (Math.log10(Math.max(1e-3, v)) + 3)) + 1)), dec: q => Math.pow(10, (q - 1) / 54 - 3) },
    growing: { enc: v => Math.max(1, Math.min(255, Math.round(20 * v) + 1)), dec: q => (q - 1) / 20 },
    rainSeason: { enc: v => Math.max(1, Math.min(255, Math.round(100 * v) + 1)), dec: q => (q - 1) / 100 },
};
export const STAT_KEYS = ['tRange', 'aridity', 'growing', 'rainSeason'];

/**
 * A collector for computeClimate's / cellClimate's onCell: `stats` (per cell, the four measures — Float32 with NaN at
 * sea, or with `bytes`, Uint8 per Q with 0 at sea) and `add(r, T12, P12)`. `latRadOf(r)`: a cell's latitude (radians).
 */
export function cellStatsCollector(n, latRadOf, { bytes = false } = {}) {
    const make = () => (bytes ? new Uint8Array(n) : new Float32Array(n).fill(NaN));
    const stats = { tRange: make(), aridity: make(), growing: make(), rainSeason: make() }, s = {};
    const add = (r, T12, P12) => {
        monthlyStats(T12, P12, latRadOf(r), s);
        if (bytes) for (const k of STAT_KEYS) stats[k][r] = Q[k].enc(s[k]);
        else for (const k of STAT_KEYS) stats[k][r] = s[k];
    };
    return { stats, add };
}

/** A stat's value from either form (Float32 or bytes): NaN where there is none. */
export function statValue(arr, key, i) {
    if (arr instanceof Uint8Array) return arr[i] ? Q[key].dec(arr[i]) : NaN;
    return arr[i];
}
