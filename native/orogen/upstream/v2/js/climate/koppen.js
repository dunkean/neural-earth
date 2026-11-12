// Köppen–Geiger classification from 12 monthly means, with the rules of Beck et al. (2018, 2023) — the modern
// standard (Peel et al. 2007 thresholds, 0 °C C/D boundary, As merged into Aw). DOM-free.
//
// Months are indexed 0 … 11 from the northern-hemisphere winter solstice month (Earth: January), so on any planet
// the two fixed halves 3…8 and 9…2 are the halves around the solstices. "Summer" is the warmer of the two for
// each cell, which is Beck's April–September / October–March rule generalised to either hemisphere.
//
// Class ids and colours are the same as V1's table (js/koppen.js), so maps look familiar and scores compare 1:1.

export const KOPPEN_CLASSES = [
    { code: 'Ocean', name: 'Ocean',                                 color: [0.29, 0.44, 0.65] },
    { code: 'Af',    name: 'Tropical rainforest',                   color: [0.00, 0.00, 1.00] },
    { code: 'Am',    name: 'Tropical monsoon',                      color: [0.00, 0.47, 1.00] },
    { code: 'Aw',    name: 'Tropical savanna',                      color: [0.27, 0.67, 0.98] },
    { code: 'BWh',   name: 'Hot desert',                            color: [1.00, 0.00, 0.00] },
    { code: 'BWk',   name: 'Cold desert',                           color: [1.00, 0.59, 0.59] },
    { code: 'BSh',   name: 'Hot steppe',                            color: [0.96, 0.65, 0.00] },
    { code: 'BSk',   name: 'Cold steppe',                           color: [1.00, 0.86, 0.39] },
    { code: 'Cfa',   name: 'Humid subtropical',                     color: [0.78, 1.00, 0.31] },
    { code: 'Cfb',   name: 'Oceanic',                               color: [0.39, 1.00, 0.31] },
    { code: 'Cfc',   name: 'Subpolar oceanic',                      color: [0.20, 0.78, 0.00] },
    { code: 'Csa',   name: 'Hot-summer Mediterranean',              color: [1.00, 1.00, 0.00] },
    { code: 'Csb',   name: 'Warm-summer Mediterranean',             color: [0.78, 0.78, 0.00] },
    { code: 'Csc',   name: 'Cold-summer Mediterranean',             color: [0.59, 0.59, 0.00] },
    { code: 'Cwa',   name: 'Humid subtropical (monsoon)',           color: [0.59, 1.00, 0.59] },
    { code: 'Cwb',   name: 'Subtropical highland',                  color: [0.39, 0.78, 0.39] },
    { code: 'Cwc',   name: 'Cold subtropical highland',             color: [0.20, 0.59, 0.20] },
    { code: 'Dfa',   name: 'Hot-summer continental',                color: [0.00, 1.00, 1.00] },
    { code: 'Dfb',   name: 'Warm-summer continental',               color: [0.22, 0.78, 1.00] },
    { code: 'Dfc',   name: 'Subarctic',                             color: [0.00, 0.49, 0.49] },
    { code: 'Dfd',   name: 'Extremely cold subarctic',              color: [0.00, 0.27, 0.37] },
    { code: 'Dsa',   name: 'Hot-summer continental (dry summer)',   color: [0.90, 0.50, 1.00] },
    { code: 'Dsb',   name: 'Warm-summer continental (dry summer)',  color: [0.70, 0.35, 0.85] },
    { code: 'Dsc',   name: 'Subarctic (dry summer)',                color: [0.50, 0.20, 0.65] },
    { code: 'Dsd',   name: 'Extremely cold subarctic (dry summer)', color: [0.35, 0.10, 0.45] },
    { code: 'Dwa',   name: 'Hot-summer continental (monsoon)',      color: [0.67, 0.69, 1.00] },
    { code: 'Dwb',   name: 'Warm-summer continental (monsoon)',     color: [0.43, 0.47, 0.78] },
    { code: 'Dwc',   name: 'Subarctic (monsoon)',                   color: [0.29, 0.31, 0.78] },
    { code: 'Dwd',   name: 'Extremely cold subarctic (monsoon)',    color: [0.20, 0.00, 0.53] },
    { code: 'ET',    name: 'Tundra',                                color: [0.70, 0.70, 0.70] },
    { code: 'EF',    name: 'Ice cap',                               color: [0.41, 0.41, 0.41] },
];

export const KOPPEN_ID = Object.fromEntries(KOPPEN_CLASSES.map((c, i) => [c.code, i]));

const SUMMER_A = [3, 4, 5, 6, 7, 8];     // months around the northern summer solstice
const SUMMER_B = [9, 10, 11, 0, 1, 2];   // months around the northern winter solstice

/**
 * Classify one cell. T, P: 12 monthly values (°C, mm/month), read as T[o + m·stride] so both
 * per-cell arrays (o = 0, stride = 1) and month-major fields (o = cell, stride = cells) work.
 * @returns {number} KOPPEN_CLASSES id (1 … 30)
 */
export function classifyCell(T, P, o = 0, stride = 1) {
    let MAT = 0, MAP = 0, Thot = -Infinity, Tcold = Infinity, Tmon10 = 0, Pdry = Infinity;
    for (let m = 0; m < 12; m++) {
        const t = T[o + m * stride], p = Math.max(0, P[o + m * stride]);
        MAT += t; MAP += p;
        if (t > Thot) Thot = t;
        if (t < Tcold) Tcold = t;
        if (t > 10) Tmon10++;
        if (p < Pdry) Pdry = p;
    }
    MAT /= 12;

    let tA = 0, tB = 0;
    for (const m of SUMMER_A) tA += T[o + m * stride];
    for (const m of SUMMER_B) tB += T[o + m * stride];
    const summer = tA >= tB ? SUMMER_A : SUMMER_B, winter = tA >= tB ? SUMMER_B : SUMMER_A;
    let Ps = 0, Psdry = Infinity, Pswet = 0, Pwdry = Infinity, Pwwet = 0;
    for (const m of summer) { const p = Math.max(0, P[o + m * stride]); Ps += p; if (p < Psdry) Psdry = p; if (p > Pswet) Pswet = p; }
    for (const m of winter) { const p = Math.max(0, P[o + m * stride]); if (p < Pwdry) Pwdry = p; if (p > Pwwet) Pwwet = p; }
    const Pw = MAP - Ps;

    // B (arid) takes precedence
    const Pth = Pw > 0.7 * MAP ? 2 * MAT : Ps > 0.7 * MAP ? 2 * MAT + 28 : 2 * MAT + 14;
    if (MAP < 10 * Pth) {
        const hot = MAT >= 18;
        if (MAP < 5 * Pth) return hot ? KOPPEN_ID.BWh : KOPPEN_ID.BWk;
        return hot ? KOPPEN_ID.BSh : KOPPEN_ID.BSk;
    }
    // A (tropical)
    if (Tcold >= 18) {
        if (Pdry >= 60) return KOPPEN_ID.Af;
        return Pdry >= 100 - MAP / 25 ? KOPPEN_ID.Am : KOPPEN_ID.Aw;
    }
    // E (polar)
    if (Thot <= 10) return Thot > 0 ? KOPPEN_ID.ET : KOPPEN_ID.EF;

    // C or D
    const C = Tcold > 0;
    let s = Psdry < 40 && Psdry < Pwwet / 3;
    let w = Pwdry < Pswet / 10;
    if (s && w) { s = Ps <= Pw; w = !s; }      // both: the wetter half decides
    const L2 = s ? 's' : w ? 'w' : 'f';
    const L3 = Thot >= 22 ? 'a' : Tmon10 >= 4 ? 'b' : (!C && Tcold < -38) ? 'd' : 'c';
    return KOPPEN_ID[(C ? 'C' : 'D') + L2 + L3];
}

/**
 * Classify every cell of month-major fields T, P (Float32Array(12 · n)). Cells where `land` is 0 (or any input is
 * NaN) get 0 (Ocean). Returns Uint8Array(n).
 */
export function classifyField(T, P, n, land = null) {
    const out = new Uint8Array(n);
    for (let c = 0; c < n; c++) {
        if (land && !land[c]) continue;
        let ok = true;
        for (let m = 0; m < 12 && ok; m++) ok = T[c + m * n] === T[c + m * n] && P[c + m * n] === P[c + m * n];
        if (ok) out[c] = classifyCell(T, P, c, n);
    }
    return out;
}

/** Major group letter of a class id ('' for Ocean). */
export function majorGroup(id) {
    return id > 0 ? KOPPEN_CLASSES[id].code[0] : '';
}
