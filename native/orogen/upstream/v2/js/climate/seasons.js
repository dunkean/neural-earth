// Twelve months from a few points in the year. The climate is computed at K "anchors" evenly spaced through the
// year; the 12 monthly values are rebuilt by trigonometric (Fourier) interpolation, which passes exactly through
// the anchors and is the smoothest periodic curve that does.
//
// Temperature is interpolated as is. Precipitation cannot go negative and its dry seasons are sharp, so it is
// interpolated in a transformed space (log or square root) and transformed back.
//
// Month m runs 0 … 11 (0 = the northern winter-solstice month; Earth: January). Anchor k sits at month
// phase + k · 12 / K (fractional months allowed). DOM-free.
//
// Trigonometric interpolation is linear in the anchor values, so for a given (K, phase, attenuation) it is a fixed
// 12 × K matrix: built once, then each cell costs 12·K multiply-adds and no allocation.

const TAU = 2 * Math.PI;

/** Forward / inverse precipitation transforms. eps (mm/month) keeps log finite for rainless months. */
export const PRECIP_TRANSFORMS = {
    linear: { fwd: p => p, inv: v => (v > 0 ? v : 0) },
    sqrt:   { fwd: p => (p > 0 ? Math.sqrt(p) : 0), inv: v => (v > 0 ? v * v : 0) },
    log:    { fwd: (p, eps) => Math.log((p > 0 ? p : 0) + eps), inv: (v, eps) => Math.max(0, Math.exp(v) - eps) },
};

const matrices = new Map();

/**
 * The 12 × K interpolation matrix (row-major Float64Array) for K equally spaced anchors starting at `phase`.
 * `attenuation`: if the anchors are running means (e.g. 3-month seasonal means) rather than point values, the
 * averaging window in months; harmonics are divided by the window's damping so rebuilt extremes are not flattened.
 */
export function interpolationMatrix(K, phase = 0, attenuation = 0) {
    const key = `${K}|${phase}|${attenuation}`;
    let M = matrices.get(key);
    if (M) return M;
    M = new Float64Array(12 * K);
    const H = Math.floor(K / 2);
    const damp = (h) => {
        if (!(attenuation > 0) || h === 0) return 1;
        const x = Math.PI * h * attenuation / 12;          // boxcar of `attenuation` months → sinc damping
        return Math.max(0.25, Math.sin(x) / x);
    };
    for (let m = 0; m < 12; m++) {
        const t = TAU * (m - phase) / 12;                   // radians since anchor 0
        for (let k = 0; k < K; k++) {
            // contribution of sample k: (1/K)·[1 + Σ_h w_h·cos(h·(t − 2πk/K)) / damp_h]
            let w = 1;
            for (let h = 1; h <= H; h++) {
                const weight = (K % 2 === 0 && h === H) ? 1 : 2;
                w += weight * Math.cos(h * (t - TAU * k / K)) / damp(h);
            }
            M[m * K + k] = w / K;
        }
    }
    matrices.set(key, M);
    return M;
}

/** Rebuild 12 values from K anchor values a[] (out: 12 values). */
export function interpolateMonths(a, phase, out, attenuation = 0) {
    const K = a.length, M = interpolationMatrix(K, phase, attenuation);
    for (let m = 0; m < 12; m++) {
        let v = 0;
        for (let k = 0; k < K; k++) v += M[m * K + k] * a[k];
        out[m] = v;
    }
    return out;
}

const scratch = new Float64Array(12);

/**
 * Rebuild 12 monthly temperatures and precipitations for one cell from K anchors.
 * @param {ArrayLike<number>} tAnch   K temperatures (°C)
 * @param {ArrayLike<number>} pAnch   K precipitations (mm/month)
 * @param {object} opts  { phase = 0, transform = 'sqrt', eps = 5, attenuation = 0 }
 * @param {Float64Array} T12 out, {Float64Array} P12 out
 */
export function synthesizeCell(tAnch, pAnch, opts, T12, P12) {
    const { phase = 0, transform = 'sqrt', eps = 5, attenuation = 0 } = opts;
    const K = tAnch.length, M = interpolationMatrix(K, phase, attenuation);
    const tf = PRECIP_TRANSFORMS[transform];
    for (let k = 0; k < K; k++) scratch[k] = tf.fwd(pAnch[k], eps);
    for (let m = 0; m < 12; m++) {
        let t = 0, p = 0;
        for (let k = 0; k < K; k++) { const w = M[m * K + k]; t += w * tAnch[k]; p += w * scratch[k]; }
        T12[m] = t;
        P12[m] = tf.inv(p, eps);
    }
    return { T12, P12 };
}
