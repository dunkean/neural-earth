// Sunlight at the top of the atmosphere: daily-mean insolation by latitude and time of year, for any axial tilt
// (circular orbit). DOM-free.
//
// Calendar: t in months from the start of the month after the northern winter solstice (Earth: 1 January); month m
// is centred on t = m + 0.5. The northern spring equinox falls at t = 2.63 (Earth: 20 March).

export const SOLAR_CONSTANT = 1361;          // W/m²
export const EQUINOX_MONTH = 2.63;

/** Daily-mean insolation (W/m²) at latitude φ (radians) when the solar declination is δ (radians). */
export function dailyInsolation(phi, delta, S0 = SOLAR_CONSTANT) {
    const x = -Math.tan(phi) * Math.tan(delta);
    const h0 = x >= 1 ? 0 : x <= -1 ? Math.PI : Math.acos(x);           // hour angle of sunset (polar night / day)
    return (S0 / Math.PI) * (h0 * Math.sin(phi) * Math.sin(delta) + Math.cos(phi) * Math.cos(delta) * Math.sin(h0));
}

/** Solar declination (radians) at time t (months), for axial tilt in degrees. */
export function declination(t, tiltDeg) {
    const L = 2 * Math.PI * (t - EQUINOX_MONTH) / 12;                    // solar longitude
    return Math.asin(Math.sin(tiltDeg * Math.PI / 180) * Math.sin(L));
}

/**
 * Insolation per latitude row at NT evenly spaced times of year (t_k = 12·k / NT months).
 * @param {Float64Array} latDeg  row latitudes
 * @returns {Float64Array} NT × rows, time-major
 */
export function insolationTable(latDeg, tiltDeg, NT = 24, S0 = SOLAR_CONSTANT) {
    const rows = latDeg.length, out = new Float64Array(NT * rows);
    for (let k = 0; k < NT; k++) {
        const delta = declination(12 * k / NT, tiltDeg);
        for (let j = 0; j < rows; j++) out[k * rows + j] = dailyInsolation(latDeg[j] * Math.PI / 180, delta, S0);
    }
    return out;
}
