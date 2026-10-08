import {climate} from './climate-config.js';
// Heuristic precipitation model: smooth zonal patterns blended with the
// complex advection model to reduce splotchiness and strengthen deserts.
// Computes precipitation from four multiplicative factors: zonal base curve
// (distance from ITCZ), seasonal modifier, continental dryness, and
// orographic rain shadow.

import { smoothstep } from './wind.js';
import { elevToHeightKm } from './color-map.js';
import { smoothField, makeItczLookup } from './climate-util.js';

const DEG = Math.PI / 180;

// ── Zonal base curve ────────────────────────────────────────────────────────
// Returns a value in [0.03, 1.0] based on distance from the ITCZ in degrees.

function zonalBase(distDeg) {
    if (distDeg < climate.heuristic_core_lat) {
        // ITCZ core: 1.0
        return climate.heuristic_core_rain;
    } else if (distDeg < climate.heuristic_outer_lat) {
        // Outer ITCZ / trades: 1.0 → 0.35 (faster falloff)
        return climate.heuristic_core_rain - climate.heuristic_outer_drop * smoothstep(climate.heuristic_core_lat, climate.heuristic_outer_lat, distDeg);
    } else if (distDeg < climate.heuristic_desert_end_lat) {
        // Subtropical highs (desert factory): 0.35 → 0.02
        // Very aggressive minimum — core of the desert belt.
        return climate.heuristic_desert_rain - climate.heuristic_desert_drop * smoothstep(climate.heuristic_outer_lat, climate.heuristic_desert_core_lat, distDeg);
    } else if (distDeg < climate.heuristic_wet_mid_lat) {
        // Mid-lat westerlies recovery: 0.02 → 0.5
        return climate.heuristic_mid_base + climate.heuristic_mid_gain * smoothstep(climate.heuristic_desert_end_lat, climate.heuristic_wet_mid_lat, distDeg);
    } else if (distDeg < climate.heuristic_polar_lat) {
        // Subpolar: 0.5 → 0.3
        return climate.heuristic_subpolar_base - climate.heuristic_subpolar_drop * smoothstep(climate.heuristic_wet_mid_lat, climate.heuristic_polar_lat, distDeg);
    } else {
        // Polar: 0.3 → 0.1
        return climate.heuristic_polar_base - climate.heuristic_polar_drop * smoothstep(climate.heuristic_polar_lat, 90, distDeg);
    }
}

// ── Heuristic zonal wind ────────────────────────────────────────────────────
// Idealized wind direction based on latitude relative to the ITCZ.
// Returns local east/north components (positive east = blowing eastward,
// positive north = blowing poleward in NH).
//
// Zonal wind belts (Earth-like):
//   ITCZ (0-5°):        light/convergent
//   Trades (5-30°):     strong easterlies, deflected equatorward by Coriolis
//   Subtropical (25-35°): weak/variable (transition)
//   Westerlies (35-60°): west→east, deflected poleward
//   Polar easterlies (60-90°): east→west, deflected equatorward

function heuristicWind(distFromItczDeg, isNorthOfItcz) {
    // Sign for hemisphere: +1 if north of ITCZ, -1 if south
    const hemiSign = isNorthOfItcz ? 1 : -1;
    let we, wn;

    if (distFromItczDeg < 5) {
        // ITCZ: light convergent winds — slight equatorward component
        we = 0;
        wn = -hemiSign * climate.heuristic_wind_itcz;
    } else if (distFromItczDeg < 30) {
        // Trade winds: easterlies (blowing westward) with equatorward component
        // Strength ramps up from ITCZ edge, peaks ~15-20°, fades toward subtropics
        const tradeStrength = smoothstep(5, 15, distFromItczDeg)
            * (1 - smoothstep(25, 32, distFromItczDeg));
        we = -tradeStrength * climate.heuristic_wind_trade_east;                 // strong westward
        wn = -hemiSign * tradeStrength * climate.heuristic_wind_trade_north;      // equatorward (toward ITCZ)
    } else if (distFromItczDeg < 60) {
        // Westerlies: blowing eastward with poleward component
        const westStrength = smoothstep(30, 40, distFromItczDeg)
            * (1 - smoothstep(55, 65, distFromItczDeg));
        we = westStrength * climate.heuristic_wind_westerly_east;                    // strong eastward
        wn = hemiSign * westStrength * climate.heuristic_wind_westerly_north;        // poleward
    } else {
        // Polar easterlies: blowing westward with equatorward component
        const polarStrength = smoothstep(60, 70, distFromItczDeg);
        we = -polarStrength * climate.heuristic_wind_polar_east;                  // moderate westward
        wn = -hemiSign * polarStrength * climate.heuristic_wind_polar_north;      // equatorward
    }

    return { we, wn };
}

// ── Heuristic wind field for a full season ──────────────────────────────────
// Computes idealized zonal wind E/N arrays for all regions.

export function computeHeuristicWindField(numRegions, r_lat, r_lon, itczLookup) {
    const hWindE = new Float32Array(numRegions);
    const hWindN = new Float32Array(numRegions);

    for (let r = 0; r < numRegions; r++) {
        const lat = r_lat[r];
        const itczLat = itczLookup(r_lon[r]) * climate.heuristic_itcz_shift; // dampened ITCZ, same as precip
        const signedDist = lat - itczLat;
        const distDeg = Math.abs(signedDist) / DEG;
        const northOfItcz = signedDist > 0;
        const { we, wn } = heuristicWind(distDeg, northOfItcz);
        hWindE[r] = we;
        hWindN[r] = wn;
    }

    return { hWindE, hWindN };
}

// ── Main entry point ─────────────────────────────────────────────────────────

/**
 * Compute heuristic precipitation for both seasons.
 * Returns raw (un-normalized) Float32Arrays.
 *
 * @param {SphereMesh} mesh
 * @param {Float32Array} r_xyz
 * @param {Float32Array} r_elevation
 * @param {object} windResult - output from computeWind()
 * @param {Float32Array} r_elevGradE - pre-computed east elevation gradient
 * @param {Float32Array} r_elevGradN - pre-computed north elevation gradient
 * @param {Int32Array} r_coastDistLand - BFS hop distance from coast through land
 * @returns {{ r_precip_summer, r_precip_winter }}
 */
export function computeHeuristicPrecipitation(mesh, r_xyz, r_elevation, windResult, r_elevGradE, r_elevGradN, r_coastDistLand) {
    const numRegions = mesh.numRegions;
    const { r_lat, r_lon, r_isLand, r_continentality } = windResult;

    const avgEdgeKm = (Math.PI * 6371) / Math.sqrt(numRegions);

    // Precompute west-coast proximity: positive = west coast, negative = east coast.
    // Coastal land cells check which side ocean is on relative to the local east
    // direction, then the signal is smoothed ~300 km inland through land only.
    const { r_eastX, r_eastY, r_eastZ } = windResult;
    const { adjOffset, adjList } = mesh;
    const r_westCoast = new Float32Array(numRegions);
    for (let r = 0; r < numRegions; r++) {
        if (!r_isLand[r] || r_coastDistLand[r] !== 0) continue;
        let oceanDotEast = 0;
        let count = 0;
        const end = adjOffset[r + 1];
        for (let ni = adjOffset[r]; ni < end; ni++) {
            const nb = adjList[ni];
            if (!r_isLand[nb]) {
                const dx = r_xyz[3 * nb] - r_xyz[3 * r];
                const dy = r_xyz[3 * nb + 1] - r_xyz[3 * r + 1];
                const dz = r_xyz[3 * nb + 2] - r_xyz[3 * r + 2];
                oceanDotEast += dx * r_eastX[r] + dy * r_eastY[r] + dz * r_eastZ[r];
                count++;
            }
        }
        if (count > 0) {
            // Negative dot = ocean is to the west = west coast
            r_westCoast[r] = oceanDotEast < 0 ? 1 : -1;
        }
    }
    // Smooth through land only (~300 km) so the signal bleeds inland
    const wcPasses = Math.max(2, Math.round(climate.heuristic_coastal_smoothing_km / avgEdgeKm));
    const wcTmp = new Float32Array(numRegions);
    for (let pass = 0; pass < wcPasses; pass++) {
        for (let r = 0; r < numRegions; r++) {
            if (!r_isLand[r]) { wcTmp[r] = 0; continue; }
            let sum = r_westCoast[r], count = 1;
            const end = adjOffset[r + 1];
            for (let ni = adjOffset[r]; ni < end; ni++) {
                const nb = adjList[ni];
                if (r_isLand[nb]) { sum += r_westCoast[nb]; count++; }
            }
            wcTmp[r] = sum / count;
        }
        r_westCoast.set(wcTmp);
    }

    const result = {};

    const seasons = [
        { name: 'summer', shift: 5 },
        { name: 'winter', shift: -5 }
    ];

    for (const { name } of seasons) {
        const isSummer = name === 'summer';

        const itczLookup = makeItczLookup(windResult.itczLons,
            isSummer ? windResult.itczLatsSummer : windResult.itczLatsWinter);

        const precip = new Float32Array(numRegions);

        for (let r = 0; r < numRegions; r++) {
            const lat = r_lat[r];
            const lon = r_lon[r];

            // ── A. Zonal base curve (distance from ITCZ) ──
            // Dampen ITCZ shift: use only 30% of the complex model's ITCZ
            // displacement so the zonal bands stay close to the geographic
            // equator. The full ITCZ swing (up to 15-20°) would drag the
            // subtropical desert belt too far, drying the true equator and
            // wetting the mid-latitudes in the shifted season.
            const itczLat = itczLookup(lon) * climate.heuristic_itcz_shift;
            const signedDist = lat - itczLat;
            const distFromItczDeg = Math.abs(signedDist) / DEG;
            const isNorthOfItcz = signedDist > 0;
            const zonal = zonalBase(distFromItczDeg);

            // ── B. Seasonal modifier + Mediterranean subtropical suppression ──
            const absLatDeg = Math.abs(lat) / DEG;
            const inSummerHemi = isSummer ? (lat >= 0) : (lat < 0);
            let seasonMod = inSummerHemi ? climate.heuristic_summer : climate.heuristic_winter;

            // Mediterranean suppression: subtropical highs expand poleward in
            // local summer, strongly suppressing rainfall at 25-42° latitude.
            // In local winter the highs retreat equatorward and westerlies
            // bring rain to these latitudes.  This seasonal contrast is the
            // primary driver of Mediterranean (Cs) climates.
            // Stronger on west coasts (subtropical highs sit over eastern ocean
            // basins, drying the adjacent western continental margins) and
            // weaker on east coasts (onshore tropical moisture counters drying).
            if (inSummerHemi && absLatDeg > climate.heuristic_med_start && absLatDeg < climate.heuristic_med_end) {
                const medSuppress = smoothstep(climate.heuristic_med_start, climate.heuristic_med_full, absLatDeg)
                    * (1 - smoothstep(climate.heuristic_med_fade, climate.heuristic_med_end, absLatDeg));
                const wc = r_westCoast[r]; // +1 west coast, -1 east coast, 0 inland
                const strength = climate.heuristic_med_drying + wc * climate.heuristic_med_coast; // 0.35 west coast, 0.15 inland, ~0 east coast
                seasonMod *= (1 - medSuppress * Math.max(0, strength));
            }

            // ── C. Continental dryness ──
            let contMod = 1.0;
            const cont = (r_isLand[r] && r_continentality) ? r_continentality[r] : 0;
            if (cont > 0) {
                contMod = 1.0 - cont * cont * climate.heuristic_continental;
            }

            // ── D. Orographic rain shadow (using heuristic zonal wind) ──
            let oroMod = 1.0;
            if (r_isLand[r] && r_elevation[r] > 0) {
                const { we, wn } = heuristicWind(distFromItczDeg, isNorthOfItcz);
                // Wind dot elevation gradient: positive = windward, negative = leeward
                const windDotGrad = we * r_elevGradE[r] + wn * r_elevGradN[r];

                if (windDotGrad > 0) {
                    // Windward: up to +60% boost
                    const uplift = Math.min(1, windDotGrad * climate.precip_uplift_scale);
                    oroMod = 1.0 + uplift * climate.heuristic_uplift;
                } else {
                    // Leeward: up to -70% suppression, scaled by mountain height
                    const heightKm = elevToHeightKm(Math.max(0, r_elevation[r]));
                    const heightScale = Math.min(1, heightKm / climate.heuristic_shadow_height); // 3km+ = full shadow
                    const shadow = Math.min(1, -windDotGrad * climate.precip_shadow_scale);
                    oroMod = Math.max(0.3, 1.0 - shadow * climate.heuristic_shadow * heightScale);
                }
            }

            // ── E. Hard distance-from-coast cutoff ──
            // Fixed 2000-3000km cutoff regardless of latitude.
            let distMod = 1.0;
            if (r_isLand[r] && r_coastDistLand[r] > 0) {
                const distKm = r_coastDistLand[r] * avgEdgeKm;
                if (distKm > climate.precip_cutoff_start_km) {
                    distMod = Math.max(0.03, 1 - smoothstep(climate.precip_cutoff_start_km, climate.precip_cutoff_end_km, distKm));
                }
            }

            // ── Final ──
            precip[r] = Math.max(0.05, zonal * seasonMod * contMod * oroMod * distMod);
        }

        // Light smoothing ~100km
        const smoothPasses = Math.max(1, Math.round(climate.precip_smoothing_km / avgEdgeKm));
        smoothField(mesh, precip, smoothPasses);

        result[`r_precip_${name}`] = precip;
    }

    return result;
}
