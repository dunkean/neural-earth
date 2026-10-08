import {climate} from './climate-config.js';
// Ocean current simulation: rule-based geographic approach with wind-belt-driven gyres.
// Wind belts drive zonal currents; continental shelves deflect them into gyres.
// Warmth is classified geographically: western coasts = warm, eastern coasts = cold.

console.log('[ocean.js] Module loaded');
import { smoothstep } from './wind.js';
import { makeItczLookup, percentile } from './climate-util.js';

const DEG = Math.PI / 180;

// ── Coast distance & classification via BFS ─────────────────────────────────

function computeCoastFields(mesh, r_xyz, r_isOcean,
    r_eastX, r_eastY, r_eastZ) {
    const { adjOffset, adjList, numRegions } = mesh;

    const westSeeds = [];
    const eastSeeds = [];
    const allCoastSeeds = [];

    for (let r = 0; r < numRegions; r++) {
        if (!r_isOcean[r]) continue;

        let landDirX = 0, landDirY = 0, landDirZ = 0;
        let hasLandNeighbor = false;

        const end = adjOffset[r + 1];
        for (let ni = adjOffset[r]; ni < end; ni++) {
            const nb = adjList[ni];
            if (!r_isOcean[nb]) {
                hasLandNeighbor = true;
                landDirX += r_xyz[3 * nb] - r_xyz[3 * r];
                landDirY += r_xyz[3 * nb + 1] - r_xyz[3 * r + 1];
                landDirZ += r_xyz[3 * nb + 2] - r_xyz[3 * r + 2];
            }
        }

        if (!hasLandNeighbor) continue;

        allCoastSeeds.push(r);

        // Project land direction into tangent frame east component
        const normalE = landDirX * r_eastX[r] + landDirY * r_eastY[r] + landDirZ * r_eastZ[r];

        // normalE < -0.2 → land is to the west → western coast seed
        // normalE > +0.2 → land is to the east → eastern coast seed
        if (normalE < -0.2) {
            westSeeds.push(r);
        } else if (normalE > 0.2) {
            eastSeeds.push(r);
        } else {
            if (normalE <= 0) westSeeds.push(r);
            else eastSeeds.push(r);
        }
    }

    // BFS: compute hop distance from seed set through ocean cells.
    // Reuses a single queue array (capacity allocated once) across all three passes.
    const bfsQueue = new Int32Array(numRegions);

    function bfsDistance(seeds) {
        const dist = new Int32Array(numRegions);
        dist.fill(-1);
        let qLen = 0;
        for (const s of seeds) {
            dist[s] = 0;
            bfsQueue[qLen++] = s;
        }
        let head = 0;
        while (head < qLen) {
            const r = bfsQueue[head++];
            const d = dist[r] + 1;
            const end = adjOffset[r + 1];
            for (let ni = adjOffset[r]; ni < end; ni++) {
                const nb = adjList[ni];
                if (r_isOcean[nb] && dist[nb] === -1) {
                    dist[nb] = d;
                    bfsQueue[qLen++] = nb;
                }
            }
        }
        return dist;
    }

    const r_coastDist = bfsDistance(allCoastSeeds);
    const r_westCoastDist = bfsDistance(westSeeds);
    const r_eastCoastDist = bfsDistance(eastSeeds);

    return { r_coastDist, r_westCoastDist, r_eastCoastDist };
}

// ── Circumpolar channel detection ───────────────────────────────────────────

function hasCircumpolarChannel(r_lat, r_lon, r_isOcean, numRegions, targetLat, bandWidth) {
    const NUM_BINS = 72;
    const binHasOcean = new Uint8Array(NUM_BINS);
    const latMin = targetLat - bandWidth;
    const latMax = targetLat + bandWidth;

    for (let r = 0; r < numRegions; r++) {
        if (!r_isOcean[r]) continue;
        const lat = r_lat[r];
        if (lat < latMin || lat > latMax) continue;

        let bin = Math.floor(((r_lon[r] + Math.PI) / (2 * Math.PI)) * NUM_BINS);
        bin = ((bin % NUM_BINS) + NUM_BINS) % NUM_BINS;
        binHasOcean[bin] = 1;
    }

    for (let i = 0; i < NUM_BINS; i++) {
        if (!binHasOcean[i]) return false;
    }
    return true;
}

// ── Geographic heat classification ──────────────────────────────────────────
// Warmth is determined by coast type and wind cell. The prevailing wind
// direction determines which side of a basin accumulates warm water:
//   Hadley cell (trades westward):   western=warm, eastern=cold
//   Ferrel cell (westerlies eastward): western=cold, eastern=warm  (flipped)
//   Polar cell (easterlies westward):  western=warm, eastern=cold  (flipped back)

function classifyWarmth(r_isOcean, r_lat, numRegions,
    r_westCoastDist, r_eastCoastDist, fadeRange, seasonalShiftDeg) {
    const r_warmth = new Float32Array(numRegions);

    for (let r = 0; r < numRegions; r++) {
        if (!r_isOcean[r]) continue;

        // Shifted latitude for cell boundaries (matches wind band shift)
        const bandLatDeg = Math.abs(r_lat[r] / DEG - seasonalShiftDeg);

        // Wind cell sign: trades/polar push water west (western=warm → +1),
        // westerlies push water east (western=cold → -1)
        let cellSign;
        if (bandLatDeg < climate.ocean_warm_trade_end) {
            cellSign = 1;
        } else if (bandLatDeg < climate.ocean_westerly_start) {
            cellSign = 1 - 2 * smoothstep(climate.ocean_warm_trade_end, climate.ocean_westerly_start, bandLatDeg);
        } else if (bandLatDeg < climate.ocean_warm_westerly_end) {
            cellSign = -1;
        } else if (bandLatDeg < climate.ocean_polar_lat) {
            cellSign = -1 + 2 * smoothstep(climate.ocean_warm_westerly_end, climate.ocean_polar_lat, bandLatDeg);
        } else {
            cellSign = 1;
        }

        const wDist = r_westCoastDist[r];
        const eDist = r_eastCoastDist[r];

        let warm = 0;

        if (wDist >= 0 && wDist < fadeRange) {
            const t = 1 - wDist / fadeRange;
            warm += cellSign * t * t;
        }

        if (eDist >= 0 && eDist < fadeRange) {
            const t = 1 - eDist / fadeRange;
            warm -= cellSign * t * t;
        }

        r_warmth[r] = Math.max(-1, Math.min(1, warm));
    }

    return r_warmth;
}

// ── Laplacian smoothing (ocean only) ────────────────────────────────────────

function smoothOcean(mesh, field, r_isOcean, passes) {
    const { adjOffset, adjList, numRegions } = mesh;
    const tmp = new Float32Array(numRegions);

    for (let pass = 0; pass < passes; pass++) {
        for (let r = 0; r < numRegions; r++) {
            if (!r_isOcean[r]) { tmp[r] = field[r]; continue; }

            let sum = field[r], count = 1;
            const end = adjOffset[r + 1];
            for (let ni = adjOffset[r]; ni < end; ni++) {
                const nb = adjList[ni];
                if (r_isOcean[nb]) {
                    sum += field[nb];
                    count++;
                }
            }
            tmp[r] = sum / count;
        }
        field.set(tmp);
    }
}

// ── Main entry point ────────────────────────────────────────────────────────

/**
 * Compute ocean surface currents using rule-based geographic approach.
 * Wind belts drive zonal currents, continental shelves deflect them into
 * gyres. Warmth is classified geographically by coast type.
 *
 * @param {SphereMesh} mesh
 * @param {Float32Array} r_xyz - per-region 3D positions
 * @param {Float32Array} r_elevation - per-region elevation
 * @param {object} windResult - output from computeWind() (includes lat, lon, sinLat, isLand, tangent frames, ITCZ arrays)
 * @returns {object} current vectors, warmth, and speed arrays for both seasons
 */
export function computeOceanCurrents(mesh, r_xyz, r_elevation, windResult) {
    console.log('[ocean.js] computeOceanCurrents called, numRegions:', mesh.numRegions);
    const numRegions = mesh.numRegions;
    const avgEdgeKm = (Math.PI * 6371) / Math.sqrt(numRegions);
    const timing = [];

    const { r_lat, r_sinLat, r_isLand,
        r_eastX, r_eastY, r_eastZ,
        r_northX, r_northY, r_northZ } = windResult;

    // Ocean mask
    const r_isOcean = new Uint8Array(numRegions);
    for (let r = 0; r < numRegions; r++) r_isOcean[r] = r_isLand[r] ? 0 : 1;

    // Step 0: Setup — r_lon and ITCZ lookups
    let t0 = performance.now();
    let r_lon = windResult.r_lon;
    if (!r_lon) {
        r_lon = new Float32Array(numRegions);
        for (let r = 0; r < numRegions; r++) {
            r_lon[r] = Math.atan2(r_xyz[3 * r], r_xyz[3 * r + 2]);
        }
    }

    const itczLookupSummer = makeItczLookup(windResult.itczLons, windResult.itczLatsSummer);
    const itczLookupWinter = makeItczLookup(windResult.itczLons, windResult.itczLatsWinter);
    timing.push({ stage: 'Ocean: setup (ITCZ lookup + lon)', ms: performance.now() - t0 });

    // Step 1: Coast distance & classification (shared between seasons)
    t0 = performance.now();
    const { r_coastDist, r_westCoastDist, r_eastCoastDist } =
        computeCoastFields(mesh, r_xyz, r_isOcean,
            r_eastX, r_eastY, r_eastZ);
    timing.push({ stage: 'Ocean: coast BFS (3 passes)', ms: performance.now() - t0 });

    // Step 2: Circumpolar channel detection
    t0 = performance.now();
    const circumpolarNH = hasCircumpolarChannel(r_lat, r_lon, r_isOcean, numRegions, 60 * DEG, 5 * DEG);
    const circumpolarSH = hasCircumpolarChannel(r_lat, r_lon, r_isOcean, numRegions, -60 * DEG, 5 * DEG);
    console.log(`[ocean.js] Circumpolar: NH=${circumpolarNH}, SH=${circumpolarSH}`);
    timing.push({ stage: 'Ocean: circumpolar detection', ms: performance.now() - t0 });

    // Coast influence threshold
    const coastThreshold = Math.max(5, Math.round(Math.sqrt(numRegions) * climate.ocean_coast_range));
    // Warmth fade range — extends beyond coast deflection zone
    const warmthRange = coastThreshold * climate.ocean_warmth_range;

    const result = {};
    const seasons = [
        { name: 'summer', itczLookup: itczLookupSummer },
        { name: 'winter', itczLookup: itczLookupWinter }
    ];

    for (const { name, itczLookup } of seasons) {
        // Seasonal shift: wind cells migrate ~5° toward summer hemisphere
        const seasonalShiftDeg = name === 'summer' ? climate.seasonal_shift : -climate.seasonal_shift;

        // Steps 3–4: Wind band classification + current vectors
        t0 = performance.now();
        const currentE = new Float32Array(numRegions);
        const currentN = new Float32Array(numRegions);

        for (let r = 0; r < numRegions; r++) {
            if (!r_isOcean[r]) continue;

            const lat = r_lat[r];
            const absLatDeg = Math.abs(lat) / DEG;
            const lon = r_lon[r];
            const hemisphereSign = lat >= 0 ? 1 : -1;

            // Shifted latitude for wind band boundaries (cells migrate with season)
            const bandLatDeg = Math.abs(lat / DEG - seasonalShiftDeg);

            // ITCZ latitude at this longitude
            const itczLat = itczLookup(lon);
            const distFromItcz = Math.abs(lat - itczLat) / DEG;

            // Step 3: Base zonal flow from wind band (using shifted boundaries)
            let baseE;
            if (distFromItcz < climate.ocean_itcz_width) {
                // ITCZ zone: eastward countercurrent at center, blends to westward at edges
                baseE = 1 - 2 * smoothstep(0, climate.ocean_itcz_width, distFromItcz);
            } else if (bandLatDeg < climate.ocean_trade_lat) {
                // Trade winds: westward
                baseE = -1;
            } else if (bandLatDeg < climate.ocean_westerly_start) {
                // Subtropical transition: blend trades → westerlies
                baseE = -1 + 2 * smoothstep(climate.ocean_trade_lat, climate.ocean_westerly_start, bandLatDeg);
            } else if (bandLatDeg < climate.ocean_westerly_end) {
                // Ferrel cell / westerlies: eastward
                baseE = 1;
            } else if (bandLatDeg < climate.ocean_polar_lat) {
                // Subpolar transition: blend westerlies → polar easterlies
                baseE = 1 - climate.ocean_polar_transition * smoothstep(climate.ocean_westerly_end, climate.ocean_polar_lat, bandLatDeg);
            } else {
                // Polar easterlies: weak westward
                baseE = -climate.ocean_polar_speed;
            }

            currentE[r] = baseE;
            currentN[r] = 0;

            // Step 4: Coast deflection
            const wDist = r_westCoastDist[r];
            const eDist = r_eastCoastDist[r];

            // Near western coast: strong poleward deflection (warm current)
            if (wDist >= 0 && wDist < coastThreshold) {
                const t = 1 - wDist / coastThreshold;
                const strength = t * t * climate.ocean_west_strength; // western intensification ×2
                currentN[r] += hemisphereSign * strength; // poleward
                currentE[r] *= (1 - t * t * climate.ocean_west_deflection);
            }

            // Near eastern coast: moderate equatorward deflection (cold current)
            if (eDist >= 0 && eDist < coastThreshold) {
                const t = 1 - eDist / coastThreshold;
                const strength = t * t * climate.ocean_east_strength; // eastern weaker ×0.8
                currentN[r] -= hemisphereSign * strength; // equatorward
                currentE[r] *= (1 - t * t * climate.ocean_east_deflection);
            }

            // Circumpolar override (55–75° with open channel)
            const isCircumpolar = (lat > 0 && circumpolarNH) || (lat < 0 && circumpolarSH);
            if (isCircumpolar && absLatDeg >= climate.ocean_circumpolar_min && absLatDeg <= climate.ocean_circumpolar_max) {
                const cStrength = 1 - Math.abs(absLatDeg - climate.ocean_circumpolar_center) / climate.ocean_circumpolar_width;
                currentE[r] = currentE[r] * (1 - cStrength) + climate.ocean_circumpolar_strength * cStrength;
                currentN[r] *= (1 - cStrength * climate.ocean_circumpolar_deflection);
            }
        }
        timing.push({ stage: `Ocean: wind bands + vectors (${name})`, ms: performance.now() - t0 });

        // Step 5: Smooth ~125 km (scale-invariant)
        t0 = performance.now();
        const oceanSmoothPasses = Math.max(2, Math.round(climate.ocean_smoothing_km / avgEdgeKm));
        smoothOcean(mesh, currentE, r_isOcean, oceanSmoothPasses);
        smoothOcean(mesh, currentN, r_isOcean, oceanSmoothPasses);

        // Zero out land
        for (let r = 0; r < numRegions; r++) {
            if (!r_isOcean[r]) { currentE[r] = 0; currentN[r] = 0; }
        }
        timing.push({ stage: `Ocean: smoothing (${name})`, ms: performance.now() - t0 });

        // Step 6: Geographic warmth classification (coast type, not flow direction)
        // Smoothed heavily to blend out jagged coastline noise and dilute
        // small island contributions (few coast cells → weak signal after smoothing).
        t0 = performance.now();
        const r_warmth = classifyWarmth(r_isOcean, r_lat, numRegions,
            r_westCoastDist, r_eastCoastDist, warmthRange, seasonalShiftDeg);
        const warmthSmoothPasses = Math.max(3, Math.round(climate.ocean_warmth_smoothing_km / avgEdgeKm));
        smoothOcean(mesh, r_warmth, r_isOcean, warmthSmoothPasses);

        // Step 7: Normalize speed (95th percentile)
        // Use speed-squared to avoid sqrt in the hot loop; sqrt is monotonic
        // so percentile on squared values gives the same ranking.
        const r_speed = new Float32Array(numRegions);
        const oceanSpeedsSq = new Float32Array(numRegions);
        let oceanCount = 0;
        for (let r = 0; r < numRegions; r++) {
            const spdSq = currentE[r] * currentE[r] + currentN[r] * currentN[r];
            r_speed[r] = spdSq;
            if (r_isOcean[r] && spdSq > 0) oceanSpeedsSq[oceanCount++] = spdSq;
        }
        const p95Sq = percentile(oceanSpeedsSq.subarray(0, oceanCount), 0.95);
        // Now convert to linear 0-1: speed/p95 = sqrt(spdSq)/sqrt(p95Sq) = sqrt(spdSq/p95Sq)
        const invP95Sq = 1 / p95Sq;
        for (let r = 0; r < numRegions; r++) {
            r_speed[r] = Math.min(1, Math.sqrt(r_speed[r] * invP95Sq));
        }

        console.log(`[Ocean ${name}] coastThreshold=${coastThreshold}, warmthRange=${warmthRange}, p95Sq=${p95Sq.toExponential(3)}, oceanCells=${oceanCount}`);
        timing.push({ stage: `Ocean: warmth + normalize (${name})`, ms: performance.now() - t0 });

        result[`r_ocean_current_east_${name}`] = currentE;
        result[`r_ocean_current_north_${name}`] = currentN;
        result[`r_ocean_speed_${name}`] = r_speed;
        result[`r_ocean_warmth_${name}`] = r_warmth;
    }

    result._oceanTiming = timing;
    return result;
}
