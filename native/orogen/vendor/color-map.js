import {climate} from './climate-config.js';
// Elevation → RGB colour mapping.

// Convert raw mesh elevation (nonlinear, 0-~1 for land) to physical height
// in kilometres. A bounded coastal term avoids a zero-slope land plateau;
// the original quartic mountain curve and its 6 km summit remain.
// Ocean (elev < 0) is mapped with a linear scale (~5 km at -0.5).
export function elevToHeightKm(elev) {
    if (elev <= 0) return elev * 10;  // ocean: -0.5 → -5 km
    const t = Math.min(elev, 1);
    const t2 = t * t;
    return 6 * t2 * t2 * (5 - 4 * t) + t * (1 - t) ** 4;
}

// Biome base colors indexed by Köppen class ID (satellite-view palette).
// 0=Ocean delegated, 1-30 = land biomes.
const BIOME_COLORS = [
    null,                        //  0 Ocean — handled separately
    climate.biome_color_af,         //  1 Af   Tropical rainforest — deep emerald
    climate.biome_color_am,         //  2 Am   Tropical monsoon — dense green
    climate.biome_color_aw,         //  3 Aw   Tropical savanna — yellow-green
    climate.biome_color_bwh,         //  4 BWh  Hot desert — sandy tan
    climate.biome_color_bwk,         //  5 BWk  Cold desert — gray-brown
    climate.biome_color_bsh,         //  6 BSh  Hot steppe — dry gold
    climate.biome_color_bsk,         //  7 BSk  Cold steppe — muted olive-tan
    climate.biome_color_cfa,         //  8 Cfa  Humid subtropical — mid green
    climate.biome_color_cfb,         //  9 Cfb  Oceanic — rich green
    climate.biome_color_cfc,         // 10 Cfc  Subpolar oceanic — dark muted green
    climate.biome_color_csa,         // 11 Csa  Hot-summer Mediterranean — khaki-green
    climate.biome_color_csb,         // 12 Csb  Warm-summer Mediterranean — chaparral
    climate.biome_color_csc,         // 13 Csc  Cold-summer Mediterranean — darker khaki
    climate.biome_color_cwa,         // 14 Cwa  Humid subtropical monsoon — mid green
    climate.biome_color_cwb,         // 15 Cwb  Subtropical highland — green
    climate.biome_color_cwc,         // 16 Cwc  Cold subtropical highland — dark green
    climate.biome_color_dfa,         // 17 Dfa  Hot-summer continental — forest green
    climate.biome_color_dfb,         // 18 Dfb  Warm-summer continental — forest green
    climate.biome_color_dfc,         // 19 Dfc  Subarctic — dark spruce green
    climate.biome_color_dfd,         // 20 Dfd  Extremely cold subarctic — very dark
    climate.biome_color_dsa,         // 21 Dsa  Hot-summer continental dry — olive-brown
    climate.biome_color_dsb,         // 22 Dsb  Warm-summer continental dry — olive-brown
    climate.biome_color_dsc,         // 23 Dsc  Subarctic dry summer — dark green
    climate.biome_color_dsd,         // 24 Dsd  Extremely cold subarctic dry — very dark
    climate.biome_color_dwa,         // 25 Dwa  Hot-summer continental monsoon — forest green
    climate.biome_color_dwb,         // 26 Dwb  Warm-summer continental monsoon
    climate.biome_color_dwc,         // 27 Dwc  Subarctic monsoon — dark spruce
    climate.biome_color_dwd,         // 28 Dwd  Extremely cold subarctic monsoon
    climate.biome_color_et,         // 29 ET   Tundra — earthy brown (sparse moss/lichen on rock)
    climate.biome_color_ef,         // 30 EF   Ice cap — blue-tinted white
];

// Rocky/alpine mountain color for high-elevation blending.
const ROCK_COLOR = climate.biome_rock_color;

// Altitude thresholds (km) by Köppen group:
//   [alpine line, snow line]
// Alpine line: vegetation gives way to rocky alpine terrain.
// Snow line: permanent snow begins.
function altitudeThresholds(classId) {
    if (classId <= 0)  return [0, 0];           // Ocean
    if (classId <= 3)  return [climate.biome_tropical_alpine, climate.biome_tropical_snow];       // Tropical (A)
    if (classId <= 7)  return [climate.biome_arid_alpine, climate.biome_arid_snow];       // Arid (B)
    if (classId <= 16) return [climate.biome_temperate_alpine, climate.biome_temperate_snow];       // Temperate (C)
    if (classId <= 18 || classId === 21 || classId === 22 ||
        classId === 25 || classId === 26) return [climate.biome_continental_alpine, climate.biome_continental_snow];  // Continental humid (D*a, D*b)
    if (classId <= 28) return [climate.biome_subarctic_alpine, climate.biome_subarctic_snow];       // Subarctic (D*c, D*d)
    if (classId === 29) return [climate.biome_tundra_alpine, climate.biome_tundra_snow];      // Tundra (ET) — rocky higher up, snow only at peaks
    return [climate.biome_ice_alpine, climate.biome_ice_snow];                             // Ice cap (EF)
}

// Satellite-view biome color: realistic land colors based on Köppen class
// and elevation, with ocean delegated to the standard ocean palette.
export function biomeColor(koppenId, elevation) {
    // Ocean
    if (koppenId === 0 || elevation <= 0) return elevationToColor(elevation);

    const base = BIOME_COLORS[koppenId] || [0.30, 0.50, 0.20];
    const hKm = elevToHeightKm(elevation);
    const [alpineLine, snowLine] = altitudeThresholds(koppenId);

    let r = base[0], g = base[1], b = base[2];

    // Low-elevation subtle darkening for depth (0-200m)
    if (hKm < climate.biome_low_height) {
        const dark = (1 - climate.biome_low_darkening) + climate.biome_low_darkening * (hKm / climate.biome_low_height);
        r *= dark; g *= dark; b *= dark;
    }

    // Mid-elevation: gentle darkening to show terrain relief (200m to alpine line)
    if (alpineLine > 0 && hKm > climate.biome_low_height && hKm < alpineLine) {
        const t = (hKm - climate.biome_low_height) / (alpineLine - climate.biome_low_height);
        const darken = 1.0 - t * climate.biome_mid_darkening; // up to 15% darker at alpine line
        r *= darken; g *= darken; b *= darken;
    }

    // Alpine zone: blend toward rocky brown-gray above the tree/vegetation line
    if (alpineLine > 0 && hKm > alpineLine) {
        const rockZone = snowLine > alpineLine ? snowLine - alpineLine : climate.biome_rock_transition;
        const rockT = Math.min(1, (hKm - alpineLine) / rockZone);
        const s = rockT * rockT; // ease-in for gradual transition
        r = r + (ROCK_COLOR[0] - r) * s;
        g = g + (ROCK_COLOR[1] - g) * s;
        b = b + (ROCK_COLOR[2] - b) * s;
    }

    // Snow zone: blend toward white above the snow line
    if (snowLine > 0 && hKm > snowLine) {
        const snowT = Math.min(1, (hKm - snowLine) / climate.biome_snow_transition);
        const s = snowT * snowT; // ease-in for gradual snow buildup
        r = r + (climate.biome_snow_color[0] - r) * s;
        g = g + (climate.biome_snow_color[1] - g) * s;
        b = b + (climate.biome_snow_color[2] - b) * s;
    }

    return [r, g, b];
}

export function elevationToColor(e) {
    if (e < -0.50) return [0.04, 0.06, 0.30];
    if (e < -0.10) { const t=(e+0.50)/0.40; return [0.04+t*0.07,0.06+t*0.14,0.30+t*0.18]; }
    if (e <  0.00) { const t=(e+0.10)/0.10; return [0.11+t*0.19,0.20+t*0.22,0.48+t*0.12]; }
    if (e <  0.03) { const t=e/0.03;         return [0.72+t*0.08,0.68-t*0.02,0.46-t*0.10]; }
    if (e <  0.25) { const t=(e-0.03)/0.22;  return [0.20-t*0.06,0.54-t*0.12,0.12+t*0.08]; }
    if (e <  0.50) { const t=(e-0.25)/0.25;  return [0.14+t*0.30,0.42-t*0.14,0.20-t*0.06]; }
    if (e <  0.75) { const t=(e-0.50)/0.25;  return [0.44+t*0.16,0.28+t*0.12,0.14+t*0.18]; }
    { const t=Math.min(1,(e-0.75)/0.20);      return [0.60+t*0.35,0.40+t*0.50,0.32+t*0.60]; }
}
