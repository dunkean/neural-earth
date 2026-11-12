// Physical monthly climate for the neural inputs and existing diagnostic layers.
import { planetClimate } from './upstream/v2/js/climate/climate.js';
import { cellClimate } from './upstream/v2/js/climate/core.js';
import { pageClimate } from './upstream/v2/js/evolution/climate-fields.js';
import { KOPPEN_CLASSES } from './upstream/v2/js/climate/koppen.js';

export function sample(grid, values, xyz, c) {
    const lon = Math.atan2(xyz[3*c+1], xyz[3*c]) / (2 * Math.PI) + .5;
    const lat = .5 - Math.asin(Math.max(-1, Math.min(1, xyz[3*c+2]))) / Math.PI;
    const x = lon * grid.w - .5, y = Math.max(0, Math.min(grid.h - 1, lat * grid.h - .5));
    const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
    const a = (ix % grid.w + grid.w) % grid.w, b = (a + 1) % grid.w, j = Math.min(iy + 1, grid.h - 1);
    return (values[iy*grid.w+a] * (1-fx) + values[iy*grid.w+b] * fx) * (1-fy) + (values[j*grid.w+a] * (1-fx) + values[j*grid.w+b] * fx) * fy;
}

// WorldClim/dismo conventions: sample SD; BIO4 is °C × 100; +1 mm in BIO15.
export function bioclim(T, P) {
    let meanT = 0, meanP = 0;
    for (let m = 0; m < 12; m++) { meanT += T[m] / 12; meanP += P[m] / 12; }
    let vt = 0, vp = 0;
    for (let m = 0; m < 12; m++) { vt += (T[m] - meanT) ** 2; vp += (P[m] - meanP) ** 2; }
    return [meanT, 100 * Math.sqrt(vt / 11), meanP * 12, 100 * Math.sqrt(vp / 11) / (meanP + 1)];
}

export function monthlyClimate(state, options, biomeColor) {
    const n = state.mesh.numCells, xyz = state.mesh.xyz, z = state.heightKm;
    const zeros = () => new Float32Array(n);
    const names = ['bio1','bio4','bio12','bio15','temperature_summer','temperature_winter','precip_summer','precip_winter'];
    const fields = Object.fromEntries(names.map(k => [k, zeros()]));
    const monthlyT = Array.from({ length: 12 }, zeros), monthlyP = Array.from({ length: 12 }, zeros);
    const land = Uint8Array.from(z, (h, c) => state.fields.ocean_mask ? 1 - state.fields.ocean_mask[c] : h > 0 ? 1 : 0);
    const res = planetClimate({ xyz, elevKm: z, axes: 'z-up', land }, options);
    const page = pageClimate(res, { cells: false });
    // Finish on all cells: the neural input must be finite over water as well.
    const out = cellClimate(res, xyz, z, new Uint8Array(n).fill(1), {}, (c, T, P) => {
        let ps = 0, pw = 0;
        for (let m = 0; m < 12; m++) {
            monthlyT[m][c] = T[m]; monthlyP[m][c] = Math.max(0, P[m]);
            if (m >= 3 && m <= 8) ps += monthlyP[m][c]; else pw += monthlyP[m][c];
        }
        const bio = bioclim(T, P);
        for (let i = 0; i < 4; i++) fields[names[i]][c] = bio[i];
        fields.temperature_summer[c] = (T[6] + 45) / 90; fields.temperature_winter[c] = (T[0] + 45) / 90;
        fields.precip_summer[c] = ps / 1000; fields.precip_winter[c] = pw / 1000;
    });
    for (let m = 0; m < 12; m++) { fields[`temperature_month_${m+1}`] = monthlyT[m]; fields[`precip_month_${m+1}`] = monthlyP[m]; }
    fields.koppen = Float32Array.from(out.koppen, (k, c) => land[c] ? k : 0);
    for (const [season, month] of [['summer','jul'], ['winter','jan']]) {
        const suffix = season === 'summer' ? 'Summer' : 'Winter';
        const gridFields = { ['pressure' + suffix]: page.slp[month], ['rainShadow' + suffix]: page.orographicSeasons[month],
            ['wind_east_' + season]: page.u[month], ['wind_north_' + season]: page.v[month],
            ['ocean_current_east_' + season]: page.currentU[month], ['ocean_current_north_' + season]: page.currentV[month] };
        for (const [name, values] of Object.entries(gridFields)) fields[name] = Float32Array.from({ length: n }, (_, c) => sample(page.grid, values, xyz, c));
    }
    for (let channel = 0; channel < 3; channel++) {
        fields['koppen_color_' + channel] = Float32Array.from(fields.koppen, k => KOPPEN_CLASSES[k].color[channel]);
        fields['biome_' + channel] = Float32Array.from(fields.koppen, (k, c) => biomeColor(k, z[c], true)[channel]);
    }
    return fields;
}
