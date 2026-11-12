// Orogen v3 CPU algorithms; Python owns CUDA projection and persistent stages.
import fs from 'node:fs';
import { serialize, deserialize } from 'node:v8';
import Delaunator from './vendor/delaunator.cjs';
import { setDelaunator, buildSphere } from './upstream/js/sphere-mesh.js';
import { makeRng } from './upstream/js/rng.js';
import { meshFromArrays, buildSimMesh } from './upstream/v2/js/core/sim-mesh.js';
import { presentTerrainMesh, presentGroundOf, presentTerrainOverrides } from './upstream/v2/js/present/world.js';
import { finishConfigOf } from './upstream/v2/js/present/knobs.js';
import { terrainConfig } from './upstream/v2/js/terrain/terrain-config.js';
import { plainCarveInputs } from './upstream/v2/js/terrain/synthesize.js';
import { finishPlanet, surfaceConfig } from './upstream/v2/js/finish/finish.js';
import { setParams } from './upstream/v2/js/climate/config.js';
import { monthlyClimate, sample } from './climate.mjs';

const request = JSON.parse(fs.readFileSync(process.argv[2], 'utf8')), directory = process.argv[3];
globalThis.OROGEN_CLIMATE = request.climateSettings ?? {};
const { biomeColor } = await import('./vendor/color-map.js');
setDelaunator(Delaunator);
setParams(Object.fromEntries(Object.entries(request.climateSettings ?? {}).filter(([k]) => k.startsWith('v3_')).map(([k,v]) => [k.slice(3).toUpperCase(),v])));
const knobs = { ...request }, F = finishConfigOf(knobs), cfg = terrainConfig(presentTerrainOverrides({ knobs }));
const climateOptions = { greenhouseWm2: request.temperatureOffset ?? 0,
    precipScale: Math.max(0, 1 + .5 * (request.precipitationOffset ?? 0)), tiltDeg: request.climateSettings?.tilt_deg ?? 23.44 };
const timing = [];
let last = performance.now();
function lap(stage) { const now = performance.now(); timing.push({ stage, ms: now-last }); last = now; }
const arrayMesh = mesh => ({ numCells: mesh.numCells, xyz: mesh.xyz, adjOffset: mesh.adjOffset, adjList: mesh.adjList });
let state;
if (request.resumeFile) {
    state = deserialize(fs.readFileSync(request.resumeFile));
    if (state.version !== 3) throw new Error('Orogen v3 requires a freshly generated world');
} else if (request.inputFile) {
    // Physical heights: direct metres -> kilometres, no nonlinear mapping or cap.
    const { mesh, r_xyz } = buildSphere(request.N, request.jitter, makeRng(request.seed));
    const tm = presentTerrainMesh(mesh, r_xyz), bytes = fs.readFileSync(request.inputFile);
    const input = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength/4), grid = { w: request.imageWidth, h: request.imageHeight };
    if (input.length !== grid.w*grid.h || !input.every(Number.isFinite)) throw new Error('Invalid physical heights');
    const heightKm = Float32Array.from({ length: tm.numCells }, (_,c) => sample(grid,input,tm.xyz,c)/1000);
    state = { version: 3, seed: request.seed, imported: true, mesh: arrayMesh(tm), triangles: mesh.triangles,
        simMesh: arrayMesh(buildSimMesh(40000,12345)), heightKm, originalHeightKm: heightKm.slice(),
        fields: { plates: new Float32Array(tm.numCells), crust: Float32Array.from(heightKm,h => h < 0 ? 1 : 0),
            tectonic: new Float32Array(tm.numCells), hotspot: new Float32Array(tm.numCells), backArc: new Float32Array(tm.numCells) } };
    lap('Import physical heights');
} else {
    let result;
    globalThis.self = { postMessage(message) {
        if (message.type === 'error') throw new Error(message.stack || message.message);
        if (message.type === 'done') result = message;
    } };
    const worker = await import('./upstream/js/planet-worker.js');
    self.onmessage({ data: { ...request, cmd: 'generate' } });
    const s = result.shared, f = s.finish, retained = worker.retainedTerrain();
    state = { version: 3, seed: request.seed, imported: false, mesh: s.mesh, triangles: retained.triangles,
        simMesh: f.simMesh, frame: f.frame, groundFrame: retained.frame, mask: retained.mask, frontKm: retained.frontKm,
        heightKm: s.heightKm, originalHeightKm: s.heightKm.slice(), fields: tectonics(result,retained) };
    lap('Orogen v3 present-day world');
}
if (request.resumeCommand === 'erosion' || !request.resumeFile && !request.skipPost) {
    if (!request.skipPost) {
        const mesh = meshFromArrays(state.mesh), sim = meshFromArrays(state.simMesh);
        let surface;
        if (state.imported) {
            surface = { heightKm: state.originalHeightKm.slice(), carveIn: plainCarveInputs(mesh.xyz,mesh.numCells,surfaceConfig(F,cfg),state.seed), recCell: null, cues: null };
        } else {
            const r_xyz = new Float32Array(mesh.xyz.length);
            for (let c = 0; c < mesh.numCells; c++) { r_xyz[3*c] = mesh.xyz[3*c+1]; r_xyz[3*c+1] = mesh.xyz[3*c+2]; r_xyz[3*c+2] = mesh.xyz[3*c]; }
            const gen = { numRegions: mesh.numCells, adjOffset: mesh.adjOffset, adjList: mesh.adjList };
            surface = presentGroundOf(state.groundFrame,sim,gen,r_xyz,
                { seed: state.seed, knobs, mask: state.mask, frontKm: state.frontKm, finish: F }).surface;
        }
        lap('Retained ground shaping');
        const out = await finishPlanet({ frame: state.frame ?? null, simMesh: sim, seed: state.seed, cells: mesh.numCells,
            mesh, surface: () => surface, cfg, F, climateOptions });
        state.heightKm = out.z;
        state.fields.ocean_mask = Float32Array.from(out.ocean);
        state.fields.lake_mask = Float32Array.from(out.lake);
        state.fields.lake_level_m = Float32Array.from(out.lakeLevel,x => Number.isFinite(x) ? x*1000 : 0);
        if (out.geology?.rock) state.fields.geology_rock = Float32Array.from(out.geology.rock);
        for (const [key,value] of Object.entries(out.farming ?? {})) if (ArrayBuffer.isView(value) && value.length === mesh.numCells) state.fields['farming_'+key] = Float32Array.from(value);
        state.finishTiming = Object.entries(out.ms).map(([stage,ms]) => ({stage,ms}));
        state.rivers = out.rivers;
        lap('Orogen v3 finish');
    }
    state.fields.erosionDelta = Float32Array.from(state.heightKm,(h,c) => h-state.originalHeightKm[c]);
}
if (request.externalErosion) {
    save(); fs.writeFileSync(`${directory}/relief.ready`,'ready');
    const response = JSON.parse(fs.readFileSync(0,'utf8')), bytes = fs.readFileSync(response.elevationFile);
    const heights = new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
    if (heights.length !== state.heightKm.length || !heights.every(Number.isFinite)) throw new Error('Invalid external heights');
    state.heightKm = heights.slice();
    state.fields.erosionDelta = Float32Array.from(heights,(h,c) => h-state.originalHeightKm[c]);
    state.fields.ocean_mask = Float32Array.from(heights,h => h < 0 ? 1 : 0);
    lap('External City erosion');
}
if (request.resumeCommand === 'climate' || !request.resumeFile && !request.skipClimate) {
    Object.assign(state.fields,monthlyClimate(state,climateOptions,biomeColor));
    lap('Orogen v3 monthly climate');
}
save();
if (request.snapshotOutput) fs.writeFileSync(request.snapshotOutput,serialize(state));

function save() {
    const n = state.mesh.numCells, xyz = new Float32Array(3*n);
    // Upstream z north, atan2(y,x) -> application y north, atan2(z,x).
    for (let c = 0; c < n; c++) { xyz[3*c] = state.mesh.xyz[3*c]; xyz[3*c+1] = state.mesh.xyz[3*c+2]; xyz[3*c+2] = state.mesh.xyz[3*c+1]; }
    const fields = { xyz, triangles: state.triangles, elevation: state.heightKm, ...state.fields };
    fields.ocean_mask ??= Float32Array.from(state.heightKm,h => h < 0 ? 1 : 0);
    for (const [name,values] of Object.entries(fields)) if (!values.every(Number.isFinite)) throw new Error('Non-finite Orogen v3 field: '+name);
    const views = Object.entries(fields), all = new Float32Array(views.reduce((sum,[,v]) => sum+v.length,0));
    let at = 0;
    for (const [,values] of views) { all.set(values,at); at += values.length; }
    fs.writeFileSync(`${directory}/fields.f32`,Buffer.from(all.buffer));
    fs.writeFileSync(`${directory}/graph.json`,JSON.stringify({ numRegions: n, fields: Object.fromEntries(views.map(([k,v]) => [k,v.length])),
        timing, postTiming: state.finishTiming ?? [], upstream: 'ed466f9db71292e912d8d0160e689a67a994f2d9', heightUnits: 'km' }));
}

function tectonics(result,retained) {
    const frame = retained.frame;
    const mesh = result.shared.mesh, n = mesh.numCells, cellOf = result.shared.regionCell, pr = result.present;
    const fields = { plates: Float32Array.from(pr.regionPlate), crust: Float32Array.from(cellOf,c => frame.type[c] === 0 ? 1 : 0),
        ...result.debugLayers, tectonic: Float32Array.from(cellOf,c => (frame.upliftKm?.[c] ?? 0)/6),
        geology_belt_kind: Float32Array.from(cellOf,c => frame.oroType[c]), geology_belt_age_myr: result.debugLayers.presentBeltAge,
        geology_craton: result.debugLayers.presentCraton,
        hotspot: Float32Array.from(cellOf,c => frame.volcWornKm[c]/6),
        geology_backarc: Float32Array.from(cellOf,c => retained.backArc[c]),
        // Legacy resource-context scale: -0.06 denotes a full back-arc footprint,
        // not an elevation contribution from the retired v1 terrain formula.
        backArc: Float32Array.from(cellOf,c => -.06*retained.backArc[c]),
        boundaries: new Float32Array(n), convergence: new Float32Array(n) };
    const B = pr.boundaries, pairs = new Map(), pair = (p,q) => p < q ? `${p},${q}` : `${q},${p}`;
    for (let b = 0; b < B.count; b++) pairs.set(pair(B.a[b],B.b[b]),b);
    for (let c = 0; c < n; c++) for (let a = mesh.adjOffset[c]; a < mesh.adjOffset[c+1]; a++) {
        const p = fields.plates[c], q = fields.plates[mesh.adjList[a]];
        if (p === q) continue;
        const b = pairs.get(pair(p,q)); if (b === undefined) continue;
        const kind = B.kind[b];
        fields.boundaries[c] = kind === 3 || kind === 4 ? 1 : kind === 1 || kind === 2 ? 2 : kind === 5 ? 3 : 0;
        fields.convergence[c] = -B.openCmYr[b];
    }
    return fields;
}
