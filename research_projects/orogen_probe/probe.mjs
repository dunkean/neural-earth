// Research-only CPU probe of the unmodified Orogen upstream pipeline.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import Delaunator from 'delaunator';
const root = path.resolve(import.meta.dirname, '../../../planet_heightmap_generation/js');
const mod = name => import(pathToFileURL(path.join(root, name)).href);
const { setDelaunator, buildSphere, computeNeighborDist } = await mod('sphere-mesh.js');
const { makeRng } = await mod('rng.js');
const { SimplexNoise } = await mod('simplex-noise.js');
const { generateCoarsePlates, projectCoarsePlates } = await mod('coarse-plates.js');
const { smoothAndReconnectPlates } = await mod('plates.js');
const { buildSuperPlates } = await mod('super-plates.js');
const { applyPlatePhysics } = await mod('plate-physics.js');
const { assignElevation } = await mod('elevation.js');
const { elevToHeightKm } = await mod('color-map.js');
const { warpTerrain, smoothElevation, applyDetailNoise, erodeComposite, sharpenRidges, applySoilCreep } = await mod('terrain-post.js');
const { SUPER_PLATE_PHYSICS_MULT, DETAIL_NOISE_DAMPEN_STRENGTH } = await mod('terrain-config.js');
setDelaunator(Delaunator);
const baseCases = [
  { name: 'default', seed: 42, P: 80, continents: 4, variety: .35, coverage: .3 },
  { name: 'continental', seed: 100, P: 8, continents: 2, variety: .35, coverage: .6 },
  { name: 'oceanic', seed: 314, P: 100, continents: 8, variety: .35, coverage: .2 },
];
const batch = process.env.OROGEN_PROBE_BATCH === '1';
const cases = batch ? baseCases.flatMap(c => Array.from({length:12},(_,i)=>({...c,seed:i*97+13}))) : baseCases;
const reports = [];
for (const c of cases) {
  const started = performance.now();
  const { mesh, r_xyz } = buildSphere(20000, .75, makeRng(c.seed));
  const neighborDist = computeNeighborDist(mesh, r_xyz);
  const coarse = generateCoarsePlates(c.seed, c.P, c.continents, c.variety, c.coverage);
  const { coarseMesh, coarse_xyz, coarse_r_plate, coarsePlateSeeds: seeds, coarsePlateVec: vectors, coarsePlateIsOcean: oceans } = coarse;
  const plates = projectCoarsePlates(mesh, r_xyz, coarseMesh, coarse_xyz, coarse_r_plate, c.seed, c.P);
  smoothAndReconnectPlates(mesh, plates, seeds, 3);
  const density = {};
  for (const id of seeds) {
    const rng = makeRng(id + 777);
    const ocean = 3 + rng() * .5;
    const land = 2.4 + rng() * .5;
    density[id] = oceans.has(id) ? ocean : land;
  }
  const { mantleField } = applyPlatePhysics(vectors, seeds, oceans, coarse_r_plate, coarseMesh, coarse_xyz, c.seed);
  const superplates = buildSuperPlates(coarseMesh, coarse_r_plate, seeds, vectors, oceans, density, plates);
  applyPlatePhysics(superplates.superPlateVec, new Set(Array.from({length: superplates.numSuperPlates}, (_,i)=>i)), superplates.superPlateIsOcean, superplates.r_superPlate, mesh, r_xyz, c.seed + 7777, SUPER_PLATE_PHYSICS_MULT);
  const sums = {}, counts = {};
  for(let r=0;r<coarseMesh.numRegions;r++) { const id=coarse_r_plate[r]; sums[id]=(sums[id]||0)+mantleField[r]; counts[id]=(counts[id]||0)+1; }
  const mantle = Float32Array.from(plates, id => counts[id] ? sums[id]/counts[id] : 0);
  const result = assignElevation(mesh,r_xyz,oceans,plates,vectors,seeds,new SimplexNoise(c.seed),.4,c.seed,5,density,superplates,mantle);
  const heights = result.r_elevation;
  warpTerrain(mesh,heights,r_xyz,c.seed,.75,result.debugLayers.hotspot);
  const oceanMask = Uint8Array.from(heights,h=>h<=0 ? 1:0);
  smoothElevation(mesh,heights,oceanMask,1,.25);
  const dampen = Float32Array.from(heights, (_,r)=>Math.max(result.debugLayers.cratonWeight[r],result.debugLayers.basinWeight[r]));
  const orogenic = Float32Array.from(heights, (_,r)=>Math.min(1,Math.max(0,result.debugLayers.orogenicPower[r]+.5)));
  const detail = { dampenField:dampen,dampenStrength:DETAIL_NOISE_DAMPEN_STRENGTH,amplitudeField:orogenic };
  applyDetailNoise(mesh,r_xyz,heights,oceanMask,c.seed,detail);
  applyDetailNoise(mesh,r_xyz,heights,oceanMask,c.seed,{...detail,amplitudeKm:.05,frequencyMult:2,warpAmpMult:2,bipolar:true,biasExponent:.4,seedOffset:13579});
  erodeComposite(mesh,heights,r_xyz,oceanMask,10,.0003,.5,1,1,1.16,.015,5,.5,neighborDist);
  sharpenRidges(mesh,heights,oceanMask,3,.04);
  applySoilCreep(mesh,heights,oceanMask,3,.1125);
  const heightMeters = Float32Array.from(heights,h=>elevToHeightKm(h)*1000);
  if (!batch) {
    fs.writeFileSync(path.join(import.meta.dirname,`${c.name}-xyz.f32`),Buffer.from(r_xyz.buffer,r_xyz.byteOffset,r_xyz.byteLength));
    fs.writeFileSync(path.join(import.meta.dirname,`${c.name}-height-m.f32`),Buffer.from(heightMeters.buffer));
  }
  const visited=new Uint8Array(mesh.numRegions), components=[];
  for(let r=0;r<mesh.numRegions;r++) {
    if(visited[r] || heightMeters[r]<=0)continue;
    const queue=[r]; visited[r]=1;
    for(let k=0;k<queue.length;k++) for(let j=mesh.adjOffset[queue[k]];j<mesh.adjOffset[queue[k]+1];j++) {const nb=mesh.adjList[j]; if(!visited[nb] && heightMeters[nb]>0) {visited[nb]=1;queue.push(nb);}}
    components.push(queue.length);
  }
  components.sort((a,b)=>b-a);
  const land=heightMeters.filter(h=>h>0).length;
  reports.push({...c,n:mesh.numRegions,elapsed_s:(performance.now()-started)/1000,min_m:Math.min(...heightMeters),max_m:Math.max(...heightMeters),finite:Array.from(heightMeters).every(Number.isFinite),land_fraction:land/heightMeters.length,land_components:components.length,large_land_components:components.filter(n=>n>=100).length,largest_land_component_fraction:components[0]/land,highland_fraction_of_land:heightMeters.filter(h=>h>2000).length/land,elevation_stages:batch ? undefined:result._timing});
  console.log(JSON.stringify(reports.at(-1)));
}
fs.writeFileSync(path.join(import.meta.dirname,batch?'batch-report.json':'report.json'),JSON.stringify({source_commit:'cc2662b4edd52231c4f65d8765f3ef12cd82d9b7',cpu_only:true,climate_computed:false,probe:'worker terrain pipeline; nearest-neighbor raster preview is research-only',cases:reports},null,2));
