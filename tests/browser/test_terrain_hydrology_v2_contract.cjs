// Pure display contract, no WebGPU adapter, terrain server or model.
const assert=require('node:assert/strict');
const hydro=require('../../web/terrain_hydrology.js');
const defaults=hydro.defaults(),lake=new Float32Array([5,1,2,1]),river=new Float32Array([5,1,3,1]);
assert(!hydro.hasRiver(lake));assert(hydro.hasInlandWater(lake));
assert(hydro.hasRiver(river));assert(hydro.hasInlandWater(river));
for(const field of [[5,0,2,1],[5,1,2,0],[0,1,1,1]])assert(!hydro.hasInlandWater(new Float32Array(field)));
assert.equal(hydro.tileIdentity(defaults),'','Legacy fast identity stays unchanged');
const params=hydro.v2TileParams(defaults,7);
assert.deepEqual(params,{hydro:'1',hydro_month:'7',v2_lakes:'1',v2_lake_min_area_m2:'1000000'});
assert.notEqual(hydro.v2DisplayIdentity(defaults,7),hydro.v2DisplayIdentity(defaults,1));
assert.notEqual(hydro.v2DisplayIdentity(defaults,7),hydro.v2DisplayIdentity({...defaults,lakes:false},7));
assert.notEqual(hydro.v2DisplayIdentity(defaults,7),hydro.v2DisplayIdentity({...defaults,minLakeArea:2},7));
assert.equal(hydro.v2DisplayIdentity({...defaults,enabled:false},7),'');
assert.equal(hydro.v2TileParams({...defaults,enabled:false},7),null);
console.log('PASS V2 lake-only authority, settings/season identity and legacy compatibility');
