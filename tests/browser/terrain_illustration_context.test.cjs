const assert=require('node:assert/strict');
const test=require('node:test');
const {summarize}=require('../../web/terrain_illustration_context.js');
test('illustration reads visible climate without requesting new terrain',()=>{
 const fields=new Float32Array(20);fields.fill(25,0,4);fields.fill(2400,8,12);fields.fill(-.006,16,20);
 const result=summarize([{tile:{heightOptions:{climate:fields,climateWidth:2,climateHeight:2},elevationMin:0,elevationMax:100}}]);
 assert.equal(result.biome,'tropical rainforest');assert.equal(result.annualRainMm,2400);assert.match(result.biomeEvidence,/Approximate/);
 assert.deepEqual(summarize([]),{});
});
