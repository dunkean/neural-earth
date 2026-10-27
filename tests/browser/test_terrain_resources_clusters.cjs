const {test}=require('node:test'),assert=require('node:assert/strict');
require('../../web/terrain_resources_gpu.js');
const RG=globalThis.TerrainResourcesGPU;
function fixture(points){
  const f=new Float32Array(points.length*12),i=new Int32Array(f.buffer);
  points.forEach(([x,y,mass,a=1],k)=>{i.set([x,y,0,k%2],k*12);f.set([a,1,0,1,0,3,0,mass],k*12+4);});
  return {header:{kind:5,lattice:[-2,-1,4,2],world_bounds:[-61440,-30720,61440,30720]},data:{deposits:new Uint32Array(f.buffer)}};
}
test('cluster masses and members are conserved, separate at close zoom and keep resolved bodies',()=>{
  const {header,data}=fixture([[0,0,10],[2,0,20],[4,0,30],[1000,0,50],[0,50,40,5000]]);
  const far=RG.aggregate(header,data,30),near=RG.aggregate(header,data,1);
  assert.equal(far.symbols,3);assert.equal(far.clusters,1);assert.equal(near.symbols,5);
  assert.equal(far.groups.reduce((n,g)=>n+g.mass,0),150);
  assert.deepEqual(far.groups.flatMap(g=>g.members).sort((a,b)=>a-b),[0,1,2,3,4]);
  assert.equal(far.buckets.filter((_,i)=>i%2).reduce((a,b)=>a+b),3);
  const words=new Float32Array(far.deposits.buffer);
  assert.equal([...words].filter((_,i)=>i%12===8).find(v=>v<0),-3);
  assert.deepEqual(RG.aggregate(header,data,30).deposits,far.deposits);
  const reverse=new Uint32Array(data.deposits.length);
  for(let k=0;k<5;k++)reverse.set(data.deposits.subarray((4-k)*12,(5-k)*12),k*12);
  assert.deepEqual(RG.aggregate(header,{deposits:reverse},30).deposits,far.deposits,'package order is irrelevant');
});
test('clustering crosses physical longitude and LOD-10 bucket boundaries',()=>{
  const {header,data}=fixture([[-2048,0,100],[2047,0,10],[1023,100,20],[1024,100,30]]);
  const a=RG.aggregate(header,data,30);
  assert.equal(a.symbols,2);assert.equal(a.clusters,2);
  assert.deepEqual(a.groups.map(g=>g.members.length),[2,2]);
});
