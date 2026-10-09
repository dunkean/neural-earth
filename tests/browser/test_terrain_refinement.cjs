const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
// Partial coverage must not schedule an expensive redundant ancestor.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const script=fs.readFileSync('index.html','utf8').split('<script>')[1].split('</script>')[0];
const prefix=script.slice(0,script.indexOf("view.addEventListener('wheel'"));
function createContext(search){return vm.createContext({URLSearchParams,AbortController,performance,Map,Set,Math,Date,
 location:{search},crypto:require('node:crypto').webcrypto,
 document:{getElementById:()=>({checked:false,dataset:{},options:[],getContext:()=>({})})},
 window:{TerrainLOD:require(NEURAL_EARTH_ROOT + '/terrain_lod.js')},TerrainLighting:{get:()=>({})},devicePixelRatio:1,
 setTimeout:()=>1,clearTimeout:()=>{},requestAnimationFrame:()=>1});}
const context=createContext('?coarse_prepare=0&latent_geometry=2');
vm.runInContext(prefix,context);
vm.runInContext(`world={seed:'42',generation_profile:'natural',cache_profile:'test',world_bounds:[-20e6,-10e6,20e6,10e6]};
 W=8192;H=256;mpp=30;cx=122880;cy=3840;publishView=()=>{};evict=()=>{};updateStatus=()=>{};
 for(let tx=0;tx<8;tx++){const t=taskFor(1,tx,0,0);tiles.set(t.key,{...t,presented:true,expiresAt:Infinity})}
 refresh();`,context);
assert.equal(vm.runInContext('refinementLevel',context),4);
const ancestors=JSON.parse(vm.runInContext('JSON.stringify([...interests.values()].map(t=>[t.lod,t.tx,t.ty]))',context));
assert.deepEqual(ancestors,[[4,1,0]],'LOD1 coverage of the left half must skip its LOD4 request');
for(let level of [4,3]){
 const count=2**(4-level),start=count;
 vm.runInContext(`for(let tx=${start};tx<${2*count};tx++){const t=taskFor(${level},tx,0,0);tiles.set(t.key,{...t,presented:true,expiresAt:Infinity})}refresh()`,context);
 const next=level===4?3:0;
 assert.equal(vm.runInContext('refinementLevel',context),next);
 const missing=JSON.parse(vm.runInContext('JSON.stringify(queue.map(t=>[t.lod,t.tx,t.ty]))',context));
 const geometry=next===3?2:next,missingCount=2**(4-geometry);
 // Existing LOD1 parents satisfy intermediate quality on the left, but the
 // final native stage must generate both halves. Compare coverage, not order.
 assert.deepEqual(missing.sort((a,b)=>a[1]-b[1]),Array.from({length:next===0?32:missingCount},(_,i)=>[geometry,(next===0?0:missingCount)+i,0]));
}
// A new strip lowers the global stage, but useful native/intermediate flights
// admitted earlier must keep their interests and non-aborted controllers.
vm.runInContext(`tiles.clear();inflight.clear();queue=[];
 held=[];for(const lod of [0,1,2,3]){const t=taskFor(lod,0,0,2000),controller=new AbortController();inflight.set(t.key,{task:t,epoch,controller});held.push([t.key,controller])}
 const away=taskFor(0,100,0,2000);awayController=new AbortController();inflight.set(away.key,{task:away,epoch,controller:awayController});refresh()`,context);
assert.equal(vm.runInContext('refinementLevel',context),4);
assert(vm.runInContext('held.every(([key,c])=>interests.has(key)&&!c.signal.aborted)',context));
assert(vm.runInContext('queue.length>0&&queue.every(t=>t.lod===refinementLevel)',context),'new finer jobs must wait for the stage barrier');
assert(vm.runInContext('awayController.signal.aborted',context),'panned-away flights must be aborted');
console.log('Partial finer coverage skips redundant ancestors: OK');

// A presented LOD3 parent is already sufficient for source3; only its coarse
// neighbour needs bounded latent patches. Equal quality is coverage.
vm.runInContext(`tiles.clear();inflight.clear();W=8192;H=256;mpp=30;cx=122880;cy=3840;
 for(let tx=0;tx<2;tx++){const t=taskFor(4,tx,0,0);tiles.set(t.key,{...t,presented:true,expiresAt:Infinity})}
 for(let tx=0;tx<2;tx++){const t=taskFor(3,tx,0,0);tiles.set(t.key,{...t,presented:true,expiresAt:Infinity})}refresh();`,context);
assert.equal(vm.runInContext('refinementLevel',context),3);
assert(vm.runInContext('queue.length>0&&queue.every(t=>t.lod===2&&t.source_lod===3&&t.tx>=4)',context),'covered LOD3 parent must not spawn duplicate subtiles');

// Geometry1 A/B survives URL reconstruction and really schedules width112
// source3 addresses; it must not silently fall back to geometry2.
const geometry1=vm.createContext({URLSearchParams,AbortController,performance,Map,Set,Math,Date,
 location:{search:'?coarse_prepare=0&latent_geometry=1'},crypto:require('node:crypto').webcrypto,
 document:{getElementById:()=>({checked:false,dataset:{},options:[],getContext:()=>({})})},
 history:{replaceState(a,b,url){geometry1.savedURL=url}},
 window:{TerrainLOD:require(NEURAL_EARTH_ROOT + '/terrain_lod.js')},TerrainLighting:{get:()=>({}),serialize:()=>''},devicePixelRatio:1,
 setTimeout:()=>1,clearTimeout:()=>{},requestAnimationFrame:()=>1});
vm.runInContext(prefix,geometry1);
vm.runInContext(`world={seed:'42',world_profile:'natural',generation_profile:'natural',cache_profile:'test',world_bounds:[-20e6,-10e6,20e6,10e6]};
 W=256;H=256;mpp=30;cx=3840;cy=3840;publishView=()=>{};evict=()=>{};updateStatus=()=>{};
 const t=taskFor(4,0,0,0);tiles.set(t.key,{...t,presented:true,expiresAt:Infinity});updateURL();refresh();`,geometry1);
assert(new URLSearchParams(geometry1.savedURL.split('?')[1]).get('latent_geometry')==='1');
assert(vm.runInContext('queue.length===1&&queue[0].lod===1&&queue[0].source_lod===3',geometry1));

const standard=createContext('?coarse_prepare=0');vm.runInContext(prefix,standard);
vm.runInContext(`world={seed:'42',generation_profile:'natural',cache_profile:'test',world_bounds:[-20e6,-10e6,20e6,10e6]};
 W=256;H=256;mpp=30;cx=3840;cy=3840;publishView=()=>{};evict=()=>{};updateStatus=()=>{};
 const coarse=taskFor(4,0,0,0);tiles.set(coarse.key,{...coarse,presented:true,expiresAt:Infinity});refresh();`,standard);
assert.equal(vm.runInContext('refinementLevel',standard),3);
assert(vm.runInContext('queue.length===1&&queue[0].lod===3&&!queue[0].source_lod',standard));
