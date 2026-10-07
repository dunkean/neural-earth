// Partial coverage must not schedule an expensive redundant ancestor.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const script=fs.readFileSync('index.html','utf8').split('<script>')[1].split('</script>')[0];
const prefix=script.slice(0,script.indexOf("view.addEventListener('wheel'"));
const context=vm.createContext({URLSearchParams,AbortController,performance,Map,Set,Math,Date,
 location:{search:'?prepare=0'},crypto:require('node:crypto').webcrypto,
 document:{getElementById:()=>({checked:false,dataset:{},getContext:()=>({})})},
 window:{TerrainLOD:require('./terrain_lod.js')},devicePixelRatio:1,
 setTimeout:()=>1,clearTimeout:()=>{},requestAnimationFrame:()=>1});
vm.runInContext(prefix,context);
vm.runInContext(`world={seed:'42',generation_profile:'natural',cache_profile:'test',world_bounds:[-20e6,-10e6,20e6,10e6]};
 W=8192;H=256;mpp=30;cx=122880;cy=3840;publishView=()=>{};evict=()=>{};updateStatus=()=>{};
 for(let tx=0;tx<8;tx++){const t=taskFor(1,tx,0,0);tiles.set(t.key,{...t,presented:true,expiresAt:Infinity})}
 refresh();`,context);
assert.equal(vm.runInContext('refinementLevel',context),4);
const ancestors=JSON.parse(vm.runInContext('JSON.stringify([...interests.values()].map(t=>[t.lod,t.tx,t.ty]))',context));
assert.deepEqual(ancestors,[[4,1,0]],'LOD1 coverage of the left half must skip its LOD4 request');
for(let level=4;level>=2;level--){
 const count=2**(4-level),start=count;
 vm.runInContext(`for(let tx=${start};tx<${2*count};tx++){const t=taskFor(${level},tx,0,0);tiles.set(t.key,{...t,presented:true,expiresAt:Infinity})}refresh()`,context);
 assert.equal(vm.runInContext('refinementLevel',context),level-1);
 const missing=JSON.parse(vm.runInContext('JSON.stringify(queue.map(t=>[t.lod,t.tx,t.ty]))',context));
 assert.deepEqual(missing,Array.from({length:count*2},(_,i)=>[level-1,count*2+i,0]));
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
