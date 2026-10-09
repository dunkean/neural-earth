const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
// Regression: camera LOD9 / depth2 must finish LOD7 without resurfacing LOD4.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const script=fs.readFileSync('index.html','utf8').split('<script>')[1].split('</script>')[0];
const elements=new Map(),drawCalls=[];
const canvas=new Proxy({}, {get:(target,key)=>target[key]??((...args)=>drawCalls.push([key,...args]))});
const context=vm.createContext({URLSearchParams,AbortController,performance,Map,Set,Math,Date,
 location:{search:''},crypto:require('node:crypto').webcrypto,
 document:{getElementById(id){if(!elements.has(id))elements.set(id,{checked:['grid','renderSea'].includes(id),
  value:id==='refinementDepth'?'2':id==='cacheLodGap'?'3':id==='seaMaxLod'?'9':'',
  dataset:{},options:[],getContext:()=>canvas,setAttribute(){}});return elements.get(id)}},
 window:{TerrainLOD:require(NEURAL_EARTH_ROOT + '/terrain_lod.js')},TerrainLOD:require(NEURAL_EARTH_ROOT + '/terrain_lod.js'),TerrainLighting:{get:()=>({})},devicePixelRatio:1,
 setTimeout:()=>1,clearTimeout(){},requestAnimationFrame:()=>1});
vm.runInContext(script.slice(0,script.indexOf("view.addEventListener('wheel'")),context);
const run=code=>vm.runInContext(code,context);
run(`world={seed:'42',generation_profile:'natural',cache_profile:'test',world_bounds:[-20e6,-10e6,20e6,10e6]};
 W=256;H=256;mpp=15360;cx=1966080;cy=1966080;publishView=()=>{};updateStatus=()=>{};
 function ready(t){const tile={...t,image:{width:256,height:256,close(){}},presented:true,expiresAt:Infinity,
  b:tileBounds(t),bytes:gpuEnabled()?304*304*4+5*33*33*4+TILE_BYTES:TILE_BYTES};tiles.set(t.key,tile);cacheBytes+=tile.bytes}
 ready(taskFor(9,0,0,0));ready(taskFor(4,0,0,0));refresh();`);
assert.equal(run('lodForCamera()'),9);
assert.equal(run('refinementProgress().active'),8);
run(`serverStatus={scheduler:{queued:0,computing:0,encoding:0}};updateActivity()`);
assert.equal(elements.get('loadingIndicator').hidden,false,'keep activity visible between refinement jobs');
assert.match(elements.get('loadingLabel').textContent,/LOD 8.*0\/4/);
run('for(const t of queue)ready(t);refresh()');
assert.equal(run('refinementProgress().active'),7);
assert(run('queue.length===16&&queue.every(t=>t.lod===7)'));
run('for(const t of queue)ready(t);refresh();drawFrame();updateActivity()');
assert(run('refinementProgress().complete'));
assert.equal(elements.get('loadingIndicator').hidden,true);
assert.match(elements.get('lodIndicator').textContent,/LOD 9.*cible 7.*terminé/);
for(let i=0;i<5;i++){
 run('refresh();drawFrame()');
 assert(run('visiblePlan.every(p=>p.tile.lod===7)'),'historical LOD4 must stay hidden at the configured depth');
 assert.equal(run('queue.length'),0,'completed refinement must not restart');
}
assert(!drawCalls.some(([method,label])=>method==='fillText'&&String(label).startsWith('LOD 4')));

// A pending neighbouring prefetch must not prevent visible refinement.
run(`tiles.clear();cacheBytes=0;inflight.clear();ready(taskFor(9,0,0,0));
 $('prefetch').checked=true;velocity={x:1,y:0};lastMotion=performance.now();refresh()`);
assert(run('queue.some(t=>t.prefetch&&!t.refinementAhead)'));
assert(run('queue.some(t=>t.lod===8&&t.refinementAhead)'));

// GPU depth2 fits a configured 384MiB cap. Keep parents and every completed
// child until all 256 visible LOD7 cells are ready; evict unrelated history.
run(`tiles.clear();cacheBytes=0;inflight.clear();visiblePlan=[];$('prefetch').checked=false;
 renderer={available:true,deleteTile(){}};$('gpuRender').checked=true;
 MAX_BYTES=.375*GIB;W=1024;H=1024;cx=0;cy=0;
 for(let i=0;i<300;i++)ready(taskFor(4,100+i,100,0));
 for(let y=-2;y<2;y++)for(let x=-2;x<2;x++)ready(taskFor(9,x,y,0));refresh();evict()`);
assert.equal(run('tileCacheBudget()'),run('MAX_BYTES'));
assert.equal(run('renderer.maxBytes'),run('tileCacheBudget()'),'GPU textures must share the working-set reservation');
let batches=0;
while(!run('refinementProgress().complete')&&batches++<20){
 assert(run('queue.length>0'),'unfinished visible stage must retain runnable jobs');
 run('for(const t of queue)ready(t);evict();refresh()');
 assert(run('refinementStageComplete(9,bounds())'),'cache pressure must not remove required camera tiles');
}
assert(batches<20,'all batches must finish');
assert(run('refinementProgress().complete'));
assert.equal(run('refinementProgress().target'),7);
assert.equal(run('refinementProgress().stages.at(-1).ready'),256);
assert(run('cacheBytes>192*1024**2&&cacheBytes<=tileCacheBudget()'));
assert(run('[...tiles.values()].filter(t=>t.lod===4).length<300'),'evict unrelated history before the working chain');
assert.equal(run('[...tiles.values()].filter(t=>t.lod>=7).length'),336);
run("$('refinementDepth').value='14';refresh()");
assert(run('refinementPolicy().budgetLimited'));
assert.equal(run('refinementPolicy().cacheBytes'),run('MAX_BYTES'));
assert.match(run('refinementProgress().text'),/limite mémoire/);
console.log('LOD9 depth2 finishes LOD7, stable grid, persistent progress, prefetch and bounded cache: OK');
