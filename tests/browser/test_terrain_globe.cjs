const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const gl=new Proxy({getShaderParameter:()=>true,getProgramParameter:()=>true,getParameter:()=>4096}, {get:(o,k)=>o[k]??(()=>({}))});
const listeners={},canvas={clientHeight:600,getContext:()=>gl,addEventListener:(name,fn)=>listeners[name]=fn,setPointerCapture(){},parentElement:{focus(){}}};
const context=vm.createContext({window:{},document:{createElement:()=>({getContext:()=>({clearRect(){},drawImage(){}})})},Math,Number,Float32Array,Uint8Array});
vm.runInContext(fs.readFileSync('terrain_globe.js','utf8'),context);
const globe=context.window.TerrainGlobe.create(canvas,{isActive:()=>true}),wb=[-20e6,-10e6,20e6,10e6];
const start=globe.snapshot();globe.orbit(.1,0);const far=globe.snapshot().yaw-start.yaw;
globe.zoom(100);const before=globe.snapshot();globe.orbit(.1,0);const near=globe.snapshot().yaw-before.yaw;
assert(Math.abs(near/far-.01)<1e-10,'orbit scales with altitude for keyboard and dragging');
const coarse=globe.camera(1000,600,wb);globe.zoom(1000);const fine=globe.camera(1000,600,wb);
assert(fine.mpp<30,'globe can reach native neural resolution');assert(fine.mpp<coarse.mpp/999);
assert(fine.bounds.every(Number.isFinite));assert(fine.regions.length===1);
globe.fit();const dragCenter=globe.camera(1000,600,wb).cx;
listeners.pointerdown({button:0,pointerId:1,clientX:100,clientY:100,stopPropagation(){}});
listeners.pointermove({pointerId:1,clientX:120,clientY:100,stopPropagation(){}});
assert(globe.camera(1000,600,wb).cx<dragCenter,'drag right grabs the terrain towards the right, as on the map');
listeners.pointerup({pointerId:1,stopPropagation(){}});
globe.fit();globe.orbit(3*Math.PI/2-globe.snapshot().yaw,0); // Aim at the longitude seam.
globe.zoom(2000);const seam=globe.camera(1000,600,wb);
assert.equal(seam.regions.length,2,'both sides of the longitude seam are visible');
assert(seam.regions.reduce((n,r)=>n+r[2]-r[0],0)<100000,'seam coverage stays local');
globe.fit();globe.orbit(0,Math.PI);const pole=globe.camera(1000,600,wb);
assert(pole.bounds.every(Number.isFinite));assert.equal(pole.uvBounds[1],0);

// Exercise the shared scheduler with real globe camera bounds and learned tiles.
const script=fs.readFileSync('index.html','utf8').split('<script>')[1].split('</script>')[0],elements=new Map();
const noopContext=new Proxy({}, {get:(o,k)=>o[k]??(()=>{})});
const lighting={get:()=>({}),serialize:()=>'',vectors:()=>[0,1,1,1,0,0,1,0]};
const mapContext=vm.createContext({console,URLSearchParams,AbortController,performance,Map,Set,Math,Date,
 location:{search:'?coarse_prepare=0'},crypto:require('node:crypto').webcrypto,
 document:{getElementById(id){if(!elements.has(id))elements.set(id,{checked:id==='renderSea',value:id==='refinementDepth'?'0':id==='cacheLodGap'?'3':'',dataset:{},options:[],getContext:()=>noopContext,setAttribute(){}});return elements.get(id)}},
 window:{TerrainLOD:require(NEURAL_EARTH_ROOT + '/terrain_lod.js')},TerrainLOD:require(NEURAL_EARTH_ROOT + '/terrain_lod.js'),TerrainLighting:lighting,devicePixelRatio:1,
 setTimeout:()=>1,clearTimeout(){},requestAnimationFrame:()=>1});
vm.runInContext(script.slice(0,script.indexOf("view.addEventListener('wheel'")),mapContext);
const run=code=>vm.runInContext(code,mapContext);
mapContext.testGlobe=globe;
globe.fit();globe.orbit(Math.PI/2-globe.snapshot().yaw,-globe.snapshot().pitch);globe.zoom(2000);
run(`world={seed:'42',generation_profile:'natural',cache_profile:'test',world_bounds:[-20e6,-10e6,20e6,10e6]};
 sceneView='globe';globe=testGlobe;W=1000;H=600;publishView=()=>{};updateStatus=()=>{};refresh();`);
assert(run('wanted.size>0&&desiredView.tiles.length>0'),'globe subscribes to neural terrain');
assert(run('requestedLod()<7'),'zoom chooses a finer LOD');
assert(run('queue.every(t=>t.nnEngine==="exact")'));
run(`for(const task of queue){tiles.set(task.key,{...task,image:{width:256,height:256},stage:'coarse',expiresAt:Infinity,presented:false,b:tileBounds(task),bytes:TILE_BYTES})}drawFrame();`);
assert(run('visiblePlan.length>0'),'learned relief is projected onto the globe');
assert(run('visiblePlan.every(p=>p.tile.presented)'),'presentation advances progressive refinement');
assert(globe.snapshot().detailReady);
run(`nnEngine='reference';tiles.clear();refresh()`);
assert(run('desiredView.nnEngine==="reference"&&queue.every(t=>t.nnEngine==="reference")'));
globe.fit();globe.orbit(3*Math.PI/2-globe.snapshot().yaw,0);globe.zoom(2000);run('refresh()');
assert(run('globeCamera.regions.length===2'));
assert(run('wanted.size<100'),'seam does not subscribe to the entire world');
assert(run('desiredView.tiles.some(t=>tileBounds(t)[0]<-19e6)&&desiredView.tiles.some(t=>tileBounds(t)[2]>19e6)'));
// Polar detail uses a local chart instead of loading every longitude.
globe.fit();globe.orbit(0,Math.PI);globe.zoom(2000);run('tiles.clear();refresh()');
assert(run('globeCamera.neuralChart==="polar"&&desiredView.neural_chart==="polar"'));
assert(run('requestedLod()<=2&&wanted.size<100'),'polar zoom keeps native detail and bounded requests');
assert(run('queue.every(t=>t.neuralChart==="polar"&&t.key.includes("/polar/"))'));
const polarFine=globe.camera(1000,600,wb);
assert(polarFine.bounds[2]-polarFine.bounds[0]<100000,'polar coverage is local');
console.log('Globe orbit, fine zoom, polar charts, seam, learned rendering and both NN engines: OK');
