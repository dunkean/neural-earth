const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('index.html','utf8'),script=html.split('<script>')[1].split('</script>')[0];
const elements=new Map(),drawCalls=[];
const canvasContext=new Proxy({}, {get(target,key){return target[key]??((...args)=>drawCalls.push([key,...args]));}});
const context=vm.createContext({URLSearchParams,AbortController,performance,Map,Set,Math,Date,
 location:{search:''},crypto:require('node:crypto').webcrypto,
 document:{getElementById(id){if(!elements.has(id))elements.set(id,{checked:id==='prefetch',value:id==='refinementDepth'?'0':id==='seaMaxLod'?'9':id==='cacheLodGap'?'3':'',dataset:{},options:[],getContext:()=>canvasContext});return elements.get(id)}},
 window:{TerrainLOD:require(NEURAL_EARTH_ROOT + '/terrain_lod.js'),TerrainRender:{serialize:()=> '{}',get:()=>({})}},TerrainLOD:require(NEURAL_EARTH_ROOT + '/terrain_lod.js'),devicePixelRatio:1,setTimeout:()=>1,clearTimeout(){},requestAnimationFrame:()=>1});
vm.runInContext(script.slice(0,script.indexOf("view.addEventListener('wheel'")),context);
const run=code=>vm.runInContext(code,context);
run(`world={seed:'42',generation_profile:'natural',cache_profile:'test',world_bounds:[-20e6,-10e6,20e6,10e6]};
 W=256;H=256;mpp=120;cx=15360;cy=15360;publishView=()=>{};evict=()=>{};updateStatus=()=>{};
 const parent=taskFor(2,0,0,0);tiles.set(parent.key,{...parent,presented:true,expiresAt:Infinity,b:tileBounds(parent)});refresh()`);
assert.equal(run('queue.length'),0,'default prefetch must not generate finer LODs at rest');
run("$('refinementDepth').value='2';refresh()");
assert.equal(run('queue.length'),4);
assert(run('queue.every(t=>t.lod===1&&t.refinementAhead)'));
run(`for(const t of queue)tiles.set(t.key,{...t,presented:false,expiresAt:Infinity,b:tileBounds(t)});refresh()`);
assert.equal(run('queue.length'),16,'completed background LOD1 must unlock LOD0 without moving');
assert(run('queue.every(t=>t.lod===0)'));
run(`for(const t of queue)tiles.set(t.key,{...t,presented:false,expiresAt:Infinity,b:tileBounds(t)});refresh()`);
assert.equal(run('queue.length'),0,'depth 2 stops at LOD0 from LOD2');
run("$('refinementDepth').value='3';refresh()");
assert(run('queue.length>0&&queue.every(t=>t.lod===-1)'),'depth is configurable');
run(`const t=queue[0];held=new AbortController();inflight.set(t.key,{task:t,controller:held});$('refinementDepth').value='0';refresh()`);
assert.equal(run('queue.length'),0);
assert(run('held.signal.aborted'),'disabling refinement cancels speculative flights');
run(`inflight.clear();tiles.clear();tiles.set(parent.key,{...parent,presented:true,expiresAt:Infinity,b:tileBounds(parent)});
 $('forcedLod').value='0';forceViewLod();refresh()`);
assert.equal(run('mpp'),120,'manual rendering must preserve zoom');
assert.equal(run('requestedLod()'),0);
assert.equal(run('wantedLod'),1);
assert(run('queue.length===4&&queue.every(t=>t.lod===1&&!t.prefetch)'),'manual rendering completes LOD1 before LOD0');
run(`for(const t of queue)tiles.set(t.key,{...t,presented:true,expiresAt:Infinity,b:tileBounds(t)});refresh()`);
assert(run('queue.length===16&&queue.every(t=>t.lod===0&&!t.prefetch)'),'one-shot then covers the full view at LOD0');
run('cx+=120;refresh()');
assert.equal(run('requestedLod()'),2,'movement returns to camera LOD');
assert(run('!queue.some(t=>t.lod<2)'),'movement cancels manually forced fine work');
assert(html.indexOf('id="grid"')<html.indexOf('id="runtimePanel"'),'tiles checkbox belongs in the top bar');
console.log('LOD opt-in, recursive depth, cancellation and manual viewport rendering: OK');

// A full first-level reservation must still allow the second configured level.
run(`forcedView=null;inflight.clear();tiles.clear();W=1024;H=1024;mpp=120;cx=61440;cy=61440;
 $('refinementDepth').value='2';$('refinementDepth').value='2';$('prefetch').checked=false;
 for(let ty=0;ty<4;ty++)for(let tx=0;tx<4;tx++){const p=taskFor(2,tx,ty,0);tiles.set(p.key,{...p,presented:true,expiresAt:Infinity,b:tileBounds(p)})}refresh()`);
assert.equal(run('queue.length'),64);
run(`for(const t of queue)tiles.set(t.key,{...t,presented:false,expiresAt:Infinity,b:tileBounds(t)});refresh()`);
assert(run('queue.length>0&&queue.every(t=>t.lod===0)'),'full LOD1 budget must not prevent depth 2');
console.log('Shared background budget progresses to the second LOD: OK');

// The real frame renders the finer image as soon as it arrives, and the tile
// overlay labels the displayed LOD rather than the camera's LOD.
run(`tiles.clear();W=256;H=256;mpp=120;cx=15360;cy=15360;
 $('grid').checked=true;$('refinementDepth').value='2';$('refinementDepth').value='2';
 for(const lod of [2,1,0]){const t=taskFor(lod,0,0,0);tiles.set(t.key,{...t,b:tileBounds(t),image:{width:256,height:256},presented:false,expiresAt:Infinity})}
 drawFrame()`);
assert.deepEqual(JSON.parse(run('JSON.stringify([...new Set(visiblePlan.map(p=>p.tile.lod))].sort())')),[0,1,2]);
assert(run('[...tiles.values()].every(t=>t.presented===true)'),'all three ready levels actually reach the renderer');
assert(drawCalls.some(([method,label])=>method==='fillText'&&label==='LOD 0 · 0, 0'));
const clips=[];
for(const [method,label] of drawCalls){
 if(method==='save')clips.push(false);
 if(method==='clip')clips[clips.length-1]=true;
 if(method==='fillText'&&typeof label==='string'&&label.startsWith('LOD '))assert.equal(clips.at(-1),true,'tile labels must stay inside their displayed patch');
 if(method==='restore')clips.pop();
}
drawCalls.length=0;
run("$('refinementDepth').value='0';$('cacheLodGap').value='0';drawFrame()");
assert(run('visiblePlan.every(p=>p.tile.lod===2)'));
assert(!drawCalls.some(([method,label])=>method==='fillText'&&label.startsWith('LOD 0')));
console.log('Continuous LOD frame rendering and actual tile grid labels: OK');

// A viewport larger than one batch must progress beyond the first 64 children.
run(`inflight.clear();tiles.clear();W=1536;H=1536;mpp=120;cx=92160;cy=92160;
 $('refinementDepth').value='2';$('refinementDepth').value='1';
 for(let ty=0;ty<6;ty++)for(let tx=0;tx<6;tx++){const p=taskFor(2,tx,ty,0);tiles.set(p.key,{...p,presented:true,expiresAt:Infinity,b:tileBounds(p)})}
 refresh();firstBatch=new Set(queue.map(t=>t.key));
 for(const t of queue)tiles.set(t.key,{...t,presented:true,expiresAt:Infinity,b:tileBounds(t)});
 refresh()`);
assert.equal(run('firstBatch.size'),64);
assert.equal(run('queue.length'),64,'completed children must yield their batch slots');
assert(run('queue.every(t=>!firstBatch.has(t.key))'));
run(`{const t=queue[0];held=new AbortController();inflight.set(t.key,{task:t,controller:held});refresh()}`);
assert(run('interests.has([...inflight.keys()][0])&&!held.signal.aborted'),'advancing a batch must retain useful fine flights');

// A near-centre sea tile must follow even a distant land tile, in the browser
// queue and in the priorities published to the compute server.
run(`inflight.clear();tiles.clear();W=1024;H=256;mpp=120;cx=61440;cy=15360;
 $('refinementDepth').value='0';viewMode='snr-elevation';
 const rows=Array.from({length:8},()=> '0000000000000001');
 $('cacheLodGap').value='3';$('renderSea').checked=true;$('seaMaxLod').value='-3';world.scheduling_land_mask={bounds:[0,0,122880,61440],width:16,height:8,rows};refresh()`);
assert(run('queue.some(t=>t.sea)&&queue.some(t=>!t.sea)'));
assert(run('queue[0].sea===false&&queue.at(-1).sea===true'));
assert(run('Math.max(...queue.filter(t=>!t.sea).map(t=>t.priority))<Math.min(...queue.filter(t=>t.sea).map(t=>t.priority))'));
assert(run('desiredView.tiles.some(t=>t.priority>=4000)&&desiredView.tiles.some(t=>t.priority<4000)'));
console.log('Large refinement batches, retained flights and land-before-sea subscription: OK');

// The switch overrides the cap, and the cap is inclusive in the direction of zoom.
run(`inflight.clear();tiles.clear();forcedView=null;viewMode='snr-elevation';
 world.scheduling_land_mask.rows.fill('0000000000000000');$('renderSea').checked=false;
 $('seaMaxLod').value='9';refresh()`);
assert.equal(run('seaMaxLod()'),9);
assert.equal(run('queue.length'),0,'filtered sea must not create replacement ocean jobs');
assert.equal(run('wanted.size'),0,'excluded ocean must not block completion');
for(const lod of [10,11])assert(run(`seaTaskAllowed({lod:${lod},tx:0,ty:0})`));
for(const lod of [9,8,3,0,-3])assert(!run(`seaTaskAllowed({lod:${lod},tx:0,ty:0})`));
run(`$('seaMaxLod').value='3'`);
assert(run('seaTaskAllowed({lod:4,tx:0,ty:0})'));
assert(!run('seaTaskAllowed({lod:3,tx:0,ty:0})'));
run(`$('renderSea').checked=true;refresh()`);
assert(run('queue.length>0&&queue.every(t=>t.lod===2&&t.sea)'),'render sea disables filtering');
run(`{const t=queue[0];held=new AbortController();inflight.set(t.key,{task:t,controller:held});
 $('renderSea').checked=false;$('seaMaxLod').value='9';refresh()}`);
assert(run('held.signal.aborted'),'new filters cancel pending ocean transfers');
assert.equal(run('queue.length'),0);
run(`$('seaMaxLod').value='0'`);
assert.equal(run('seaMaxLod()'),0);

// Measured relief is authoritative, including shallow sea above -10m.
run(`inflight.clear();tiles.clear();delete world.scheduling_land_mask;
 $('seaMaxLod').value='9';const measured=taskFor(2,0,0,0);
 tiles.set(measured.key,{...measured,elevationMin:-100,elevationMax:-10,expiresAt:0,b:tileBounds(measured)});`);
assert(!run('seaTaskAllowed(measured)'));
assert(!run('seaTaskAllowed({lod:1,tx:0,ty:0})'),'empty parent filters its descendants');
assert(!run('needsValidation(tiles.get(measured.key))'),'frozen sea is not revalidated');
run('tiles.get(measured.key).elevationMax=-9.99');
assert(run('seaTaskAllowed(measured)'),'relief strictly above -10m remains eligible');
assert(run("terrainClass(measured)==='sea'"),'shallow sea still sorts last');

// Centre distance sorts inside each category; coasts precede land and sea on
// both the download queue and the server subscription.
run(`tiles.clear();W=768;H=256;mpp=120;cx=46080;cy=15360;
 $('renderSea').checked=true;
 for(let tx=0;tx<3;tx++){const t=taskFor(2,tx,0,0),ranges=[[-100,100],[100,200],[-100,-10]];
 tiles.set(t.key,{...t,b:tileBounds(t),expiresAt:0,elevationMin:ranges[tx][0],elevationMax:ranges[tx][1]})}refresh()`);
assert.deepEqual(JSON.parse(run('JSON.stringify(queue.map(t=>t.terrain))')),['coast','land','sea']);
assert(run('desiredView.tiles[0].priority<desiredView.tiles[1].priority&&desiredView.tiles[1].priority<desiredView.tiles[2].priority'));

// A partially completed first finer LOD must never admit deeper children.
run(`tiles.clear();W=512;H=256;mpp=960;cx=245760;cy=122880;viewMode='relief';
 $('refinementDepth').value='2';
 for(let tx=0;tx<2;tx++){const t=taskFor(5,tx,0,0);tiles.set(t.key,{...t,expiresAt:Infinity,b:tileBounds(t)})}refresh();
 tiles.set(queue[0].key,{...queue[0],expiresAt:Infinity,b:tileBounds(queue[0])});refresh()`);
assert(run('queue.length===7&&queue.every(t=>t.lod===4)'),'finish all LOD4 tiles before LOD3');
run(`for(const t of queue)tiles.set(t.key,{...t,expiresAt:Infinity,b:tileBounds(t)});refresh()`);
assert(run('queue.length===32&&queue.every(t=>t.lod===3)'),'LOD5 depth2 reaches LOD3 via LOD4');
run(`for(const t of queue)tiles.set(t.key,{...t,expiresAt:Infinity,b:tileBounds(t)});refresh()`);
assert.equal(run('queue.length'),0,'depth stops after two levels');

// Cached LOD3 is displayed from a LOD6 camera only with a gap of at least 3.
run(`tiles.clear();W=256;H=256;mpp=1920;cx=245760;cy=245760;
 $('refinementDepth').value='0';$('grid').checked=true;$('cacheLodGap').value='3';
 for(const lod of [6,3]){const t=taskFor(lod,0,0,0);tiles.set(t.key,{...t,image:{width:256,height:256},expiresAt:Infinity,b:tileBounds(t)})}drawFrame()`);
assert(run('visiblePlan.some(p=>p.tile.lod===3)'));
run(`$('cacheLodGap').value='2';drawFrame()`);
assert(run('visiblePlan.every(p=>p.tile.lod===6)'));
run(`$('cacheLodGap').value='0';drawFrame()`);
assert(run('visiblePlan.every(p=>p.tile.lod===6)'));

// One actual coarser tile may cover many target cells: the grid draws its real
// footprint, never the artificial target cell rectangles. Undrawable entries
// cannot hide that fallback or create a grid square.
drawCalls.length=0;
run(`tiles.clear();W=256;H=256;mpp=120;cx=15360;cy=15360;$('cacheLodGap').value='3';
 const fallback=taskFor(4,0,0,0),missing=taskFor(2,0,0,0);
 tiles.set(fallback.key,{...fallback,image:{width:256,height:256},b:tileBounds(fallback)});
 tiles.set(missing.key,{...missing,b:tileBounds(missing)});drawFrame()`);
assert(run('visiblePlan.length>0&&visiblePlan.every(p=>p.tile.lod===4)'));
assert(drawCalls.filter(([method])=>method==='strokeRect').some(([,x,y,w,h])=>w===1024&&h===1024));
assert(!drawCalls.some(([method,label])=>method==='fillText'&&typeof label==='string'&&label.startsWith('LOD 2')));
assert(!html.includes('id="idleRefinement"'));
assert(html.includes('id="refinementDepth" type="number" min="0"'));
const rendering=html.slice(html.indexOf('<details id="renderPanel">'),html.indexOf('<details id="runtimePanel">'));
for(const id of ['renderSea','seaMaxLod','forcedLod','cacheLodGap','gpuRender','nnEngine','prefetch'])assert(rendering.includes(`id="${id}"`));
console.log('Inclusive sea cap, -10m threshold, coast ordering, stage barrier, cache gap and actual grid: OK');

// Zooming must keep the continental tile and its real outline, even across a
// larger LOD jump than the cache gap, until all its visible area is replaced.
drawCalls.length=0;
run(`tiles.clear();visiblePlan=[];W=256;H=256;mpp=15360;cx=1966080;cy=1966080;
 $('cacheLodGap').value='0';$('refinementDepth').value='0';$('grid').checked=true;
 const continental=taskFor(9,0,0,0);tiles.set(continental.key,{...continental,image:{width:256,height:256},expiresAt:Infinity,b:tileBounds(continental)});drawFrame();
 mpp=960;drawFrame()`);
assert(run('visiblePlan.length>0&&visiblePlan.every(p=>p.tile.lod===9)'),'zoom keeps the previously drawn continental parent');
assert(drawCalls.some(([method,label])=>method==='fillText'&&label==='LOD 9 · 0, 0'));
run(`const replacement=taskFor(5,7,7,0);tiles.set(replacement.key,{...replacement,image:{width:256,height:256},expiresAt:Infinity,b:tileBounds(replacement)});drawFrame()`);
assert(run('visiblePlan.some(p=>p.tile.lod===9)&&visiblePlan.some(p=>p.tile.lod===5)'),'partial arrival only replaces its own area');
run(`for(let ty=7;ty<=8;ty++)for(let tx=7;tx<=8;tx++){const t=taskFor(5,tx,ty,0);tiles.set(t.key,{...t,image:{width:256,height:256},expiresAt:Infinity,b:tileBounds(t)})}drawFrame()`);
assert(run('visiblePlan.length===4&&visiblePlan.every(p=>p.tile.lod===5)'),'continental frame disappears only after real replacement');
console.log('Continental tiles and frames survive zoom until complete visible replacement: OK');

// Newly available intermediate parents must refine the picture without a
// camera change, even when cached fine history is disabled.
run(`tiles.clear();visiblePlan=[];inflight.clear();forcedView=null;W=256;H=256;mpp=480;cx=61440;cy=61440;
 $('cacheLodGap').value='0';$('refinementDepth').value='0';
 const coarseTransition=taskFor(4,0,0,0);tiles.set(coarseTransition.key,{...coarseTransition,image:{width:256,height:256},expiresAt:Infinity,b:tileBounds(coarseTransition)});drawFrame();
 mpp=120;drawFrame();
 const intermediateTransition=taskFor(3,0,0,0);tiles.set(intermediateTransition.key,{...intermediateTransition,image:{width:256,height:256},presented:false,expiresAt:Infinity,b:tileBounds(intermediateTransition)});drawFrame()`);
assert.equal(run('lodForCamera()'),2);
assert(run('visiblePlan.some(p=>p.tile.lod===3)&&visiblePlan.some(p=>p.tile.lod===4)'),
 'a ready LOD3 replaces its part of LOD4 while camera remains at LOD2');
assert(run('tiles.get(intermediateTransition.key).presented'),
 'intermediate tiles are actually presented without dezooming');
run(`for(let ty=0;ty<=1;ty++)for(let tx=0;tx<=1;tx++){const t=taskFor(3,tx,ty,0);tiles.set(t.key,{...t,image:{width:256,height:256},expiresAt:Infinity,b:tileBounds(t)})}drawFrame()`);
assert(run('visiblePlan.length===4&&visiblePlan.every(p=>p.tile.lod===3)'),
 'complete LOD3 coverage replaces the coarse image at the fixed LOD2 camera');
run(`const finalTransition=taskFor(2,1,1,0);tiles.set(finalTransition.key,{...finalTransition,image:{width:256,height:256},expiresAt:Infinity,b:tileBounds(finalTransition)});drawFrame()`);
assert(run('visiblePlan.some(p=>p.tile.lod===2)&&visiblePlan.some(p=>p.tile.lod===3)'),
 'final tiles progressively replace intermediate coverage');

// A GPU tile received after the zoom uses the same intermediate selection.
run(`tiles.clear();visiblePlan=[];
 renderer={available:true,hasTile:()=>true,draw:()=>true};$('gpuRender').checked=true;
 for(const lod of [4,3]){const t=taskFor(lod,0,0,0);tiles.set(t.key,{...t,gpu:true,heights:new Float32Array([100]),presented:false,expiresAt:Infinity,b:tileBounds(t)})}drawFrame()`);
assert(run('visiblePlan.some(p=>p.tile.lod===3)&&tiles.get(taskFor(3,0,0,0).key).presented'),
 'GPU intermediate parents appear immediately with cache gap zero');
console.log('Ready intermediate LODs progressively replace fallback at a fixed camera LOD: OK');

// Climate must cover open sea at LOD9 even at world view, and stay there when
// zoomed or when the relief controls request a finer level.
run(`inflight.clear();tiles.clear();visiblePlan=[];renderer=null;overview=null;initialImage=null;
 world.world_bounds=[0,0,16e6,8e6];world.scheduling_land_mask={bounds:world.world_bounds,width:16,height:8,rows:Array(8).fill('0000000000000000')};
 W=256;H=128;cx=8e6;cy=4e6;$('renderSea').checked=false;$('seaMaxLod').value='9';
 $('refinementDepth').value='14';$('cacheLodGap').value='0';$('prefetch').checked=false;zoomUntil=0;`);
const climateModes=['temperature','precipitation',
 'orogen-temperature-summer','orogen-temperature-winter','orogen-precip-summer','orogen-precip-winter',
 'orogen-pressure-summer','orogen-pressure-winter','orogen-rain-shadow','orogen-continentality',
 'orogen-wind-summer','orogen-wind-winter','orogen-currents-summer','orogen-currents-winter'];
for(const mode of climateModes)for(const cameraLod of [11,9,3,-3]){
 run(`viewMode=${JSON.stringify(mode)};tiles.clear();mpp=30*2**${cameraLod};
  $('forcedLod').value='0';forceViewLod();refresh()`);
 assert.equal(run('requestedLod()'),9,`${mode}: fixed resolution`);
 assert.equal(run('wantedLod'),9);
 assert(run('!continuousRefinement()'));
 assert(run('queue.length>0&&queue.every(t=>t.lod===9&&t.sea)'),`${mode}: sea coverage without parent or finer jobs`);
 assert(run('desiredView.tiles.every(t=>t.lod===9)'),`${mode}: server receives only LOD9`);
 run(`for(const t of queue)tiles.set(t.key,{...t,image:{width:256,height:256},elevationMax:-100,expiresAt:Infinity,b:tileBounds(t)});
  for(const lod of [8,11]){const t=taskFor(lod,0,0,0);tiles.set(t.key,{...t,image:{width:256,height:256},expiresAt:Infinity,b:tileBounds(t)})}
  drawFrame();refresh()`);
 assert(run('visiblePlan.length>0&&visiblePlan.every(p=>p.tile.lod===9)'),`${mode}: draw only LOD9`);
 assert.equal(run('queue.length'),0,`${mode}: completion does not start refinement`);
}
run(`viewMode='relief';forcedView=null;mpp=120;refresh()`);
assert.equal(run('requestedLod()'),2,'relief returns to camera resolution');
assert(run('seaFilteringAt(9)'),'relief keeps its sea constraint');
assert(run('continuousRefinement()'),'relief keeps its depth control');
console.log('Climate layers cover sea at fixed LOD9 across zoom, manual LOD and refinement controls: OK');


for(const mode of ['biomes','orogen-biomes','orogen-koppen','render','soil']){
 run(`viewMode=${JSON.stringify(mode)};forcedView=null;$('renderSea').checked=true;$('refinementDepth').value='0';mpp=120;refresh()`);
 assert.equal(run('fixedClimateLod()'),false);assert.equal(run('requestedLod()'),2,mode+' follows terrain camera LOD');
 run(`$('forcedLod').value='0';forceViewLod()`);assert.equal(run('requestedLod()'),0,mode+' permits terrain LOD override');
}
console.log('Biomes and Köppen follow camera and requested terrain LOD: OK');
