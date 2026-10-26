const assert=require('node:assert/strict');
const hydro=require('../../web/terrain_hydrology.js');
const tick=()=>new Promise(ok=>setTimeout(ok,0));
const ctx=()=>{const strokes=[],stack=[];let visible=true;return {strokes,
 save(){stack.push(visible);},restore(){visible=stack.pop();},beginPath(){},rect(){},
 clip(rule){visible=rule!=='evenodd';},moveTo(){},lineTo(){},setLineDash(){},
 stroke(){if(visible)strokes.push({width:this.lineWidth,colour:this.strokeStyle});}};};
(async()=>{
 let state={seed:42,profile:'same',worldIdentity:'world-a',worldBounds:[-1e6,-1e6,1e6,1e6],bounds:[0,0,128,128],
  lod:0,mpp:1,width:128,height:128,view:'map',renderV2:true,physicalWater:true,physicalRects:[{x:0,y:0,width:128,height:128}],project:(x,y)=>[x,y]};
 const months=Array(12).fill(100);months[0]=0;
 const network={status:'ready',hydro_id:'network-a',reaches:[{id:'regional',xy:[[0,20],[64,40],[128,20]],water_m:[-10,-8,-6],
  width_m:4,flow_m3s:100,area_m2:1e10,monthly_flow_m3s:months}],lakes:[]};
 const v=hydro.create({getState:()=>state,initial:{enabled:true,month:7,intermittent:true},fetchJSON:async()=>({ok:true,status:200,json:async()=>network})});
 assert.equal(hydro.riverDisplay().ready,false,'Pending network is an identified overlay fallback');
 v.draw(ctx(),state);await tick();v.draw(ctx(),state);
 const a=hydro.riverDisplay();assert(a.ready);assert.equal(hydro.riverDisplay(),a,'Unchanged frames reuse the same projected Float32Array');assert.equal(a.segments.length,16);assert.deepEqual(Array.from(a.segments.slice(0,8)),[0,20,64,40,2,-10,-8,0]);
 const cpu=ctx();v.draw(cpu,state);assert.equal(cpu.strokes.length,0,'PNG/CPU without a live GPU surface keeps previous raster/overlay policy');
 hydro.acknowledgeRiverDisplay(null,[],a.key);
 const fallback=ctx();v.draw(fallback,state);assert(fallback.strokes.length,'Live GPU surface without ready mask uses regional R3 fallback');
 hydro.acknowledgeRiverDisplay(a.key,state.physicalRects);assert.equal(v.snapshot().riverDisplay.frameKey,a.key);
 const ready=ctx();v.draw(ready,state);assert.equal(ready.strokes.length,0,'Ready mask suppresses duplicate wet vector inside its live GPU surface');
 hydro.acknowledgeRiverDisplay(null,[]);const detached=ctx();v.draw(detached,state);assert.equal(detached.strokes.length,0,'GPU detached/compiling: previous CPU/PNG behavior is restored');
 const oldState=state;state={...state,renderV2:false,physicalRects:[],physicalWater:false};const legacy=ctx();v.draw(legacy,state);assert(legacy.strokes.length,'Legacy R3 remains its previous vector policy');assert.equal(hydro.riverDisplay(),null);state=oldState;
 v.set({month:1});v.draw(ctx(),state);await tick();const dry=hydro.riverDisplay();assert(dry.ready);assert.equal(dry.segments.length,0,'Dry month is a valid empty geometry, never stale July water');assert.notEqual(dry.key,a.key);
 hydro.acknowledgeRiverDisplay(dry.key,state.physicalRects);const dryDraw=ctx();v.draw(dryDraw,state);assert(dryDraw.strokes.length&&dryDraw.strokes.every(x=>x.colour==='#9b7a50'),'Dry-month overlay stays brown, never stale wet blue');
 v.set({month:7,widthScale:2});const scaled=hydro.riverDisplay();assert.equal(scaled.segments[4],2,'Cartographic line width does not inflate physical water');
 state={...state,width:256};assert.notEqual(hydro.riverDisplay().key,scaled.key,'Resize invalidates screen geometry');
 state={...state,worldIdentity:'world-b'};assert(!hydro.riverDisplay().ready,'Another world cannot inherit the previous network');
 v.reset();assert(!hydro.riverDisplay().ready);v.dispose();assert.equal(hydro.riverDisplay(),null);assert(!hydro.regionalRiverDisplayEnabled());
 console.log('PASS regional display: shared projected geometry/physical width, monthly dry, fallback, resize, world reset/dispose');
})().catch(e=>{console.error(e);process.exitCode=1;});
