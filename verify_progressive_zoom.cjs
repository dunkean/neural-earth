// Real inline client and coverage planner, mocked transport/renderer only.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM0LJrHwMDAxAAGAA7JAUW48M0QAAAAAElFTkSuQmCC','base64');
const physical=Buffer.from(new Float32Array(304*304+5*33*33).fill(100).buffer);
const html=fs.readFileSync('index.html','utf8'),lodScript=fs.readFileSync('terrain_lod.js','utf8');
const renderer=`window.createTerrainRenderer=async()=>{const tiles=new Set();return{available:true,clear(){tiles.clear()},deleteTile(k){tiles.delete(k)},hasTile:k=>tiles.has(k),uploadTile(k){tiles.add(k)},draw(){},getStats(){return{}}}}`;
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--disable-gpu']});
 try{
  const page=await browser.newPage({viewport:{width:1000,height:720}}),requests=[],errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  let phase='overview',holdLOD3=false,releasesLOD3=[],releasesCancel=[],holdNative=false,releasesNative=[];
  await page.route('https://refinement.test/**',async route=>{
   const u=new URL(route.request().url());
   if(u.pathname==='/')return route.fulfill({contentType:'text/html',body:html});
   if(u.pathname==='/terrain_renderer.js')return route.fulfill({contentType:'application/javascript',body:renderer});
   if(u.pathname==='/terrain_lod.js')return route.fulfill({contentType:'application/javascript',body:lodScript});
   if(u.pathname==='/api/world')return route.fulfill({json:{seed:u.searchParams.get('seed'),version:'natural-v1',world_profile:'natural',generation_profile:'natural',cache_profile:'test',world_bounds:[-20e6,-10e6,20e6,10e6],overview_bounds:[-20e6,-10e6,20e6,10e6],overview:'/api/overview/test.png',gpu:'Fake'}});
   if(u.pathname==='/api/view'){
    const body=route.request().postDataJSON();assert(body.tiles.every(t=>t.lod<=11));return route.fulfill({json:{accepted:true}});
   }
   if(u.pathname==='/api/status')return route.fulfill({json:{scheduler:{queued:0,computing:0,encoding:0}}});
   if(u.pathname.startsWith('/height/')){
    const lod=Number(u.pathname.split('/')[4]);assert(lod<=11);
    const entry={phase,lod,path:u.pathname,before:null};requests.push(entry);
    entry.before=await page.evaluate(()=>terrainDebug.snapshot()).catch(()=>null);
    if(holdLOD3&&lod===3)await new Promise(resolve=>{releasesLOD3.push(resolve)});
    if(phase==='cancel'&&lod===4)await new Promise(resolve=>{releasesCancel.push(resolve)});
    if(holdNative&&lod===0)await new Promise(resolve=>{releasesNative.push(resolve)});
    await sleep(25);
    return route.fulfill({contentType:'application/octet-stream',body:physical,headers:{'X-Terrain-Width':'304','X-Terrain-Halo':'24','X-Terrain-Climate-Width':'33','X-Terrain-Climate-Height':'33','X-Terrain-Resolution':String(30*2**lod),'X-Terrain-Stage':lod>=4?'coarse':lod===3?'latent':'decoder','X-Terrain-Cache':'miss'}}).catch(()=>{});
   }
   if(u.pathname.startsWith('/tiles/'))requests.push({phase,lod:Number(u.pathname.split('/')[4]),png:true});
   return route.fulfill({contentType:'image/png',body:png});
  });
  await page.goto('https://refinement.test/?seed=42&profile=natural&prepare=0');
  await page.waitForFunction(()=>terrainDebug.snapshot().rendererReady&&terrainDebug.snapshot().overview&&terrainDebug.snapshot().visible.pending===0);
  await page.locator('#prefetch').uncheck();
  // All zoom entry points, including fit on a tiny viewport, must cap scale.
  await page.evaluate(()=>zoom(1e-9));
  await page.waitForFunction(()=>terrainDebug.snapshot().camera.mpp===61440&&terrainDebug.snapshot().visible.pending===0);
  assert.equal((await page.evaluate(()=>terrainDebug.snapshot())).camera.lod,11);
  phase='native';holdLOD3=true;
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  await page.evaluate(()=>{window.savedRAF=requestAnimationFrame;window.heldFrames=[];window.requestAnimationFrame=f=>{heldFrames.push(f);return 10000+heldFrames.length};cx=110000;cy=110000;mpp=30;draw();schedule()});
  await page.waitForFunction(()=>terrainDebug.snapshot().cache.keys.some(k=>k.includes('/42/4/'))&&terrainDebug.snapshot().scheduler.inflight===0);
  await page.waitForTimeout(150);
  assert(!requests.some(r=>r.phase==='native'&&r.lod<4),'without an animation frame, receipt must not start the expensive stage');
  await page.evaluate(()=>{window.requestAnimationFrame=savedRAF;for(const frame of heldFrames)savedRAF(frame)});
  await page.waitForFunction(()=>terrainDebug.snapshot().refinement.active===3&&terrainDebug.snapshot().rendered.lods.includes(4));
  // LOD3 is intentionally blocked: coarse must be displayed and finer NN jobs
  // must not already be occupying the server lane.
  await page.waitForTimeout(150);
  assert(requests.some(r=>r.phase==='native'&&r.lod===3));
  assert(!requests.some(r=>r.phase==='native'&&r.lod<3));
  holdLOD3=false;for(const release of releasesLOD3)release();
  await page.waitForFunction(()=>terrainDebug.snapshot().visible.pending===0&&terrainDebug.snapshot().camera.lod===0);
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  const native=requests.filter(r=>r.phase==='native'),order=[...new Set(native.map(r=>r.lod))];
  assert.deepEqual(order,[4,3,2,1,0]);
  for(const lod of [3,2,1,0]){
   const first=native.find(r=>r.lod===lod);
   assert(first.before.rendered.lods.length&&Math.max(...first.before.rendered.lods)<=lod+1,'the previous stage must cover the full view before the next stage');
  }
  // A cache-hot refresh should not invent ancestor requests.
  const received=requests.length;
  await page.evaluate(()=>refresh());await page.waitForTimeout(200);
  assert.equal(requests.length,received);const hotRefreshRequests=requests.length-received;
  phase='cancel';
  const beforeAbort=(await page.evaluate(()=>terrainDebug.snapshot())).counters.aborts;
  await page.evaluate(()=>{cx=400000;cy=400000;mpp=30;draw();schedule()});
  while(!releasesCancel.length)await page.waitForTimeout(20);
  const fitMark=requests.length;await page.locator('#fit').click();
  await page.waitForFunction(before=>terrainDebug.snapshot().counters.aborts>before,beforeAbort);
  phase='after-fit';for(const release of releasesCancel)release();
  await page.waitForFunction(()=>terrainDebug.snapshot().visible.pending===0&&terrainDebug.snapshot().camera.lod>=9);
  assert.equal((await page.evaluate(()=>terrainDebug.snapshot())).rendered.hasFineHistory,false);
  assert(requests.slice(fitMark).every(r=>r.lod>=9));
  assert(!(await page.evaluate(()=>terrainDebug.snapshot())).cache.keys.some(k=>k.endsWith('/42/4/3/3')));
  phase='pan-native';holdNative=true;releasesNative=[];
  await page.evaluate(()=>{cx=900000;cy=900000;mpp=30;draw();schedule()});
  while(!releasesNative.length)await page.waitForTimeout(20);
  holdLOD3=true;releasesLOD3=[];
  const retained=await page.evaluate(()=>{
   window.panFlights=[...inflight].map(([key,r])=>({key,flight:r}));
   cx+=16000;draw();refresh();const b=bounds();
   return panFlights.filter(({flight})=>intersects(tileBounds(flight.task),b)).map(({key})=>key);
  });
  assert(retained.length>0,'test must retain some overlapping native flights');
  assert(await page.evaluate(keys=>keys.every(key=>inflight.has(key)&&!inflight.get(key).controller.signal.aborted&&desiredView.tiles.some(t=>taskFor(t.lod,t.tx,t.ty,0).key===key)),retained));
  assert((await page.evaluate(()=>terrainDebug.snapshot())).refinement.active>0,'new strip must need an ancestor');
  assert(await page.evaluate(keys=>{
   const away=panFlights.filter(f=>!keys.includes(f.key));
   return away.length>0&&away.every(f=>f.flight.controller.signal.aborted);
  },retained),'panned-away native flights must be aborted');
  const keptPaths=await page.evaluate(keys=>panFlights.filter(f=>keys.includes(f.key)).map(f=>f.flight.task.path),retained);
  const panAbortCount=(await page.evaluate(()=>terrainDebug.snapshot())).counters.aborts;
  const panStage=(await page.evaluate(()=>terrainDebug.snapshot())).refinement.active;
  const beforeRelease=requests.length;await page.waitForTimeout(150);
  assert(requests.slice(beforeRelease).filter(r=>r.phase==='pan-native').every(r=>r.lod>=panStage||keptPaths.some(path=>r.path.endsWith('/'+path+'.bin'))),'new finer jobs must wait for the stage barrier');
  holdLOD3=false;for(const release of releasesLOD3)release();
  holdNative=false;for(const release of releasesNative)release();
  await page.waitForFunction(()=>terrainDebug.snapshot().visible.pending===0&&terrainDebug.snapshot().camera.lod===0);
  assert.equal((await page.evaluate(()=>terrainDebug.snapshot())).counters.aborts,panAbortCount,'kept flights must settle without further aborts');
  for(const path of keptPaths)assert.equal(requests.filter(r=>r.phase==='pan-native'&&r.path?.endsWith('/'+path+'.bin')).length,1,'kept native request must not restart');
  // A supplied native image remains visible while native packets arrive; no
  // coarse fallback may cover it or waste NN work wholly inside the image.
  phase='initial';
  holdNative=true;releasesNative=[];
  await page.evaluate(()=>{initialImage=overview;world.initial_bounds=[-15000,-10000,15000,10000];cx=0;cy=0;mpp=24;draw();schedule()});
  while(!releasesNative.length)await page.waitForTimeout(20);
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  assert((await page.evaluate(()=>terrainDebug.snapshot())).visible.pending>0);
  assert(requests.filter(r=>r.phase==='initial').length>0&&requests.filter(r=>r.phase==='initial').every(r=>r.lod===0));
  assert(await page.evaluate(()=>visiblePlan.some(p=>p.fallback)&&visiblePlan.every(p=>!p.fallback||!intersects(p.bounds,world.initial_bounds))));
  holdNative=false;for(const release of releasesNative)release();
  await page.waitForFunction(()=>terrainDebug.snapshot().visible.pending===0&&terrainDebug.snapshot().camera.lod===0);
  assert(requests.every(r=>r.lod<=11));
  assert.deepEqual(errors,[]);
  const report={passed:true,transport:'mock',gpu:'mock',sourceHashes:Object.fromEntries(['index.html','terrain_lod.js','verify_progressive_zoom.cjs'].map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')])),nativeRequestOrder:order,noLOD12:requests.every(r=>r.lod<=11),slowLOD3HasPresentedLOD4:native.find(r=>r.lod===3).before.rendered.lods.includes(4),heldRAFBlocksRefinement:true,nativeImageProtected:requests.filter(r=>r.phase==='initial').every(r=>r.lod===0),panRetainedFlights:retained.length,cancelAborts:(await page.evaluate(()=>terrainDebug.snapshot())).counters.aborts-beforeAbort,hotRefreshRequests,requests:requests.length};
  if(process.env.REPORT_PATH)fs.writeFileSync(process.env.REPORT_PATH,JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
