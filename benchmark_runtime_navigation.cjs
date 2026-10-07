// Actual server/Chrome probe: time to successive submitted terrain stages.
// HTTP and snapshots are measured; GPU execution/presentation is not timed.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');
const fs=require('node:fs'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const output=process.env.TERRAIN_RUNTIME_NAV_REPORT||'E:/TerrainDiffusionRuntime/runtime-lod-optimization/browser-progressive.json';
const profile=process.env.TERRAIN_RUNTIME_PROFILE||'natural',seed=process.env.TERRAIN_RUNTIME_SEED||'20261007117';
const x=Number(process.env.TERRAIN_RUNTIME_X||'-103680'),y=Number(process.env.TERRAIN_RUNTIME_Y||'80640');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--enable-unsafe-webgpu']});
 const page=await browser.newPage({viewport:{width:900,height:650}}),responses=[],worldResponses=[],headerPromises=[],errors=[],requestPhases=new WeakMap();
 let phase='startup';
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>requestPhases.set(r,phase));
 page.on('response',r=>{if(new URL(r.url()).pathname==='/api/world'){
  const entry={phase:requestPhases.get(r.request())||'unknown',url:r.url(),status:r.status()};worldResponses.push(entry);
  headerPromises.push(r.body().then(body=>{entry.rawUTF8=body.toString('utf8');entry.sha256=crypto.createHash('sha256').update(body).digest('hex')}));
 }});
 page.on('response',r=>{if(r.url().includes('/height/')){const entry={phase:requestPhases.get(r.request())||'unknown',url:r.url(),status:r.status(),receivedMs:Date.now(),headers:null};responses.push(entry);headerPromises.push(r.allHeaders().then(h=>{entry.headers=h}));if(r.status()>=400)headerPromises.push(r.text().then(body=>{entry.errorBody=body}).catch(()=>{}))}});
 const report={profile,seed,centre:[x,y],sourceHashes:Object.fromEntries(['index.html','terrain_lod.js','terrain_renderer.js','terrain_inference.py','terrain_nn_constants.py','terrain_server.py','benchmark_runtime_navigation.cjs'].map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')])),notes:['Actual Chrome/server; response cache headers record physical cache reuse.','Automatic world preparation and predictive prefetch disabled for this probe.','Other clients and desktop GPU work may still be active.','Stage times measure first sampled submitted draw, not GPU presentation.','Completion confirms final camera coverage after draining two animation frames.','Reconstruct the Python manifest from worldResponses.rawUTF8; worldManifest is the client JSON representation and may normalize integral floats.'],stages:[],responses,worldResponses,errors};
 try{
  report.serverBefore=await (await fetch('http://127.0.0.1:8765/api/status')).json();
  const opened=Date.now();
  await page.goto(`http://127.0.0.1:8765/?seed=${seed}&profile=${profile}&prepare=0`);
  await page.waitForFunction(()=>terrainDebug.snapshot().overview&&terrainDebug.snapshot().rendererReady,{},{timeout:120000});
  report.overviewReadyMs=Date.now()-opened;
  report.worldManifest=await page.evaluate(()=>world?.world_manifest);
  await page.locator('#prefetch').uncheck();
  assert.equal((await page.evaluate(()=>terrainDebug.snapshot())).backend,'webgpu');
  if(process.env.TERRAIN_RUNTIME_LOD11==='1'){
   phase='lod11';const begin=Date.now();await page.evaluate(()=>zoom(1e-9));
   await page.waitForFunction(()=>{const s=terrainDebug.snapshot();return s.camera.lod===11&&s.visible.pending===0&&s.scheduler.inflight===0&&s.scheduler.queued===0},{},{timeout:120000});
   report.lod11=await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>resolve(terrainDebug.snapshot())))));
   report.lod11ReadyMs=Date.now()-begin;assert.equal(report.lod11.camera.mpp,61440);
   report.lod11Screenshot=output.replace(/\.json$/,'-world.png');await page.screenshot({path:report.lod11Screenshot});
   report.lod11ScreenshotSHA256=crypto.createHash('sha256').update(fs.readFileSync(report.lod11Screenshot)).digest('hex');
   report.notes.push('LOD11 screenshot shows the source stages in its snapshot; conditioning-preview is an input initialization, not the fully computed learned world.');
  }
  phase='native';const start=Date.now();
  const desiredEpoch=await page.evaluate(({x,y})=>{clearTimeout(timer);timer=null;cx=x;cy=y;mpp=30;draw();refresh();return cameraEpoch},{x,y});
  const seen=new Set(),deadline=start+180000;
  while(Date.now()<deadline){
   const s=await page.evaluate(()=>terrainDebug.snapshot());report.lastSnapshot=s;
   if(s.cameraEpoch>=desiredEpoch&&s.camera.lod===0){
    for(const lod of s.rendered.lods)if(lod<=4&&!seen.has(lod)){seen.add(lod);report.stages.push({lod,elapsedMs:Date.now()-start,snapshot:s})}
    if(s.visible.pending===0&&s.visible.lod===0&&s.scheduler.inflight===0&&s.scheduler.queued===0&&s.rendered.lods.length&&s.rendered.lods.every(l=>l===0)){
     const confirmed=await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>resolve(terrainDebug.snapshot())))));
     if(confirmed.visible.pending===0&&confirmed.visible.lod===0&&confirmed.scheduler.inflight===0&&confirmed.scheduler.queued===0&&confirmed.rendered.lods.every(l=>l===0)){
      report.completedMs=Date.now()-start;report.final=confirmed;break;
     }
    }
   }
   await page.waitForTimeout(50);
  }
  assert(report.final,'native view did not complete');
  await Promise.all(headerPromises);
  const nativeResponses=responses.filter(r=>r.phase==='native'&&r.status===200);
  const order=[...new Set(nativeResponses.map(r=>Number(new URL(r.url).pathname.split('/')[4])))];
  report.requestOrder=order;assert.deepEqual(order,[4,3,2,1,0]);
  assert(nativeResponses.every(r=>Number(new URL(r.url).pathname.split('/')[4])<=11));
  report.physicalCacheCounts=nativeResponses.reduce((a,r)=>{const k=r.headers['x-terrain-cache']||'unknown';a[k]=(a[k]||0)+1;return a},{});
  report.serverAfter=await (await fetch('http://127.0.0.1:8765/api/status')).json();
  report.screenshot=output.replace(/\.json$/,'.png');await page.screenshot({path:report.screenshot});
  report.screenshotSHA256=crypto.createHash('sha256').update(fs.readFileSync(report.screenshot)).digest('hex');
  assert.deepEqual(errors,[]);report.passed=true;
 }catch(e){report.passed=false;report.failure=String(e);throw e}
 finally{await Promise.all(headerPromises);fs.mkdirSync(require('node:path').dirname(output),{recursive:true});fs.writeFileSync(output,JSON.stringify(report,null,2));await browser.close();console.log(JSON.stringify({passed:report.passed,report:output,overviewReadyMs:report.overviewReadyMs,completedMs:report.completedMs,stageTimes:report.stages.map(s=>[s.lod,s.elapsedMs]),cache:report.physicalCacheCounts},null,2))}
})().catch(e=>{console.error(e);process.exitCode=1});
