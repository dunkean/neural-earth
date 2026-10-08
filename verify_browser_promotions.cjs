// Real browser: a cached conditioning preview must eventually promote after
// whole-world coarse preparation; presentation remains available meanwhile.
const {chromium}=require('E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');
const fs=require('node:fs'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--enable-unsafe-webgpu']});
 const page=await browser.newPage({viewport:{width:900,height:650}}),errors=[],samples=[];
 page.on('pageerror',e=>errors.push(e.message));
 const start=Date.now();
 try {
  await page.goto('http://127.0.0.1:8765/?seed=42&profile=natural&coarse_prepare=0');
  await page.waitForFunction(()=>typeof terrainDebug!=='undefined'&&terrainDebug.snapshot().overview,null,{timeout:120000});
  const statusBefore=await(await page.request.get('http://127.0.0.1:8765/api/status')).json();
  let snapshot;
  while(Date.now()-start<180000){
   snapshot=await page.evaluate(()=>terrainDebug.snapshot());
   samples.push({wall_ms:Date.now()-start,snapshot});
   if(snapshot.visible.pending===0&&snapshot.rendered.sources.length===1&&snapshot.rendered.sources[0]==='coarse-area-mean')break;
   await page.waitForTimeout(1000);
  }
  const after=await(await page.request.get('http://127.0.0.1:8765/api/status')).json();
  const passed=snapshot.visible.pending===0&&snapshot.rendered.sources.length===1&&snapshot.rendered.sources[0]==='coarse-area-mean'&&errors.length===0;
  const report={passed,wall_ms:Date.now()-start,samples,errors,nn_before:statusBefore.cuda_forward_calls,nn_after:after.cuda_forward_calls,chrome_version:browser.version(),implementation_sha256:Object.fromEntries(['index.html','terrain_lod.js','terrain_renderer.js','verify_browser_promotions.cjs'].map(n=>[n,require('node:crypto').createHash('sha256').update(fs.readFileSync(n)).digest('hex')])),note:'Seed 42 learned coarse already fully persisted; automatic preparation disabled. Polls live rendered source, not merely HTTP arrival. Existing preview and overview are retained during replacement. No cold-preparation or sustained-pan claim.'};
  fs.writeFileSync('E:/TerrainDiffusionRuntime/audit-implementation/browser-promotions.json',JSON.stringify(report,null,2));
  await page.screenshot({path:'E:/TerrainDiffusionRuntime/audit-implementation/browser-promotions.png'});
  assert(passed,'Visible world tiles did not all promote to learned coarse');
  assert.deepEqual(report.nn_after,report.nn_before,'Promotion required additional NN work');
  console.log(JSON.stringify({passed,wall_ms:report.wall_ms,samples:samples.length,initial_sources:samples[0].snapshot.rendered.sources,final_sources:snapshot.rendered.sources}));
 } finally {await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
