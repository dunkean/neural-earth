const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
// Actual Chrome selector changes, equal camera/seed and binary height comparison.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const out=process.env.TERRAIN_COMPARISON_OUTPUT||'E:/TerrainDiffusionRuntime/inference-engine-20261008/ui-comparison';
const base=process.env.TERRAIN_RUNTIME_URL||'http://127.0.0.1:8765';
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--enable-unsafe-webgpu']});
 const page=await browser.newPage({viewport:{width:900,height:650}}),errors=[],payloads={},pending=[];
 let phase='startup';const phases=new WeakMap(),report={status:'running',states:[],errors};
 const save=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>phases.set(r,phase));
 page.on('response',r=>{const u=new URL(r.url());if(r.status()===200&&/\/height\/natural-v1\/\d+\/4\//.test(u.pathname)){
  const label=phases.get(r.request());pending.push((async()=>{const bytes=await r.body(),headers=await r.allHeaders(),parts=u.pathname.split('/'),key=parts.slice(-2).join('/').replace('.bin','');
   (payloads[label]??={})[key]={bytes,width:Number(headers['x-terrain-width']),climateWidth:Number(headers['x-terrain-climate-width']),method:headers['x-terrain-coarse-interpolation']};
   fs.writeFileSync(path.join(out,label+'-'+key.replace('/','_')+'.bin'),bytes);
  })());}});
 try{
  await page.goto(base+'/?seed=42&profile=natural&coarse_prepare=0&nn_engine=exact&coarse_interpolation=bilinear');
  await page.waitForFunction(()=>terrainDebug.snapshot().overview&&terrainDebug.snapshot().rendererReady,{},{timeout:120000});
  await page.locator('#runtimePanel').evaluate(el=>el.open=true);await page.locator('#prefetch').uncheck();
  phase='exact-bilinear';
  await page.evaluate(()=>{cx=-921600;cy=614400;mpp=480;draw();schedule()});
  async function settled(engine,method){
   await page.waitForFunction(({engine,method})=>{const s=terrainDebug.snapshot();return s.world==='42'&&!document.getElementById('nnEngine').disabled&&s.nnEngine===engine&&s.coarseInterpolation===method&&s.camera.lod===4&&s.visible.pending===0&&s.scheduler.inflight===0&&s.scheduler.queued===0&&s.rendered.qualityLods.length===1&&s.rendered.qualityLods[0]===4},{engine,method},{timeout:120000});
   await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
   await Promise.all(pending.splice(0));
   const snapshot=await page.evaluate(()=>terrainDebug.snapshot());
   assert.equal(snapshot.world,'42');assert.equal(snapshot.camera.cx,-921600);assert.equal(snapshot.camera.cy,614400);assert.equal(snapshot.camera.mpp,480);
   assert.equal(errors.length,0);report.states.push(snapshot);save();
   await page.screenshot({path:path.join(out,phase+'.png')});
  }
  await settled('exact','bilinear');
  phase='exact-monotone';await page.locator('#runtimePanel').evaluate(el=>el.open=true);await page.selectOption('#coarseInterpolation','monotone');await settled('exact','monotone');
  phase='reference-monotone';await page.locator('#runtimePanel').evaluate(el=>el.open=true);await page.selectOption('#nnEngine','reference');await settled('reference','monotone');
  phase='reference-bilinear';await page.locator('#runtimePanel').evaluate(el=>el.open=true);await page.selectOption('#coarseInterpolation','bilinear');await settled('reference','bilinear');
  report.fidelity={};
  for(const method of ['bilinear','monotone']){
   const a=payloads['exact-'+method],b=payloads['reference-'+method],keys=Object.keys(a||{}).filter(k=>b?.[k]);assert(keys.length>0,'No matched actual LOD4 payload');
   for(const k of keys){assert.equal(a[k].method,method);assert.equal(b[k].method,method);assert(a[k].bytes.equals(b[k].bytes),'NN backend changed LOD4 height/climate');}
   report.fidelity[method]={matchedTiles:keys.length,entireHeightClimateByteExact:true};
  }
  report.methodsDiffer=Object.keys(payloads['exact-bilinear']).some(k=>payloads['exact-monotone']?.[k]&&!payloads['exact-bilinear'][k].bytes.equals(payloads['exact-monotone'][k].bytes));
  assert(report.methodsDiffer,'Interpolation choice had no effect');
  phase='restored-exact';await page.locator('#runtimePanel').evaluate(el=>el.open=true);await page.selectOption('#nnEngine','exact');await settled('exact','bilinear');
  report.backends=await (await fetch(base+'/api/inference-backends')).json();report.status='complete';
 }catch(e){report.status='failed';report.error=String(e.stack||e);await page.screenshot({path:path.join(out,'failed.png')}).catch(()=>{});throw e;}
 finally{save();await browser.close();}
 console.log(JSON.stringify({status:report.status,fidelity:report.fidelity,methodsDiffer:report.methodsDiffer}));
})().catch(e=>{console.error(e);process.exitCode=1});
