// Crossed REAL matrix: six biome families x four requested reliefs x four
// extents, always forced decoder LOD0. Source relief labels remain provisional
// until native measurements and human visual review confirm them.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const {loadPlaywright,browserExecutable,pythonExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..'),out=path.resolve(root,process.env.REVIEW_OUT||'output/render-v2-40km/real-matrix');
const selection=JSON.parse(fs.readFileSync(process.env.REVIEW_SELECTION||path.join(root,'output/render-v2-40km/real-matrix-24-selection.json'),'utf8'));
const wanted=process.env.REVIEW_CASES?.split(','),patches=selection.slots.filter(p=>p.available&&(!wanted||wanted.includes(p.id))).map(p=>({...p,relief:p.relief??p.requestedRelief}));
// Reuse complete decoder caches first; never call their old source labels a
// verified morphology simply because a capture is available.
patches.sort((a,b)=>Number(!!b.native_stats)-Number(!!a.native_stats));
const size=1024,allZooms=[['40km',40000],['10km',10000],['2.56km',2560],['640m',640]];
const requestedZooms=process.env.REVIEW_ZOOMS?.split(',');
const zooms=allZooms.filter(z=>!requestedZooms||requestedZooms.includes(z[0])),origin='http://127.0.0.1:8765';
assert(zooms.length,'No valid zoom requested');
const expectedFrames=zooms.length*3;
const sourceFiles=['web/terrain_render_v2.js','web/terrain_render3d.js','web/terrain_renderer.js','web/index.html'];
const identity=()=>Object.fromEntries(sourceFiles.map(n=>[n,crypto.createHash('sha256').update(fs.readFileSync(path.join(root,n))).digest('hex')]));
const codeIdentity=identity(),sameCode=x=>JSON.stringify(x)===JSON.stringify(codeIdentity);
const reportFile=path.join(out,'report.json'),previous=fs.existsSync(reportFile)?JSON.parse(fs.readFileSync(reportFile,'utf8')):null;
const results=previous?.worldIdentity===selection.worldIdentity&&sameCode(previous.codeIdentity)?previous.results:[];
const started=Date.now();let browser;
function save(){fs.writeFileSync(reportFile,JSON.stringify({label:`REAL ${patches.length} candidate sites x${zooms.length} zooms, decoder30m LOD0 enlarged; morphology acceptance is separate from render protocol`,worldIdentity:selection.worldIdentity,sourceCacheKey:selection.sourceCacheKey,codeIdentity,size,zooms,selection,results,reassignments:previous?.worldIdentity===selection.worldIdentity?previous.reassignments||[]:[],seconds:(Date.now()-started)/1000},null,2));}
(async()=>{
 fs.mkdirSync(out,{recursive:true});
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
  for(const patch of patches){
   assert(sameCode(identity()),'Production code changed during review; stop to avoid mixing versions');
   const old=results.find(r=>r.id===patch.id);
   const sameCenter=old&&JSON.stringify(old.selection.center_m)===JSON.stringify(patch.center_m);
   if(old?.protocolOk&&sameCenter&&old.frames.length===expectedFrames&&old.frames.every(f=>zooms.some(z=>z[0]===f.zoom)&&fs.existsSync(path.join(out,f.file)))){console.log(JSON.stringify({id:patch.id,reused:true,acceptance:old.acceptance}));continue;}
   if(old){
    const archive=path.join(out,'attempts',patch.id,String(Date.now()));fs.mkdirSync(archive,{recursive:true});
    fs.writeFileSync(path.join(archive,'entry.json'),JSON.stringify(old,null,2));
    for(const f of old.frames)if(fs.existsSync(path.join(out,f.file)))fs.copyFileSync(path.join(out,f.file),path.join(archive,f.file));
    results.splice(results.indexOf(old),1);
   }
   const page=await browser.newPage({viewport:{width:1200,height:1200}}),entry={id:patch.id,selection:patch,frames:[],errors:[],responses:[],negativeLodBlocked:[]};results.push(entry);save();
   page.on('pageerror',e=>entry.errors.push(e.stack));
   page.on('response',r=>{if(r.status()>=400&&!r.url().endsWith('favicon.ico'))entry.responses.push({status:r.status(),url:r.url()});});
   // A wheel event changes the camera before the LOD selector is reapplied.
   // Both controls run in one JS turn below; refuse any accidental transient
   // refinement subscription as an additional guard against new fine inference.
   await page.route('**/api/view',async route=>{
    const d=route.request().postDataJSON();
    if((d.tiles||[]).some(t=>t.lod<0)){entry.negativeLodBlocked.push(d.tiles.filter(t=>t.lod<0));return route.fulfill({status:409,contentType:'application/json',body:'{"accepted":false,"error":"Verification forbids negative LOD"}'});}
    await route.continue();
   });
   await page.route('**/height/**',async route=>{
    const match=new URL(route.request().url()).pathname.match(/\/height\/[^/]+\/\d+\/(-?\d+)\//);
    if(match&&+match[1]<0){entry.negativeLodBlocked.push(route.request().url());return route.fulfill({status:409,body:'Negative LOD refused'});}
    await route.continue();
   });
   const params=new URLSearchParams({share:'gHY4ZFzncjegep8JTpf.rulIR78fFpNhu4Tj2Y2',gpu:1,layer:'render-v2',view:'map',zoom:zooms[0][1]/size,position:patch.center_m.join(','),lod:0,size:[size,size].join(','),v2season:.5,hydro:0,hydroMonth:7,cache_gib:1,coarse_gpu:1});
   entry.url=origin+'/?'+params;
   try{
    await page.goto(entry.url,{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>window.terrainDebug?.snapshot().renderer?.render3d?.ready,null,{timeout:120000});
    await page.addStyleTag({content:'#hud,#mini,#legend{display:none!important}'});
    // Disable peripheral prefetch through its real control. Existing root
    // captures may still be completing elsewhere; this page requests only its view.
    await page.evaluate(()=>{const p=document.getElementById('prefetch');if(p.checked){p.checked=false;p.dispatchEvent(new Event('change'));}});
    async function settle(){
     const deadline=Date.now()+360000;let logged=0;
     while(Date.now()<deadline){
      const s=await page.evaluate(()=>terrainDebug.snapshot());
      if(s.visible.pending===0&&s.refinement.forced===0&&s.rendered.patches>0&&s.rendered.qualityLods.length&&s.rendered.lods.every(l=>l===0)&&s.rendered.qualityLods.every(l=>l===0)&&s.rendered.sources.every(t=>t==='decoder')&&s.rendered.sourceResolutions.every(r=>r===30)){
       await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));return page.evaluate(()=>terrainDebug.snapshot());
      }
      if(Date.now()-logged>20000){logged=Date.now();console.log(JSON.stringify({id:patch.id,pending:s.visible.pending,sources:s.rendered.sources,quality:s.rendered.qualityLods,elapsed:Math.round((Date.now()-started)/1000)}));}
      await page.waitForTimeout(500);
     }
     throw Error('Exact decoder LOD0 did not finish in6 minutes');
    }
    async function mode(next){
     if((await page.evaluate(()=>terrainDebug.snapshot().mode))===next)return;
     await page.locator('#layerPanel > summary').click();await page.locator(`#layerPanel button[data-mode="${next}"]`).click();await page.waitForFunction(m=>terrainDebug.snapshot().mode===m,next);
    }
    async function season(value){await page.evaluate(value=>{const c=document.getElementById('renderV2Season');c.value=String(value);c.dispatchEvent(new Event('input'));},value);await page.waitForFunction(v=>terrainDebug.snapshot().renderV2.season===v,value);}
    async function capture(zoom,extent,tag){
     const state=await settle(),dimensions=await page.locator('#gpuMap').evaluate(c=>({width:c.width,height:c.height}));
     assert.deepEqual(dimensions,{width:size,height:size});assert(Math.abs(state.camera.mpp*size-extent)<1e-5);assert(Math.abs(state.camera.cx-patch.center_m[0])<1e-5&&Math.abs(state.camera.cy-patch.center_m[1])<1e-5);
     assert(state.cache.keys.every(k=>k.includes(selection.worldIdentity)));assert(sameCode(identity()),'Production changed before capture');
     const file=`${patch.id}-${zoom}-${tag}.png`;await page.locator('#gpuMap').screenshot({path:path.join(out,file)});
     entry.frames.push({zoom,tag,file,dimensions,extentMetres:extent,decoderResolutionMetres:30,state});save();
    }
    for(const [zoom,extent] of zooms){
     if(zoom!==zooms[0][0]){
      await page.evaluate(target=>{
       const s=terrainDebug.snapshot(),view=document.getElementById('viewport'),r=view.getBoundingClientRect();
       view.dispatchEvent(new WheelEvent('wheel',{deltaY:Math.log(target/s.camera.mpp)/.0015,clientX:r.left+r.width/2,clientY:r.top+r.height/2,bubbles:true,cancelable:true}));
       const forced=document.getElementById('forcedLod');forced.value='0';forced.dispatchEvent(new Event('change'));
      },extent/size);
     }
     await mode('render-v2');await season(.5);await capture(zoom,extent,'v2-july');
     await mode('render3d');await capture(zoom,extent,'render3d-july');
     await mode('render-v2');await season(0);await capture(zoom,extent,'v2-january');
     if(zoom==='40km'){
      // Measure the actual completed 40km DEM and reconstruct source climate
      // at its heights. This never asks the model/server to generate data.
      const input=path.join(out,'native-input.json');fs.writeFileSync(input,JSON.stringify({results:[{id:patch.id,ok:true,selection:{bounds_m:patch.bounds_m,center_m:patch.center_m},frames:[entry.frames[0]]}]}));
      execFileSync(pythonExecutable(),['tools/verification/select_render_v2_real_matrix.py','--measure',input],{cwd:root,env:{...process.env,TERRAIN_RENDER_DEVICE:'cpu'},timeout:120000,maxBuffer:1024*1024});
      const measured=JSON.parse(fs.readFileSync(path.join(out,'native-metrics.json'),'utf8'))[0];assert(measured,'No native morphology measurement');
      entry.nativeStats=measured.stats;entry.nativeRelief=measured.relief;
      entry.familyDominance=patch.biome_codes.reduce((n,c)=>n+(measured.stats.biome_fractions_land[c]||0),0);
      entry.representativeFamily=entry.familyDominance>=.70;entry.usableMixedBiome=entry.representativeFamily&&entry.familyDominance<.8;
      if(entry.usableMixedBiome)entry.mixedBiomeNote='Representative mixed transition; strict80% biome acceptance remains false. Fractions are in nativeStats.biome_fractions_land.';
      entry.acceptance={biomeFamilyMatched:entry.familyDominance>=.8,nativeReliefMatched:measured.relief===patch.relief,visualReliefReview:'pending',
       status:entry.familyDominance<.8?'biome-mismatch':measured.relief!==patch.relief?'relief-mismatch-needs-replacement':'native-metrics-match-visual-review-pending'};save();
     }
    }
    entry.protocolOk=entry.frames.length===expectedFrames&&!entry.errors.length&&!entry.responses.some(r=>r.status!==409)&&!entry.negativeLodBlocked.length;
   }catch(error){entry.protocolOk=false;entry.failure=error.stack;entry.last=await page.evaluate(()=>window.terrainDebug?.snapshot()).catch(()=>null);console.log(JSON.stringify({id:patch.id,failure:String(error)}));}
   finally{await page.close();save();}
   console.log(JSON.stringify({id:patch.id,protocolOk:entry.protocolOk,frames:entry.frames.length,acceptance:entry.acceptance,seconds:Math.round((Date.now()-started)/1000)}));
  }
 }finally{if(browser)await browser.close();save();}
 const requested=results.filter(r=>patches.some(p=>p.id===r.id));
 console.log(JSON.stringify({protocolOk:requested.every(r=>r.protocolOk),sites:requested.length,frames:requested.reduce((n,r)=>n+r.frames.length,0),nativeMatched:requested.filter(r=>r.acceptance?.biomeFamilyMatched&&r.acceptance.nativeReliefMatched).length,out}));
 if(requested.some(r=>!r.protocolOk))process.exitCode=1;
})().catch(error=>{console.error(error);process.exitCode=1;});
