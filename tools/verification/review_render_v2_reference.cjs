// Final REAL terrain corpus. Default --plan does not launch a browser/model.
// --freeze records immutable source hashes; --run is used only after root GO.
// Captures live in a new versioned directory and never overwrite old baselines.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const {loadPlaywright,browserExecutable,pythonExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..'),base=path.resolve(root,process.env.REVIEW_REFERENCE_BASE||'output/render-v2-reference');
const manifestFile=path.join(base,'manifest.json'),freezeFile=path.join(base,'freeze.json');
const manifest=JSON.parse(fs.readFileSync(manifestFile,'utf8'));
const digest=x=>crypto.createHash('sha256').update(x).digest('hex');
const manifestHash=digest(fs.readFileSync(manifestFile));
const productionFiles=[...fs.readdirSync(path.join(root,'web')).filter(f=>f.endsWith('.js')||f==='index.html').map(f=>'web/'+f),
 ...fs.readdirSync(path.join(root,'backend')).filter(f=>/^terrain_(render|hydrology).*\.py$/.test(f)||['terrain_server.py','terrain_koppen.py','terrain_soil.py','terrain_glacier.py'].includes(f)).map(f=>'backend/'+f)].sort();
const identity=()=>Object.fromEntries(productionFiles.map(f=>[f,digest(fs.readFileSync(path.join(root,f)))]));
const codeIdentity=identity(),codeHash=digest(JSON.stringify(codeIdentity));
const protocolFiles=['tools/verification/review_render_v2_reference.cjs','tools/verification/select_render_v2_reference.py','tools/verification/select_render_v2_real_matrix.py',
 'tools/verification/review_render_v2_hydrology_reference.cjs','tools/verification/select_render_v2_hydrology_reference.py'].filter(f=>fs.existsSync(path.join(root,f)));
const protocolIdentity=()=>Object.fromEntries(protocolFiles.map(f=>[f,digest(fs.readFileSync(path.join(root,f)))]));
const protocolHashes=protocolIdentity(),protocolHash=digest(JSON.stringify(protocolHashes));
const wanted=process.env.REVIEW_CASES?.split(',');
const sites=manifest.sites.filter(s=>!wanted||wanted.includes(s.id));
assert(sites.length,'No requested sites');
if(wanted)assert.equal(sites.length,wanted.length,'Unknown/duplicate requested site id');
assert.deepEqual(manifest.styles,['v2-july','render3d-july','v2-january']);
const expectedSiteFrames=manifest.zooms.length*manifest.styles.length;
const args=new Set(process.argv.slice(2)),plan={label:manifest.label,sites:sites.length,zooms:manifest.zooms,
 styles:manifest.styles,frames:sites.length*manifest.zooms.length*manifest.styles.length,
 sourceIdentity:manifest.worldIdentity,codeHash,protocolHash,manifestHash,output:base,
 command:'--freeze then --run only after production code is approved/frozen'};
if(args.has('--freeze')){
 fs.mkdirSync(base,{recursive:true});
 const frozen={frozenAt:new Date().toISOString(),codeHash,protocolHash,manifestHash,worldIdentity:manifest.worldIdentity,codeIdentity,protocolIdentity:protocolHashes};
 fs.writeFileSync(freezeFile,JSON.stringify(frozen,null,2));console.log(JSON.stringify({...plan,frozen:true}));
}else if(!args.has('--run'))console.log(JSON.stringify({...plan,planOnly:true}));
else (async()=>{
 assert(fs.existsSync(freezeFile),'Missing freeze.json; do not start a final pass on changing source');
 const frozen=JSON.parse(fs.readFileSync(freezeFile,'utf8'));
 assert.equal(frozen.codeHash,codeHash,'Production differs from frozen version');
 assert.equal(frozen.manifestHash,manifestHash,'Manifest differs from frozen selection');
 assert.equal(frozen.protocolHash,protocolHash,'Capture/measurement protocol differs from frozen version');
 const runId=codeHash.slice(0,12)+'-'+manifestHash.slice(0,8)+'-'+protocolHash.slice(0,8),out=path.join(base,'captures',runId);
 fs.mkdirSync(out,{recursive:true});
 const reportFile=path.join(out,'report.json'),previous=fs.existsSync(reportFile)?JSON.parse(fs.readFileSync(reportFile,'utf8')):null;
 const results=previous?.results||[],started=Date.now();
 const origin=process.env.REVIEW_ORIGIN||'http://127.0.0.1:8765',size=manifest.size;
 let browser,worldEvidence;
 const assertFrozen=()=>{
  assert.equal(digest(JSON.stringify(identity())),codeHash,'Production code changed: stop rather than mix image versions');
  assert.equal(digest(fs.readFileSync(manifestFile)),manifestHash,'Manifest changed during capture');
  assert.equal(digest(JSON.stringify(protocolIdentity())),protocolHash,'Capture/measurement protocol changed during capture');
 };
 function save(){
  const report={label:manifest.label,runId,frozen,manifest,worldEvidence,size,zooms:manifest.zooms,
   results,secondsThisInvocation:(Date.now()-started)/1000,
   summary:{sites:results.length,protocolPassed:results.filter(r=>r.protocolOk).length,
    frames:results.reduce((n,r)=>n+r.frames.length,0),expectedSites:manifest.sites.length,expectedFrames:manifest.expectedFrames,
    nativeMorphologyMatched:results.filter(r=>r.acceptance?.nativeReliefMatched).length,
    visualAcceptance:'pending-human-review-not-inferred-from-coverage-or-protocol'}};
  fs.writeFileSync(reportFile,JSON.stringify(report,null,2));
  fs.writeFileSync(path.join(base,'report.json'),JSON.stringify(report,null,2));
 }
 const info=await fetch(origin+'/api/world?'+new URLSearchParams({seed:manifest.seed,world_profile:manifest.generationProfile}));
 assert(info.ok,'World metadata GET failed');const world=await info.json();
 assert.equal(world.world_identity,manifest.worldIdentity,'World identity mismatch');
 assert.deepEqual(world.generation_stages,manifest.generationStages,'Generation/source stage mismatch');
 worldEvidence={world_identity:world.world_identity,generation_profile:world.generation_profile,
  generation_stages:world.generation_stages,cache_profile:world.cache_profile,
  world_bounds:world.world_bounds,native_resolution:world.native_resolution,verifiedAt:new Date().toISOString()};
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
  for(const site of sites){
   assertFrozen();
   const old=results.find(r=>r.id===site.id);
   if(old?.protocolOk&&old.frames.length===expectedSiteFrames&&old.frames.every(f=>fs.existsSync(path.join(base,f.file))&&digest(fs.readFileSync(path.join(base,f.file)))===f.sha256)){
    console.log(JSON.stringify({id:site.id,reused:true,frames:expectedSiteFrames}));continue;
   }
   if(old){
    const archive=path.join(out,'attempts',site.id,String(Date.now()));fs.mkdirSync(archive,{recursive:true});
    fs.writeFileSync(path.join(archive,'entry.json'),JSON.stringify(old,null,2));
    for(const f of old.frames)if(fs.existsSync(path.join(base,f.file)))fs.copyFileSync(path.join(base,f.file),path.join(archive,path.basename(f.file)));
    results.splice(results.indexOf(old),1);
   }
   const page=await browser.newPage({viewport:{width:1200,height:1200},deviceScaleFactor:1});
   const entry={id:site.id,site,frames:[],errors:[],responses:[],forbiddenRefinements:[],startedAt:new Date().toISOString()};results.push(entry);save();
   page.on('pageerror',e=>entry.errors.push(e.stack));
   page.on('response',r=>{if(r.status()>=400&&!r.url().endsWith('favicon.ico'))entry.responses.push({status:r.status(),url:r.url()});});
   if(manifest.hydrologyRequired)await page.addInitScript(()=>{
    // Link the response headers to the exact ArrayBuffer uploaded to WebGPU,
    // and retain read-only references for class/authority measurements.
    const buffers=new WeakMap(),uploads=new Map();
    const originalFetch=window.fetch;
    window.fetch=async(...args)=>{
     const response=await originalFetch(...args);
     if(new URL(response.url,location.href).pathname.startsWith('/height/')){
      const read=response.arrayBuffer.bind(response),headers=Object.fromEntries(response.headers.entries()),url=response.url;
      response.arrayBuffer=async()=>{const bytes=await read();buffers.set(bytes,{url,headers});return bytes;};
     }return response;
    };
    let create;
    Object.defineProperty(window,'createTerrainRenderer',{configurable:true,get(){return create;},set(fn){
     create=async(...args)=>{
      const renderer=await fn(...args);if(!renderer)return renderer;
      const upload=renderer.uploadTile;
      renderer.uploadTile=function(key,heights,options={}){
       const result=upload.call(this,key,heights,options);
       uploads.set(key,{...buffers.get(heights.buffer),options,hydrology:options.hydrology});return result;
      };
      window.__referenceHydro={renderer,uploads};return renderer;
     };
    }});
   });
   await page.route('**/api/view',async route=>{
    const d=route.request().postDataJSON();
    if((d.tiles||[]).some(t=>t.lod<0)){
     entry.forbiddenRefinements.push(d.tiles.filter(t=>t.lod<0));
     return route.fulfill({status:409,contentType:'application/json',body:'{"accepted":false,"error":"Reference forbids negativeLOD"}'});
    }await route.continue();
   });
   await page.route('**/height/**',async route=>{
    const m=new URL(route.request().url()).pathname.match(/\/height\/[^/]+\/\d+\/(-?\d+)\//);
    if(m&&+m[1]<0){entry.forbiddenRefinements.push(route.request().url());return route.fulfill({status:409,body:'NegativeLOD forbidden'});}
    await route.continue();
   });
   const initial=manifest.zooms[0];
   const params=new URLSearchParams({share:manifest.share,gpu:1,layer:'render-v2',view:'map',zoom:initial.extentMetres/size,
    position:site.center_m.join(','),lod:initial.lod,size:[size,size].join(','),v2season:.5,hydro:site.hydro||0,hydroMonth:7,
    ...(site.hydroIntermittent?{hydroIntermittent:1}:{}),cache_gib:1,coarse_gpu:1});
   entry.url=origin+'/?'+params;
   try{
    await page.goto(entry.url,{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>window.terrainDebug?.snapshot().renderer?.render3d?.ready,null,{timeout:120000});
    await page.addStyleTag({content:'#hud,#mini,#minimap,#legend,#stats,#status,.mapHud{visibility:hidden!important}'});
    await page.evaluate(({hydro,intermittent})=>{
     for(const [id,checked] of [['prefetch',false],['grid',false],['renderSea',true]]){
      const control=document.getElementById(id);if(control.checked!==checked){control.checked=checked;control.dispatchEvent(new Event('change'));}
     }
     const depth=document.getElementById('refinementDepth');depth.value='0';depth.dispatchEvent(new Event('change'));
     for(const [name,value] of Object.entries({enabled:hydro,intermittent,lakes:true,terrainDetail:false})){
      const control=document.querySelector(`[data-hydro="${name}"]`);
      if(control.checked!==value){control.checked=value;control.dispatchEvent(new Event('change'));}
     }
    },{hydro:!!site.hydro,intermittent:!!site.hydroIntermittent});
    function hydrologySettled(s){
     if(!manifest.hydrologyRequired)return true;
     const h=s.hydrology,d=s.renderer?.riverDisplay;
     if(!h||h.status!=='ready'||h.fetching||!h.settings.enabled||!h.geometryCurrent)return false;
     if(s.mode==='render3d')return true;
     return d?.ready===true&&d.mode==='regional-mask'&&!d.error&&h.riverDisplay?.active===true
      &&d.frameKey===h.riverDisplay.frameKey&&d.month===h.settings.month;
    }
    async function settle(z){
     const deadline=Date.now()+Number(process.env.REVIEW_TIMEOUT_MS||480000);let logged=0;
     while(Date.now()<deadline){
      assertFrozen();const s=await page.evaluate(()=>terrainDebug.snapshot());
      const expectedStage=z.lod===3?'latent':'decoder';
      const hydroReady=hydrologySettled(s);
      if(hydroReady&&s.visible.pending===0&&s.refinement.forced===z.lod&&s.refinement.renderSea&&s.rendered.patches>0
       &&s.rendered.lods.length&&s.rendered.lods.every(l=>l===z.lod)
       &&s.rendered.qualityLods.length&&s.rendered.qualityLods.every(l=>l===z.lod)
       &&s.rendered.sources.length&&s.rendered.sources.every(t=>t===expectedStage)
       &&s.rendered.sourceResolutions.length&&s.rendered.sourceResolutions.every(r=>r===z.sourceResolutionMetres)){
       await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
       const final=await page.evaluate(()=>terrainDebug.snapshot());
       if(final.visible.pending===0&&final.mode===s.mode&&hydrologySettled(final))return final;
      }
      if(Date.now()-logged>20000){logged=Date.now();console.log(JSON.stringify({id:site.id,zoom:z.id,pending:s.visible.pending,
       quality:s.rendered.qualityLods,sources:s.rendered.sources,resolutions:s.rendered.sourceResolutions,
       hydrology:s.hydrology,forced:s.refinement.forced,renderSea:s.refinement.renderSea,
       elapsed:Math.round((Date.now()-started)/1000)}));}
      await page.waitForTimeout(500);
     }throw Error('Requested real '+z.sourceResolutionMetres+'m source did not settle in time');
    }
    async function mode(next){
     if(await page.evaluate(()=>terrainDebug.snapshot().mode)===next)return;
     const panel=page.locator('#layerPanel');if(!await panel.evaluate(e=>e.open))await panel.locator(':scope > summary').click();
     await page.locator(`#layerPanel button[data-mode="${next}"]`).click();
     await page.waitForFunction(m=>terrainDebug.snapshot().mode===m,next);
    }
    async function season(value){
     await page.evaluate(v=>{const c=document.getElementById('renderV2Season');c.value=String(v);c.dispatchEvent(new Event('input'));},value);
     await page.waitForFunction(v=>terrainDebug.snapshot().renderV2.season===v,value);
    }
    async function capture(z,tag){
     const state=await settle(z),dimensions=await page.locator('#gpuMap').evaluate(c=>({width:c.width,height:c.height}));
     assert.deepEqual(dimensions,{width:size,height:size});
     assert(Math.abs(state.camera.mpp*size-z.extentMetres)<1e-5,'Wrong extent');
     assert(Math.abs(state.camera.cx-site.center_m[0])<1e-5&&Math.abs(state.camera.cy-site.center_m[1])<1e-5,'Camera moved');
     assert.equal(state.world,manifest.seed);assert.equal(state.generationProfile,manifest.generationProfile);
     assert(state.cache.keys.length&&state.cache.keys.every(k=>k.includes(manifest.worldIdentity)),'Wrong cache identity');
     assert.equal(state.backend,'webgpu');assert.equal(state.refinement.forced,z.lod);
     assert.equal(state.mode,tag==='render3d-july'?'render3d':'render-v2');
     assert.equal(state.renderV2.season,tag==='v2-january'?0:.5);
     let hydrologyEvidence;
     if(manifest.hydrologyRequired){
      hydrologyEvidence=await page.evaluate(async({extent,tag})=>{
       const probe=window.__referenceHydro,s=terrainDebug.snapshot();
       const bounds=[s.camera.cx-extent/2,s.camera.cy-extent/2,s.camera.cx+extent/2,s.camera.cy+extent/2];
       const summaries=[],seen=new Set();let lakes=0,rivers=0,dry=0,authority=0;
       for(const rect of probe.renderer.lastDrawnRects||[]){
        if(seen.has(rect.key))continue;seen.add(rect.key);
        const upload=probe.uploads.get(rect.key);if(!upload){summaries.push({key:rect.key,missing:true});continue;}
        const o=upload.options,water=upload.hydrology,width=o.width||304,halo=o.halo??24,res=o.metresPerSample;
        let localLake=0,localRiver=0,localDry=0,localAuthority=0;
        if(water)for(let y=halo;y<width-halo;y++)for(let x=halo;x<width-halo;x++){
         const wx=o.origin[0]+(x+.5)*res,wy=o.origin[1]+(y+.5)*res;
         if(wx<bounds[0]||wx>=bounds[2]||wy<bounds[1]||wy>=bounds[3])continue;
         const i=(y*width+x)*4;if(water[i+3]>.5)localAuthority++;
         if(water[i+1]>0&&water[i+3]>.5){if(water[i+2]>2.5)localRiver++;else if(water[i+2]>1.5)localLake++;else localDry++;}else localDry++;
        }
        lakes+=localLake;rivers+=localRiver;dry+=localDry;authority+=localAuthority;
        summaries.push({key:rect.key,url:upload.url,hydroState:upload.headers?.['x-terrain-hydro-state'],
         hydroId:upload.headers?.['x-terrain-hydro-id'],month:upload.headers?.['x-terrain-hydro-month'],
         hydroLayers:upload.headers?.['x-terrain-hydro-layers'],maxCut:upload.headers?.['x-terrain-hydro-max-cut-m'],
         lakeSamples:localLake,riverSamples:localRiver,drySamples:localDry,authoritySamples:localAuthority});
       }
       const canvas=document.getElementById('annotations'),pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
       let overlayPixels=0;for(let i=3;i<pixels.length;i+=4)if(pixels[i])overlayPixels++;
       const physical=typeof hydrologyState==='function'?hydrologyState():null;
       // Read the actual submitted regional mask, including the valid empty
       // 1x1 case. Source class3 samples no longer prove displayed river water.
       let maskReadback=null;
       if(tag!=='render3d-july'){
        const r=probe.renderer,mask=r.riverMask;
        if(!mask)throw Error('Missing actual regional river mask texture');
        const width=mask.width,height=mask.height,rowBytes=Math.ceil(width*8/256)*256;
        const buffer=r.device.createBuffer({size:rowBytes*height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        try{
         const encoder=r.device.createCommandEncoder();
         encoder.copyTextureToBuffer({texture:mask},{buffer,bytesPerRow:rowBytes,rowsPerImage:height},[width,height]);
         r.device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);
         const pixels=new Uint16Array(buffer.getMappedRange());let wetPixels=0,solidPixels=0,coveragePixels=0,maxCoverage=0;
         const half=v=>{const e=(v>>10)&31,m=v&1023;return e?Math.pow(2,e-15)*(1+m/1024):m*Math.pow(2,-24);};
         for(let y=0;y<height;y++)for(let x=0;x<width;x++){
          const value=pixels[y*rowBytes/2+x*4];if(!value)continue;
          const cover=half(value);wetPixels++;if(cover>=.5)solidPixels++;coveragePixels+=cover;maxCoverage=Math.max(maxCoverage,cover);
         }
         maskReadback={dimensions:[width,height],format:'rgba16float',wetPixels,solidPixels,coveragePixels,maxCoverage};
        }finally{buffer.unmap();buffer.destroy();}
       }
       return{status:s.hydrology.status,fetching:s.hydrology.fetching,month:s.hydrology.settings.month,
        summaries,lakeSamples:lakes,riverSamples:rivers,drySamples:dry,authoritySamples:authority,
        overlayPixels,physicalWater:physical?.physicalWater,physicalRects:physical?.physicalRects?.length||0,
        lakeOnly:lakes>0&&rivers===0,geometryCurrent:s.hydrology.geometryCurrent,
        queryRevision:s.hydrology.queryRevision,currentTerrainRevision:s.hydrology.currentTerrainRevision,
        regionalDisplay:s.renderer?.riverDisplay,networkDisplay:s.hydrology.riverDisplay,maskReadback};
      },{extent:z.extentMetres,tag});
      assert.equal(hydrologyEvidence.status,'ready');assert.equal(hydrologyEvidence.fetching,false);
      if(tag!=='render3d-july'){
       assert(hydrologyEvidence.geometryCurrent,'Regional geometry is stale relative to actual terrain arrivals');
       const display=hydrologyEvidence.regionalDisplay;
       assert(display?.ready&&display.mode==='regional-mask'&&!display.error,'Regional mask is pending or in legacy fallback');
       assert.equal(display.frameKey,hydrologyEvidence.networkDisplay?.frameKey,'Overlay and GPU regional geometry disagree');
       assert.equal(display.month,tag==='v2-january'?1:7,'Regional mask month mismatch');
       assert(hydrologyEvidence.maskReadback,'Missing readback of actual regional mask');
       if(display.empty){assert.deepEqual(display.dimensions,[1,1]);assert.equal(hydrologyEvidence.maskReadback.wetPixels,0);}
       assert(hydrologyEvidence.summaries.length&&hydrologyEvidence.summaries.every(s=>s.hydroState==='ready'),'Actual uploaded V2 tile headers must all be hydro ready');
       assert(hydrologyEvidence.summaries.every(s=>s.hydroLayers==='4'&&Number(s.month)===(tag==='v2-january'?1:7)),'V2 uploaded water raster/month mismatch');
       assert(hydrologyEvidence.summaries.every(s=>Number(s.maxCut)===0),'V2 lake display must not carve');
       if(site.id==='lake-temperate'&&z.id==='160m')assert(hydrologyEvidence.lakeSamples>0,'Targeted Dfc lake has no actual submerged raster samples');
       if(site.record_type==='lake'&&z.id==='160m'&&hydrologyEvidence.lakeOnly){
        assert(hydrologyEvidence.physicalWater,'Lake-only uploads must suppress the vector overlay');
        assert.equal(hydrologyEvidence.overlayPixels,0,'Lake-only view still has painted vector water');
       }
      }
     }
     const filename=`${site.id}-${z.id}-${tag}.png`,file=path.relative(base,path.join(out,filename)).replaceAll('\\','/');
     await page.locator(manifest.hydrologyRequired?'#viewport':'#gpuMap').screenshot({path:path.join(base,file)});assertFrozen();
     const png=fs.readFileSync(path.join(base,file));assert.equal(png.subarray(1,4).toString('ascii'),'PNG');
     const imageDimensions={width:png.readUInt32BE(16),height:png.readUInt32BE(20)};assert.deepEqual(imageDimensions,{width:size,height:size});
     entry.frames.push({zoom:z.id,tag,file,sha256:digest(png),dimensions,imageDimensions,
      extentMetres:z.extentMetres,metresPerPixel:z.extentMetres/size,requestedLod:z.lod,
      requiredSourceResolutionMetres:z.sourceResolutionMetres,state,...(hydrologyEvidence?{hydrologyEvidence}:{})});save();
    }
    for(const z of manifest.zooms){
     if(z!==initial)await page.evaluate(({mpp,lod})=>{
      const s=terrainDebug.snapshot(),view=document.getElementById('viewport'),r=view.getBoundingClientRect();
      view.dispatchEvent(new WheelEvent('wheel',{deltaY:Math.log(mpp/s.camera.mpp)/.0015,clientX:r.left+r.width/2,clientY:r.top+r.height/2,bubbles:true,cancelable:true}));
      const forced=document.getElementById('forcedLod');forced.value=String(lod);forced.dispatchEvent(new Event('change'));
     },{mpp:z.extentMetres/size,lod:z.lod});
     await mode('render-v2');await season(.5);await capture(z,'v2-july');
     await mode('render3d');await capture(z,'render3d-july');
     await mode('render-v2');await season(0);await capture(z,'v2-january');
     if(z.id==='40km'){
      const input=path.join(out,site.id+'-native-input.json');
      const frame=entry.frames.find(f=>f.zoom==='40km');
      fs.writeFileSync(input,JSON.stringify({results:[{id:site.id,site,frames:[frame]}]}));
      execFileSync(pythonExecutable(),['tools/verification/select_render_v2_reference.py','--measure',input,
       ...(manifest.worldInfoFile?['--world-info',manifest.worldInfoFile]:[])],
       {cwd:root,env:{...process.env,TERRAIN_RENDER_DEVICE:'cpu'},timeout:120000,maxBuffer:1024*1024});
      const measurement=JSON.parse(fs.readFileSync(input.replace('.json','-metrics.json'),'utf8'))[0];
      entry.nativeStats=measurement.stats;entry.nativeRelief=measurement.relief;
      entry.familyDominance=site.biome_codes.reduce((n,c)=>n+(measurement.stats.biome_fractions_land[c]||0),0);
      const pureWater=measurement.stats.land_pixels===0;
      entry.acceptance={biomeFamilyMatched:pureWater?site.biomeFamily==='OCEAN':entry.familyDominance>=.8,
       representativeMixedBiome:!pureWater&&entry.familyDominance>=.7&&entry.familyDominance<.8,
       nativeReliefMatched:site.requestedRelief===null?null:measurement.relief===site.requestedRelief,
       requestedRelief:site.requestedRelief,measuredRelief:measurement.relief,
       visualAcceptance:'pending-human-review',status:'native-measured-separate-from-visual-quality'};save();
     }
     if(z.id==='160m'&&site.featureProvenance){
      const [x,y]=site.center_m,focusSite={...site,bounds_m:[x-80,y-80,x+80,y+80]};
      const input=path.join(out,site.id+'-near-focus-input.json');
      fs.writeFileSync(input,JSON.stringify({results:[{id:site.id,site:focusSite,frames:[entry.frames.find(f=>f.zoom==='160m')]}]}));
      execFileSync(pythonExecutable(),['tools/verification/select_render_v2_reference.py','--measure',input,
       ...(manifest.worldInfoFile?['--world-info',manifest.worldInfoFile]:[])],
       {cwd:root,env:{...process.env,TERRAIN_RENDER_DEVICE:'cpu'},timeout:120000,maxBuffer:1024*1024});
      entry.nearFocus=JSON.parse(fs.readFileSync(input.replace('.json','-metrics.json'),'utf8'))[0];
      entry.nearFocus.extentMetres=160;
      entry.nearFocus.geometryTargetMatched=site.id==='user-pack-ocean'?entry.nearFocus.stats.land_fraction<.1:
       entry.nearFocus.stats.land_fraction>.05&&entry.nearFocus.stats.land_fraction<.95;
      entry.nearFocus.visualMaterialReview='pending; ocean geometry alone does not prove visible sea ice or a rendered beach';save();
     }
    }
    entry.protocolOk=entry.frames.length===expectedSiteFrames&&!entry.errors.length&&!entry.responses.some(r=>r.status!==409)&&!entry.forbiddenRefinements.length;
   }catch(error){
    entry.protocolOk=false;entry.failure=error.stack;entry.last=await page.evaluate(()=>window.terrainDebug?.snapshot()).catch(()=>null);
    await page.screenshot({path:path.join(out,site.id+'-failure.png')}).catch(()=>{});
    console.log(JSON.stringify({id:site.id,failure:String(error)}));
   }finally{await page.close();entry.finishedAt=new Date().toISOString();save();}
   console.log(JSON.stringify({id:site.id,protocolOk:entry.protocolOk,frames:entry.frames.length,acceptance:entry.acceptance,seconds:Math.round((Date.now()-started)/1000)}));
   assertFrozen();
  }
 }finally{if(browser)await browser.close();save();}
 const requested=results.filter(r=>sites.some(s=>s.id===r.id));
 console.log(JSON.stringify({ok:requested.every(r=>r.protocolOk),sites:requested.length,frames:requested.reduce((n,r)=>n+r.frames.length,0),report:reportFile}));
 if(requested.some(r=>!r.protocolOk))process.exitCode=1;
})().catch(error=>{console.error(error);process.exitCode=1;});
