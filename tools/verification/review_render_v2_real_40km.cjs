// Review selected real source-atlas sites through the live viewer and require
// completed decoder tiles. A source/conditioning preview cannot satisfy this.
const fs=require('node:fs'),path=require('node:path');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..'),out=path.join(root,process.env.REVIEW_OUT||'output/render-v2-40km/real');
const selection=JSON.parse(fs.readFileSync(path.resolve(root,process.env.REVIEW_SELECTION||'output/render-v2-review/real-patches-40km.json'),'utf8'));
const ids=process.env.REVIEW_CASES?.split(',');
const patches=selection.patches.filter(p=>p.available&&(!ids||ids.includes(p.id)));
const size=Number(process.env.REVIEW_SIZE||1024),lod=Number(process.env.REVIEW_LOD||0),mpp=Number(process.env.REVIEW_MPP||40000/size),width=mpp*size;
const warmOnly=process.env.REVIEW_WARM_ONLY==='1',expectedFrames=warmOnly?1:3;
function warmCachePresent(entry){
 if(!warmOnly)return true;
 return entry.frames.every(f=>f.state.cache.keys.filter(k=>k.includes('/physical/')).every(k=>{
  const p=k.split('/'),i=p.indexOf('physical');if(p[i+2]!=='0')return true;
  const tile=path.join(root,'generated/natural-v1',p[3],f.state.generationProfile,'physical-v1',String(f.state.world),'0',p[i+3]+'_'+p[i+4]);
  if(!fs.existsSync(tile+'.npy')||!fs.existsSync(tile+'.json'))return false;
  const receipt=JSON.parse(fs.readFileSync(tile+'.json','utf8'));
  return receipt.world_identity===selection.world_identity&&receipt.stage==='decoder'&&receipt.resolution===30;
 }));
}
(async()=>{
 fs.mkdirSync(out,{recursive:true});let browser;const results=[],started=Date.now();
 const previous=fs.existsSync(path.join(out,'report.json'))?JSON.parse(fs.readFileSync(path.join(out,'report.json'),'utf8')).results:[];
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
  for(const patch of patches){
   const cached=previous.find(p=>p.id===patch.id&&p.frames.length===expectedFrames&&!p.errors.length&&!p.bad.some(b=>b.status!==409)&&p.frames.every(f=>f.extentMetres===width&&f.state.refinement.forced===lod&&f.state.cache.keys.some(k=>k.includes(selection.world_identity))&&fs.existsSync(path.join(out,f.file))));
   if(cached&&warmCachePresent(cached)){cached.ok=true;cached.cancelledResponses=cached.bad.filter(b=>b.status===409).length;results.push(cached);console.log(JSON.stringify({id:patch.id,reused:true}));continue;}
   const page=await browser.newPage({viewport:{width:1200,height:1200}}),errors=[],bad=[];let lastLog=0;
   page.on('pageerror',e=>errors.push(e.stack));page.on('response',r=>{if(r.status()>=400&&!r.url().endsWith('favicon.ico'))bad.push({status:r.status(),url:r.url()});});
   if(warmOnly){
    await page.route('**/api/view',async route=>{
     const data=route.request().postDataJSON();
     if((data.tiles||[]).some(t=>t.lod<0))return route.fulfill({status:409,body:'Negative LOD forbidden during warming'});
     await route.continue();
    });
    await page.route('**/height/**',async route=>{
     const m=new URL(route.request().url()).pathname.match(/\/height\/[^/]+\/\d+\/(-?\d+)\//);
     if(m&&+m[1]<0)return route.fulfill({status:409,body:'Negative LOD forbidden during warming'});
     await route.continue();
    });
   }
   const params=new URLSearchParams({share:'gHY4ZFzncjegep8JTpf.rulIR78fFpNhu4Tj2Y2',gpu:1,layer:'render-v2',view:'map',zoom:mpp,position:patch.center_m.join(','),lod,size:[size,size].join(','),v2season:.5,hydro:0,hydroMonth:7,cache_gib:1,coarse_gpu:1});
   const url='http://127.0.0.1:8765/?'+params,entry={id:patch.id,selection:patch,url,frames:[],errors,bad};
   try{
    await page.goto(url,{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>window.terrainDebug?.snapshot().world&&terrainDebug.snapshot().renderer?.render3d?.ready,null,{timeout:120000});
    await page.evaluate(()=>{
     for(const [id,value] of [['prefetch',false],['renderSea',true],['grid',false]]){
      const control=document.getElementById(id);if(control.checked!==value){control.checked=value;control.dispatchEvent(new Event('change'));}
     }
     const depth=document.getElementById('refinementDepth');depth.value='0';depth.dispatchEvent(new Event('change'));
     const hydro=document.querySelector('[data-hydro="enabled"]');if(hydro.checked){hydro.checked=false;hydro.dispatchEvent(new Event('change'));}
    });
    async function settle(){
     const deadline=Date.now()+300000;
     while(Date.now()<deadline){
      const s=await page.evaluate(()=>terrainDebug.snapshot());
      if(s.visible.pending===0&&s.rendered.patches>0&&s.rendered.qualityLods.length&&s.rendered.qualityLods.every(l=>l<=lod)&&(lod>0||s.rendered.sources.every(t=>t==='decoder'||t==='final-dem-mip')))return s;
      if(Date.now()-lastLog>20000){lastLog=Date.now();console.log(JSON.stringify({id:patch.id,pending:s.visible.pending,sources:s.rendered.sources,received:s.counters.received,elapsed:Math.round((Date.now()-started)/1000)}));}
      await page.waitForTimeout(1000);
     }
     throw Error('Decoder plate did not finish within five minutes');
    }
    async function capture(tag){
     const state=await settle(),file=patch.id+'-'+tag+'.png';
     if(warmOnly&&(!state.cache.keys.length||!state.cache.keys.every(k=>k.includes(selection.world_identity))||state.refinement.forced!==0||state.rendered.sources.some(s=>s!=='decoder')||state.rendered.sourceResolutions.some(r=>r!==30)))throw Error('Warm cache identity/decoder30m source mismatch');
     const dimensions=await page.locator('#gpuMap').evaluate(c=>({width:c.width,height:c.height}));
     if(dimensions.width!==size||dimensions.height!==size)throw Error('Logical plate/canvas dimensions do not match 40 km contract: '+JSON.stringify(dimensions));
     if(Math.abs(state.camera.mpp*size-width)>1e-6)throw Error('Camera extent differs from 40 km');
     await page.addStyleTag({content:'#mini,#minimap,#hud,#stats,#status,.mapHud{visibility:hidden !important}'});
     await page.locator('#gpuMap').screenshot({path:path.join(out,file)});entry.frames.push({tag,file,state,dimensions,extentMetres:width});
    }
    await capture('v2-july');
    if(!warmOnly){
    await page.locator('#layerPanel > summary').click();await page.locator('#layerPanel button[data-mode="render3d"]').click();await page.waitForFunction(()=>terrainDebug.snapshot().mode==='render3d');await capture('render3d-july');
    await page.locator('#layerPanel > summary').click();await page.locator('#layerPanel button[data-mode="render-v2"]').click();
    await page.locator('#renderPanel > summary').click();await page.locator('#tabRendering').click();await page.locator('#renderV2Season').fill('0');await page.locator('#renderV2Season').dispatchEvent('input');await page.locator('#renderPanel > summary').click();
    await page.waitForFunction(()=>terrainDebug.snapshot().renderV2.season===0);await capture('v2-january');
    }
    // A 409 cancels an obsolete view epoch; completed decoder pixels above
    // are the acceptance criterion. Keep those cancellations in the report.
    entry.cancelledResponses=bad.filter(b=>b.status===409).length;
    entry.ok=!errors.length&&!bad.some(b=>b.status!==409);console.log(JSON.stringify({id:patch.id,ok:entry.ok,frames:entry.frames.length,seconds:Math.round((Date.now()-started)/1000)}));
   }catch(error){entry.ok=false;entry.failure=error.stack;entry.last=await page.evaluate(()=>window.terrainDebug?.snapshot()).catch(()=>null);await page.screenshot({path:path.join(out,patch.id+'-failure.png')}).catch(()=>{});console.log(JSON.stringify({id:patch.id,failure:String(error)}));}
   finally{results.push(entry);await page.close();fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({selection,plateMetres:width,results,seconds:(Date.now()-started)/1000,warmOnly,label:warmOnly?'CACHE WARMING ONLY; working images, not final visual validation':undefined},null,2));}
  }
 }finally{if(browser)await browser.close();}
 console.log(JSON.stringify({ok:results.every(r=>r.ok),sites:results.length,out,seconds:(Date.now()-started)/1000}));
 if(results.some(r=>!r.ok))process.exitCode=1;
})().catch(error=>{console.error(error);process.exitCode=1;});
