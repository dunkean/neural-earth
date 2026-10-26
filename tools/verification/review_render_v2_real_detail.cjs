// Close views of completed real plates: 512 * 2.5 m = 1.28 km, NOT 40 km.
// Force decoder LOD0 and refuse uncached physical requests before inference.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/render-v2-40km/real-detail');
const selection=JSON.parse(fs.readFileSync(path.join(root,'output/render-v2-review/real-patches-40km.json'),'utf8'));
const world=JSON.parse(fs.readFileSync(path.join(root,'output/render-v2-review/selection-world-info.json'),'utf8').replace(/^\uFEFF/,''));
const complete=JSON.parse(fs.readFileSync(path.join(root,'output/render-v2-40km/real/report.json'),'utf8'));
const wanted=(process.env.REVIEW_CASES||'tropical-rainforest-low,humid-temperate-low,taiga-low,sandy-desert,alpine-tundra,polar-ice').split(',');
const patches=wanted.map(id=>selection.patches.find(p=>p.id===id));
const size=512,mpp=2.5,extent=size*mpp,origin='http://127.0.0.1:8765';
const cache=path.join(root,'generated/natural-v1',world.cache_profile,world.generation_profile,'physical-v1',String(selection.seed),'0');
function cached(tx,ty){
 const base=path.join(cache,`${tx}_${ty}`);
 if(!fs.existsSync(base+'.npy')||!fs.existsSync(base+'.json'))return false;
 const receipt=JSON.parse(fs.readFileSync(base+'.json','utf8'));
 return receipt.stage==='decoder'&&receipt.resolution===30&&receipt.world_identity===selection.world_identity;
}
async function status(){const r=await fetch(origin+'/api/status');assert(r.ok);const s=await r.json();return {generated:s.metrics.generated_tiles,calls:s.cuda_forward_calls};}
(async()=>{
 fs.mkdirSync(out,{recursive:true});const results=[],started=Date.now(),before=await status();let browser;
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
  for(const patch of patches){
   assert(patch);assert(complete.results.some(p=>p.id===patch.id&&p.ok),'40 km parent plate not complete');
   const page=await browser.newPage({viewport:{width:900,height:900}}),entry={id:patch.id,center:patch.center_m,extentMetres:extent,mpp,decoderResolutionMetres:30,frames:[],errors:[],responses:[],blocked:[]};
   const reject=async(route,reason)=>{entry.blocked.push({url:route.request().url(),reason});await route.fulfill({status:409,contentType:'application/json',body:JSON.stringify({accepted:false,error:reason})});};
   await page.route('**/api/view',async route=>{
    const data=route.request().postDataJSON();
    const invalid=(data.tiles||[]).find(t=>t.lod!==0||t.source_lod||t.native_coarse||!cached(t.tx,t.ty));
    if(data.neural_chart||invalid)return reject(route,'Uncached/non-LOD0 subscription refused: '+JSON.stringify(invalid||data.neural_chart));
    await route.continue();
   });
   await page.route('**/height/**',async route=>{
    const url=new URL(route.request().url()),match=url.pathname.match(/\/height\/natural-v1\/(\d+)\/(-?\d+)\/(-?\d+)\/(-?\d+)\.bin$/);
    if(!match||match[1]!==String(selection.seed)||+match[2]!==0||url.searchParams.has('source_lod')||url.searchParams.has('neural_chart')||!cached(+match[3],+match[4]))return reject(route,'Uncached/non-LOD0 height refused');
    await route.continue();
   });
   page.on('pageerror',e=>entry.errors.push(e.stack));
   page.on('response',r=>{if(r.status()>=400&&!r.url().endsWith('favicon.ico'))entry.responses.push({status:r.status(),url:r.url()});});
   const params=new URLSearchParams({share:'gHY4ZFzncjegep8JTpf.rulIR78fFpNhu4Tj2Y2',gpu:1,layer:'render-v2',view:'map',zoom:mpp,position:patch.center_m.join(','),lod:0,size:[size,size].join(','),v2season:.5,hydro:0,hydroMonth:7,cache_gib:1,coarse_gpu:1});
   entry.url=origin+'/?'+params;
   try{
    await page.goto(entry.url,{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>window.terrainDebug?.snapshot().renderer?.render3d?.ready,null,{timeout:120000});
    async function settle(){
     await page.waitForFunction(()=>{const s=terrainDebug.snapshot();return s.visible.pending===0&&s.rendered.patches>0&&s.refinement.forced===0&&s.rendered.lods.every(l=>l===0)&&s.rendered.qualityLods.every(l=>l===0)&&s.rendered.sources.every(t=>t==='decoder')&&s.rendered.sourceResolutions.every(r=>r===30);},null,{timeout:60000});
     return page.evaluate(()=>terrainDebug.snapshot());
    }
    async function capture(tag){
     const state=await settle(),dimensions=await page.locator('#gpuMap').evaluate(c=>({width:c.width,height:c.height}));
     assert.deepEqual(dimensions,{width:size,height:size});assert.equal(state.camera.mpp,mpp);
     assert.equal(state.camera.cx,patch.center_m[0]);assert.equal(state.camera.cy,patch.center_m[1]);
     assert(state.cache.keys.every(k=>k.includes(selection.world_identity)));
     const file=patch.id+'-'+tag+'.png';await page.locator('#gpuMap').screenshot({path:path.join(out,file)});
     entry.frames.push({tag,file,dimensions,extentMetres:extent,state});
    }
    await capture('v2-july');
    await page.locator('#layerPanel > summary').click();await page.locator('#layerPanel button[data-mode="render3d"]').click();await page.waitForFunction(()=>terrainDebug.snapshot().mode==='render3d');await capture('render3d-july');
    if(['humid-temperate-low','taiga-low','alpine-tundra','polar-ice'].includes(patch.id)){
     await page.locator('#layerPanel > summary').click();await page.locator('#layerPanel button[data-mode="render-v2"]').click();
     await page.locator('#renderPanel > summary').click();await page.locator('#tabRendering').click();await page.locator('#renderV2Season').fill('0');await page.locator('#renderV2Season').dispatchEvent('input');await page.locator('#renderPanel > summary').click();
     await page.waitForFunction(()=>terrainDebug.snapshot().renderV2.season===0);await capture('v2-january');
    }
    entry.ok=!entry.errors.length&&!entry.blocked.length&&!entry.responses.some(r=>r.status!==409);
   }catch(error){entry.ok=false;entry.failure=error.stack;entry.last=await page.evaluate(()=>window.terrainDebug?.snapshot()).catch(()=>null);}
   finally{await page.close();results.push(entry);fs.writeFileSync(path.join(out,'report.json'),JSON.stringify({label:'Real close views, 1.28 km squares; NOT 40 km plates; cached 30 m decoder enlarged at 2.5 m/px',worldIdentity:selection.world_identity,sourceCacheKey:selection.source_cache_key,extentMetres:extent,before,results},null,2));}
   console.log(JSON.stringify({id:patch.id,ok:entry.ok,frames:entry.frames.length,blocked:entry.blocked,failure:entry.failure}));
  }
 }finally{if(browser)await browser.close();}
 const after=await status(),noInference=JSON.stringify(before.calls)===JSON.stringify(after.calls)&&before.generated===after.generated;
 const report={label:'Real close views, 1.28 km squares; NOT 40 km plates; cached 30 m decoder enlarged at 2.5 m/px',worldIdentity:selection.world_identity,sourceCacheKey:selection.source_cache_key,extentMetres:extent,seconds:(Date.now()-started)/1000,before,after,noInference,results};
 report.ok=noInference&&results.every(r=>r.ok);fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify({ok:report.ok,noInference,sites:results.length,seconds:report.seconds,out}));if(!report.ok)process.exitCode=1;
})().catch(error=>{console.error(error);process.exitCode=1;});
