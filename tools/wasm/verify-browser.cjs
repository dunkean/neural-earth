// Real checkpoints and a static-only server: this is intentionally an explicit,
// relatively expensive smoke test rather than part of the unit suite.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {browserExecutable,browserArguments}=require('../platform.cjs');
(async()=>{
  const profile=process.env.NEURAL_EARTH_WEBGPU_PROFILE||'default';
  fs.mkdirSync(path.join(__dirname,'../../tmp'),{recursive:true});
  assert.ok(['default','portable'].includes(profile),'WebGPU profile must be default or portable');
  const browser=await chromium.launch({executablePath:browserExecutable(),headless:true,args:browserArguments()});
  let inferenceProgress;
  try {
    const context=await browser.newContext({viewport:{width:1000,height:720}});
    if(profile==='portable')await context.route('**/runtime/runtime-worker.mjs',async route=>{
      const response=await route.fetch();
      // Test a real GPUDevice without the optional extensions. Advertising only
      // core adapter features makes wgpu request a baseline device; actual WGSL
      // compilation and all numerical operations still execute on the GPU.
      const mask=`const featureGetter=Object.getOwnPropertyDescriptor(GPUAdapter.prototype,'features').get;
Object.defineProperty(GPUAdapter.prototype,'features',{get(){return new Set([...featureGetter.call(this)].filter(f=>f!=='shader-f16'&&!f.includes('subgroup')));}});
const requestDevice=GPUAdapter.prototype.requestDevice;
GPUAdapter.prototype.requestDevice=function(descriptor={}){
  if((descriptor.requiredFeatures||[]).some(f=>f==='shader-f16'||f.includes('subgroup')))throw Error('Portable runtime requested an unavailable extension');
  return requestDevice.call(this,descriptor);
};
const createShaderModule=GPUDevice.prototype.createShaderModule;
GPUDevice.prototype.createShaderModule=function(descriptor){
  if(/enable\\s+(f16|subgroups)|@builtin\\(subgroup/.test(descriptor.code))throw Error('Portable runtime compiled an extension-dependent shader');
  return createShaderModule.call(this,descriptor);
};\n`;
      await route.fulfill({response,body:mask+await response.text()});
    });
    const page=await context.newPage(),errors=[],network=[];
    page.on('pageerror',e=>{errors.push(e.message);console.log('pageerror:',e.message);});
    page.on('request',request=>{const url=new URL(request.url());if(/^\/(api|tiles|height|coarse|reference)\//.test(url.pathname))network.push(request.url());});
    page.on('response',response=>{if(response.status()>=400)console.log('HTTP',response.status(),response.url());});
    page.on('console',message=>{if(message.type()==='error')errors.push(message.text());else if(message.text().startsWith('[verify]'))console.log(message.text());});
    const url=process.env.NEURAL_EARTH_WASM_URL||'http://127.0.0.1:8780';
    await page.goto(`${url}/?seed=42&gpu=0&mode=render`,{waitUntil:'domcontentloaded',timeout:60000});
    const progressTimer=setInterval(()=>page.locator('#status').textContent().then(s=>console.log('Status:',s)).catch(()=>{}),10000);
    try {await page.waitForFunction(()=>window.terrainDebug?.snapshot().overview||document.getElementById('status')?.textContent.startsWith('Error:'),null,{timeout:120000});}
    catch(error){console.log('Startup status:',await page.locator('#status').textContent(),errors);throw error;}
    finally{clearInterval(progressTimer);}
    if(!(await page.evaluate(()=>window.terrainDebug?.snapshot().overview)))throw Error(await page.locator('#status').textContent());
    const overview=await page.evaluate(()=>window.terrainDebug.snapshot());
    assert.equal(overview.world,'42');assert.equal(overview.coarseGpu,false);
    const models=await page.evaluate(async()=>await (await TerrainTransport.fetch('/api/distill/models')).json());
    assert.equal(models.selection.base,'nf4_b75','Browser must use the distilled base');
    const manifest=await (await page.request.get(`${url}/runtime/models/manifest.json`)).json();
    assert.match(manifest.profile,/nf4/);
    assert.ok(manifest.models.base.bytes<160000000,'Do not ship the original base');
    const world=await page.evaluate(async()=>await (await TerrainTransport.fetch('/api/world?seed=42')).json());
    if(profile==='portable')assert.match(world.cache_profile,/fp32/);
    console.log('Shared viewer and Orogen overview ready');
    inferenceProgress=setInterval(()=>page.locator('#status').textContent().then(s=>console.log(`${profile}:`,s)).catch(()=>{}),20000);
    const report=await page.evaluate(async profile=>{
      const seed='42',params='?mode=relief';
      const tile=async(lod,x,y)=>{
        console.log('[verify] tile',lod,x,y);
        const start=performance.now(),response=await TerrainTransport.fetch(`/height/wasm-v1/${seed}/${lod}/${x}/${y}.bin${params}`);
        if(!response.ok)throw Error(await response.text());
        const h=new Float32Array(await response.arrayBuffer());
        if(h.length!==304*304||!h.every(Number.isFinite))throw Error('Invalid heights');
        const ms=performance.now()-start;console.log('[verify] tile ready',lod,x,y,Math.round(ms),'ms');
        return {heights:h,ms,stage:response.headers.get('X-Terrain-Stage')};
      };
      const a=await tile(4,-1,2),b=await tile(4,0,2),again=await tile(4,-1,2);
      if(a.heights.some((h,i)=>h!==again.heights[i]))throw Error('Repeated inference changed heights');
      // Select actual high land from the source, at unequal global X/Y origins.
      const source=await tile(11,0,-1),bounds=terrainDebug.snapshot().worldBounds,step=30*2**11,span=256*step;
      let peak=-Infinity,point=null;
      for(let y=24;y<280;y++)for(let x=24;x<280;x++) {
        const px=(x-24+.5)*step,py=-span+(y-24+.5)*step;
        if(px>=bounds[0]&&px<bounds[2]&&py>=bounds[1]&&py<bounds[3]&&source.heights[y*304+x]>peak){peak=source.heights[y*304+x];point=[px,py];}
      }
      if(!point||peak<=0)throw Error('No land in source fixture');
      const [tx,ty]=point.map(v=>Math.floor(v/7680));
      console.log('Fine land tile',tx,ty,'source elevation',peak);
      const fine=await tile(0,tx,ty);
      const fineAgain=await tile(0,tx,ty);
      if(fine.heights.some((h,i)=>h!==fineAgain.heights[i]))throw Error('Repeated decoder changed heights');
      let fineNeighbour=null,fineSeam=null,recomputed=null;
      if(profile==='default') {
        fineNeighbour=await tile(0,tx+1,ty);fineSeam=0;
        for(let y=0;y<304;y++)for(let x=0;x<48;x++)fineSeam=Math.max(fineSeam,Math.abs(fine.heights[y*304+256+x]-fineNeighbour.heights[y*304+x]));
        if(fineSeam>.01)throw Error(`Two-step base/decoder halo seam ${fineSeam} m`);
        const reset=await TerrainTransport.fetch('/api/distill/models',{method:'POST',body:JSON.stringify({reset:true})});
        if(!reset.ok)throw Error(await reset.text());
        recomputed=await tile(0,tx,ty);
        if(fine.heights.some((h,i)=>h!==recomputed.heights[i]))throw Error('Recomputed decoder changed heights after clearing all window caches');
      }
      // The two coarse neighbours overlap by 48 samples (the shared halos).
      let seam=0;for(let y=0;y<304;y++)for(let x=0;x<48;x++)seam=Math.max(seam,Math.abs(a.heights[y*304+256+x]-b.heights[y*304+x]));
      if(seam>.01)throw Error(`Coarse halo seam ${seam} m`);
      const png=await TerrainTransport.fetch(`/tiles/wasm-v1/${seed}/0/${tx}/${ty}.png?mode=render`);
      if(!png.ok)throw Error(await png.text());
      const bytes=new Uint8Array(await png.arrayBuffer()),image=btoa(Array.from(bytes,c=>String.fromCharCode(c)).join(''));
      return {coarse:[a.ms,b.ms,again.ms],decoder:[fine.ms,fineAgain.ms,recomputed?.ms??null],fineNeighbourMs:fineNeighbour?.ms??null,coarseHaloError:seam,fineHaloError:fineSeam,fineTile:[tx,ty],sourceElevation:peak,stages:[a.stage,fine.stage],min:fine.heights.reduce((a,b)=>Math.min(a,b),Infinity),max:fine.heights.reduce((a,b)=>Math.max(a,b),-Infinity),image};
    },profile);
    fs.writeFileSync(path.join(__dirname,'../../tmp/wasm-fine-land.png'),Buffer.from(report.image,'base64'));delete report.image;
    console.log(JSON.stringify(report,null,2));
    await page.locator('#globeView').click();
    await page.waitForTimeout(1000);
    const globe=await page.evaluate(()=>window.terrainDebug.snapshot());assert.equal(globe.view,'globe');assert.equal(globe.globe.ready,true);
    const suffix=profile==='default'?'':'-portable';
    const output=path.join(__dirname,`../../tmp/wasm-browser${suffix}.png`);await page.screenshot({path:output});
    assert.deepEqual(network,[],'API traffic must remain inside the browser');
    assert.deepEqual(errors,[],'Browser errors');
    fs.writeFileSync(path.join(__dirname,`../../tmp/wasm-browser${suffix}.json`),JSON.stringify({profile,models,cacheProfile:world.cache_profile,overview,report,globe,network,errors},null,2));
    console.log(`Rust/WASM coarse, distilled base, decoder, cached repeats, globe and zero backend requests passed (${profile}).`);
  }finally{clearInterval(inferenceProgress);await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
