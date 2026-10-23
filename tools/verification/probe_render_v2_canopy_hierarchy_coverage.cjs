// Read actual canopy coverage from the WebGPU texture. No PNGs or terrain edits.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'output/render-v2-floor/canopy-hierarchy-causal'),hash=x=>crypto.createHash('sha256').update(x).digest('hex');
const manifest=JSON.parse(fs.readFileSync(path.join(base,'manifest.json'),'utf8')),records=[],receipts=[];let browser;
(async()=>{
 browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
 try{for(const variant of ['production','hierarchical-lobes']){
  const report=JSON.parse(fs.readFileSync(path.join(base,variant,'report.json'),'utf8')),receipt=JSON.parse(fs.readFileSync(path.join(base,variant,'ABLATION.json'),'utf8'));
  const frozen=report.frozen;const assertSources=()=>{for(const [f,h]of Object.entries(frozen.codeIdentity))assert.equal(hash(fs.readFileSync(path.join(root,f))),h,'Production changed: '+f);};assertSources();
  const from='if(!r)return r;window.__canopyProbe=',to="if(!r)return r;r.context.configure({device:r.device,format:r.format,alphaMode:'premultiplied',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});window.__canopyProbe=";assert(receipt.override.injection.includes(from));const injection=receipt.override.injection.replace(from,to);
  receipts.push({variant,parentVisualOverrideSha256:receipt.override.sha256,readbackOverrideSha256:hash(injection),readbackOverride:injection,productionCodeHash:frozen.codeHash,method:'Only adds COPY_SRC usage to the GPU canvas; coverage diagnostic901 already exists in the visual shader. Read from current GPU texture before presentation.'});
  for(const site of manifest.sites){
   const entry=report.results.find(s=>s.id===site.id);assert(entry?.protocolOk);
   const page=await browser.newPage({viewport:{width:1200,height:1200},deviceScaleFactor:1}),errors=[],blocked=[];page.on('pageerror',e=>errors.push(e.message));
   await page.route('**/terrain_render_v2.js*',route=>route.fulfill({status:200,contentType:'application/javascript',body:fs.readFileSync(path.join(root,'web/terrain_render_v2.js'),'utf8')+injection}));
   await page.route('**/api/view',route=>{const d=route.request().postDataJSON();if((d.tiles||[]).some(t=>t.lod<0)){blocked.push(d);return route.fulfill({status:409,body:'Negative LOD forbidden'});}return route.continue();});
   await page.route('**/height/**',route=>{const m=new URL(route.request().url()).pathname.match(/\/height\/[^/]+\/\d+\/(-?\d+)\//);if(m&&+m[1]<0){blocked.push(route.request().url());return route.fulfill({status:409,body:'Negative LOD forbidden'});}return route.continue();});
   try{
    await page.goto(entry.url,{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>terrainDebug?.snapshot().renderer?.render3d?.ready,null,{timeout:120000});
    await page.evaluate(()=>{for(const id of ['prefetch','grid']){const e=document.getElementById(id);if(e.checked){e.checked=false;e.dispatchEvent(new Event('change'));}}const d=document.getElementById('refinementDepth');d.value='0';d.dispatchEvent(new Event('change'));});
    for(const z of manifest.zooms){
     await page.evaluate(mpp=>{const s=terrainDebug.snapshot(),v=document.getElementById('viewport'),r=v.getBoundingClientRect();v.dispatchEvent(new WheelEvent('wheel',{deltaY:Math.log(mpp/s.camera.mpp)/.0015,clientX:r.left+r.width/2,clientY:r.top+r.height/2,bubbles:true,cancelable:true}));const e=document.getElementById('forcedLod');e.value='0';e.dispatchEvent(new Event('change'));},z.extentMetres/manifest.size);
     await page.waitForFunction(()=>{const s=terrainDebug.snapshot();return s.visible.pending===0&&s.refinement.forced===0&&s.rendered.patches>0&&s.rendered.lods.every(x=>x===0)&&s.rendered.qualityLods.every(x=>x===0)&&s.rendered.sources.every(x=>x==='decoder')&&s.rendered.sourceResolutions.every(x=>x===30);},null,{timeout:120000});
     await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
     const measurement=await page.evaluate(async()=>{
      const p=window.__canopyProbe,r=p.renderer,last=p.last,device=r.device;if(!last)throw Error('No actual submitted view');p.flag=901;r.draw(...last);
      const width=r.canvas.width,height=r.canvas.height,bytesPerRow=Math.ceil(width*4/256)*256;
      const b=device.createBuffer({size:bytesPerRow*height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),encoder=device.createCommandEncoder();encoder.copyTextureToBuffer({texture:r.context.getCurrentTexture()},{buffer:b,bytesPerRow,rowsPerImage:height},[width,height]);device.queue.submit([encoder.finish()]);await b.mapAsync(GPUMapMode.READ);
      const a=new Uint8Array(b.getMappedRange());let sum=0,nonzero=0,opaque=0,grayError=0;for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=y*bytesPerRow+x*4;sum+=a[i]/255;if(a[i])nonzero++;if(a[i]===255)opaque++;grayError=Math.max(grayError,Math.abs(a[i]-a[i+1]),Math.abs(a[i]-a[i+2]));}
      b.unmap();b.destroy();p.flag=0;r.draw(...last);await device.queue.onSubmittedWorkDone();return{mean:sum/(width*height),nonzero,opaque,pixels:width*height,grayError,width,height,state:terrainDebug.snapshot()};
     });
     assert(measurement.mean>0,'Coverage readback unexpectedly empty');assert.equal(measurement.grayError,0,'Coverage output is not actual grayscale diagnostic');assertSources();records.push({variant,id:site.id,zoom:z.id,extentMetres:z.extentMetres,displayFootprint:z.extentMetres/manifest.size,requiredPhysicalResolutionMetres:30,...measurement,errors:[...errors],blocked:[...blocked]});console.log(JSON.stringify({variant,id:site.id,zoom:z.id,mean:measurement.mean,opaque:measurement.opaque}));
    }
    assert.deepEqual(errors,[]);assert.deepEqual(blocked,[]);
   }finally{await page.close();fs.writeFileSync(path.join(base,'coverage-readback.json'),JSON.stringify({kind:'Actual GPU texture canopy coverage; replaces invalid Canvas2D readbacks in image reports',receipts,records},null,2));}
  }
 }}finally{if(browser)await browser.close();}
 const comparisons=records.filter(x=>x.variant==='production').map(a=>{const b=records.find(b=>b.variant==='hierarchical-lobes'&&b.id===a.id&&b.zoom===a.zoom);assert(b);return{id:a.id,zoom:a.zoom,baseline:a.mean,hierarchy:b.mean,absoluteDifference:b.mean-a.mean,relativeDifference:b.mean/a.mean-1};});
 fs.writeFileSync(path.join(base,'coverage-readback.json'),JSON.stringify({kind:'Actual GPU texture canopy coverage; replaces invalid Canvas2D readbacks in image reports',receipts,comparisons,records},null,2));console.log(JSON.stringify({ok:true,comparisons}));
})().catch(e=>{console.error(e);process.exitCode=1;});
