const {loadPlaywright, browserExecutable} = require('../../tools/platform.cjs');
const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {readRepositoryFile} = require(NEURAL_EARTH_ROOT + '/tools/repository-files.cjs');
// Real WebGPU shader and camera integration; no checkpoint/network inference.
const assert=require('node:assert/strict'),fs=require('node:fs');
const {chromium}=loadPlaywright();
const {sampleHeight}=require(NEURAL_EARTH_ROOT + '/web/terrain_map_tools.js');
assert.equal(sampleHeight({heights:new Float32Array(16).fill(-10),heightOptions:{width:4,halo:1,encoding:'signed-sqrt'},b:[0,0,2,2]},{x:1,y:1}),-100);
(async()=>{
  const browser=await chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
  try{
    const page=await browser.newPage({viewport:{width:1024,height:768}}),errors=[],requests=[],subscriptions=[];
    page.on('pageerror',e=>errors.push(e.message));
    const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM0LJrHwMDAxAAGAA7JAUW48M0QAAAAAElFTkSuQmCC','base64');
    await page.route('https://coarse.test/**',async route=>{
      const u=new URL(route.request().url());
      if(u.pathname==='/')return route.fulfill({contentType:'text/html',body:readRepositoryFile('index.html','utf8')});
      if(u.pathname==='/terrain_styles.js')return route.fulfill({contentType:'application/javascript',body:'window.TerrainStyles='+readRepositoryFile('terrain_styles.json','utf8')+';\n'+readRepositoryFile('terrain_style_rendering.js','utf8')});
      if(u.pathname.endsWith('.js'))return route.fulfill({contentType:'application/javascript',body:readRepositoryFile(u.pathname.slice(1),'utf8')});
      if(u.pathname==='/api/world')return route.fulfill({json:{version:'natural-v1',cache_profile:'test',world_identity:'test-world',world_profile:'natural',generation_profile:'natural',seed:u.searchParams.get('seed'),orogen_layers:{'orogen-biomes':'Adaptive Orogen biomes'},world_bounds:[-2e6,-1e6,2e6,1e6],overview_bounds:[-2e6,-1e6,2e6,1e6],overview:'/overview.png',gpu:'GPU fixture'}});
      if(u.pathname==='/api/view'){subscriptions.push(route.request().postDataJSON());return route.fulfill({json:{accepted:true}})}
      if(u.pathname==='/api/status')return route.fulfill({json:{scheduler:{queued:0,computing:0,encoding:0}}});
      if(u.pathname.startsWith('/coarse/')||u.pathname.startsWith('/height/')){
        requests.push(u.pathname);
        const native=u.pathname.startsWith('/coarse/'),source=u.searchParams.get('source_lod'),lod=Number(u.pathname.split('/')[4]);
        const width=native?160:source?256/2**(3-lod)+48:304,halo=native?16:24;
        const climateSize=native?41:33,values=new Float32Array(width*width+21*climateSize**2);values.fill(native?25:625,0,width*width);values.fill(20,width*width,width*width+climateSize**2);
        [.12,.38,.10,2,3.5,.2,.07,.15,2,2.5,.42,.38,.32,.92,.93,.96].forEach((v,i)=>values.fill(v,width*width+(5+i)*climateSize**2,width*width+(6+i)*climateSize**2));
        return route.fulfill({contentType:'application/octet-stream',body:Buffer.from(values.buffer),headers:{'X-Terrain-Width':String(width),'X-Terrain-Halo':String(halo),'X-Terrain-Encoding':native?'signed-sqrt':'metres','X-Terrain-Climate-Width':String(climateSize),'X-Terrain-Climate-Height':String(climateSize),'X-Terrain-Climate-Layers':'21','X-Terrain-Resolution':native?'7680':String(30*2**lod),'X-Terrain-Stage':native||lod>=4?'coarse':source||lod===3?'latent':'decoder','X-Terrain-Source-LOD':source||String(native?4:lod),'X-Terrain-Elevation-Min':'625','X-Terrain-Elevation-Max':'625','X-Terrain-Source-Resolution':native?'7680':source||lod===3?'240':'30'}});
      }
      return route.fulfill({contentType:'image/png',body:png});
    });
    await page.goto('https://coarse.test/?seed=42&profile=natural&coarse_prepare=0');
    const settle=()=>page.waitForFunction(()=>terrainDebug.snapshot().visible.ready>0&&!terrainDebug.snapshot().visible.pending&&!terrainDebug.snapshot().scheduler.inflight&&!terrainDebug.snapshot().scheduler.queued);
    await settle();
    assert.equal(await page.locator('#coarseGpu').isChecked(),true);
    assert.equal(await page.evaluate(()=>terrainDebug.snapshot().coarseGpu),true);
    assert(requests.every(p=>p.startsWith('/coarse/')),'Coarse camera LODs use source endpoint only');
    const before=requests.length,uploads=await page.evaluate(()=>terrainDebug.snapshot().renderer.uploads);
    for(const lod of [6,5,4,7,6]){
      await page.evaluate(lod=>{mpp=30*2**lod;draw();refresh()},lod);await settle();
      assert.equal(requests.length,before,'Zoom must reuse the same loaded native cells');
      assert.equal(await page.evaluate(()=>terrainDebug.snapshot().renderer.uploads),uploads);
    }
    assert(subscriptions.some(v=>v.tiles.every(t=>t.native_coarse&&t.lod===7)));
    await page.evaluate(()=>{document.getElementById('renderPanel').open=true});
    await page.locator('#tabRendering').click();
    await page.locator('#coarseGpu').uncheck();await settle();
    assert(requests.some(p=>p.startsWith('/height/')),'Unchecked option retains the original endpoint');
    assert.equal(await page.evaluate(()=>terrainDebug.snapshot().coarseGpu),false);
    assert.equal(new URL(page.url()).searchParams.get('coarse_gpu'),'0');
    await page.locator('#coarseGpu').check();await settle();
    await page.evaluate(()=>{mpp=240;draw();refresh()});await settle();
    assert(requests.some(p=>p.startsWith('/height/')&&Number(p.split('/')[4])===3),'Base model still takes over at LOD 3');
    assert.deepEqual(await page.evaluate(()=>terrainDebug.snapshot().rendered.sources),['latent']);
    // Crossing from native coarse into the base stage also works when depth
    // refinement requests it without changing the camera.
    await page.evaluate(()=>{mpp=480;document.getElementById('refinementDepth').value='1';resetBackend();draw();refresh()});
    await page.waitForFunction(()=>refinementProgress()?.complete&&terrainDebug.snapshot().rendered.sources.includes('latent')&&terrainDebug.snapshot().scheduler.inflight===0).catch(async e=>{console.error(JSON.stringify(await page.evaluate(()=>({snapshot:terrainDebug.snapshot(),progress:refinementProgress(),queue:queue.map(t=>({lod:t.lod,native:t.native_coarse,source:t.source_lod})),interests:interests.size})),null,2));throw e});
    await page.screenshot({path:'tmp/coarse-gpu-ui.png'});
    // The visible Biomes layer follows camera/refinement LODs and remains GPU.
    await page.evaluate(()=>{document.getElementById('mapMode').value='orogen-biomes';document.getElementById('mapMode').dispatchEvent(new Event('change'));document.getElementById('refinementDepth').value='0';mpp=30;draw();refresh()});
    await settle();
    assert.equal(await page.evaluate(()=>terrainDebug.snapshot().mode),'orogen-biomes');
    assert.equal(await page.evaluate(()=>terrainDebug.snapshot().backend),'webgpu');
    assert.equal(await page.evaluate(()=>terrainDebug.snapshot().refinement.target),0);
    assert((await page.evaluate(()=>terrainDebug.snapshot().rendered.lods)).includes(0),'Biomes render the detailed DEM');
    await page.evaluate(()=>{mpp=15;draw();refresh()});await settle();
    assert.equal(await page.evaluate(()=>terrainDebug.snapshot().refinement.target),-1);
    assert((await page.evaluate(()=>terrainDebug.snapshot().rendered.lods)).includes(-1),'Biomes also render experimental fine LODs');
    // Validate signed units, palette, continuous drawing, physical mip means,
    // and shared-halo seams directly on the real GPU.
    const shader=await page.evaluate(async()=>{
      const canvas=document.createElement('canvas');document.body.append(canvas);
      const r=await createTerrainRenderer(canvas,{maxBytes:16*1024**2});
      r.context.configure({device:r.device,format:r.format,alphaMode:'premultiplied',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
      const roots=new Float32Array(160*160),climate=new Float32Array(5*41*41);climate.fill(20,0,41*41);climate.fill(-.0065,4*41*41);
      const upload=(key,values,mode='temperature')=>r.uploadCoarse(key,values,{climate,mode,interpolation:'bilinear'});
      const read=async(rects,width=256,height=128)=>{
        r.draw(rects,{width,height,dpr:1});
        const row=Math.ceil(width*4/256)*256,buffer=r.device.createBuffer({size:row*height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const encoder=r.device.createCommandEncoder();encoder.copyTextureToBuffer({texture:r.context.getCurrentTexture()},{buffer,bytesPerRow:row},[width,height]);r.device.queue.submit([encoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);const values=new Uint8Array(buffer.getMappedRange()).slice();buffer.unmap();buffer.destroy();return{x:width,y:height,row,values};
      };
      const pixel=(img,x,y)=>{const v=Array.from(img.values.slice(y*img.row+x*4,y*img.row+x*4+4));return r.format.startsWith('bgra')?[v[2],v[1],v[0],v[3]]:v};
      roots.fill(Math.sqrt(1000));upload('flat',roots);
      const flat=await read([{key:'flat',x:0,y:0,width:256,height:128}]);
      const initialUploads=r.uploads;
      for(const size of [512,768,1536])await read([{key:'flat',x:-size/3,y:-size/4,width:size,height:size}]);
      const afterZoom=r.uploads;
      for(let y=0;y<160;y++)for(let x=0;x<160;x++)roots[y*160+x]=x%2?Math.sqrt(3000):-Math.sqrt(1000);
      upload('checker',roots);
      const mip=await read([{key:'checker',x:0,y:0,width:64,height:64}],64,64);
      for(const tx of [0,1]){for(let y=0;y<41;y++)for(let x=0;x<41;x++)climate[y*41+x]=20+(tx*128-16+x*4+.5)*.015+(y*4-16+.5)*.01;const values=new Float32Array(160*160);for(let y=0;y<160;y++)for(let x=0;x<160;x++)values[y*160+x]=30+12*Math.sin((tx*128+x-16)*.02)+7*Math.sin((y-16)*.05);upload('seam'+tx,values)}
      const seams=[];
      for(const size of [512,128,32]){const img=await read([{key:'seam0',x:0,y:0,width:size/2,height:128},{key:'seam1',x:size/2,y:0,width:size/2,height:128}],size,128);seams.push([pixel(img,size/2-1,64),pixel(img,size/2,64)])}
      const report={flat:pixel(flat,128,64),mip:pixel(mip,32,32),initialUploads,afterZoom,seams,status:r.status};r.dispose();canvas.remove();return report;
    });
    const t=(13.5+35)/70,expected=[.18,.38,.88].map((v,i)=>Math.round((v+([.95,.22,.08][i]-v)*t)*255)).concat(255);
    for(const values of [shader.flat,shader.mip])values.forEach((v,i)=>assert(Math.abs(v-expected[i])<=1,`${values} versus ${expected}`));
    assert.equal(shader.initialUploads,shader.afterZoom);
    for(const [a,b] of shader.seams)a.forEach((v,i)=>assert(Math.abs(v-b[i])<=3,`Discontinuous shared halo: ${a} vs ${b}`));
    assert.equal(shader.status,'webgpu');assert.deepEqual(errors,[]);
    console.log(JSON.stringify({ok:true,nativeRequests:before,shader},null,2));
  }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
