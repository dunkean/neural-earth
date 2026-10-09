// Real WebGPU pixel parity against the CPU biome implementation and UI routing.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');
const fixtures=JSON.parse(execFileSync(path.join(__dirname,'.venv/Scripts/python.exe'),['test_terrain_biomes.py','--fixtures'],{cwd:__dirname,encoding:'utf8'}));
const koppenFixtures=JSON.parse(execFileSync(path.join(__dirname,'.venv/Scripts/python.exe'),['test_terrain_koppen.py','--fixtures'],{cwd:__dirname,encoding:'utf8'}));
(async()=>{
  const server=http.createServer((req,res)=>{
    res.setHeader('Content-Type',req.url.endsWith('.js')?'application/javascript':'text/html');
    res.end(req.url==='/terrain_renderer.js'?fs.readFileSync(path.join(__dirname,'terrain_renderer.js')):
      '<canvas id="map"></canvas><script>window.biomeTestLight=0;window.TerrainLighting={vectors:()=>[window.biomeTestLight,.35,1,1,-.5,-.5,.70710678,0]};</script><script src="/terrain_renderer.js"></script>');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
  try{
    browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--enable-unsafe-webgpu']});
    const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    const report=await page.evaluate(async ({fixtures,koppenFixtures})=>{
      const renderer=await createTerrainRenderer(document.getElementById('map'),{maxBytes:32*1024**2,onStatus:(state,detail)=>{if(state==='fallback')throw Error(detail)}});
      if(!renderer?.available)throw Error('WebGPU unavailable');
      renderer.context.configure({device:renderer.device,format:renderer.format,alphaMode:'premultiplied',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
      const climate=new Float32Array(21*33*33);
      fixtures.fields.forEach((v,i)=>climate.fill(v,(5+i)*33*33,(6+i)*33*33));
      const pixels=[];
      async function read(key){
        const buffer=renderer.device.createBuffer({size:256*256*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const encoder=renderer.device.createCommandEncoder();encoder.copyTextureToBuffer({texture:renderer.tiles.get(key).color},{buffer,bytesPerRow:1024},[256,256]);
        renderer.device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);
        const pixel=Array.from(new Uint8Array(buffer.getMappedRange()).slice((128*256+128)*4,(128*256+128)*4+3));buffer.unmap();buffer.destroy();return pixel;
      }
      async function screenPixel(){
        const buffer=renderer.device.createBuffer({size:256*256*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const encoder=renderer.device.createCommandEncoder();encoder.copyTextureToBuffer({texture:renderer.context.getCurrentTexture()},{buffer,bytesPerRow:1024},[256,256]);
        renderer.device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);
        let pixel=Array.from(new Uint8Array(buffer.getMappedRange()).slice((128*256+128)*4,(128*256+128)*4+3));buffer.unmap();buffer.destroy();
        if(renderer.format.startsWith('bgra'))pixel.reverse();return pixel;
      }
      for(const c of fixtures.cases){
        window.biomeTestLight=c.lighting?1:0;
        const h=new Float32Array(304*304),gradient=Math.tan(c.slope*Math.PI/180);
        for(let y=0;y<304;y++)for(let x=0;x<304;x++)h[y*304+x]=c.height+(x-152)*c.resolution*gradient;
        renderer.setBiomeRockSlope(c.threshold);
        renderer.uploadTile(c.name,h,{mode:'orogen-biomes',climate,metresPerSample:c.resolution,lod:Math.log2(c.resolution/30)});
        pixels.push({name:c.name,pixel:await read(c.name)});
      }
      // Legacy slope preferences must no longer change informational Biomes.
      window.biomeTestLight=0;
      renderer.setBiomeRockSlope(40);renderer.setMode('orogen-biomes');
      renderer.draw([{key:'changed-threshold',lod:0,x:0,y:0,width:256,height:256}],{width:256,height:256,dpr:1});
      const recolored=await read('changed-threshold');
      // The native coarse fragment path accepts the same 21-channel payload.
      const coarseClimate=new Float32Array(21*41*41);
      fixtures.fields.forEach((v,i)=>coarseClimate.fill(v,(5+i)*41*41,(6+i)*41*41));
      renderer.uploadCoarse('coarse',new Float32Array(160*160).fill(Math.sqrt(6000)),{mode:'orogen-biomes',climate:coarseClimate});
      renderer.draw([{key:'coarse',lod:7,x:0,y:0,width:256,height:256}],{width:256,height:256,dpr:1});
      const coarse=await screenPixel();
      window.biomeTestLight=1;
      renderer.draw([{key:'coarse',lod:7,x:0,y:0,width:256,height:256}],{width:256,height:256,dpr:1});
      const coarseLit=await screenPixel();
      window.biomeTestLight=0;
      const terrainClimate=new Float32Array(koppenFixtures.climate),classified=[];
      for(const c of koppenFixtures.cases){
        if(c.seasons)c.seasons.forEach((v,i)=>terrainClimate.fill(v,(21+i)*33*33,(22+i)*33*33));
        for(const [mode,expected] of [['orogen-koppen',c.koppen],['orogen-biomes',c.biome]]){
          const key=mode+'/'+classified.length;
          renderer.setMode(mode);
          renderer.uploadTile(key,new Float32Array(304*304).fill(c.height),{mode,climate:terrainClimate});
          classified.push({mode,height:c.height,pixel:await read(key),expected});
          // The fragment path classifies the interpolated coarse relief too.
          renderer.uploadCoarse(key+'/coarse',new Float32Array(160*160).fill(Math.sign(c.height)*Math.sqrt(Math.abs(c.height))),{mode,climate:terrainClimate,climateWidth:33,climateHeight:33});
          renderer.setMode(mode);renderer.draw([{key:key+'/coarse',lod:7,x:0,y:0,width:256,height:256}],{width:256,height:256,dpr:1});
          classified.push({mode:mode+'/coarse',height:c.height,pixel:await screenPixel(),expected});
          renderer.deleteTile(key);renderer.deleteTile(key+'/coarse');
        }
      }
      const stats=renderer.getStats();renderer.dispose();return {pixels,recolored,coarse,coarseLit,stats,classified};
    },{fixtures,koppenFixtures});
    for(const c of fixtures.cases){const result=report.pixels.find(p=>p.name===c.name);result.pixel.forEach((v,i)=>assert(Math.abs(v-c.expected[i])<=1,`${c.name}: GPU ${result.pixel}, CPU ${c.expected}`));}
    report.recolored.forEach((v,i)=>assert(Math.abs(v-[.92,.93,.96][i]*255)<=1,'Informational Biomes keeps snow on steep terrain'));
    report.coarse.forEach((v,i)=>assert(Math.abs(v-[.92,.93,.96][i]*255)<=1,'Native coarse uses adaptive snow colors'));
    const flatLit=fixtures.cases.find(c=>c.name==='snow-flat-lit').expected;
    report.coarseLit.forEach((v,i)=>assert(Math.abs(v-flatLit[i])<=1,`Native coarse preserves snow illumination: ${report.coarseLit} vs ${flatLit}`));
    const shadow=report.pixels.find(p=>p.name==='snow-shaded').pixel;
    const sun=report.pixels.find(p=>p.name==='snow-sunlit').pixel;
    assert(sun[0]-shadow[0]>75,'Snow retains the relief shadow contrast');
    assert.deepEqual(errors,[]);assert.equal(report.stats.status,'webgpu');
    for(const c of report.classified)c.pixel.forEach((v,i)=>assert(Math.abs(v-c.expected[i])<=1,`${c.mode} at ${c.height}m: GPU ${c.pixel}, CPU ${c.expected}`));
    console.log(JSON.stringify({ok:true,cases:report.pixels.length,classified:report.classified.length,recolored:report.recolored,coarse:report.coarse,coarseLit:report.coarseLit,shadow,sun}));
  }finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
