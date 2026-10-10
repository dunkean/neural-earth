const {loadPlaywright, browserExecutable, pythonExecutable} = require('../platform.cjs');
const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {readRepositoryFile} = require(NEURAL_EARTH_ROOT + '/tools/repository-files.cjs');
// Real GPU oracle, material reset and coarse-frame completion benchmark.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const {chromium}=loadPlaywright();
const fixtures=JSON.parse(execFileSync(pythonExecutable(),['tests/python/test_terrain_render.py','--fixtures'],{cwd:NEURAL_EARTH_ROOT,encoding:'utf8'}));
(async()=>{
  const server=http.createServer((req,res)=>{
    res.setHeader('Content-Type',req.url.endsWith('.js')?'application/javascript':'text/html');
    res.end(req.url.endsWith('.js')?readRepositoryFile(path.join(NEURAL_EARTH_ROOT,req.url)):'<meta charset="utf-8"><div id="controls"></div><canvas id="map"></canvas><script>window.TerrainLighting={vectors:()=>[0,.35,1,1,-.5,-.5,.70710678,0]};</script><script src="/terrain_render_controls.js"></script><script src="/terrain_renderer.js"></script>');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
  try{
    browser=await chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
    const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    const report=await page.evaluate(async ({fixtures})=>{
      let failure='';const renderer=await createTerrainRenderer(document.getElementById('map'),{maxBytes:64*1024**2,onStatus:(s,d)=>{if(s==='fallback')failure=d;}});
      if(!renderer?.available)throw Error(failure||'WebGPU unavailable');
      renderer.context.configure({device:renderer.device,format:renderer.format,alphaMode:'premultiplied',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
      const climate=new Float32Array(fixtures.climate),pixels=[];
      async function read(texture,n,x,y,bgra=false){
        const bytesPerRow=Math.ceil(n*4/256)*256;
        const buffer=renderer.device.createBuffer({size:bytesPerRow*n,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const encoder=renderer.device.createCommandEncoder();encoder.copyTextureToBuffer({texture},{buffer,bytesPerRow},[n,n]);renderer.device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);
        const pixel=Array.from(new Uint8Array(buffer.getMappedRange()).slice(y*bytesPerRow+x*4,y*bytesPerRow+x*4+3));if(bgra)pixel.reverse();buffer.unmap();buffer.destroy();return pixel;
      }
      renderer.setMode('render');
      for(const [i,c] of fixtures.cases.entries()){
        renderer.setRenderSettings({...TerrainRender.defaults(),...c.settings});
        for(const [i,v] of c.seasons.entries())climate.fill(v,(21+i)*33*33,(22+i)*33*33);
        const coarse=c.resolution===7680,n=coarse?160:304,dem=new Float32Array(n*n);
        for(let y=0;y<n;y++)for(let x=0;x<n;x++)dem[y*n+x]=c.height+(x-n/2)*c.resolution*Math.tan(c.slope*Math.PI/180);
        climate[45*33*33+4]=c.plane?0:1;climate[45*33*33+5]=c.polar?1:0;
        if(coarse){
          renderer.uploadCoarse('case',Float32Array.from(dem,h=>Math.sign(h)*Math.sqrt(Math.abs(h))),{climate,climateWidth:33,climateHeight:33,origin:c.origin});
          renderer.draw([{key:'case',lod:7,x:0,y:0,width:128,height:128}],{width:128,height:128,dpr:1});
          pixels.push(await read(renderer.context.getCurrentTexture(),128,64,64,renderer.format.startsWith('bgra')));
        }else{
          renderer.uploadTile('case',dem,{climate,origin:c.origin,metresPerSample:c.resolution,lod:Math.log2(c.resolution/30)});
          pixels.push(await read(renderer.tiles.get('case').color,256,128,128));
        }renderer.deleteTile('case');
      }
      climate[45*33*33+4]=1;climate[45*33*33+5]=0;
      for(const [i,v] of fixtures.seasons.entries())climate.fill(v,(21+i)*33*33,(22+i)*33*33);
      renderer.setRenderSettings(TerrainRender.defaults());renderer.setMode('soil');
      renderer.uploadTile('soil',new Float32Array(304*304).fill(1000),{climate});
      const soil=await read(renderer.tiles.get('soil').color,256,128,128);
      renderer.setMode('pedology');
      renderer.draw([{key:'soil',lod:0,x:0,y:0,width:256,height:256}],{width:256,height:256,dpr:1});
      const pedology=await read(renderer.tiles.get('soil').color,256,128,128);
      // Recolor cached geometry, including seasonal and exact color reset.
      let changes=0;TerrainRender.mount(document.getElementById('controls'),()=>{changes++;renderer.setRenderSettings(TerrainRender.get());});
      for(const [key,value] of [['season',0],['variation',2],['forest',.2],['moisture',-.5],['rock_slope',65],['snow',.3]]){
        const input=document.getElementById('render_'+key);input.value=value;input.dispatchEvent(new Event('input'));
      }
      for(const key of ['vegetation_tint','rock_tint','snow_color']){const input=document.getElementById('render_'+key);input.value='#aa1122';input.dispatchEvent(new Event('input'));}
      const changed=TerrainRender.get();document.getElementById('renderMaterialReset').click();
      const reset=TerrainRender.get(),exact=Object.fromEntries(['vegetation_tint','rock_tint','snow_color'].map(k=>[k,JSON.parse(document.getElementById('render_'+k).dataset.rgb)]));
      renderer.setMode('render');renderer.draw([{key:'soil',lod:0,x:0,y:0,width:256,height:256}],{width:256,height:256,dpr:1});const afterReset=await read(renderer.tiles.get('soil').color,256,128,128);
      renderer.clear();
      // Completion includes browser submission + GPU work; it is not an
      // isolated GPU timer. Same geometry, resolution and color inputs.
      const roots=new Float32Array(160*160);
      for(let y=0;y<160;y++)for(let x=0;x<160;x++)roots[y*160+x]=Math.sqrt(2000+800*Math.sin(x*.08)+500*Math.cos(y*.11));
      renderer.uploadCoarse('bench',roots,{climate,climateWidth:33,climateHeight:33,origin:[-5e5,-4e5]});
      const bench={};
      for(const mpp of [1000,30,3.75])for(const mode of ['orogen-biomes','render']){
        const rects=[{key:'bench',lod:7,x:0,y:0,width:960,height:540,uv:[.25,.25,960*mpp/(128*7680),540*mpp/(128*7680)]}];
        renderer.setMode(mode);
        for(let i=0;i<8;i++){renderer.draw(rects,{width:960,height:540,dpr:1});await renderer.device.queue.onSubmittedWorkDone();}
        const samples=[];for(let i=0;i<30;i++){const start=performance.now();renderer.draw(rects,{width:960,height:540,dpr:1});await renderer.device.queue.onSubmittedWorkDone();samples.push(performance.now()-start);}
        samples.sort((a,b)=>a-b);bench[mode+'/'+mpp+'m']={medianMs:samples[15],p90Ms:samples[27]};
      }
      // Visual QA at actual fine DEM resolution: wooded hills, bare ridges and
      // snow, with physical shadows. Keep the context alive for the screenshot.
      renderer.clear();renderer.setMode('render');renderer.setRenderSettings(TerrainRender.defaults());
      for(const [i,v]of [[21,30],[23,.006],[24,8],[26,.006]])climate.fill(v,i*33*33,(i+1)*33*33);
      window.TerrainLighting.vectors=()=>[1,.35,1,1,-.5,-.5,.70710678,0];
      const mountain=new Float32Array(304*304);
      for(let y=0;y<304;y++)for(let x=0;x<304;x++)mountain[y*304+x]=1200+3800*Math.exp(-((x-190)**2+(y-125)**2)/4200)+180*Math.sin(x*.09)*Math.cos(y*.07);
      renderer.uploadTile('visual',mountain,{climate,origin:[-5000,-4000]});renderer.draw([{key:'visual',lod:0,x:0,y:0,width:768,height:768}],{width:768,height:768,dpr:1});
      await renderer.device.queue.onSubmittedWorkDone();if(failure)throw Error(failure);window.renderTestRenderer=renderer;return{pixels,soil,pedology,changed,reset,exact,afterReset,changes,bench};
    },{fixtures});
    for(const [i,c] of fixtures.cases.entries())report.pixels[i].forEach((v,k)=>assert(Math.abs(v-c.expected[k])<=2,`Render case ${i}: GPU ${report.pixels[i]} CPU ${c.expected}`));
    report.soil.forEach((v,k)=>assert(Math.abs(v-fixtures.soil[k])<=2,`Soil: GPU ${report.soil} CPU ${fixtures.soil}`));
    report.pedology.forEach((v,k)=>assert(Math.abs(v-fixtures.pedology[k])<=1,`Pedology: GPU ${report.pedology} CPU ${fixtures.pedology}`));
    assert.deepEqual(report.reset,{season:.5,variation:1,forest:1,moisture:0,rock_slope:40,snow:1,vegetation_tint:[1,1,1],rock_tint:[1,1,1],snow_color:[.94,.96,.97]});
    assert.deepEqual(report.exact,{vegetation_tint:[1,1,1],rock_tint:[1,1,1],snow_color:[.94,.96,.97]});assert.equal(report.changes,10);assert.deepEqual(errors,[]);
    await page.locator('#map').screenshot({path:path.join(NEURAL_EARTH_ROOT,'output/render-material-gpu-test.png')});
    await page.evaluate(()=>window.renderTestRenderer.dispose());
    console.log(JSON.stringify({ok:true,cases:report.pixels.length,soil:report.soil,resetExactColors:true,benchCompletion:report.bench}));
  }finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
