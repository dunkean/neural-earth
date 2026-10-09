const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {readRepositoryFile} = require(NEURAL_EARTH_ROOT + '/tools/repository-files.cjs');
// Isolated WebGPU shader QA, without requests to the model/server. Readback only here.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const http=require('node:http');
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');
(async()=>{
  const source=readRepositoryFile(path.join(NEURAL_EARTH_ROOT,'terrain_renderer.js'));
  const server=http.createServer((req,res)=>{
    if(req.url==='/terrain_renderer.js'){res.setHeader('Content-Type','application/javascript');res.end(source);}
    else {res.setHeader('Content-Type','text/html');res.end('<canvas id="map" style="width:512px;height:256px"></canvas><script src="/terrain_renderer.js"></script>');}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try {
    browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--enable-unsafe-webgpu']});
    const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    const report=await page.evaluate(async()=>{
      const statuses=[];const renderer=await createTerrainRenderer(document.getElementById('map'),{maxBytes:3*1024**2,onStatus:(status,detail)=>statuses.push({status,detail})});
      if(!renderer)return {available:false,statuses};
      const readPixel=async(key,x=128,y=128)=>{
        const buffer=renderer.device.createBuffer({size:256*256*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const encoder=renderer.device.createCommandEncoder();encoder.copyTextureToBuffer({texture:renderer.tiles.get(key).color},{buffer,bytesPerRow:1024},[256,256]);
        renderer.device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);const pixel=Array.from(new Uint8Array(buffer.getMappedRange()).slice((y*256+x)*4,(y*256+x)*4+4));buffer.unmap();buffer.destroy();return pixel;
      };
      const height=new Float32Array(304*304).fill(1000);
      renderer.uploadTile('flat',height,{});const flat=await readPixel('flat');
      renderer.uploadTile('ocean',new Float32Array(304*304).fill(-1000),{});const ocean=await readPixel('ocean');
      renderer.draw([{key:'flat',x:0,y:0,width:256,height:256},{key:'ocean',x:256,y:0,width:256,height:256}],{width:512,height:256,dpr:1});
      for(let y=0;y<304;y++)for(let x=0;x<304;x++)height[y*304+x]=1000+(x-152)*20+(y-152)*10;
      renderer.uploadTile('plane',height,{});const plane=await readPixel('plane');
      renderer.draw([{key:'plane',x:-20,y:0,width:512,height:256}],{width:512,height:256,dpr:2});
      for(let y=0;y<304;y++)for(let x=0;x<304;x++)height[y*304+x]=1000+30*Math.sin((x-152)*.25)+15*Math.sin((y-152)*.17);
      renderer.uploadTile('waves',height,{});const waves=await readPixel('waves');
      const climate=new Float32Array(5*33*33);
      climate.fill(20,0,33*33);climate.fill(1000,2*33*33,3*33*33);climate.fill(-.0065,4*33*33);
      renderer.uploadTile('temperature',new Float32Array(304*304).fill(1000),{climate,mode:'temperature'});
      const temperature=await readPixel('temperature');
      renderer.uploadTile('precipitation',new Float32Array(304*304).fill(1000),{climate,mode:'precipitation'});
      const precipitation=await readPixel('precipitation');
      renderer.uploadTile('biomes',new Float32Array(304*304).fill(1000),{climate,mode:'biomes'});
      const biomes=await readPixel('biomes');
      climate.fill(-25,0,33*33);
      renderer.uploadTile('polar',new Float32Array(304*304).fill(-1000),{climate,mode:'biomes'});
      const polar=await readPixel('polar');
      await renderer.device.queue.onSubmittedWorkDone();
      const stats=renderer.getStats();const cacheKeys=[...renderer.tiles.keys()];
      renderer.clear();renderer.draw([],{width:512,height:256});const cleared=renderer.getStats();
      renderer.device.destroy();await renderer.device.lost;await new Promise(resolve=>setTimeout(resolve,50));
      return {available:true,adapter:{vendor:renderer.adapter.info.vendor,architecture:renderer.adapter.info.architecture},flat,ocean,plane,waves,temperature,precipitation,biomes,polar,stats,cacheKeys,cleared,lost:renderer.status,statuses};
    });
    assert.equal(report.available,true,JSON.stringify(report));assert.deepEqual(errors,[]);
    const terrain=t=>t<.5?[0,.8,.4].map((v,i)=>v+([1,1,.6][i]-v)*(t-.25)*4):t<.75?[1,1,.6].map((v,i)=>v+([.5,.36,.33][i]-v)*(t-.5)*4):[.5,.36,.33].map(v=>v+(1-v)*(t-.75)*4);
    const land=(height,dx,dy)=>{const hs=Math.max(0,(-.5*dx-.5*dy+Math.SQRT1_2)/Math.hypot(dx,dy,1))**.85;return terrain(.25+.75*(height/4500)**.7).map(v=>Math.round(v*(.35+.65*hs)*255)).concat(255);};
    const close=(actual,expected)=>actual.forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<=1,`${actual} versus ${expected}`));
    close(report.flat,land(1000,0,0));close(report.plane,land(1000,4,2));
    const attenuation=(sigma,radius,frequency)=>{let sum=0,response=0;for(let i=-radius;i<=radius;i++){const w=Math.exp(-i*i/(2*sigma*sigma));sum+=w;response+=w*Math.cos(i*frequency);}return response/sum;};
    const hs=(dx,dy)=>Math.max(0,(-.5*dx-.5*dy+Math.SQRT1_2)/Math.hypot(dx,dy,1));
    const waveShade=(sigma,radius)=>hs(30*Math.sin(.25)*attenuation(sigma,radius,.25)/5,15*Math.sin(.17)*attenuation(sigma,radius,.17)/5);
    const waveIntensity=.35+.65*(.75*waveShade(6,24)+.25*waveShade(1.2,5))**.85;
    close(report.waves,terrain(.25+.75*(1000/4500)**.7).map(v=>Math.round(v*waveIntensity*255)).concat(255));
    const t=.1**.7;close(report.ocean,[.68,.88,1].map((v,i)=>Math.round((v+([0,.1,.45][i]-v)*t)*255)).concat(255));
    const temp=(13.5+35)/70;close(report.temperature,[.18,.38,.88].map((v,i)=>Math.round((v+([.95,.22,.08][i]-v)*temp)*255)).concat(255));
    const rain=Math.log(1001)/Math.log(4001);close(report.precipitation,[.78,.61,.32].map((v,i)=>Math.round((v+([.08,.36,.72][i]-v)*rain)*255)).concat(255));
    close(report.polar,[.82,.93,.98].map(v=>Math.round(v*255)).concat(255));
    assert.ok(report.biomes[1]>report.biomes[0],'Wet temperate land is green');
    assert.ok(report.stats.totalBytes<=report.stats.maxBytes);assert.ok(report.cacheKeys.length<3,'LRU should evict at this deliberately small budget');
    assert.equal(report.cleared.tiles,0);assert.equal(report.cleared.bytes,0);assert.equal(report.lost,'fallback');
    console.log(JSON.stringify({ok:true,...report},null,2));
  } finally {await browser?.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
