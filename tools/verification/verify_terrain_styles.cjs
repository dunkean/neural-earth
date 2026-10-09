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
  const styles='window.TerrainStyles='+readRepositoryFile(path.join(NEURAL_EARTH_ROOT,'terrain_styles.json'),'utf8')+';\n'+readRepositoryFile(path.join(NEURAL_EARTH_ROOT,'terrain_style_rendering.js'),'utf8');
  const server=http.createServer((req,res)=>{
    if(req.url==='/terrain_styles.js'){res.setHeader('Content-Type','application/javascript');res.end(styles);}
    else if(req.url==='/terrain_renderer.js'){res.setHeader('Content-Type','application/javascript');res.end(source);}
    else {res.setHeader('Content-Type','text/html');res.end('<meta charset="utf-8"><canvas id="map" style="width:512px;height:256px"></canvas><script src="/terrain_styles.js"></script><script src="/terrain_renderer.js"></script>');}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try {
    browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--enable-unsafe-webgpu']});
    const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    const report=await page.evaluate(async()=>{
      const statuses=[];const renderer=await createTerrainRenderer(document.getElementById('map'),{maxBytes:64*1024**2,onStatus:(status,detail)=>statuses.push({status,detail})});
      if(!renderer)return {available:false,statuses};
      const readColors=async key=>{
        const buffer=renderer.device.createBuffer({size:256*256*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const encoder=renderer.device.createCommandEncoder();encoder.copyTextureToBuffer({texture:renderer.tiles.get(key).color},{buffer,bytesPerRow:1024},[256,256]);
        renderer.device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);const colors=new Uint8Array(buffer.getMappedRange()).slice();buffer.unmap();buffer.destroy();return colors;
      };
      const readPixel=async(key,x=128,y=128)=>Array.from((await readColors(key)).slice((y*256+x)*4,(y*256+x)*4+4));
      const readScreen=()=>{
        const c=document.createElement('canvas');c.width=256;c.height=256;
        const ctx=c.getContext('2d');ctx.drawImage(renderer.canvas,0,0);
        return ctx.getImageData(0,0,256,256).data;
      };
      const styles={};
      for(const mode of Object.keys(TerrainStyles)){
        renderer.uploadTile(mode,new Float32Array(304*304).fill(1000),{mode});
        styles[mode]=await readPixel(mode);
      }
      // A flat field isolates paper grain from relief and intentional engraving hatches.
      // Planet-scale coordinates exposed diagonal bands in the old f32 sine hash.
      const grain=[];
      for(const mode of ['illuminated','blueprint','cadastre','engraving','parchment'])for(const lod of [4,3,2,1,0,-1,-2,-3]){
        const resolution=30*2**lod,step=1,key=`grain-${mode}-${lod}`;
        renderer.uploadTile(key,new Float32Array(304*304).fill(1000),{mode,lod,metresPerSample:resolution,origin:[-20000000,-10000000]});
        for(const scale of [1,8]){
        // Enlarging a cached parent tile must enlarge only relief, never its grain.
        renderer.draw([{key,lod,x:128-128*scale,y:128-128*scale,width:256*scale,height:256*scale}],{width:256,height:256,dpr:1});
        const colors=readScreen(),correlations=[];
        for(const [dx,dy]of [[1,0],[0,1],[1,1],[6,-1],[6,1],[1,6],[1,-6]]){
          let n=0,a=0,b=0,aa=0,bb=0,ab=0;
          for(let y=0;y<256;y+=step)for(let x=0;x<256;x+=step){
            const xx=x+dx*step,yy=y+dy*step;if(xx<0||xx>=256||yy<0||yy>=256)continue;
            const u=colors[(y*256+x)*4+2],v=colors[(yy*256+xx)*4+2];
            n++;a+=u;b+=v;aa+=u*u;bb+=v*v;ab+=u*v;
          }
          correlations.push((n*ab-a*b)/Math.sqrt((n*aa-a*a)*(n*bb-b*b)));
        }
        grain.push({mode,lod,scale,maxCorrelation:Math.max(...correlations.map(Math.abs))});
        }
        renderer.deleteTile(key);
      }
      // Screen-space curves change without uploading another physical tile.
      const ramp=new Float32Array(304*304);for(let y=0;y<304;y++)for(let x=0;x<304;x++)ramp[y*304+x]=500+(x-152)*2;
      renderer.uploadTile('contour-ramp',ramp,{mode:'topographic'});
      const screenPixel=(x,y)=>Array.from(readScreen().slice((y*256+x)*4,(y*256+x)*4+4));
      const rect=[{key:'contour-ramp',x:0,y:0,width:256,height:256}];
      TerrainStyleRendering.set({enabled:false,interval:100});renderer.draw(rect,{width:256,height:256,dpr:1});const noLine=screenPixel(128,128),uploads=renderer.uploads;
      TerrainStyleRendering.set({enabled:true,interval:100});renderer.draw(rect,{width:256,height:256,dpr:1});const mainLine=screenPixel(128,128),between=screenPixel(153,128);
      const contourUploads=renderer.uploads-uploads;
      const contourBuffer=renderer.tiles.get('contour-ramp').contourBuffer,contourCount=renderer.tiles.get('contour-ramp').contourCount;
      TerrainStyleRendering.set({width:.3});renderer.draw(rect,{width:256,height:256,dpr:1});const thinEdge=screenPixel(130,128);
      TerrainStyleRendering.set({width:4});renderer.draw(rect,{width:256,height:256,dpr:1});const wideEdge=screenPixel(130,128);
      const widthReuse=renderer.tiles.get('contour-ramp').contourBuffer===contourBuffer;
      TerrainStyleRendering.set({interval:200});renderer.draw(rect,{width:256,height:256,dpr:1});
      const halfDensityCount=renderer.tiles.get('contour-ramp').contourCount;
      TerrainStyleRendering.set({width:.9,interval:100});
      // Native coarse uses the same palettes and isolines after interpolation.
      const roots=new Float32Array(304*304);for(let i=0;i<roots.length;i++)roots[i]=Math.sqrt(ramp[i]);
      const climateNative=new Float32Array(5*33*33);
      renderer.uploadCoarse('coarse-style',roots,{mode:'topographic',width:304,height:304,halo:24,metresPerSample:30,climate:climateNative,climateWidth:33,climateHeight:33});
      renderer.draw([{key:'coarse-style',x:0,y:0,width:256,height:256}],{width:256,height:256,dpr:1});const coarseLine=screenPixel(128,128);
      const gallery=document.createElement('div');gallery.style='display:grid;grid-template-columns:repeat(5,256px);gap:10px;background:#172129;color:white;font:14px system-ui';document.body.append(gallery);
      const landscape=new Float32Array(304*304);
      for(let y=0;y<304;y++)for(let x=0;x<304;x++)landscape[y*304+x]=3200*Math.exp(-((x-190)**2+(y-130)**2)/6000)+1200*Math.sin(x/65)*Math.cos(y/75)-250;
      TerrainStyleRendering.set({enabled:true,interval:200});
      for(const mode of Object.keys(TerrainStyles)){
        renderer.uploadTile('gallery-'+mode,landscape,{mode});renderer.draw([{key:'gallery-'+mode,x:0,y:0,width:256,height:256}],{width:256,height:256,dpr:1});
        const cell=document.createElement('div'),img=document.createElement('img');img.src=renderer.canvas.toDataURL();cell.textContent=TerrainStyles[mode].label;cell.append(img);gallery.append(cell);
      }
      TerrainStyleRendering.set({enabled:false});
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
      const physicalTexture=renderer.tiles.get('contour-ramp').elevation,physicalColor=renderer.tiles.get('contour-ramp').color,styleUploads=renderer.uploads;
      renderer.setMode('blueprint');renderer.draw(rect,{width:256,height:256,dpr:1});const blueStyle=screenPixel(128,128);
      renderer.setMode('parchment');renderer.draw(rect,{width:256,height:256,dpr:1});const parchmentStyle=screenPixel(128,128);
      const styleReuse=renderer.uploads===styleUploads&&renderer.tiles.get('contour-ramp').elevation===physicalTexture&&renderer.tiles.get('contour-ramp').color===physicalColor;
      await renderer.device.queue.onSubmittedWorkDone();
      const stats=renderer.getStats();const cacheKeys=[...renderer.tiles.keys()];
      renderer.clear();renderer.draw([],{width:512,height:256});const cleared=renderer.getStats();
      renderer.device.destroy();await renderer.device.lost;await new Promise(resolve=>setTimeout(resolve,50));
      return {available:true,grain,styleReuse,blueStyle,parchmentStyle,styles,noLine,mainLine,between,coarseLine,contourUploads,thinEdge,wideEdge,widthReuse,contourCount,halfDensityCount,adapter:{vendor:renderer.adapter.info.vendor,architecture:renderer.adapter.info.architecture},flat,ocean,plane,waves,temperature,precipitation,biomes,polar,stats,cacheKeys,cleared,lost:renderer.status,statuses};
    });
    assert.equal(report.available,true,JSON.stringify(report));assert.deepEqual(errors,[]);
    for(const sample of report.grain)assert(sample.maxCorrelation<.06,`${sample.mode} LOD ${sample.lod} at ${sample.scale}x: enlarged or directional paper noise (${sample.maxCorrelation})`);
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
    assert.ok(report.stats.totalBytes<=report.stats.maxBytes);
    assert.equal(Object.keys(report.styles).length,10);assert.equal(new Set(Object.values(report.styles).map(p=>p.join(','))).size,10);
    const palettes=JSON.parse(readRepositoryFile(path.join(NEURAL_EARTH_ROOT,'terrain_styles.json'),'utf8'));
    for(const [mode,pixel]of Object.entries(report.styles)){
      const pal=palettes[mode],t=(1000/4500)**(mode==='copernicus'?1:.85),upper=pal.hypso.findIndex(s=>s[0]>=t),a=pal.hypso[upper-1],b=pal.hypso[upper],u=(t-a[0])/(b[0]-a[0]);
      for(let c=0;c<3;c++){const start=parseInt(a[1].slice(1+c*2,3+c*2),16),end=parseInt(b[1].slice(1+c*2,3+c*2),16);assert(Math.abs(pixel[c]-(start+(end-start)*u))<=Math.ceil(255*pal.grain)+2,`${mode} must use the shared physical palette`);}
    }
    assert(report.styleReuse,'fine palettes reuse existing physical and color textures');assert(report.blueStyle[2]>report.blueStyle[0]);assert(report.parchmentStyle[0]>report.parchmentStyle[2]);
    assert.equal(report.contourUploads,0,'contour settings reuse the physical GPU tile');
    assert(report.wideEdge[0]<report.thinEdge[0]-20,'width changes the stroke on the GPU');
    assert(report.widthReuse,'changing width keeps the contour geometry');
    assert(report.halfDensityCount<report.contourCount,'double interval reduces GPU segments on the visible ramp');
    assert(report.mainLine[0]<report.noLine[0]-20,'main contour is visible');assert(report.between[0]>report.mainLine[0]+20,'contours do not fill the bands');
    close(report.coarseLine,report.mainLine);
    await page.setViewportSize({width:1350,height:900});await page.screenshot({path:path.join(NEURAL_EARTH_ROOT,'tmp','terrain-styles.png')});
    assert.equal(report.cleared.tiles,0);assert.equal(report.cleared.bytes,0);assert.equal(report.lost,'fallback');
    console.log(JSON.stringify({ok:true,...report},null,2));
  } finally {await browser?.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
