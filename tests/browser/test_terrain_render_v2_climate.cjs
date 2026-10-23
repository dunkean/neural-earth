// Technical GPU round-trip only. Visual acceptance uses generated-world tiles.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {loadPlaywright,browserExecutable}=require('../../tools/platform.cjs');
const root=path.resolve(__dirname,'../..');
const scripts=['terrain_render_controls.js','terrain_lighting.js','terrain_forest_brushes.js','terrain_render3d.js','terrain_render_v2.js','terrain_renderer.js'];
const html='<canvas id="map"></canvas>'+scripts.map(s=>`<script src="/${s}"></script>`).join('');
(async()=>{
 const server=http.createServer((req,res)=>{const f=new URL(req.url,'http://x').pathname.slice(1);res.setHeader('Content-Type',scripts.includes(f)?'application/javascript':'text/html');res.end(scripts.includes(f)?fs.readFileSync(path.join(root,'web',f)):html);});await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result=await page.evaluate(async()=>{
   let fallback='';const r=await createTerrainRenderer(document.getElementById('map'),{maxBytes:128*1024**2,onStatus:(s,d)=>{if(s==='fallback')fallback=d;}});if(!r)throw Error('WebGPU unavailable: '+fallback);r.setMode('render-v2');await r.r3Promise;if(r.r3Error)throw Error(r.r3Error);
   const n=304,c=33,L=c*c,climate=new Float32Array(51*L),fill=(l,v)=>climate.fill(v,l*L,(l+1)*L);
   [28,30,.0045,12,150,.0093].forEach((v,i)=>fill(21+i,v));fill(0,20);fill(2,180);for(const l of [36,37,38,39,40,41,46,47,48])fill(l,.4);fill(42,.3);fill(49,1);climate.set([-2e7,-1e7,2e7,1e7,0,0,17331],45*L);
   const heights=new Float32Array(n*n),water=new Float32Array(n*n*4);for(let y=0;y<n;y++)for(let x=0;x<n;x++){heights[y*n+x]=1500+5*x;water[(y*n+x)*4+3]=1;}
   TerrainRenderV2.set({season:.5});r.uploadTile('roundtrip',heights,{climate,climateWidth:c,climateHeight:c,hydrology:water,mode:'render-v2',metresPerSample:30,origin:[100000,200000]});
   const draw=()=>r.draw([{key:'roundtrip',x:0,y:0,width:128,height:128,lod:0}],{width:128,height:128,dpr:1});draw();
   const tile=r.tiles.get('roundtrip'),initialAux=tile.aux,uploads=r.uploads;
   const row=Math.ceil(n*8/256)*256,buffer=r.device.createBuffer({size:row*n,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),encoder=r.device.createCommandEncoder();encoder.copyTextureToBuffer({texture:tile.aux,origin:[0,0,3]},{buffer,bytesPerRow:row},[n,n,1]);r.device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);const data=new DataView(buffer.getMappedRange());
   function half(v){const s=(v&0x8000)?-1:1,e=(v>>10)&31,m=v&1023;return e===0?s*2**-14*m/1024:e===31?(m?NaN:s*Infinity):s*2**(e-15)*(1+m/1024);}
   const samples=[];for(const x of [24,25,26,101,152,230,279]){const i=140*row+x*8,ts=half(data.getUint16(i,true))*100-50,tw=half(data.getUint16(i+4,true))*100-50,potential=half(data.getUint16(i+2,true))*5;const h=heights[140*n+x];samples.push({x,height:h,ts,tw,rain:potential*Math.max(120,20*(ts+tw)*.5+280),expectedTs:28-.0045*h,expectedTw:12-.0093*h});}buffer.unmap();buffer.destroy();
   TerrainRenderV2.set({season:0});draw();await r.device.queue.onSubmittedWorkDone();const calendar={date:tile.v2BakeSeason,sameAux:tile.aux===initialAux,sameUploads:r.uploads===uploads,uniformDate:r.renderVectors(-8)[0],legacyDate:r.renderVectors(-6)[0]};
   const preciseBytes=tile.r3Bytes;r.setMode('render3d');draw();await r.device.queue.onSubmittedWorkDone();const legacy={format:tile.auxFormat,bytes:tile.r3Bytes};r.setMode('render-v2');draw();await r.device.queue.onSubmittedWorkDone();const precise={format:tile.auxFormat,bytes:tile.r3Bytes};
   // Observe the actual native coarse material bake, not just its key. A
   // calendar change must update its snow/soil while reusing physical roots.
   const makeTexture=r.makeTexture.bind(r);r.makeTexture=(w,h,f,u=0)=>makeTexture(w,h,f,u|GPUTextureUsage.COPY_SRC);
   fill(22,1000);fill(25,1000);const roots=new Float32Array(160*160).fill(Math.sqrt(1500));
   r.uploadCoarse('coarse',roots,{climate,climateWidth:c,climateHeight:c,mode:'render-v2',origin:[100000,200000]});
   const native=r.tiles.get('coarse'),nativeUploads=r.uploads;
   const drawNative=()=>r.draw([{key:'coarse',x:0,y:0,width:128,height:128,lod:4}],{width:128,height:128,dpr:1});
   async function pixel(){const b=r.device.createBuffer({size:256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e=r.device.createCommandEncoder();e.copyTextureToBuffer({texture:native.albedo,origin:[160,160]},{buffer:b,bytesPerRow:256},[1,1]);r.device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);const range=b.getMappedRange(),p=native.albedoFormat==='rgba16float'?Array.from(new Uint16Array(range).slice(0,4),v=>half(v)*255):Array.from(new Uint8Array(range).slice(0,4));b.unmap();b.destroy();return p;}
   TerrainRenderV2.set({season:0});drawNative();await r.device.queue.onSubmittedWorkDone();const winter=await pixel(),nativeAlbedo=native.albedo,winterKey=native.bakedKey;
   TerrainRenderV2.set({season:.5});drawNative();await r.device.queue.onSubmittedWorkDone();const summer=await pixel();
   const summerKey=native.bakedKey,savedLighting=JSON.parse(TerrainLighting.serialize());
   TerrainLighting.set({global:{...TerrainLighting.get('global'),strength:0}});drawNative();await r.device.queue.onSubmittedWorkDone();const unlit=await pixel();
   const coarse={winter,summer,unlit,sameAlbedo:native.albedo===nativeAlbedo,sameUploads:r.uploads===nativeUploads,calendarChanged:summerKey!==winterKey,lightChanged:native.bakedKey!==summerKey,format:native.albedoFormat,bytes:native.albedoBytes};TerrainLighting.set(savedLighting);
   TerrainLighting.set({...savedLighting,lods:{...savedLighting.lods,4:{...TerrainLighting.get(4),strength:0}}});drawNative();await r.device.queue.onSubmittedWorkDone();coarse.lodUnlit=await pixel();TerrainLighting.set(savedLighting);
   const savedForest=TerrainRender3D.get();TerrainRender3D.set({...savedForest,enabled:false});drawNative();await r.device.queue.onSubmittedWorkDone();coarse.disabledCanopy=(await pixel())[3];TerrainRender3D.set(savedForest);
   drawNative();await r.device.queue.onSubmittedWorkDone();const preciseTotal=r.bytes;
   r.setMode('render3d');drawNative();await r.device.queue.onSubmittedWorkDone();const legacyPixel=await pixel(),legacyTotal=r.bytes;coarse.legacyFormat=native.albedoFormat;coarse.legacyBytes=native.albedoBytes;
   r.setMode('render-v2');drawNative();await r.device.queue.onSubmittedWorkDone();coarse.preciseRoundtripBytes=r.bytes;coarse.preciseTotal=preciseTotal;coarse.allocationDelta=preciseTotal-legacyTotal;
   r.setMode('render3d');drawNative();await r.device.queue.onSubmittedWorkDone();coarse.legacyRoundtripPixel=await pixel();coarse.legacyPixel=legacyPixel;
   r.dispose();return {samples,calendar,preciseBytes,legacy,precise,coarse};
  });
  // Half-float temperature error also enters the decoded rain denominator;
  // allow <0.34% here, while the former lapse mismatch changes it by >5%.
  for(const s of result.samples){assert(Math.abs(s.ts-s.expectedTs)<.055,JSON.stringify(s));assert(Math.abs(s.tw-s.expectedTw)<.055,JSON.stringify(s));assert(Math.abs(s.rain-180)<.6,JSON.stringify(s));}assert.notEqual(result.samples[0].ts,result.samples[1].ts,'Temperature must retain sub-8bit elevation changes');assert.deepEqual(result.calendar,{date:0,sameAux:true,sameUploads:true,uniformDate:0,legacyDate:.5});assert.equal(result.legacy.format,'rgba8unorm');assert.equal(result.precise.format,'rgba16float');assert.equal(result.precise.bytes,result.preciseBytes);assert(result.precise.bytes>result.legacy.bytes*1.9);
  assert(result.coarse.sameAlbedo&&result.coarse.sameUploads&&result.coarse.calendarChanged&&result.coarse.lightChanged,JSON.stringify(result.coarse));assert.notDeepEqual(result.coarse.unlit,result.coarse.summer,'Mean canopy was not updated when lighting changed');assert.deepEqual(result.coarse.lodUnlit,result.coarse.unlit,'Coarse bake must use the displayed LOD lighting');assert.equal(result.coarse.disabledCanopy,0,'Forest settings must invalidate the coarse population');assert(result.coarse.winter[0]>result.coarse.summer[0]+20,'Native coarse snow did not follow the calendar: '+JSON.stringify(result.coarse));assert.equal(result.coarse.format,'rgba16float');assert.equal(result.coarse.legacyFormat,'rgba8unorm');assert.equal(result.coarse.bytes,2*result.coarse.legacyBytes);assert.equal(result.coarse.preciseRoundtripBytes,result.coarse.preciseTotal);assert.equal(result.coarse.allocationDelta,result.coarse.legacyBytes);assert.deepEqual(result.coarse.legacyRoundtripPixel,result.coarse.legacyPixel);assert.deepEqual(errors,[]);console.log(JSON.stringify({ok:true,...result}));
 }finally{if(browser)await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
