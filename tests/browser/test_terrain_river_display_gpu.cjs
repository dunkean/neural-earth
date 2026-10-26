// Production GPU ribbon/union test; analytical geometry is a technical gate,
// not a substitute for the three real regional river captures.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../../tools/platform.cjs');
const root=path.resolve(__dirname,'../..');
(async()=>{
 const html='<canvas id="map"></canvas>'+['terrain_render_controls.js','terrain_forest_brushes.js','terrain_render3d.js','terrain_render_v2.js','terrain_renderer.js'].map(f=>`<script src="/${f}"></script>`).join('');
 const server=http.createServer((q,s)=>{const f=new URL(q.url,'http://local').pathname.slice(1);s.setHeader('Content-Type',f.endsWith('.js')?'application/javascript':'text/html');s.end(f.endsWith('.js')?fs.readFileSync(path.join(root,'web',f)):html);});
 await new Promise(ok=>server.listen(0,'127.0.0.1',ok));
 const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
 try{
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result=await page.evaluate(async()=>{
   const r=await createTerrainRenderer(document.getElementById('map'));if(!r)throw Error('Renderer unavailable');
   let frame={key:'cross',ready:true,month:7,networkRevision:1,hydroId:'test',authority:'test-regional',segments:new Float32Array([16,64.5,112,64.5,5,-123.456,-123.456,0,64.5,16,64.5,112,5,8123.123,8123.123,0])};
   window.TerrainHydrology={riverDisplay:()=>frame,acknowledgeRiverDisplay:()=>{},regionalRiverDisplayEnabled:()=>false};
   r.tiles.set('test',{native:false,shadedMode:-8,params:[304,304,-8,0]});const rects=[{key:'test',x:0,y:0,width:128,height:128}];
   async function draw(width=128,height=128){const e=r.device.createCommandEncoder();r.prepareRiverDisplay(e,width,height,width,height,rects);r.device.queue.submit([e.finish()]);r.finishRiverDisplay();if(r.riverDisplayPromise)await r.riverDisplayPromise;await r.device.queue.onSubmittedWorkDone();}
   function half(h){const sign=h&32768?-1:1,exp=h>>10&31,f=h&1023;return sign*(exp?exp===31?Infinity:2**(exp-15)*(1+f/1024):2**-14*f/1024);}
   async function read(){const w=r.riverMaskWidth,h=r.riverMaskHeight,row=Math.ceil(w*8/256)*256,b=r.device.createBuffer({size:row*h,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e=r.device.createCommandEncoder();e.copyTextureToBuffer({texture:r.riverMask},{buffer:b,bytesPerRow:row},[w,h]);r.device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);const a=new Uint16Array(b.getMappedRange()).slice();b.unmap();b.destroy();return {w,h,at:(x,y)=>Array.from(a.slice(y*row/2+x*4,y*row/2+x*4+4),half)};}
   await draw();await draw();if(r.riverDisplayError)throw Error(r.riverDisplayError);
   let a=await read(),cross=a.at(64,64),edge=a.at(30,58),signed=a.at(30,64),other=a.at(64,30);
   if(cross[0]!==1||edge[0]!==0)throw Error('Confluence hole or excessive edge '+JSON.stringify({cross,edge}));
   if(Math.abs(signed[1]+signed[2]+123.456)>.002||Math.abs(other[1]+other[2]-8123.123)>.002)throw Error('Signed level precision lost');
   let partial=0;for(let y=45;y<85;y++)if(a.at(30,y)[0]>0&&a.at(30,y)[0]<1)partial++;if(partial>2)throw Error('Bank AA wider than two screen pixels');
   frame={...frame,key:'thin',segments:new Float32Array([16,64.5,112,64.5,.01,0,0,0])};await draw();a=await read();const thin=a.at(64,64)[0];if(Math.abs(thin-.02)>.0001)throw Error('Subpixel width inflated '+thin);
   frame={...frame,key:'empty',segments:new Float32Array()};await draw();a=await read();if(a.w!==1||a.h!==1||a.at(0,0)[0]!==0||a.at(0,0)[3]!==2||!r.riverDisplayStats.ready)throw Error('Empty mask not constant-size ready');
   frame={...frame,key:'fallback',ready:false};await draw();a=await read();if(a.at(0,0)[3]!==1||r.riverDisplayStats.mode!=='Render3D-overlay-fallback')throw Error('Unidentified fallback');
   frame={...frame,key:'resize',ready:true,segments:new Float32Array([16,32.5,48,32.5,4,0,0,0])};await draw(64,64);a=await read();if(a.w!==64||a.h!==64||a.at(32,32)[0]!==1)throw Error('Resize misalignment');
   frame={...frame,key:'translation',segments:new Float32Array([16,35.5,48,35.5,1,0,0,0])};await draw(64,64);a=await read();if(a.at(32,35)[0]!==1||a.at(32,32)[0]!==0)throw Error('Translated geometry stale');
   const stats={...r.riverDisplayStats};r.tiles.delete('test');
   // The obsolete source ribbon must not carve a straight population gap.
   TerrainHydrology.regionalRiverDisplayEnabled=()=>true;
   r.setMode('render-v2');await r.r3Promise;if(r.r3Error)throw Error(r.r3Error);
   const n=33,L=n*n,climate=new Float32Array(51*L),fill=(i,v)=>climate.fill(v,i*L,(i+1)*L);
   fill(0,27.5);fill(2,2700);[29,1400,.0065,26,1300,.0065].forEach((v,i)=>fill(21+i,v));
   for(const i of [36,37,38,39,40,41,46,47,48])fill(i,.4);fill(42,.2);climate.set([-2e7,-1e7,2e7,1e7,0,0,17331],45*L);
   const heights=new Float32Array(304*304).fill(80),water=new Float32Array(304*304*4);
   for(let y=0;y<304;y++)for(let x=0;x<304;x++)water.set([85,x>=100&&x<130?1:0,x>=100&&x<130?3:0,1],(y*304+x)*4);
   async function population(w){r.uploadTile('population',heights,{climate,hydrology:w,climateWidth:n,climateHeight:n,metresPerSample:30,origin:[1e7,-8e6],mode:'render-v2'});
    const b=r.device.createBuffer({size:2560*304,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e=r.device.createCommandEncoder();
    e.copyTextureToBuffer({texture:r.tiles.get('population').aux,origin:[0,0,2]},{buffer:b,bytesPerRow:2560},[304,304]);r.device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);const a=new Uint16Array(b.getMappedRange()).slice();b.unmap();b.destroy();return a;}
   const withOld=await population(water);for(let i=0;i<water.length;i+=4){water[i+1]=0;water[i+2]=0;}
   const withoutOld=await population(water);if(withOld.some((v,i)=>v!==withoutOld[i]))throw Error('Obsolete source river still changes population/bank');
   const oldLinePopulation=half(withOld[150*1280+115*4]);if(oldLinePopulation<=.1)throw Error('Population gate had no live forest witness');
   r.clear();if(r.riverMask||r.riverDisplayStats.mode!=='inactive')throw Error('Clear retained stale mask');r.dispose();
   return {cross,edge,signedLevel:signed[1]+signed[2],positiveLevel:other[1]+other[2],partialBankPixels:partial,thin,oldLinePopulation,stats};
  });assert.deepEqual(errors,[]);console.log(JSON.stringify({pass:true,result,errors}));
 }finally{await browser.close();await new Promise(ok=>server.close(ok));}
})().catch(e=>{console.error(e);process.exitCode=1;});
