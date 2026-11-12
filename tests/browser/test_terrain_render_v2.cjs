const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {loadPlaywright,browserExecutable}=require('../../tools/platform.cjs');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/render-v2');fs.mkdirSync(out,{recursive:true});
require('../../web/terrain_render_v2.js');const V2=globalThis.TerrainRenderV2;
assert(V2.isStyle('Render_v2'));assert(V2.isStyle('render-v2-snow'));assert(!V2.isStyle('render3d'));
V2.set({season:1});assert.equal(V2.season(),0);V2.set({season:.25});const params=new URLSearchParams();V2.writeParams(params);assert.equal(params.get('v2season'),'.25'.replace(/^\./,'0.'));
const html='<meta charset="utf-8"><style>body{margin:0;background:#151b1e;color:white;font:14px sans-serif}canvas{display:block}</style><div id="label"></div><canvas id="map"></canvas><script>window.TerrainLighting={vectors:()=>[1,.35,1,1,-.5,-.5,.70710678,0]};</script>'+['terrain_render_controls.js','terrain_forest_brushes.js','terrain_render3d.js','terrain_magna_satellite.js','terrain_render_v2.js','terrain_renderer.js'].map(s=>`<script src="/${s}"></script>`).join('');
(async()=>{
 const server=http.createServer((req,res)=>{const file=new URL(req.url,'http://x').pathname.slice(1);res.setHeader('Content-Type',file.endsWith('.js')?'application/javascript':'text/html');res.end(file.endsWith('.js')?fs.readFileSync(path.join(root,'web',file)):html);});await new Promise(ok=>server.listen(0,'127.0.0.1',ok));
 const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
 try{
  const page=await browser.newPage({viewport:{width:768,height:600}}),errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const report=await page.evaluate(async()=>{
   if(!navigator.gpu||!await navigator.gpu.requestAdapter())return {skipped:true};
   let failure=null;const r=await createTerrainRenderer(document.getElementById('map'),{maxBytes:256*1024**2,onStatus:(s,e)=>{if(s==='fallback')failure=e;}});if(!r)throw Error(failure||'Renderer failed');
   let drawSource;const compile=r.compileShader.bind(r);r.compileShader=(code,label)=>{if(label==='Render3D quads')drawSource=code;return compile(code,label);};
   r.context.configure({device:r.device,format:r.format,alphaMode:'premultiplied',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
   r.setMode('render-v2');await r.r3Promise;if(r.r3Error)throw Error(r.r3Error);
   const definitions=[['arid-mineral',38,13,35,60],['temperate',24,-8,600,500],['humid-forest',29,24,1400,1300],['sandy-coast',30,13,280,260],['snow-mountain',-2,-20,500,500],['cold-coast',-4,-24,350,350]];
   const results=[];window.v2Probe={r,definitions,results};
   const mean=a=>a.reduce((v,x)=>v+x,0)/a.length;
   const difference=(a,b)=>mean(a.map((x,i)=>Math.abs(x-b[i])));
   async function read(uv=[0,0,1,1],key='fixture'){
    r.draw([{key,x:0,y:0,width:512,height:512,uv,lod:0}],{width:512,height:512,dpr:1});
    const row=2048,b=r.device.createBuffer({size:row*512,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e=r.device.createCommandEncoder();e.copyTextureToBuffer({texture:r.context.getCurrentTexture()},{buffer:b,bytesPerRow:row},[512,512]);r.device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);
    const bytes=new Uint8Array(b.getMappedRange()).slice(),rgb=new Uint8Array(512*512*3);for(let i=0;i<512*512;i++)for(let c=0;c<3;c++)rgb[i*3+c]=bytes[i*4+(r.format.startsWith('bgra')?2-c:c)];b.unmap();b.destroy();return rgb;
   }
   function put(def,glacier=true){
    const n=33,L=n*n,climate=new Float32Array(51*L),fill=(l,v)=>climate.fill(v,l*L,(l+1)*L);
    fill(0,(def[1]+def[2])*.5);fill(2,def[3]+def[4]);[def[1],def[3],.0065,def[2],def[4],.0065].forEach((v,i)=>fill(21+i,v));for(const l of [36,37,38,39,40,41,46,47,48])fill(l,.4);fill(42,def[0].includes('sandy')?.85:.3);fill(49,1);fill(50,def[0]==='cold-coast'&&glacier?1:0);climate.set([-2e7,-1e7,2e7,1e7,0,0,17331],45*L);
    const h=new Float32Array(304*304),water=new Float32Array(304*304*4);
    for(let y=0;y<304;y++)for(let x=0;x<304;x++){const i=y*304+x;h[i]=60+35*Math.sin(x/40)*Math.cos(y/60);let kind=0,cover=0,level=0;
     if(x>250){h[i]=-30-(x-250)*3;kind=1;cover=1;}
     if(x<45&&y<90)h[i]=-12; // authoritative dry below-sea-level basin
     if(Math.abs(x-145)<7){kind=3;cover=Math.max(0,1-Math.abs(x-145)/7);level=h[i]+8*cover;}
     if((x-78)**2+(y-222)**2<32**2){kind=2;cover=1;level=h[i]+15;}
     water.set([level,cover,kind,1],i*4);
    }
    // Rare glacial fragments require a kilometre-scale coastal sample, not a 110 m strip.
    window.v2Probe.climate=climate;
    const start=performance.now();r.uploadTile('fixture',h,{climate,climateWidth:n,climateHeight:n,hydrology:water,metresPerSample:def[0]==='cold-coast'?100:3.75,origin:[100000,-300000],mode:'render-v2'});return performance.now()-start;
   }
   window.v2Probe.put=put;window.v2Probe.read=read;
   for(const def of definitions){
    const uploadSubmitMs=put(def);TerrainRenderV2.set({season:.5});const summer=await read(),repeat=await read();
    TerrainRenderV2.set({season:0});const winter=await read();TerrainRenderV2.set({season:1});const loop=await read();
    let icebergDifference=0;if(def[0]==='cold-coast'){put(def,false);TerrainRenderV2.set({season:.5});const noSource=await read();icebergDifference=difference(summer,noSource);put(def,true);}
    const field=r.tiles.get('fixture').aux;TerrainRenderV2.set({season:.5});await read();const stableField=field===r.tiles.get('fixture').aux;
    r.setMode('render-v2-vegetation');const vegetation=await read();r.setMode('render-v2-substrate');const substrate=await read();r.setMode('render-v2');
    // Pixel centres map to full-tile coordinate halo + uv*256.
    const pixel=(a,x,y)=>{const px=Math.floor((x-24)/256*512),py=Math.floor((y-24)/256*512);return Array.from(a.slice((py*512+px)*3,(py*512+px)*3+3));};
    results.push({name:def[0],uploadSubmitMs,icebergDifference,seasonDifference:difference(summer,winter),diagnosticDifference:difference(summer,substrate),repeatExact:summer.every((v,i)=>v===repeat[i]),annualLoopExact:winter.every((v,i)=>v===loop[i]),stableField,waterVegetation:pixel(vegetation,78,222),dryBasin:pixel(summer,35,55),lakeSummer:pixel(summer,78,222),lakeWinter:pixel(winter,78,222)});
   }
   // Probe the actual imported functions in compute, with two different
   // tile origins representing identical physical points. This isolates the
   // procedural phase from the independently sampled DEM/climate textures.
   const probeModule=await compile(drawSource+`
    @group(1) @binding(0) var<storage,read_write> precisionOut:array<vec4<f32>>;
    @group(1) @binding(1) var<uniform> precisionOffset:vec4<f32>;
    @compute @workgroup_size(32) fn precisionProbe(@builtin(global_invocation_id) id:vec3<u32>){
      if(id.x>=64u){return;}
      let p=vec2<f32>(13.5+f32(id.x)*.125,21.5+f32(id.x)*.09375)+precisionOffset.xy;
      mu_global.uGroundCover=vec4<f32>(.5,.707,.5,1.);mu_global.uMaterialClimate=vec2<f32>(14.,900.);
      mu_surfaceGroundHeight=500.;mu_surfaceGroundGradient=vec2<f32>(.2,.1);mu_surfaceGroundEnv=vec4<f32>(.8,.2,.7,0.);
      let value=magnaMicro(p,.25,.1,.7,.2,17331u);
      precisionOut[id.x]=vec4<f32>(value,magnaPattern(vec3<f32>(p,0.),3.2,.1,17331u));
    }`,'Satellite precision probe');
   const probePipeline=await r.device.createComputePipelineAsync({layout:'auto',compute:{module:probeModule,entryPoint:'precisionProbe'}});
   async function probe(origin,offset){
    const device=r.device,uniform=device.createBuffer({size:TerrainRender3D.tileUniform(origin,17331).byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(uniform,0,TerrainRender3D.tileUniform(origin,17331));
    const input=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,new Float32Array([...offset,0,0]));
    const output=device.createBuffer({size:1024,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),readback=device.createBuffer({size:1024,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const group0=device.createBindGroup({layout:probePipeline.getBindGroupLayout(0),entries:[{binding:5,resource:{buffer:uniform}}]}),group1=device.createBindGroup({layout:probePipeline.getBindGroupLayout(1),entries:[{binding:0,resource:{buffer:output}},{binding:1,resource:{buffer:input}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(probePipeline);pass.setBindGroup(0,group0);pass.setBindGroup(1,group1);pass.dispatchWorkgroups(2);pass.end();encoder.copyBufferToBuffer(output,0,readback,0,1024);device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
    const values=new Float32Array(readback.getMappedRange()).slice();readback.unmap();for(const b of [uniform,input,output,readback])b.destroy();return values;
   }
   const precision=[];
   for(const origin of [[0,0],[-12300000,8730000],[20000000,-10000000],[-1037.5,-1045.5]]){
    const a=await probe(origin,[0,0]),b=await probe([origin[0]+37.5,origin[1]-22.5],[-37.5,22.5]);
    precision.push({origin,maxError:Math.max(...a.map((v,i)=>Math.abs(v-b[i]))),finite:a.every(Number.isFinite)&&b.every(Number.isFinite)});
   }
   // Execute the separate native-coarse material path too: merely compiling
   // it does not detect a missing population or black material response.
   put(definitions[1]);const coarse=new Float32Array(160*160);
   for(let y=0;y<160;y++)for(let x=0;x<160;x++)coarse[y*160+x]=Math.sqrt(500+200*Math.sin(x/19)*Math.cos(y/23));
   r.uploadCoarse('native',coarse,{width:160,height:160,halo:16,encoding:'signed-sqrt',metresPerSample:7680,climate:window.v2Probe.climate,climateWidth:33,climateHeight:33,origin:[-12000000,6000000],mode:'render-v2'});
   TerrainRenderV2.set({season:.5});const nativeSummer=await read(undefined,'native'),nativeRepeat=await read(undefined,'native');
   TerrainRenderV2.set({season:0});const nativeWinter=await read(undefined,'native');
   return {results,precision,native:{mean:mean(nativeSummer),repeatExact:nativeSummer.every((v,i)=>v===nativeRepeat[i]),seasonDifference:difference(nativeSummer,nativeWinter)},stats:r.getStats().render3d,errors:failure};
  });
  if(report.skipped){console.log('SKIP: actual WebGPU adapter unavailable');return;}
  assert.equal(report.errors,null);assert.equal(report.stats.ready,true);for(const row of report.results){assert(row.repeatExact,row.name);assert(row.annualLoopExact,row.name);assert(row.stableField,row.name);assert(row.diagnosticDifference>3,row.name);assert.deepEqual(row.waterVegetation,[0,0,33],row.name);}
  assert(report.results.find(x=>x.name==='cold-coast').icebergDifference>.001);assert(report.results.find(x=>x.name==='temperate').seasonDifference>3);assert(report.results.find(x=>x.name==='snow-mountain').seasonDifference>1);
  const cold=report.results.find(x=>x.name==='temperate');assert(cold.lakeWinter.reduce((s,x)=>s+x,0)>cold.lakeSummer.reduce((s,x)=>s+x,0)+100);
  assert(report.native.repeatExact);assert(report.native.mean>10&&report.native.mean<240);assert(report.native.seasonDifference>3);
  for(const row of report.precision){assert(row.finite);assert(row.maxError<.006,JSON.stringify(row));}
  for(const [tag,date] of [['summer',.5],['winter',0]]){
   await page.evaluate(async({tag,date})=>{TerrainRenderV2.set({season:date});document.getElementById('label').textContent='Native coarse Satellite / '+tag;await v2Probe.read(undefined,'native');},{tag,date});
   await page.locator('body').screenshot({path:path.join(out,`native-${tag}.png`)});
  }
  for(const index of [0,1,2,3,4,5])for(const [tag,date,uv] of [['summer',.5,[0,0,1,1]],['winter',0,[0,0,1,1]],['close',.5,[.13,.35,.12,.12]],['bank',.5,[.35,.35,.12,.12]]]){
   await page.evaluate(async({index,date,uv,tag})=>{const p=window.v2Probe;p.put(p.definitions[index]);TerrainRenderV2.set({season:date});document.getElementById('label').textContent=p.definitions[index][0]+' / '+tag;await p.read(uv);},{index,date,uv,tag});
   await page.locator('body').screenshot({path:path.join(out,`${report.results[index].name}-${tag}.png`)});
  }
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
 }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
