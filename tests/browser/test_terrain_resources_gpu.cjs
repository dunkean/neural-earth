// Resource layers in the real WebGPU renderer: a synthetic physical tile shaded
// with the packages of tests/fixtures/resources_gpu_parity.json (continuous,
// seasonal, realized geology, deposits). Checks repeatability, season response, zone
// coherence and package switching. Requires a real adapter.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {loadPlaywright,browserExecutable}=require('../../tools/platform.cjs');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/resources-gpu');fs.mkdirSync(out,{recursive:true});
const fixture=JSON.parse(fs.readFileSync(path.join(root,'tests/fixtures/resources_gpu_parity.json'),'utf8'));
const scripts=['terrain_render_controls.js','terrain_forest_brushes.js','terrain_render3d.js','terrain_render_v2.js','terrain_resources_gpu.js','terrain_renderer.js'];
const html='<meta charset="utf-8"><style>body{margin:0;background:#151b1e}canvas{display:block}</style><canvas id="map"></canvas><script>window.TerrainLighting={vectors:()=>[1,.35,1,1,-.5,-.5,.70710678,0]};</script>'+scripts.map(s=>`<script src="/${s}"></script>`).join('');
const configs=Object.fromEntries(fixture.configs.filter(c=>['positive','seasonal','geology','points'].includes(c.name)).map(c=>[c.name,{header:c.header,field:c.field,tau:c.tau||null,deposits:c.deposits||null,buckets:c.buckets||null}]));
(async()=>{
 const server=http.createServer((req,res)=>{const file=new URL(req.url,'http://x').pathname.slice(1);res.setHeader('Content-Type',file.endsWith('.js')?'application/javascript':'text/html');res.end(file.endsWith('.js')?fs.readFileSync(path.join(root,'web',file)):html);});
 await new Promise(ok=>server.listen(0,'127.0.0.1',ok));
 const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
 try{
  const page=await browser.newPage({viewport:{width:512,height:512}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const report=await page.evaluate(async configs=>{
   if(!navigator.gpu||!await navigator.gpu.requestAdapter())return {skipped:true};
   let failure=null;const r=await createTerrainRenderer(document.getElementById('map'),{maxBytes:256*1024**2,onStatus:(s,e)=>{if(s==='fallback')failure=e;}});
   if(!r)throw Error(failure||'Renderer failed');
   r.context.configure({device:r.device,format:r.format,alphaMode:'premultiplied',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
   const RG=window.TerrainResourcesGPU;
   async function read(){
    r.draw([{key:'fixture',x:0,y:0,width:512,height:512,uv:[0,0,1,1],lod:2}],{width:512,height:512,dpr:1});
    const row=2048,b=r.device.createBuffer({size:row*512,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e=r.device.createCommandEncoder();
    e.copyTextureToBuffer({texture:r.context.getCurrentTexture()},{buffer:b,bytesPerRow:row},[512,512]);r.device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);
    const bytes=new Uint8Array(b.getMappedRange()).slice(),rgb=new Uint8Array(512*512*3);
    for(let i=0;i<512*512;i++)for(let c=0;c<3;c++)rgb[i*3+c]=bytes[i*4+(r.format.startsWith('bgra')?2-c:c)];b.unmap();b.destroy();return rgb;
   }
   const mean=a=>a.reduce((v,x)=>v+x,0)/a.length,difference=(a,b)=>mean(Array.from(a,(x,i)=>Math.abs(x-b[i])));
   // Hilly land tile, LOD 2 (120 m), inside the fixture lattice (LOD-10 cells -8..7, -4..3).
   const n=33,L=n*n,climate=new Float32Array(51*L),fill=(l,v)=>climate.fill(v,l*L,(l+1)*L);
   fill(0,12);fill(2,800);[22,8,.0065,2,6,.0065].forEach((v,i)=>fill(21+i,v));fill(49,1);
   const h=new Float32Array(304*304),water=new Float32Array(304*304*4);
   for(let y=0;y<304;y++)for(let x=0;x<304;x++){const i=y*304+x;h[i]=500+300*Math.sin(x/23)*Math.cos(y/31)+120*Math.sin((x+y)/9);water.set([0,0,0,1],i*4);}
   r.setMode('resources-fixture');
   r.uploadTile('fixture',h,{climate,climateWidth:n,climateHeight:n,hydrology:water,metresPerSample:120,origin:[-160000,-40000],mode:'resources-fixture'});
   const images={},timings={};
   const load=c=>{const t0=performance.now();r.setResources(c.header,{field:new Uint16Array(c.field),fieldShape:[c.header.field_shape[2],c.header.field_shape[1],c.header.field_shape[0]],
     tau:c.tau?new Uint16Array(c.tau):null,deposits:c.deposits?new Uint32Array(c.deposits):null,buckets:c.buckets?new Uint32Array(c.buckets):null},{season:.5});return performance.now()-t0;};
   for(const [name,c] of Object.entries(configs)){timings[name]=load(c);images[name]=await read();}
   const repeat=await read();
   load(configs.seasonal);r.setResourceSeason(.05);const january=await read();r.setResourceSeason(.55);const july=await read();
   // Zones: coherent regions, not per-pixel noise.
   load(configs.geology);const zones=await read();let changes=0,pairs=0;const colours=new Set();
   for(let y=0;y<511;y+=2)for(let x=0;x<511;x+=2){const i=(y*512+x)*3,j=i+3,k=i+512*3;colours.add(zones[i]<<16|zones[i+1]<<8|zones[i+2]);
     for(const o of [j,k]){pairs++;if(Math.abs(zones[i]-zones[o])+Math.abs(zones[i+1]-zones[o+1])+Math.abs(zones[i+2]-zones[o+2])>40)changes++;}}
   const names=Object.keys(images),distinct=[];
   for(let a=0;a<names.length;a++)for(let b=a+1;b<names.length;b++)distinct.push([names[a],names[b],difference(images[names[a]],images[names[b]])]);
   return {skipped:false,status:r.status,failure,timings,repeatExact:images.points.every((v,i)=>v===repeat[i]),seasonDifference:difference(january,july),
     zoneBoundaryShare:changes/pairs,zoneColours:colours.size,distinct};
  },configs);
  if(report.skipped){console.log('SKIP: actual WebGPU adapter unavailable');return;}
  assert.equal(report.status,'webgpu');assert.equal(report.failure,null);
  assert(report.repeatExact,'same package, same pixels');
  assert(report.seasonDifference>2,'seasonal layers follow the month');
  for(const [a,b,d] of report.distinct)assert(d>2,`${a} and ${b} must differ`);
  assert(report.zoneColours>=3,'several zone classes');
  assert(report.zoneBoundaryShare<.15,`zones are regions (boundary share ${report.zoneBoundaryShare})`);
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'browser-report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
 }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
