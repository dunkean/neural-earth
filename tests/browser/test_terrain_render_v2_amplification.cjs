// Merge contract: erosion, fine slopes and regional rivers have independent
// resources/uniforms in Relief, legacy Render3D and seasonal V2.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../../tools/platform.cjs');
const root=path.resolve(__dirname,'../..');
(async()=>{
 const html='<canvas id="map"></canvas>'+['terrain_render_controls.js','terrain_forest_brushes.js','terrain_render3d.js','terrain_render_v2.js','terrain_renderer.js'].map(f=>`<script src="/${f}"></script>`).join('');
 const server=http.createServer((q,s)=>{const f=new URL(q.url,'http://local').pathname.slice(1);s.setHeader('Content-Type',f.endsWith('.js')?'application/javascript':'text/html');s.end(f.endsWith('.js')?fs.readFileSync(path.join(root,'web',f)):html);});await new Promise(ok=>server.listen(0,'127.0.0.1',ok));
 const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
 try{
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result=await page.evaluate(async()=>{
   let regional=false;window.TerrainHydrology={regionalRiverDisplayEnabled:()=>regional};
   const r=await createTerrainRenderer(document.getElementById('map'));if(!r)throw Error('Renderer unavailable');
   r.setMode('render-v2');await r.r3Promise;if(r.r3Error)throw Error(r.r3Error);
   const width=304,L=33*33,h=new Float32Array(width*width).fill(800),climate=new Float32Array(50*L),fill=(i,v)=>climate.fill(v,i*L,(i+1)*L);
   [24,900,.0065,6,800,.0065].forEach((v,i)=>fill(21+i,v));for(const i of [36,37,38])fill(i,.4);for(const i of [39,40,41])fill(i,.5);fill(42,.3);for(const i of [46,47,48])fill(i,.4);fill(49,1);climate.set([-2e7,-1e7,2e7,1e7,0,0,1234],45*L);
   const gradient=new Float32Array(width*width*2);for(let i=0;i<width*width;i++)gradient.set([.125,-.25],i*2);
   const surface=new Uint8Array(width*width*4).fill(200),options={climate,climateWidth:33,climateHeight:33,metresPerSample:30*2**-8,lod:-8,origin:[-8232421.875,-400390.625]};
   async function read(key){const texture=r.tiles.get(key).color,b=r.device.createBuffer({size:256*256*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e=r.device.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer:b,bytesPerRow:1024},[256,256]);r.device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);const bytes=new Uint8Array(b.getMappedRange()).slice();b.unmap();b.destroy();return bytes;}
   const results=[];
   for(const mode of ['relief','render','render3d','render-v2','render-v2-vegetation']){
    r.clear();r.setMode(mode);regional=false;r.uploadTile('plain',h,options);const plain=await read('plain');
    regional=true;r.uploadTile('river',h,options);const river=await read('river');
    r.uploadTile('slope',h,{...options,gradient});const slope=await read('slope');
    r.uploadTile('erosion',h,{...options,gradient,surface});const erosion=await read('erosion');
    const t=r.tiles.get('erosion');if(!t.surface||!t.gradient||t.materialMeta?.[6]!==1234)throw Error('Lost tile resources/metadata');
    r.draw([{key:'erosion',x:0,y:0,width:256,height:256,uv:[0,0,1,1],lod:-8}],{width:256,height:256,dpr:1});await r.device.queue.onSubmittedWorkDone();
    results.push({mode,riverFlagAloneUnchanged:plain.every((v,i)=>v===river[i]),slopeChanged:plain.some((v,i)=>v!==slope[i]),erosionChanged:slope.some((v,i)=>v!==erosion[i])});
   }
   const stats=r.getStats();r.dispose();return {results,stats};
  });
  for(const v of result.results){assert(v.riverFlagAloneUnchanged,JSON.stringify(v));assert(v.slopeChanged,JSON.stringify(v));if(v.mode!=='relief')assert(v.erosionChanged,JSON.stringify(v));}
  assert.deepEqual(errors,[]);console.log(JSON.stringify(result));
 }finally{await browser.close();await new Promise(ok=>server.close(ok));}
})().catch(e=>{console.error(e);process.exitCode=1;});
