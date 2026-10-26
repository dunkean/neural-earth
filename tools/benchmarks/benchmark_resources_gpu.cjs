// GPU cost of the resource layers next to Render v2, in the same renderer and
// page: bake (blur + shade passes) per 256-sample tile, native coarse
// full-screen draw, and package switch. GPU pass durations via timestamp
// queries injected around the unmodified renderer (as benchmark_render_layer.cjs).
//
// usage: node tools/benchmarks/benchmark_resources_gpu.cjs [--packages DIR] [--origin X,Y] [--json OUT]
// DIR holds packages saved from /api/resources/<seed>/gpu?mode=... as <mode>.bin
// (default output/resources-gpu/packages); without them, the parity fixture is used.
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const root=path.resolve(__dirname,'../..');
const arg=(name,fallback)=>{const i=process.argv.indexOf(name);return i>0?process.argv[i+1]:fallback;};
const dir=arg('--packages',path.join(root,'output/resources-gpu/packages')),jsonOut=arg('--json',null);
const origin=arg('--origin','-2100000,-4100000').split(',').map(Number);
const scripts=['terrain_render_controls.js','terrain_forest_brushes.js','terrain_render3d.js','terrain_render_v2.js','terrain_resources_gpu.js','terrain_renderer.js'];
const html='<meta charset="utf-8"><canvas id="map"></canvas><script>window.TerrainLighting={vectors:()=>[1,.35,1,1,-.5,-.5,.70710678,0]};</script>'+scripts.map(s=>`<script src="/${s}"></script>`).join('');
function packages(){
  const out={};
  if(fs.existsSync(dir))for(const file of fs.readdirSync(dir).filter(f=>f.endsWith('.bin'))){
    const bytes=fs.readFileSync(path.join(dir,file)),length=bytes.readUInt32LE(0);
    out[file.slice(0,-4)]={header:JSON.parse(bytes.subarray(4,4+length).toString('utf8')),body:bytes.subarray(4+length)};
  }
  if(Object.keys(out).length)return out;
  // Fallback: the parity fixture packages (small lattice; same binary layout).
  const fixture=JSON.parse(fs.readFileSync(path.join(root,'tests/fixtures/resources_gpu_parity.json'),'utf8'));
  for(const c of fixture.configs.filter(c=>['positive','zones','points','stock'].includes(c.name))){
    const parts=[Buffer.from(new Uint16Array(c.field).buffer)];
    if(c.tau)parts.push(Buffer.from(new Uint16Array(c.tau).buffer));
    if(c.deposits){const size=parts.reduce((n,b)=>n+b.length,0);parts.push(Buffer.alloc((4-size%4)%4),Buffer.from(new Uint32Array(c.deposits).buffer),Buffer.from(new Uint32Array(c.buckets).buffer));}
    out['resources-fixture-'+c.name]={header:{...c.header,mode:'resources-fixture-'+c.name},body:Buffer.concat(parts)};
  }
  return out;
}
(async()=>{
  const pkgs=packages();
  const server=http.createServer((req,res)=>{
    if(req.url.startsWith('/package/')){res.setHeader('Content-Type','application/octet-stream');return res.end(pkgs[decodeURIComponent(req.url.slice(9))].body);}
    if(req.url==='/forest-brushes.bin'){res.setHeader('Content-Type','application/octet-stream');return res.end(fs.readFileSync(path.join(root,'web/forest-brushes.bin')));}
    const file=new URL(req.url,'http://x').pathname.slice(1);
    res.setHeader('Content-Type',file.endsWith('.js')?'application/javascript':'text/html');res.end(file.endsWith('.js')?fs.readFileSync(path.join(root,'web',file)):html);
  });
  await new Promise(ok=>server.listen(0,'127.0.0.1',ok));let browser;
  try{
    browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu','--enable-webgpu-developer-features','--enable-dawn-features=allow_unsafe_apis','--disable-dawn-features=timestamp_quantization']});
    const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.addInitScript(()=>{
      const request=GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice=function(descriptor={}){return request.call(this,{...descriptor,requiredFeatures:[...(descriptor.requiredFeatures||[]),...(this.features.has('timestamp-query')?['timestamp-query']:[])]});};
      const timer=window.gpuTimer={active:false,set:null,count:0,capacity:4096};
      for(const name of ['beginComputePass','beginRenderPass']){
        const begin=GPUCommandEncoder.prototype[name];
        GPUCommandEncoder.prototype[name]=function(descriptor={}){
          if(!timer.active||timer.count+2>timer.capacity)return begin.call(this,descriptor);
          const index=timer.count;timer.count+=2;
          return begin.call(this,{...descriptor,timestampWrites:{querySet:timer.set,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});
        };
      }
    });
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    const report=await page.evaluate(async({pkgs,origin})=>{
      const N=33,L=N*N,climate=new Float32Array(51*L),med=a=>[...a].sort((x,y)=>x-y)[a.length>>1],r3=v=>Math.round(v*1000)/1000;
      const fill=(a,v)=>climate.fill(v,a*L,(a+1)*L);
      fill(0,15);fill(2,800);fill(4,-.0065);[22,400,.0065,2,300,.0065].forEach((v,i)=>fill(21+i,v));
      for(const l of [36,37,38,39,40,41,46,47,48])fill(l,.45);fill(42,.3);fill(49,1);climate.set([-2e7,-1e7,2e7,1e7,1,0,1234],45*L);
      const heights=k=>{const h=new Float32Array(304*304);for(let y=0;y<304;y++)for(let x=0;x<304;x++)h[y*304+x]=1200+3800*Math.exp(-((x-190)**2+(y-125)**2)/4200)+180*Math.sin(x*.09+k)*Math.cos(y*.07);return h;};
      const tiles=Array.from({length:32},(_,i)=>heights(i*.37));
      const roots=new Float32Array(160*160);for(let y=0;y<160;y++)for(let x=0;x<160;x++)roots[y*160+x]=Math.sqrt(2000+800*Math.sin(x*.08)+500*Math.cos(y*.11));
      const renderer=await createTerrainRenderer(document.getElementById('map'),{maxBytes:512*1024**2,onStatus:(s,d)=>{if(s==='fallback')window.rendererFailure=d;}});
      if(!renderer?.available)throw Error('WebGPU unavailable: '+window.rendererFailure);
      const RG=window.TerrainResourcesGPU,done=()=>renderer.device.queue.onSubmittedWorkDone();
      const gpu=async f=>{
        if(!renderer.device.features.has('timestamp-query'))return null;
        const device=renderer.device;gpuTimer.set??=device.createQuerySet({type:'timestamp',count:gpuTimer.capacity});
        await done();gpuTimer.count=0;gpuTimer.active=true;try{await f();}finally{gpuTimer.active=false;}
        const count=gpuTimer.count;if(!count)return 0;
        const resolve=device.createBuffer({size:count*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:count*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const e=device.createCommandEncoder();e.resolveQuerySet(gpuTimer.set,0,count,resolve,0);e.copyBufferToBuffer(resolve,0,read,0,count*8);device.queue.submit([e.finish()]);
        await read.mapAsync(GPUMapMode.READ);const v=new BigUint64Array(read.getMappedRange());let total=0n;for(let i=0;i<count;i+=2)if(v[i+1]>v[i])total+=v[i+1]-v[i];
        read.unmap();read.destroy();resolve.destroy();await done();return Number(total)/1e6;
      };
      const gpuBench=async(n,f)=>{const s=[];for(let i=0;i<n;i++)s.push(await gpu(()=>f(i)));return s[0]===null?null:r3(med(s));};
      const time=async f=>{const s=performance.now();await f();await done();return performance.now()-s;};
      const bench=async(n,f)=>{await f(-1);const s=[];for(let i=0;i<n;i++)s.push(await f(i));return r3(med(s));};
      const upload=(mode,mpp)=>{for(let i=0;i<32;i++)renderer.uploadTile('t'+i,tiles[i],{climate,climateWidth:N,climateHeight:N,metresPerSample:mpp,lod:Math.log2(mpp/30),origin:[origin[0]+(i%8)*256*mpp,origin[1]+Math.floor(i/8)*256*mpp],mode});};
      const out={gpu:renderer.adapter?.info?.description||renderer.adapter?.info?.vendor||'',bake:{},coarseDraw:{},packageSwitchMs:{},packageBytes:{}};
      // Render v2 needs its material atlas.
      renderer.setMode('render-v2');await renderer.r3Promise;
      const modes=[['render-v2',null],...Object.entries(pkgs)];
      const load=async(name,pkg)=>{const bytes=await (await fetch('/package/'+encodeURIComponent(name))).arrayBuffer(),data=RG.parse(pkg.header,bytes);return ()=>renderer.setResources(pkg.header,data,{season:.5});};
      const big=[{key:'c0',lod:7,x:0,y:0,width:1920,height:1080,uv:[.25,.25,.3,.2]}],view={width:1920,height:1080,dpr:1};
      for(const [name,pkg] of modes){
        const mode=pkg?name:'render-v2';renderer.clear();renderer.setMode(mode);
        if(pkg){const apply=await load(name,pkg);out.packageBytes[name]=pkg.size;out.packageSwitchMs[name]=await bench(5,()=>time(apply));}
        await done();
        for(const mpp of [30,120,1920]){
          const wall=await bench(3,async()=>{renderer.clear();await done();return time(()=>upload(mode,mpp));});
          const g=await gpuBench(5,async()=>{renderer.clear();upload(mode,mpp);});
          out.bake[name+'/'+mpp+'m']={wallPerTileMs:r3(wall/32),gpuPerTileMs:g===null?null:r3(g/32)};
        }
        renderer.clear();await done();
        renderer.uploadCoarse('c0',roots,{climate,climateWidth:N,climateHeight:N,origin:[origin[0],origin[1]],mode});
        renderer.draw(big,view);await done();
        out.coarseDraw[name]={gpuMs:await gpuBench(7,()=>renderer.draw(big,view)),wallMs:await bench(9,()=>time(()=>renderer.draw(big,view)))};
      }
      out.failure=window.rendererFailure||null;return out;
    },{pkgs:Object.fromEntries(Object.entries(pkgs).map(([k,v])=>[k,{header:v.header,size:v.body.length}])),origin});
    if(errors.length)throw Error(errors.join('\n'));
    console.log(JSON.stringify(report,null,1));
    if(jsonOut){fs.mkdirSync(path.dirname(jsonOut),{recursive:true});fs.writeFileSync(jsonOut,JSON.stringify(report,null,2));}
  }finally{await browser?.close();await new Promise(ok=>server.close(ok));}
})().catch(e=>{console.error(e);process.exitCode=1;});
