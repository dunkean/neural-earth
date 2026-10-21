const {loadPlaywright, browserExecutable} = require('../platform.cjs');
const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {readRepositoryFile} = require(NEURAL_EARTH_ROOT + '/tools/repository-files.cjs');
// Wall-clock (submission + onSubmittedWorkDone) benchmark of the Render material layer vs Relief.
const fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=loadPlaywright();
const jsonArg=process.argv.indexOf('--json'),jsonOut=jsonArg>0?process.argv[jsonArg+1]:null;
// Compare a saved shader revision without changing the working renderer.
const rendererArg=process.argv.indexOf('--renderer');
const rendererSource=rendererArg>0?fs.readFileSync(path.resolve(process.argv[rendererArg+1]),'utf8'):null;
(async()=>{
  const server=http.createServer((req,res)=>{
    if(req.url==='/forest-brushes.bin'){res.setHeader('Content-Type','application/octet-stream');return res.end(fs.readFileSync(path.join(NEURAL_EARTH_ROOT,'web/forest-brushes.bin')));}
    res.setHeader('Content-Type',req.url.endsWith('.js')?'application/javascript':'text/html');
    res.end(req.url==='/terrain_renderer.js'&&rendererSource!==null?rendererSource:req.url.endsWith('.js')?readRepositoryFile(path.join(NEURAL_EARTH_ROOT,req.url)):'<meta charset="utf-8"><div id="controls"></div><canvas id="map"></canvas><script>window.TerrainLighting={vectors:()=>[1,.35,1,1,-.5,-.5,.70710678,0]};</script><script src="/terrain_render_controls.js"></script><script src="/terrain_forest_brushes.js"></script><script src="/terrain_render3d.js"></script><script src="/terrain_renderer.js"></script>');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
  try{
    browser=await chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu','--enable-webgpu-developer-features','--enable-dawn-features=allow_unsafe_apis','--disable-dawn-features=timestamp_quantization']});
    const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
    // GPU pass durations through timestamp queries, injected around the unmodified renderer.
    await page.addInitScript(()=>{
      const request=GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice=function(descriptor={}){return request.call(this,{...descriptor,requiredFeatures:[...(descriptor.requiredFeatures||[]),...(this.features.has('timestamp-query')?['timestamp-query']:[])]});};
      const timer=window.gpuTimer={active:false,device:null,set:null,count:0,capacity:4096};
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
    const report=await page.evaluate(async(N)=>{
      const L=N*N,climate=new Float32Array(50*L),med=a=>[...a].sort((x,y)=>x-y)[a.length>>1],r3=v=>Math.round(v*1000)/1000;
      const fill=(a,b,v)=>climate.fill(v,a*L,(b+1)*L);
      fill(0,0,15);fill(2,2,800);fill(4,4,-.0065);
      [22,400,.0065,2,300,.0065].forEach((v,i)=>fill(21+i,21+i,v));
      fill(36,41,.45);fill(42,42,.3);fill(46,48,.4);fill(49,49,1);
      climate.set([-2e7,-1e7,2e7,1e7,1,0,1234],45*L);
      const heights=k=>{const h=new Float32Array(304*304);for(let y=0;y<304;y++)for(let x=0;x<304;x++)h[y*304+x]=1200+3800*Math.exp(-((x-190)**2+(y-125)**2)/4200)+180*Math.sin(x*.09+k)*Math.cos(y*.07);return h;};
      const tiles=Array.from({length:32},(_,i)=>heights(i*.37));
      const roots=new Float32Array(160*160);for(let y=0;y<160;y++)for(let x=0;x<160;x++)roots[y*160+x]=Math.sqrt(2000+800*Math.sin(x*.08)+500*Math.cos(y*.11));
      const canvas=document.getElementById('map');
      let t0=performance.now();const renderer=await createTerrainRenderer(canvas,{maxBytes:512*1024**2,onStatus:(state,detail)=>{if(state==='fallback')window.rendererFailure=detail;}});const initMs=performance.now()-t0;
      if(!renderer?.available)throw Error('WebGPU unavailable');
      const done=()=>renderer.device.queue.onSubmittedWorkDone();
      const timestamps=renderer.device.features.has('timestamp-query');
      // Sum of GPU pass durations (ms) for the work submitted by f.
      const gpu=async f=>{
        if(!timestamps)return null;
        const device=renderer.device;gpuTimer.set??=device.createQuerySet({type:'timestamp',count:gpuTimer.capacity});
        await done();gpuTimer.count=0;gpuTimer.active=true;try{await f();}finally{gpuTimer.active=false;}
        const count=gpuTimer.count;if(!count)return 0;
        const resolve=device.createBuffer({size:count*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:count*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const encoder=device.createCommandEncoder();encoder.resolveQuerySet(gpuTimer.set,0,count,resolve,0);encoder.copyBufferToBuffer(resolve,0,read,0,count*8);device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);const values=new BigUint64Array(read.getMappedRange());let total=0n;for(let i=0;i<count;i+=2)if(values[i+1]>values[i])total+=values[i+1]-values[i];
        read.unmap();read.destroy();resolve.destroy();return r3(Number(total)/1e6);
      };
      const gpuBench=async(n,f)=>{const s=[];for(let i=0;i<n;i++)s.push(await gpu(()=>f(i)));return s.every(v=>v===null)?null:r3(med(s));};
      const time=async f=>{const s=performance.now();await f();await done();return performance.now()-s;};
      const bench=async(n,warm,f)=>{for(let i=0;i<warm;i++)await f(i);const s=[];for(let i=0;i<n;i++)s.push(await f(warm+i));return r3(med(s));};
      const out={initMs:r3(initMs),gpu:renderer.adapter?.info?.description||renderer.adapter?.info?.device||renderer.adapter?.info?.vendor||'',upload:{}};
      const upload=(mpp)=>{for(let i=0;i<32;i++)renderer.uploadTile('t'+i,tiles[i],{climate,climateWidth:N,climateHeight:N,metresPerSample:mpp,lod:Math.log2(mpp/30),origin:[i*256*mpp,0]});};
      for(const mode of ['relief','render'])for(const mpp of [30,60,120,240,1920,15360]){
        renderer.clear();renderer.setMode(mode);await done();
        const total=await bench(5,1,async()=>{renderer.clear();await done();return time(()=>upload(mpp));});
        const gpuMs=await gpuBench(5,async()=>{renderer.clear();upload(mpp);});
        out.upload[mode+'/'+mpp+'m']={totalMs:total,perTileMs:r3(total/32),gpuPerTileMs:gpuMs===null?null:r3(gpuMs/32)};
      }
      // Low coastal land: every sample below 12 m searches its shore distance.
      const coast=k=>{const h=new Float32Array(304*304);for(let y=0;y<304;y++)for(let x=0;x<304;x++)h[y*304+x]=5+7*Math.sin(x*.05+k)*Math.cos(y*.043)+4*Math.sin((x+y)*.11)-(x>230||(x-80)**2+(y-150)**2<400?14:0);return h;};
      const coastTiles=Array.from({length:32},(_,i)=>coast(i*.37));
      const uploadCoast=mpp=>{for(let i=0;i<32;i++)renderer.uploadTile('t'+i,coastTiles[i],{climate,climateWidth:N,climateHeight:N,metresPerSample:mpp,lod:Math.log2(mpp/30),origin:[i*256*mpp,0]});};
      const digest=async key=>{
        const texture=renderer.tiles.get(key).color,bytesPerRow=256*4,buffer=renderer.device.createBuffer({size:bytesPerRow*256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const encoder=renderer.device.createCommandEncoder();encoder.copyTextureToBuffer({texture},{buffer,bytesPerRow},[256,256]);renderer.device.queue.submit([encoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);const data=new Uint8Array(buffer.getMappedRange().slice(0));buffer.destroy();
        let h=0x811c9dc5;for(let i=0;i<data.length;i++)h=Math.imul(h^data[i],0x01000193);return (h>>>0).toString(16);
      };
      out.coast={};
      for(const mpp of [3.75,30]){
        renderer.clear();renderer.setMode('render');await done();
        const total=await bench(5,1,async()=>{renderer.clear();await done();return time(()=>uploadCoast(mpp));});
        const gpuMs=await gpuBench(5,async()=>{renderer.clear();uploadCoast(mpp);});
        out.coast['render/'+mpp+'m']={totalMs:total,perTileMs:r3(total/32),gpuPerTileMs:gpuMs===null?null:r3(gpuMs/32),colorDigest:await digest('t3')};
      }
      // Render3D: lazy compile, bake (Render material + forest field on the full tile) and per-pixel draws.
      if(window.TerrainRender3D&&renderer.ensureRender3d){
        renderer.clear();renderer.setMode('render3d');renderer.uploadTile('w',coastTiles[0],{climate,climateWidth:N,climateHeight:N,metresPerSample:30,origin:[0,0]});
        const started=performance.now();while(!renderer.getStats().render3d.ready&&!renderer.getStats().render3d.error&&performance.now()-started<120000)await new Promise(r=>setTimeout(r,20));
        out.render3d={compileMs:r3(renderer.getStats().render3d.compileMs),error:renderer.getStats().render3d.error,upload:{},draw:{}};
        for(const mpp of [3.75,30]){
          renderer.clear();await done();
          const total=await bench(5,1,async()=>{renderer.clear();await done();return time(()=>upload(mpp));});
          const gpuMs=await gpuBench(5,async()=>{renderer.clear();upload(mpp);});
          out.render3d.upload[mpp+'m']={totalMs:total,perTileMs:r3(total/32),gpuPerTileMs:gpuMs===null?null:r3(gpuMs/32)};
        }
        // Batched full-screen draws of one 30 m tile at several footprints (median of 20 per wait).
        const fullscreen={width:1920,height:1080,dpr:1};
        for(const mode of ['render','render3d','render3d-brush']){
          renderer.clear();renderer.setMode(mode);renderer.uploadTile('d',tiles[0],{climate,climateWidth:N,climateHeight:N,metresPerSample:30,origin:[0,0]});
          if(mode==='render3d-brush'){const s=performance.now();while(!renderer.getStats().render3d.brushes&&!renderer.getStats().render3d.brushError&&performance.now()-s<30000)await new Promise(r=>setTimeout(r,20));}
          for(const fp of [16,4,1,.25,.02]){
            const span=fp*1920/(256*30),rect=[{key:'d',lod:0,x:0,y:0,width:1920,height:1080,uv:[.3,.3,span,span*1080/1920]}];
            renderer.draw(rect,fullscreen);await done();
            out.render3d.draw[mode+'/'+fp+'m']={wallPerDrawMs:r3(await bench(5,1,()=>time(()=>{for(let i=0;i<20;i++)renderer.draw(rect,fullscreen);}))/20),gpuMs:await gpuBench(7,()=>renderer.draw(rect,fullscreen))};
          }
        }
      }
      // Reshade: 32 render-mode 30 m tiles drawn on a grid.
      renderer.clear();renderer.setMode('render');renderer.setRenderSettings(TerrainRender.defaults());upload(30);await done();
      const rects=Array.from({length:32},(_,i)=>({key:'t'+i,lod:0,x:(i%8)*128,y:Math.floor(i/8)*128,width:128,height:128})),view={width:1024,height:1024,dpr:1};
      out.gpu={timestamps,drawRender32:await gpuBench(7,()=>renderer.draw(rects,view)),reshadeRender32:await gpuBench(7,j=>{renderer.setRenderSettings({...TerrainRender.defaults(),season:(j%10)/10+.05});renderer.draw(rects,view);})};
      out.reshade={reshadeMs:await bench(10,2,j=>time(()=>{renderer.setRenderSettings({...TerrainRender.defaults(),season:(j%10)/10+.05});renderer.draw(rects,view);})),
        drawOnlyMs:await bench(10,3,()=>time(()=>renderer.draw(rects,view)))};
      // Coarse bake: 16 blocks, then rebake after season change.
      renderer.clear();renderer.setMode('render');renderer.setRenderSettings(TerrainRender.defaults());
      const crects=Array.from({length:16},(_,i)=>({key:'c'+i,lod:7,x:(i%4)*128,y:Math.floor(i/4)*128,width:128,height:128})),cview={width:512,height:512,dpr:1};
      const upl=[],bake=[];
      for(let rep=0;rep<4;rep++){
        renderer.clear();await done();
        upl.push(await time(()=>{for(let i=0;i<16;i++)renderer.uploadCoarse('c'+i,roots,{climate,climateWidth:N,climateHeight:N,origin:[i*1e6,0]});}));
        bake.push(await time(()=>renderer.draw(crects,cview)));
      }
      const one=[{key:'c0',lod:7,x:0,y:0,width:512,height:512}];
      renderer.clear();await done();renderer.uploadCoarse('c0',roots,{climate,climateWidth:N,climateHeight:N,origin:[0,0]});await done();
      const single=await time(()=>renderer.draw(one,cview));
      renderer.clear();await done();for(let i=0;i<16;i++)renderer.uploadCoarse('c'+i,roots,{climate,climateWidth:N,climateHeight:N,origin:[i*1e6,0]});renderer.draw(crects,cview);await done();
      const rebake=await bench(5,1,j=>time(()=>{renderer.setRenderSettings({...TerrainRender.defaults(),season:.1+.15*(j%6)});renderer.draw(crects,cview);}));
      out.coarseBake={uploadMs16:r3(med(upl.slice(1))),firstDrawBakeMs16:r3(med(bake.slice(1))),firstDrawBakeMs1:r3(single),rebakeMs16:rebake,rebakePerBlockMs:r3(rebake/16)};
      // Coarse zoomed full-screen draw.
      renderer.clear();await done();renderer.uploadCoarse('c0',roots,{climate,climateWidth:N,climateHeight:N,origin:[0,0]});
      const big=[{key:'c0',lod:7,x:0,y:0,width:1920,height:1080,uv:[.25,.25,.3,.2]}],bview={width:1920,height:1080,dpr:1};out.coarseDraw={};
      for(const mode of ['relief','render']){renderer.setMode(mode);renderer.draw(big,bview);await done();out.coarseDraw[mode]=await bench(30,5,()=>time(()=>renderer.draw(big,bview)));out.coarseDraw[mode+'Gpu']=await gpuBench(7,()=>renderer.draw(big,bview));out.coarseDraw[mode+'Batched20PerDraw']=r3(await bench(5,1,()=>time(()=>{for(let i=0;i<20;i++)renderer.draw(big,bview);}))/20);}
      out.failure=window.rendererFailure||null;renderer.dispose();return out;
    },Number(process.env.CLIMATE_N||33));
    if(errors.length)throw Error(errors.join('\n'));
    const text=JSON.stringify(report);console.log(text);if(jsonOut)fs.writeFileSync(jsonOut,JSON.stringify(report,null,2));
  }finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
