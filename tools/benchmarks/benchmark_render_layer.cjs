const {loadPlaywright, browserExecutable} = require('../platform.cjs');
const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {readRepositoryFile} = require(NEURAL_EARTH_ROOT + '/tools/repository-files.cjs');
// Wall-clock (submission + onSubmittedWorkDone) benchmark of the Render material layer vs Relief.
const fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {chromium}=loadPlaywright();
const jsonArg=process.argv.indexOf('--json'),jsonOut=jsonArg>0?process.argv[jsonArg+1]:null;
(async()=>{
  const server=http.createServer((req,res)=>{
    res.setHeader('Content-Type',req.url.endsWith('.js')?'application/javascript':'text/html');
    res.end(req.url.endsWith('.js')?readRepositoryFile(path.join(NEURAL_EARTH_ROOT,req.url)):'<meta charset="utf-8"><div id="controls"></div><canvas id="map"></canvas><script>window.TerrainLighting={vectors:()=>[1,.35,1,1,-.5,-.5,.70710678,0]};</script><script src="/terrain_render_controls.js"></script><script src="/terrain_renderer.js"></script>');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
  try{
    browser=await chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
    const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
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
      let t0=performance.now();const renderer=await createTerrainRenderer(canvas,{maxBytes:512*1024**2});const initMs=performance.now()-t0;
      if(!renderer?.available)throw Error('WebGPU unavailable');
      const done=()=>renderer.device.queue.onSubmittedWorkDone();
      const time=async f=>{const s=performance.now();await f();await done();return performance.now()-s;};
      const bench=async(n,warm,f)=>{for(let i=0;i<warm;i++)await f(i);const s=[];for(let i=0;i<n;i++)s.push(await f(warm+i));return r3(med(s));};
      const out={initMs:r3(initMs),gpu:renderer.adapter?.info?.description||renderer.adapter?.info?.device||renderer.adapter?.info?.vendor||'',upload:{}};
      const upload=(mpp)=>{for(let i=0;i<32;i++)renderer.uploadTile('t'+i,tiles[i],{climate,climateWidth:N,climateHeight:N,metresPerSample:mpp,lod:Math.log2(mpp/30),origin:[i*256*mpp,0]});};
      for(const mode of ['relief','render'])for(const mpp of [30,240,1920,15360]){
        renderer.clear();renderer.setMode(mode);await done();
        const total=await bench(5,1,async()=>{renderer.clear();await done();return time(()=>upload(mpp));});
        out.upload[mode+'/'+mpp+'m']={totalMs:total,perTileMs:r3(total/32)};
      }
      // Reshade: 32 render-mode 30 m tiles drawn on a grid.
      renderer.clear();renderer.setMode('render');renderer.setRenderSettings(TerrainRender.defaults());upload(30);await done();
      const rects=Array.from({length:32},(_,i)=>({key:'t'+i,lod:0,x:(i%8)*128,y:Math.floor(i/8)*128,width:128,height:128})),view={width:1024,height:1024,dpr:1};
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
      for(const mode of ['relief','render']){renderer.setMode(mode);renderer.draw(big,bview);await done();out.coarseDraw[mode]=await bench(30,5,()=>time(()=>renderer.draw(big,bview)));out.coarseDraw[mode+'Batched20PerDraw']=r3(await bench(5,1,()=>time(()=>{for(let i=0;i<20;i++)renderer.draw(big,bview);}))/20);}
      renderer.dispose();return out;
    },Number(process.env.CLIMATE_N||33));
    if(errors.length)throw Error(errors.join('\n'));
    const text=JSON.stringify(report);console.log(text);if(jsonOut)fs.writeFileSync(jsonOut,JSON.stringify(report,null,2));
  }finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
