const {loadPlaywright, browserExecutable} = require('../../tools/platform.cjs');
const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {readRepositoryFile} = require(NEURAL_EARTH_ROOT + '/tools/repository-files.cjs');
// Render3D / Render3DBrush: exact world lattices, settings, real WebGPU shaders,
// Render reference at distance, canonical forest across tile halos and page wiring.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {chromium}=loadPlaywright();
const R3=require(NEURAL_EARTH_ROOT+'/web/terrain_render3d.js');

// Lattice splits: the same world point gets the same exact coordinate from two tiles.
{
  const n=R3.SPLIT.entries,coordinate=(buffer,entry,local)=>{
    const cells=new Int32Array(buffer,0,n*4),fracs=new Float32Array(buffer,n*16,n*4);
    return [0,1].map(i=>cells[entry*4+i]+fracs[entry*4+i]+local[i]*fracs[entry*4+3]);
  };
  for(const [origin,step] of [[[-19993600,-4917888],[7680,7680]],[[123456.5,-98765.25],[960,-960]],[[0,0],[120,240]]]){
    const a=R3.tileUniform(origin,1234),b=R3.tileUniform([origin[0]+step[0],origin[1]+step[1]],1234);
    for(const entry of [0,7,16,R3.SPLIT.crown-1,R3.SPLIT.crown,R3.SPLIT.crown+6,R3.SPLIT.patch]){
      const point=[step[0]*1.37+3.1,step[1]*.61-2.7],ca=coordinate(a,entry,point),cb=coordinate(b,entry,[point[0]-step[0],point[1]-step[1]]);
      ca.forEach((v,i)=>assert(Math.abs(v-cb[i])<2e-3,`entry ${entry}: ${ca} vs ${cb}`));
    }
    assert.equal(new Uint32Array(a,n*32,4)[0],1234);
  }
  // Fractions stay in [0, 1) and scales match the shader's lattice spacing.
  const fracs=new Float32Array(R3.tileUniform([-1e7,3e6],7),R3.SPLIT.entries*16,R3.SPLIT.entries*4);
  for(let e=0;e<R3.SPLIT.patch+1;e++)assert(fracs[e*4]>=0&&fracs[e*4]<1&&fracs[e*4+1]>=0&&fracs[e*4+1]<1);
  assert(Math.abs(fracs[(R3.SPLIT.crown+3)*4+2]-2*2**1.5)<1e-5);
  // Display settings and links.
  assert.deepEqual(R3.normalize({density:5,clearings:-1,enabled:false}),{enabled:false,density:2,clearings:0});
  const params=new URLSearchParams();R3.set({enabled:true,density:1.5,clearings:.45});R3.writeParams(params,true);
  assert.equal(params.toString(),'forest=1&forestDensity=1.5');
  assert.deepEqual(R3.fromParams(params),{enabled:true,density:1.5,clearings:.45});
  R3.set(R3.defaults());const hidden=new URLSearchParams('forest=1&forestDensity=2');R3.writeParams(hidden,false);assert.equal(hidden.toString(),'');
  assert.equal(R3.isStyle('render3d-brush'),true);assert.equal(R3.isStyle('render'),false);
  // The Render material exposes the guide/field capture points Render3D compiles.
  const renderer=readRepositoryFile('terrain_renderer.js','utf8');
  for(const marker of ['/*R3:open*/','/*R3:beach*/','/*R3:end*/'])assert.equal(renderer.split(marker).length,2,marker);
  assert.throws(()=>R3.withCaptures('fn surfaceMaterial(){}'),/marker missing/);
  assert.equal(fs.statSync(path.join(NEURAL_EARTH_ROOT,'web/forest-brushes.bin')).size>100000,true);
}

const html='<meta charset="utf-8"><canvas id="map" style="width:512px;height:384px"></canvas><script>window.TerrainLighting={vectors:()=>[1,.35,1,1,-.5,-.5,.70710678,0]};</script>'
  +['terrain_render_controls.js','terrain_forest_brushes.js','terrain_render3d.js','terrain_renderer.js'].map(name=>`<script src="/${name}"></script>`).join('');
(async()=>{
  const server=http.createServer((req,res)=>{
    const name=new URL(req.url,'http://x').pathname.slice(1);
    if(name==='forest-brushes.bin'){res.setHeader('Content-Type','application/octet-stream');return res.end(fs.readFileSync(path.join(NEURAL_EARTH_ROOT,'web',name)));}
    res.setHeader('Content-Type',name.endsWith('.js')?'application/javascript':'text/html');
    res.end(name.endsWith('.js')?readRepositoryFile(name):html);
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
  try{
    const page=await browser.newPage({viewport:{width:640,height:480}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const report=await page.evaluate(async()=>{
      let failure=null;const canvas=document.getElementById('map');
      const r=await createTerrainRenderer(canvas,{maxBytes:256*1024**2,onStatus:(state,detail)=>{if(state==='fallback')failure=detail;}});
      if(!r)return {skipped:true};
      r.context.configure({device:r.device,format:r.format,alphaMode:'premultiplied',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
      const N=33,L=N*N,climate=new Float32Array(50*L),fill=(layer,v)=>climate.fill(v,layer*L,(layer+1)*L);
      fill(0,12);fill(2,900);fill(4,.0065);[24,520,.0065,4,420,.0065].forEach((v,i)=>fill(21+i,v));
      for(const i of [36,37,38])fill(i,.42);for(const i of [39,40,41])fill(i,.5);fill(42,.3);for(const i of [46,47,48])fill(i,.4);fill(49,1);
      climate.set([-2e7,-1e7,2e7,1e7,0,0,1234],45*L);
      // Rolling temperate lowland with a coast: forest, clearings and shoreline.
      const tile=(r,ox,oy)=>{const h=new Float32Array(304*304);for(let y=0;y<304;y++)for(let x=0;x<304;x++){const wx=ox+(x+.5)*r,wy=oy+(y+.5)*r;h[y*304+x]=60+40*Math.sin(wx/700)*Math.cos(wy/530)+25*Math.sin(wx/170+wy/230)-(wx>1800?90:0);}return h;};
      const put=(key,res,origin)=>r.uploadTile(key,tile(res,origin[0],origin[1]),{climate,climateWidth:N,climateHeight:N,metresPerSample:res,lod:Math.log2(res/30),origin});
      const read=async(rects,width=512,height=384)=>{
        r.draw(rects,{width,height,dpr:1});
        const row=Math.ceil(width*4/256)*256,buffer=r.device.createBuffer({size:row*height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const encoder=r.device.createCommandEncoder();encoder.copyTextureToBuffer({texture:r.context.getCurrentTexture()},{buffer,bytesPerRow:row},[width,height]);r.device.queue.submit([encoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);const values=new Uint8Array(buffer.getMappedRange()).slice();buffer.unmap();buffer.destroy();
        const rgb=new Float32Array(width*height*3);for(let y=0;y<height;y++)for(let x=0;x<width;x++)for(let c=0;c<3;c++)rgb[(y*width+x)*3+c]=values[y*row+x*4+(r.format.startsWith('bgra')?2-c:c)];
        return rgb;
      };
      const wait=async test=>{for(let i=0;i<600&&!test();i++){r.draw([],{width:512,height:384,dpr:1});await new Promise(resolve=>setTimeout(resolve,25));}};
      const view=(key,uv)=>[{key,lod:0,x:0,y:0,width:512,height:384,uv}];
      const mean=a=>a.reduce((s,v)=>s+v,0)/a.length,std=a=>{const m=mean(a);return Math.sqrt(mean(a.map(v=>(v-m)**2)));};
      const out={};
      // Render reference and Render3D beyond its 32 m/px reveal.
      r.setMode('render');put('far',30,[0,0]);const render=await read(view('far',[0,0,1,1]));
      r.setMode('render3d');await wait(()=>r.getStats().render3d.ready||r.getStats().render3d.error);
      out.stats=r.getStats().render3d;
      const far=await read(view('far',[0,0,1,1]));
      let total=0,count=0;for(let i=0;i<far.length;i++){total+=Math.abs(far[i]-render[i]);count++;}
      out.farDifference=total/count;
      // Near: crowns resolve (strong local contrast) and repeated frames are identical.
      put('near',3.75,[600,600]);
      const near=await read(view('near',[.4,.4,.08,.06])),again=await read(view('near',[.4,.4,.08,.06]));
      out.nearStd=std(Array.from({length:near.length/3},(_,i)=>near[i*3+1]));out.deterministic=near.every((v,i)=>v===again[i]);
      // Forest disabled keeps the ground material only.
      TerrainRender3D.set({enabled:false});const bare=await read(view('near',[.4,.4,.08,.06]));TerrainRender3D.set(TerrainRender3D.defaults());
      out.bareStd=std(Array.from({length:bare.length/3},(_,i)=>bare[i*3+1]));
      // Brushes load and change the resolved crowns.
      r.setMode('render3d-brush');await wait(()=>r.getStats().render3d.brushes||r.getStats().render3d.brushError);
      const brush=await read(view('near',[.4,.4,.08,.06]));
      out.brushStats=r.getStats().render3d;out.brushDiffers=brush.some((v,i)=>Math.abs(v-near[i])>8);
      // Canonical field: a neighbour's halo samples the same world forest as its own interior.
      r.setMode('render3d');put('a',30,[0,0]);put('b',30,[256*30,0]);await read(view('a',[0,0,1,1]));
      // Layer 2 (forest field) of the full-tile aux texture; rows padded to 256 bytes.
      const field=async key=>{const t=r.tiles.get(key).aux,row=1280,buffer=r.device.createBuffer({size:row*304,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const e=r.device.createCommandEncoder();e.copyTextureToBuffer({texture:t,origin:[0,0,2]},{buffer,bytesPerRow:row,rowsPerImage:304},[304,304,1]);r.device.queue.submit([e.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);const padded=new Uint8Array(buffer.getMappedRange()),v=new Uint8Array(304*304*4);
        for(let y=0;y<304;y++)v.set(padded.subarray(y*row,y*row+304*4),y*304*4);buffer.unmap();buffer.destroy();return v;};
      const fa=await field('a'),fb=await field('b');let worst=0,cells=0;
      // Columns 264..279 of a are columns 8..23 of b (halo 24): compare away from b's own outer halo edge.
      for(let y=40;y<264;y++)for(let x=264;x<280;x++){const ia=(y*304+x)*4,ib=(y*304+x-256)*4;for(const c of [0,1,2,3])worst=Math.max(worst,Math.abs(fa[ia+c]-fb[ib+c]));cells++;}
      out.haloWorst=worst;out.haloCells=cells;out.failure=failure;out.status=r.status;
      r.dispose();return out;
    });
    if(report.skipped){console.log('SKIP: WebGPU adapter unavailable');return;}
    assert.equal(report.failure,null);assert.equal(report.status,'webgpu');
    assert.equal(report.stats.ready,true,JSON.stringify(report.stats));
    assert(report.farDifference<6,`Render3D must stay on Render beyond 32 m/px: ${report.farDifference}`);
    assert(report.nearStd>report.bareStd*1.5,`Resolved crowns add structure: ${report.nearStd} vs ${report.bareStd}`);
    assert.equal(report.deterministic,true);
    assert.equal(report.brushStats.brushes,true,JSON.stringify(report.brushStats));assert.equal(report.brushDiffers,true);
    assert(report.haloWorst<=24,`Forest field differs across a tile halo by ${report.haloWorst}/255`);
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify(report));
  }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
