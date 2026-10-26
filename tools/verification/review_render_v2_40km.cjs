// Visual audit of genuine 40,000 x 40,000 m plates through the production GPU
// renderer. Controlled physical fixtures, not neural terrain or satellite truth.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),crypto=require('node:crypto');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const crossMatrix=process.argv.includes('--cross-matrix');
const microOnly=process.argv.includes('--micro-insets');
const root=path.resolve(__dirname,'../..'),out=path.join(root,crossMatrix?'output/render-v2-40km-matrix':'output/render-v2-40km');
const fixture=JSON.parse(fs.readFileSync(path.join(__dirname,'render_v2_40km_cases.json'),'utf8'));
const files=['terrain_render_controls.js','terrain_lighting.js','terrain_forest_brushes.js','terrain_render3d.js','terrain_render_v2.js','terrain_renderer.js'];
const sources=Object.fromEntries(files.map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'web',name))).digest('hex')]));
const html='<meta charset="utf-8"><canvas id="map"></canvas>'+files.map(name=>`<script src="/${name}"></script>`).join('');
const selected=process.env.REVIEW_CASES?.split(',');
if(selected)fixture.cases=fixture.cases.filter(c=>selected.includes(c.id));
if(crossMatrix){fixture.cases=fixture.cases.filter(c=>['af','cfb','bsk','bwh','dfc','et'].includes(c.id));fixture.reliefs=['plain','rolling','plateau','mountains'];fixture.altitudeMeaning='base elevation; report actual visible min/mean/max, not a constant-height biome';fixture.units.altitudes='base elevation added to the fixed relief, metres; actual visible min/mean/max are recorded per case';}
if(!fixture.cases.length)throw Error('No audit cases selected');
const onlyOverview=process.argv.includes('--overview-only');
(async()=>{
 fs.mkdirSync(path.join(out,'frames'),{recursive:true});
 const server=http.createServer((req,res)=>{const name=new URL(req.url,'http://x').pathname.slice(1);res.setHeader('Content-Type',files.includes(name)?'application/javascript':'text/html');res.end(files.includes(name)?fs.readFileSync(path.join(root,'web',name)):html);});
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const errors=[],frames=[],started=Date.now();let browser;
 if(microOnly){const previous=JSON.parse(fs.readFileSync(path.join(out,'report.json'),'utf8'));if(Object.keys(sources).some(k=>previous.sources[k]!==sources[k]))throw Error('Source hashes changed; regenerate the main matrix before appending micro insets');frames.push(...previous.frames.filter(f=>f.view!=='micro320m'));}
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage({viewport:{width:512,height:512}});
  page.on('pageerror',e=>errors.push(e.stack));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.exposeFunction('auditFrame',data=>{
   const {png,...row}=data;fs.writeFileSync(path.join(out,'frames',row.file),Buffer.from(png,'base64'));frames.push(row);
   if(frames.length%40===0)console.log(JSON.stringify({frames:frames.length,last:row.id,elapsedSeconds:Math.round((Date.now()-started)/1000)}));
  });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result=await page.evaluate(async({fixture,onlyOverview,crossMatrix,microOnly})=>{
   const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('A real WebGPU adapter is required');
   let failure=null;const r=await createTerrainRenderer(document.getElementById('map'),{maxBytes:384*1024**2,onStatus:(s,e)=>{if(s==='fallback')failure=e;}});if(!r)throw Error(failure||'Renderer unavailable');
   r.context.configure({device:r.device,format:r.format,alphaMode:'premultiplied',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
   TerrainLighting.set({global:{strength:1,ambient:.35,contrast:1,exaggeration:6,azimuth:315,altitude:45}});
   TerrainRender3D.set({enabled:true,density:1,clearings:.45});r.setMode('render-v2');await r.r3Promise;if(r.r3Error)throw Error(r.r3Error);
   const size=512,res=30,inner=256,halo=24,n=304,span=7680;
   const encode=document.createElement('canvas');encode.width=encode.height=size;const ctx=encode.getContext('2d');
   const coordinates=[];for(let ty=0;ty<6;ty++)for(let tx=0;tx<6;tx++)coordinates.push({tx,ty,x:tx*span-3*span,y:ty*span-3*span});
   const geometry=new Map();
   function heightAt(x,y,alt,relief){
    if(relief==='plain')return alt+12+8*Math.sin(x/9000)*Math.cos(y/8000)+2*Math.sin((x+y)/1500);
    if(relief==='rolling')return alt+220+150*Math.sin(x/3500)*Math.cos(y/4200)+50*Math.sin((x+y)/1300)+20*Math.sin(x/400)*Math.cos(y/700);
    if(relief==='plateau')return alt+400+300*Math.tanh((x-180*Math.sin(y/1400))/400)+50*Math.sin(y/5000)+20*Math.sin((x+2*y)/1200);
    if(relief==='mountains')return alt+1250+700*Math.sin(x/1900)*Math.cos(y/2400)+350*Math.sin((x+y)/750)+100*Math.sin(x/400)*Math.cos(y/600);
    // Common continuous relief. Same formula across all tiles and climates;
    // broad valleys, rolling land, a rugged ridge, and a genuine smooth plain.
    const roll=110*Math.sin(x/4200)*Math.cos(y/3900)+45*Math.sin((x+y)/1350);
    const ridge=430*Math.exp(-(((x+7800)/2200)**2))*Math.sin(y/2300);
    const plateau=1-Math.exp(-((x-8000)**2+(y-6000)**2)/(2*6000**2));
    const amplitude=alt<200?.10:alt<1000?.4:1;
    return Math.max(2,alt+(roll+ridge)*amplitude*plateau);
   }
   function makeGeometry(alt,relief){
    const geometryKey=relief+'|'+alt;if(geometry.has(geometryKey))return geometry.get(geometryKey);
    const tiles=coordinates.map(c=>{
     const origin=[c.x-halo*res,c.y-halo*res],height=new Float32Array(n*n),water=new Float32Array(n*n*4);
     let min=Infinity,max=-Infinity,sum=0,count=0;
     for(let y=0;y<n;y++)for(let x=0;x<n;x++){
      const wx=origin[0]+(x+.5)*res,wy=origin[1]+(y+.5)*res,i=y*n+x;
      const v=heightAt(wx,wy,alt,relief);height[i]=v;
      // Explicit dry authority keeps water out of this land-only matrix.
      water[i*4+3]=1;
      if(x>=halo&&x<halo+inner&&y>=halo&&y<halo+inner&&Math.abs(wx)<20000&&Math.abs(wy)<20000){min=Math.min(min,v);max=Math.max(max,v);sum+=v;count++;}
     }
     return {...c,origin,height,water,min,max,sum,count};
    });geometry.set(geometryKey,tiles);return tiles;
   }
   function climate(c){
    const width=33,L=width*width,a=new Float32Array(51*L),fill=(layer,v)=>a.fill(v,layer*L,(layer+1)*L);
    fill(0,(c.ts+c.tw)*.5);fill(2,c.ps+c.pw);[c.ts,c.ps,c.ls,c.tw,c.pw,c.lw].forEach((v,i)=>fill(21+i,v));
    [.48,.36,.23,.55,.40,.29].forEach((v,i)=>fill(36+i,v));fill(42,c.sand??.3);fill(46,.35);fill(47,.30);fill(48,.35);fill(49,1);
    a.set([-2e7,-1e7,2e7,1e7,0,0,17331],45*L);return a;
   }
   function rects(fp,cx=0,cy=0){return coordinates.map(c=>({key:`${c.tx}/${c.ty}`,x:(c.x-cx)/fp+size/2,y:(c.y-cy)/fp+size/2,width:span/fp,height:span/fp,uv:[0,0,1,1],lod:0})).filter(t=>t.x<size&&t.y<size&&t.x+t.width>0&&t.y+t.height>0);}
   async function read(fp,cx=0,cy=0){
    const patches=rects(fp,cx,cy);r.draw(patches,{width:size,height:size,dpr:1});
    const bytes=size*size*4,b=r.device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),enc=r.device.createCommandEncoder();
    enc.copyTextureToBuffer({texture:r.context.getCurrentTexture()},{buffer:b,bytesPerRow:size*4},[size,size]);r.device.queue.submit([enc.finish()]);await b.mapAsync(GPUMapMode.READ);
    const raw=new Uint8Array(b.getMappedRange()),rgba=new Uint8ClampedArray(bytes);let sum=0,sum2=0,green=0,red=0,black=0;
    for(let i=0;i<size*size;i++){
     const rr=raw[i*4+(r.format.startsWith('bgra')?2:0)],g=raw[i*4+1],bb=raw[i*4+(r.format.startsWith('bgra')?0:2)];
     rgba.set([rr,g,bb,255],i*4);const l=.2126*rr+.7152*g+.0722*bb;sum+=l;sum2+=l*l;
     if(g>rr*1.025&&g>bb*1.1)green++;if(rr>g*1.45&&rr>bb*1.5)red++;if(rr+g+bb<3)black++;
    }
    b.unmap();b.destroy();ctx.putImageData(new ImageData(rgba,size,size),0,0);
    const total=size*size,mean=sum/total;return {png:encode.toDataURL('image/png').split(',')[1],mean,std:Math.sqrt(Math.max(0,sum2/total-mean*mean)),green:green/total,red:red/total,black:black/total,patches:patches.length};
   }
   const cases=[];
   for(const c of fixture.cases)for(const relief of fixture.reliefs||['common'])for(const altitude of fixture.altitudes){
    const physical=makeGeometry(altitude,relief),a=climate(c);r.clear();r.setMode('render-v2');
    for(const t of physical)r.uploadTile(`${t.tx}/${t.ty}`,t.height,{hydrology:t.water,climate:a,climateWidth:33,climateHeight:33,metresPerSample:res,origin:t.origin,mode:'render-v2'});
    await r.device.queue.onSubmittedWorkDone();
    const land={min:Math.min(...physical.map(t=>t.min)),max:Math.max(...physical.map(t=>t.max)),mean:physical.reduce((s,t)=>s+t.sum,0)/physical.reduce((s,t)=>s+t.count,0),scope:'visible 40 km physical samples'};
    cases.push({id:c.id,relief,altitude,elevation:land,localTs:c.ts-c.ls*land.mean,localTw:c.tw-c.lw*land.mean});
    for(const mode of ['render-v2','render3d']){
     r.setMode(mode);
     for(const season of [0,.5]){
      TerrainRenderV2.set({season});
      // Match the real app: the independent V2 slider leaves legacy material
      // settings at their existing July default. Render3D uses its own slider.
      const legacySeason=mode==='render-v2'?.5:season;
      TerrainRender.set({...TerrainRender.get(),season:legacySeason});
      r.setRenderSettings(TerrainRender.get());
      const views=microOnly?[['micro320m',.625]]:crossMatrix?[['40km',40000/size],['10km',10000/size],['2.56km',5],['640m',1.25]]:(onlyOverview?[['plate',40000/size]]:[['plate',40000/size],['detail',2.5]]);
      for(const [view,fp] of views){
       const cx=crossMatrix||view==='plate'?0:-7800,cy=crossMatrix||view==='plate'?0:1800;
       const frame=await read(fp,cx,cy),file=`${c.id}-${crossMatrix?relief+'-':''}${altitude}-${mode}-${season===0?'winter':'summer'}-${view}.png`;
       await window.auditFrame({...frame,file,id:c.id,label:c.label,group:c.group,relief,altitude,mode,season,legacySeason,view,metresPerPixel:fp,extentMetres:size*fp,centre:[cx,cy],sourceResolution:res,elevation:land});
      }
     }
    }
   }
   r.clear();return {cases,adapter:adapter.info?.description||adapter.info?.device||null,rendererFailure:failure,sourceResolution:res,plateMetres:40000,tilesPerPlate:36};
  },{fixture,onlyOverview,crossMatrix,microOnly});
  const report={...result,controlled:true,fixture,sources,frames,errors,seconds:(Date.now()-started)/1000};
  fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));
  const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
  const items=frames.map(f=>`<article data-biome="${esc(f.id)}" data-alt="${f.altitude}" data-relief="${f.relief}" data-mode="${f.mode}" data-season="${f.season}" data-view="${f.view}"><a href="frames/${f.file}"><img loading="lazy" src="frames/${f.file}" width="256" height="256"></a><p>${esc(f.label)} · ${esc(f.relief)} · base ${f.altitude} m<br>H ${Math.round(f.elevation.min)}–${Math.round(f.elevation.max)} m<br>${f.mode} · ${f.season===0?'January':'July'} · ${f.extentMetres/1000} km</p></article>`).join('');
  const choices=(id,values)=>`<label>${id}<select id="${id}"><option value="">All</option>${values.map(([v,l])=>`<option value="${esc(v)}">${esc(l)}</option>`).join('')}</select></label>`;
  const controls=choices('biome',fixture.cases.map(c=>[c.id,c.label]))+choices('relief',(fixture.reliefs||['common']).map(x=>[x,x]))+choices('alt',fixture.altitudes.map(a=>[a,a+' m']))+choices('mode',[['render-v2','Render v2'],['render3d','Render3D']])+choices('season',[[0,'January'],[.5,'July']])+choices('view',crossMatrix?[['40km','40 km'],['10km','10 km'],['2.56km','2.56 km'],['640m','640 m'],['micro320m','320 m microtexture']]:[['plate','40 km plate'],['detail','1.28 km detail']]);
  fs.writeFileSync(path.join(out,'index.html'),`<!doctype html><meta charset="utf-8"><title>Render v2 · 40 km audit</title><style>body{background:#182127;color:#eee;font:14px system-ui;margin:20px}header{position:sticky;top:0;background:#182127;padding:12px;z-index:1}label{display:inline-flex;gap:5px;margin-right:12px}main{display:flex;flex-wrap:wrap;gap:12px}article{width:256px}img{display:block}article[hidden]{display:none}p{line-height:1.5}a{color:#b9d9df}</style><header><h1>Render v2 · 40 km plates</h1><p>Controlled physical grids, 30 m samples; production WebGPU. Each overview is exactly 40 × 40 km. These are not neural or real satellite plates. Climate rows are source climates; altitude applies their lapse rates. <a href="report.json">Measurements and source hashes</a></p>${controls}</header><main>${items}</main><script>const controls=[...document.querySelectorAll('select')];function filter(){document.querySelectorAll('article').forEach(a=>a.hidden=controls.some(c=>c.value!==''&&a.dataset[c.id]!==c.value))}controls.forEach(c=>c.onchange=filter);document.querySelector('#mode').value='render-v2';document.querySelector('#season').value='0.5';document.querySelector('#view').value='${crossMatrix?'40km':'plate'}';filter();</script>`);
  console.log(JSON.stringify({ok:!errors.length&&!result.rendererFailure,frames:frames.length,cases:result.cases.length,seconds:report.seconds,errors,out}));
  if(errors.length||result.rendererFailure)process.exitCode=1;
 }finally{if(browser)await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
