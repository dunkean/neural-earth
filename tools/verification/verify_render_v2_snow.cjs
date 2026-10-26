// Synthetic appearance QA through the production WebGPU renderer. No model,
// server state, terrain cache or shader source is changed. Atlas coordinates
// are chart coordinates; this is not a screenshot of the reprojected globe.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/render-v2-snow');
const files=['terrain_render_controls.js','terrain_lighting.js','terrain_forest_brushes.js','terrain_render3d.js','terrain_render_v2.js','terrain_renderer.js'];
const html='<meta charset="utf-8"><style>body{margin:0;background:#182127}canvas{display:block}</style><canvas id="map"></canvas><canvas id="atlas"></canvas><canvas id="contracts"></canvas>'+files.map(file=>`<script src="/${file}"></script>`).join('');
(async()=>{
 fs.mkdirSync(out,{recursive:true});
 const server=http.createServer((req,res)=>{const file=new URL(req.url,'http://x').pathname.slice(1);if(files.includes(file)){res.setHeader('Content-Type','application/javascript');res.end(fs.readFileSync(path.join(root,'web',file)));}else{res.setHeader('Content-Type','text/html');res.end(html);}});
 await new Promise(ok=>server.listen(0,'127.0.0.1',ok));
 let browser;
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage({viewport:{width:1024,height:900}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const report=await page.evaluate(async()=>{
   const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('Actual WebGPU adapter unavailable');
   let fallback=null;const r=await createTerrainRenderer(document.getElementById('map'),{maxBytes:64*1024**2,onStatus:(s,e)=>{if(s==='fallback')fallback=e;}});if(!r)throw Error(fallback||'Renderer unavailable');
   r.context.configure({device:r.device,format:r.format,alphaMode:'premultiplied',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
   TerrainRender3D.set({enabled:false});TerrainRenderV2.set({season:.5});r.setMode('render-v2');await r.r3Promise;if(r.r3Error)throw Error(r.r3Error);
   const size=192,footprints=[.1,1,41.9,240],rows=[];
   for(const lat of [0,67.8,85])for(const slope of [0,.25,-.25])rows.push({name:`snow / chart lat ${lat} / dHdx ${slope}`,lat,slope,cold:true,polar:false});
   for(const lat of [0,67.8,85])rows.push({name:`partial snow / chart lat ${lat} / flat`,lat,slope:0,cold:true,temperature:-1.5,polar:false});
   for(const lat of [0,67.8,85])rows.push({name:`bare reference / chart lat ${lat} / flat`,lat,slope:0,cold:false,polar:false});
   // Rotated chart latitude zero maps to geographic latitude 85 at this
   // longitude. It exercises the real metadata bit, not a cos(geographicLat).
   rows.push({name:'snow / geographic lat 85 / rotated chart lat 0',lat:0,lon:-85,slope:0,cold:true,polar:true});
   const atlas=document.getElementById('atlas'),ctx=atlas.getContext('2d');atlas.width=size*4;atlas.height=32+rows.length*(size+24);ctx.fillStyle='#182127';ctx.fillRect(0,0,atlas.width,atlas.height);ctx.font='13px sans-serif';ctx.fillStyle='white';footprints.forEach((fp,i)=>ctx.fillText(`${fp} m / pixel`,i*size+8,20));
   const defaultLight={strength:1,ambient:.35,contrast:1,exaggeration:6,azimuth:315,altitude:45};
   function put(row,fp){
    const n=33,L=n*n,res=row.sourceResolution??Math.max(30,fp*size/256),climate=new Float32Array(51*L),fill=(l,v)=>climate.fill(v,l*L,(l+1)*L),t=row.temperature??(row.cold?-25:18);
    const rain=row.rain??(row.cold?900:5); // The dry warm reference cannot grow a canopy.
    fill(0,t);fill(2,rain*2);[t,rain,0,t-5,rain,0].forEach((v,i)=>fill(21+i,v));for(const l of [36,37,38,39,40,41,46,47,48])fill(l,.4);fill(42,.3);fill(49,1);climate.set([-2e7,-1e7,2e7,1e7,1,row.polar?1:0,17331],45*L);
    const origin=[(row.lon??20)/360*4e7-152*res,-row.lat/180*2e7-152*res],h=new Float32Array(304*304),hydrology=row.river?new Float32Array(304*304*4):null;
    for(let y=0;y<304;y++)for(let x=0;x<304;x++){
     const px=(x+.5-152)*res,py=(y+.5-152)*res;
     h[y*304+x]=row.relief?1500+800*Math.sin(px/1800)*Math.cos(py/2300)+350*Math.exp(-(px*px+py*py)/(2*650**2)):20000+row.slope*(x-152)*res;
     if(hydrology){const river=Math.abs(px)<60,cover=river?Math.max(0,1-Math.abs(px)/60):0;hydrology.set([river?h[y*304+x]+2:0,cover,river?3:0,1],(y*304+x)*4);}
    }
    r.uploadTile('snow',h,{climate,climateWidth:n,climateHeight:n,hydrology,origin,metresPerSample:res,mode:'render-v2'});
    const span=fp*size/(256*res);return [.5-span/2,.5-span/2,span,span];
   }
   async function read(uv,light=defaultLight){
    TerrainLighting.set({global:light});r.shadeTile(r.tiles.get('snow'));
    r.draw([{key:'snow',x:0,y:0,width:size,height:size,uv,lod:0}],{width:size,height:size,dpr:1});
    const stride=size*4,b=r.device.createBuffer({size:stride*size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e=r.device.createCommandEncoder();e.copyTextureToBuffer({texture:r.context.getCurrentTexture()},{buffer:b,bytesPerRow:stride},[size,size]);r.device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);
    const source=new Uint8Array(b.getMappedRange()),rgba=new Uint8ClampedArray(size*size*4),luma=[];
    for(let i=0;i<size*size;i++){for(let c=0;c<3;c++)rgba[i*4+c]=source[i*4+(r.format.startsWith('bgra')?2-c:c)];rgba[i*4+3]=255;if(i%size>=12&&i%size<size-12&&Math.floor(i/size)>=12&&Math.floor(i/size)<size-12)luma.push(.2126*rgba[i*4]+.7152*rgba[i*4+1]+.0722*rgba[i*4+2]);}
    b.unmap();b.destroy();const mean=luma.reduce((a,b)=>a+b,0)/luma.length,std=Math.sqrt(luma.reduce((a,b)=>a+(b-mean)**2,0)/luma.length);
    let variationX=0,variationY=0;const inner=size-24;for(let y=0;y<inner-1;y++)for(let x=0;x<inner-1;x++){const i=y*inner+x;variationX+=Math.abs(luma[i+1]-luma[i]);variationY+=Math.abs(luma[i+inner]-luma[i]);}variationX/=(inner-1)**2;variationY/=(inner-1)**2;
    return {rgba,mean,std,min:Math.min(...luma),max:Math.max(...luma),variationX,variationY};
   }
   const results=[];
   for(let y=0;y<rows.length;y++){
    ctx.fillStyle='white';ctx.fillText(rows[y].name,8,32+y*(size+24)+17);
    for(let x=0;x<footprints.length;x++){const fp=footprints[x],uv=put(rows[y],fp),v=await read(uv);ctx.putImageData(new ImageData(v.rgba,size,size),x*size,32+y*(size+24)+24);results.push({...rows[y],fp,mean:v.mean,std:v.std,min:v.min,max:v.max,variationX:v.variationX,variationY:v.variationY});}
   }
   const lighting=[];
   for(const fp of [.1,41.9,240]){
    // A gentle slope keeps direct light away from its zero clamp, so changing
    // contrast has a measurable effect instead of testing a saturated shadow.
    const row={lat:67.8,slope:.05,cold:true,polar:false},uv=put(row,fp);
    for(const [name,change] of [['default',{}],['ambient',{ambient:.8}],['contrast',{contrast:2}],['azimuth',{azimuth:135}],['no-relief',{exaggeration:0}],['unlit',{strength:0}]]){const v=await read(uv,{...defaultLight,...change});lighting.push({fp,name,mean:v.mean,std:v.std});}
   }
   // Same continuous terrain, camera, climate and material at two DEM sample
   // spacings. Red dominance is reported separately from dark relief edges;
   // a per-channel BRDF ratio can otherwise resemble a mineral contour halo.
   const contracts=document.getElementById('contracts'),cc=contracts.getContext('2d');contracts.width=size*2;contracts.height=2*(size+24);cc.fillStyle='#182127';cc.fillRect(0,0,contracts.width,contracts.height);cc.font='13px sans-serif';const inheritance=[];
   TerrainRender3D.set({enabled:true});
   for(const [y,mode] of ['render3d','render-v2'].entries())for(const [x,sourceResolution] of [30,240].entries()){
    r.setMode(mode);const uv=put({lat:0,slope:0,cold:false,rain:900,relief:true,sourceResolution},30),v=await read(uv);cc.fillStyle='white';cc.fillText(`${mode} / DEM ${sourceResolution}m`,x*size+6,y*(size+24)+17);cc.putImageData(new ImageData(v.rgba,size,size),x*size,y*(size+24)+24);
    let green=0,red=0,edgeRed=0,edges=0;const total=(size-24)**2;
    for(let j=12;j<size-12;j++)for(let i=12;i<size-12;i++){
     const p=(j*size+i)*4,R=v.rgba[p],G=v.rgba[p+1],B=v.rgba[p+2],isRed=R>G+8&&R>B+8;
     if(G>R+2&&G>B+2)green++;if(isRed)red++;
     const px=(i+.5-size/2)*30,py=(j+.5-size/2)*30,grade=Math.hypot(800/1800*Math.cos(px/1800)*Math.cos(py/2300),-800/2300*Math.sin(px/1800)*Math.sin(py/2300));
     if(grade>.3){edges++;if(isRed)edgeRed++;}
    }
    inheritance.push({mode,sourceResolution,mean:v.mean,std:v.std,greenFraction:green/total,redFraction:red/total,redFractionOnSteepGround:edges?edgeRed/edges:0});
   }
   TerrainRender3D.set({enabled:false});r.setMode('render-v2');
   // A frozen river may enrich moisture, but cannot create woody candidates
   // where the same production forest evaluator reports zero thermal height.
   TerrainRender3D.set({enabled:true});
   await read(put({lat:67.8,slope:0,cold:true,river:true},.1));
   const auxStride=1280,auxBuffer=r.device.createBuffer({size:auxStride*304,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),auxEncoder=r.device.createCommandEncoder();
   auxEncoder.copyTextureToBuffer({texture:r.tiles.get('snow').aux,origin:[0,0,2]},{buffer:auxBuffer,bytesPerRow:auxStride,rowsPerImage:304},[304,304,1]);r.device.queue.submit([auxEncoder.finish()]);await auxBuffer.mapAsync(GPUMapMode.READ);
   const auxBytes=new Uint8Array(auxBuffer.getMappedRange());let coldCoverage=0,zeroHeightCoverage=0;
   for(let y=24;y<280;y++)for(let x=24;x<280;x++){const p=y*auxStride+x*4;coldCoverage=Math.max(coldCoverage,auxBytes[p]);if(auxBytes[p+1]===0)zeroHeightCoverage=Math.max(zeroHeightCoverage,auxBytes[p]);}
   const thermal={maxColdWoodyCoverage:coldCoverage/255,maxZeroHeightWoodyCoverage:zeroHeightCoverage/255};auxBuffer.unmap();auxBuffer.destroy();
   TerrainRender3D.set({enabled:false});
   // Verify periodic local precision against a freshly uploaded identical tile.
   const uv=put(rows[0],.1),a=await read(uv),b=await read(put(rows[0],.1));
   document.getElementById('map').hidden=true;
   const info=r.adapter.info;
   return {adapter:info?{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,isFallback:r.adapter.isFallbackAdapter??null}:null,results,lighting,inheritance,thermal,repeatExact:a.rgba.every((v,i)=>v===b.rgba[i]),errors:fallback,notes:['Chart-coordinate atlas; globe reprojection is outside this probe.','Snow atlas fixtures are synthetic positive-height planes with lapse rate zero and no canopy/water.','Inheritance fixture is the same analytic warm humid relief, camera and footprint at DEM spacings 30/240m.','The thermal fixture adds a frozen river to a cold plane and reads the production crown field.','Every lighting change explicitly rebakes the production tile.']};
  });
  report.consoleErrors=errors;report.checks=[];
  const check=(name,passed,value)=>report.checks.push({name,passed,value});
  check('actual renderer ready',report.errors===null,report.errors);check('repeat exact after upload',report.repeatExact,report.repeatExact);check('no console errors',errors.length===0,errors);
  const snow=report.results.find(v=>v.cold&&v.lat===67.8&&v.slope===0&&v.fp===.1),bare=report.results.find(v=>!v.cold&&v.lat===67.8&&v.fp===.1);
  check('snow microtexture subdued',snow.std<2,snow.std);check('snow no rougher than bare ground',snow.std<=bare.std+.5,{snow:snow.std,bare:bare.std});
  for(const lat of [67.8,85]){const v=report.results.find(x=>x.temperature!==undefined&&x.lat===lat&&x.fp===41.9),ratio=v.variationX/Math.max(v.variationY,1e-8);check(`partial snow preserves chart metric at latitude ${lat}`,ratio>.4&&ratio<2.5,ratio);}
  for(const fp of [.1,41.9,240]){const cases=report.lighting.filter(v=>v.fp===fp),base=cases.find(v=>v.name==='default');for(const name of ['ambient','contrast','azimuth','no-relief','unlit']){const value=Math.abs(cases.find(v=>v.name===name).mean-base.mean);check(`lighting ${name} affects known slope at ${fp} m/px`,value>.5,value);}}
  for(const mode of ['render3d','render-v2']){const fine=report.inheritance.find(v=>v.mode===mode&&v.sourceResolution===30),coarse=report.inheritance.find(v=>v.mode===mode&&v.sourceResolution===240);check(`${mode} retains green coverage across DEM spacing`,Math.abs(fine.greenFraction-coarse.greenFraction)<.2,{fine:fine.greenFraction,coarse:coarse.greenFraction});check(`${mode} avoids new red contours across DEM spacing`,Math.abs(fine.redFractionOnSteepGround-coarse.redFractionOnSteepGround)<.1,{fine:fine.redFractionOnSteepGround,coarse:coarse.redFractionOnSteepGround});}
  for(const res of [30,240]){const legacy=report.inheritance.find(v=>v.mode==='render3d'&&v.sourceResolution===res),v2=report.inheritance.find(v=>v.mode==='render-v2'&&v.sourceResolution===res);check(`v2 retains legacy green coverage at DEM ${res}m`,v2.greenFraction>=legacy.greenFraction*.65,{legacy:legacy.greenFraction,v2:v2.greenFraction});}
  check('frozen river cannot create thermally impossible trees',report.thermal.maxColdWoodyCoverage<=1/255,report.thermal);
  fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));await page.locator('#atlas').screenshot({path:path.join(out,'snow-atlas.png')});await page.locator('#contracts').screenshot({path:path.join(out,'inheritance-atlas.png')});
  console.log(JSON.stringify({report:path.join(out,'report.json'),atlas:path.join(out,'snow-atlas.png'),checks:report.checks},null,2));
  if(!process.argv.includes('--report-only'))assert(report.checks.every(c=>c.passed),'Snow QA failed; inspect report.json and snow-atlas.png');
 }finally{await browser?.close();await new Promise(ok=>server.close(ok));}
})().catch(e=>{console.error(e);process.exitCode=1;});
