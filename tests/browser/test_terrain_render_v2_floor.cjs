// Small GPU material regressions, not synthetic visual acceptance plates.
// Ablate the production floor response and check where that response survives.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {loadPlaywright,browserExecutable}=require('../../tools/platform.cjs');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/render-v2-floor');
const files=['terrain_render_controls.js','terrain_lighting.js','terrain_forest_brushes.js','terrain_render3d.js','terrain_render_v2.js','terrain_renderer.js'];
const html='<meta charset="utf-8"><canvas id="map"></canvas>'+files.map(f=>`<script src="/${f}"></script>`).join('');
(async()=>{
 const server=http.createServer((req,res)=>{const f=new URL(req.url,'http://x').pathname.slice(1);res.setHeader('Content-Type',files.includes(f)?'application/javascript':'text/html');res.end(files.includes(f)?fs.readFileSync(path.join(root,'web',f)):html);});
 await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const report=await page.evaluate(async()=>{
   if(!await navigator.gpu?.requestAdapter())throw Error('Actual WebGPU adapter required for floor regression');
   const production=TerrainRenderV2.drawShader;
   TerrainRenderV2.drawShader=(...args)=>{
    let s=production(...args);
    const replace=(a,b)=>{if(!s.includes(a))throw Error('Floor ablation hook missing: '+a);s=s.replace(a,b);};
    // Unused production uniform component becomes a test-only ablation flag.
    replace('v2FloorModulation=1.+form+swardTone;','v2FloorModulation=select(1.+form+swardTone,1.,(u32(params.render4.w)&1u)!=0u);');
    replace('detail.y*gain,1.,','detail.y*gain*select(1.,0.,(u32(params.render4.w)&1u)!=0u),1.,');
    replace('detail.w*gain));','detail.w*gain*select(1.,0.,(u32(params.render4.w)&1u)!=0u)));');
    replace('gain*gain*groundSlopeVariance(fp),.25)','gain*gain*groundSlopeVariance(fp)*select(1.,0.,(u32(params.render4.w)&1u)!=0u),.25)');
    replace('applyForest(&m,p,fp,l,RenderGuide(distantColor,open,closed,e.woody*e.leaf),climate,false);','if((u32(params.render4.w)&32u)!=0u){return vec4<f32>(distantColor,1.);}if((u32(params.render4.w)&64u)!=0u){return vec4<f32>(vec3<f32>(e.woody*e.leaf),1.);}applyForest(&m,p,fp,l,RenderGuide(distantColor,open,closed,e.woody*e.leaf),climate,false);');
    replace('let weights=vec3<f32>(1.-share,max(share-canopy,0.),canopy);','let weights=vec3<f32>(1.-share,max(share-canopy,0.),canopy);if((u32(params.render4.w)&128u)!=0u){return vec4<f32>(weights,1.);}');
    replace('return vec4<f32>(v2Final(v2DisplayShoulder(color),s,p,fp,e),1.);','if((u32(params.render4.w)&2u)!=0u){return vec4<f32>(vec3<f32>(forestResolvedCover),1.);}if((u32(params.render4.w)&4u)!=0u){return vec4<f32>(groundToDisplay(m.albedo),1.);}if((u32(params.render4.w)&8u)!=0u){return vec4<f32>(groundToDisplay(m.albedo)*macroLight,1.);}if((u32(params.render4.w)&16u)!=0u){return vec4<f32>(v2DisplayShoulder(shaded),1.);}return vec4<f32>(v2Final(v2DisplayShoulder(color),s,p,fp,e),1.);');
    return s;
   };
   TerrainRender3D.set({enabled:true});TerrainRenderV2.set({season:.5});
   let failure=null;const r=await createTerrainRenderer(document.getElementById('map'),{maxBytes:64*1024**2,onStatus:(s,e)=>{if(s==='fallback')failure=e;}});if(!r)throw Error(failure||'Renderer failed');
   r.context.configure({device:r.device,format:r.format,alphaMode:'premultiplied',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});r.setMode('render-v2');await r.r3Promise;if(r.r3Error)throw Error(r.r3Error);
   const createTexture=r.device.createTexture.bind(r.device);r.device.createTexture=d=>createTexture({...d,usage:d.usage|GPUTextureUsage.COPY_DST});
   let flags=0;const vectors=r.renderVectors.bind(r);r.renderVectors=()=>{const a=vectors();a[19]=flags;return a;};
   r.syncForest();
   const size=128,res=30,fp=.1,span=fp*size/(256*res),uv=[.5-span/2,.5-span/2,span,span];
   const half=x=>{if(x===0)return 0;const e=Math.floor(Math.log2(x));return e< -14?Math.round(x/2**-24):((e+15)<<10)+Math.round((x/2**e-1)*1024);};
   function field(layer,values,allMips=false){
    const tile=r.tiles.get('floor');if(!tile)throw Error('Material tile unavailable: '+failure+'; '+JSON.stringify(r.getStats()));
    const texture=tile.aux,float=texture.format==='rgba16float',v=values.map(x=>float?half(x):Math.round(x*255));
    for(let mipLevel=0;mipLevel<(allMips?texture.mipLevelCount:1);mipLevel++){
     const n=Math.max(1,304>>mipLevel),a=float?new Uint16Array(n*n*4):new Uint8Array(n*n*4);
     for(let i=0;i<n*n;i++)a.set(v,i*4);
     r.device.queue.writeTexture({texture,mipLevel,origin:[0,0,layer]},a,{bytesPerRow:n*4*(float?2:1),rowsPerImage:n},[n,n,1]);
    }
   }
   function put({forest=false,beach=false,flats=false,lit=false}){
    flags=0;TerrainLighting.set({global:{strength:lit?1:0,ambient:.35,contrast:1,exaggeration:6,azimuth:315,altitude:45}});
    const n=33,L=n*n,c=new Float32Array(51*L),fill=(l,v)=>c.fill(v,l*L,(l+1)*L);
    fill(0,26);fill(2,forest?2400:20);[29,forest?1200:10,0,24,forest?1200:10,0].forEach((v,i)=>fill(21+i,v));for(const l of [36,37,38,39,40,41,46,47,48])fill(l,.4);fill(42,.3);fill(49,1);c.set([-2e7,-1e7,2e7,1e7,0,0,17331],45*L);
    r.uploadTile('floor',new Float32Array(304*304).fill(100),{climate:c,climateWidth:n,climateHeight:n,origin:[100000,-300000],metresPerSample:res,mode:'render-v2'});
    // A fixed material unit input isolates the production forest compositor;
    // these fields are not presented as generated terrain or visual evidence.
    field(0,[0,flats?1:0,0,.5]);field(1,[.7,.6,.4,.5]);
    field(2,[forest?.6:0,forest?.24:0,0,0],true);field(4,[forest?.6:0,.6,0,beach?.8:0]);
   }
   async function read(value,sampleFp=fp){
    const sampleSpan=sampleFp*size/(256*res),sampleUv=[.5-sampleSpan/2,.5-sampleSpan/2,sampleSpan,sampleSpan];
    flags=value;r.draw([{key:'floor',x:0,y:0,width:size,height:size,uv:sampleUv,lod:0}],{width:size,height:size,dpr:1});
    const stride=size*4,b=r.device.createBuffer({size:stride*size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e=r.device.createCommandEncoder();e.copyTextureToBuffer({texture:r.context.getCurrentTexture()},{buffer:b,bytesPerRow:stride},[size,size]);r.device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);
    const bytes=new Uint8Array(b.getMappedRange()),a=new Uint8Array(size*size*3);for(let i=0;i<size*size;i++)for(let c=0;c<3;c++)a[i*3+c]=bytes[i*4+(r.format.startsWith('bgra')?2-c:c)];b.unmap();b.destroy();return a;
   }
   const delta=(a,b,mask=()=>true)=>{let sum=0,count=0,max=0;for(let i=0;i<size*size;i++){if(!mask(i))continue;for(let c=0;c<3;c++){const d=Math.abs(a[i*3+c]-b[i*3+c]);sum+=d;max=Math.max(max,d);}count++;}return {mean:sum/Math.max(1,count*3),pixels:count,max};};
   put({});const bareOn=await read(0),bareOff=await read(1),bare=delta(bareOn,bareOff);
   put({forest:true});const litterOn=await read(0),litterOff=await read(1),mask=await read(2),litter=delta(litterOn,litterOff,i=>mask[i*3]<3);
   put({forest:true,lit:true});const crownOn=await read(0),crownOff=await read(1),litMask=await read(2),crown=delta(crownOn,crownOff,i=>litMask[i*3]===255),crownAlbedo=await read(4),crownMacro=await read(8);
   const crownY=a=>{let sum=0,n=0;for(let i=0;i<size*size;i++)if(litMask[i*3]===255){sum+=.2126*a[i*3]+.7152*a[i*3+1]+.0722*a[i*3+2];n++;}return sum/Math.max(1,n);};
   const crownExposure={albedoY:crownY(crownAlbedo),macroOnlyY:crownY(crownMacro),finalY:crownY(crownOn)};
   field(2,[.6,.24,0,1],true);const bankPayloadOn=await read(0),bankToneLeak=delta(bankPayloadOn,crownOn,i=>litMask[i*3]===255);
   put({forest:true,beach:true,lit:true});const beachMask=await read(2),beachMax=Math.max(...beachMask);
   // Fractional beach coverage rejects roots, never fades a surviving crown.
   // The fixed field and seed make this independent of camera footprint.
   put({forest:true,lit:true});field(4,[.6,.6,0,.4]);
   const beachPartialMask=await read(2),beachPartialOpaque=Array.from({length:size*size},(_,i)=>beachPartialMask[i*3]===255).filter(Boolean).length;
   const beachPartialColor=await read(0),beachPartialFloorOff=await read(1),beachPartialCrown=delta(beachPartialColor,beachPartialFloorOff,i=>beachPartialMask[i*3]===255);
   put({flats:true});const flatColor=await read(1);let flatTintError=0;for(let i=0;i<flatColor.length;i++)flatTintError+=Math.abs(flatColor[i]-(bareOff[i]*.2+[.7,.6,.4][i%3]*255*.9*.8));flatTintError/=flatColor.length;
   put({forest:true,flats:true,lit:true});const estuaryMask=await read(2),estuaryOpaquePixels=Array.from({length:size*size},(_,i)=>estuaryMask[i*3]===255).filter(Boolean).length;
   put({lit:true});const flatExposed=await read(1),flatDirect=await read(17),flatExposure=delta(flatExposed,flatDirect);
   put({forest:true});const meanMaterial=await read(4,40),meanGuide=await read(32,40),meanCover=await read(64,40),unresolvedMean=delta(meanMaterial,meanGuide),unresolvedCover=meanCover.reduce((a,b)=>a+b,0)/meanCover.length/255;
   const area=await read(128,40),meanArea=[0,0,0];let areaSumError=0;for(let i=0;i<size*size;i++){let sum=0;for(let c=0;c<3;c++){meanArea[c]+=area[i*3+c]/255/(size*size);sum+=area[i*3+c]/255;}areaSumError=Math.max(areaSumError,Math.abs(sum-1));}
   // Fully snow-covered forest floor must keep the snow albedo in gaps and
   // under nearby crowns; choosing brown litter after snow used to erase it.
   put({forest:true});field(3,[.3,.8,.3,.3]);field(2,[.6,.24,1,0],true);
   const snowForest=await read(4),snowMask=await read(2);field(2,[0,.24,1,0],true);const snowBare=await read(4);
   const snowLitter=delta(snowForest,snowBare,i=>snowMask[i*3]===0);
   return {kind:'GPU material unit regression; not visual acceptance of terrain',adapter:await r.device.adapterInfo,format:r.tiles.get('floor').aux.format,auxLayers:r.tiles.get('floor').auxLayers,bare,litter,crown,crownExposure,bankToneLeak,beachMax,beachPartialOpaque,beachPartialCrown,flatTintError,estuaryOpaquePixels,flatExposure,unresolvedMean,unresolvedCover,meanArea,areaSumError,snowLitter,errors:failure};
  });
  assert.equal(report.errors,null);assert.deepEqual(errors,[]);assert(report.bare.mean>.2,`Bare floor lost its albedo detail: ${JSON.stringify(report.bare)}`);
  assert(report.bare.mean<3&&report.bare.max<8,`Bare soil detail became a high-contrast colour mask: ${JSON.stringify(report.bare)}`);
  assert(report.litter.pixels>20,`No visible forest floor sampled: ${JSON.stringify(report.litter)}`);assert(report.litter.mean>.1,`Litter lost floor detail: ${JSON.stringify(report.litter)}`);
  assert(report.crown.pixels>20,`No opaque crowns sampled: ${JSON.stringify(report.crown)}`);assert(report.crown.mean<.2,`Floor detail leaked onto crowns: ${JSON.stringify(report.crown)}`);assert.equal(report.beachMax,0,'Forest covers a retained legacy beach');
  assert(report.beachPartialOpaque>20,`Fractional beach coverage made all surviving crowns transparent: ${report.beachPartialOpaque}`);
  assert(report.beachPartialCrown.mean<.2,`Fractional beach coverage mixed floor into opaque crowns: ${JSON.stringify(report.beachPartialCrown)}`);
  assert(report.bankToneLeak.mean<.2,`Bank moisture payload changed the colour of an unchanged opaque crown: ${JSON.stringify(report.bankToneLeak)}`);
  assert(report.flatTintError<1.1,`Legacy bar sediment lost to the near material: ${report.flatTintError}`);assert(report.estuaryOpaquePixels>20,'Low flats incorrectly forbid estuary forests');
  assert(report.flatExposure.mean<.4,`Flat terrain received the hillshade's base exposure twice: ${JSON.stringify(report.flatExposure)}`);
  assert(report.unresolvedCover>.05&&report.unresolvedCover<.95,`Mean regression did not exercise mixed open/canopy coverage: ${report.unresolvedCover}`);
  assert(report.unresolvedMean.mean<.4,`Unresolved material changed the inherited display-colour mean: ${JSON.stringify(report.unresolvedMean)}`);
  assert(report.areaSumError<2/255,`Forest area responses do not conserve total area: ${report.areaSumError}`);
  assert(report.meanArea[2]>.55&&report.meanArea[2]<.65,`Mean canopy did not use the painted actual population: ${report.meanArea}`);
  assert(report.snowLitter.pixels>20,`No snowy understorey sampled: ${JSON.stringify(report.snowLitter)}`);
  assert(report.snowLitter.max<=1,`Litter repainted brown ground over a full snow mantle: ${JSON.stringify(report.snowLitter)}`);
  fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
 }finally{await browser?.close();await new Promise(r=>server.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
