// Real camera history, deliberately unforced: forced-LOD checks cannot expose this bug.
const {loadPlaywright,browserExecutable,pythonExecutable}=require('../platform.cjs');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const ROOT=path.resolve(__dirname,'../..');process.chdir(ROOT);
const BASE=process.env.BASE_URL||'http://[::1]:8765',stage=process.env.E2E_STAGE||'before';
const OUT=path.resolve('tmp/e2e/lod8bug',stage);fs.mkdirSync(OUT,{recursive:true});
const centre=[-8232421.875,-400390.625],pixel=lod=>30*2**lod*1.01;
const summary={stage,base:BASE,centre,views:[],steps:[],errors:[],tileErrors:[],responses:[],comparisons:[]};
const save=()=>fs.writeFileSync(path.join(OUT,'summary.json'),JSON.stringify(summary,null,2));
let browser,page,context,label='startup';
async function ready(lod){
  await page.waitForFunction(lod=>{
    if(!window.terrainDebug||!world||!rendererReady||worldLoading||generationPending||error)return false;
    const s=terrainDebug.snapshot();return forcedView===null&&s.camera.lod===lod&&s.backend==='webgpu'&&s.visible.wanted>0&&s.visible.pending===0&&s.scheduler.queued===0&&s.scheduler.inflight===0&&publishedEpoch===cameraEpoch;
  },lod,{timeout:180000});
  await page.evaluate(async()=>{draw();await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));await renderer.device.queue.onSubmittedWorkDone();});
}
async function snapshot(){return page.evaluate(()=>({snapshot:terrainDebug.snapshot(),floor:world.relief_floors,patches:visiblePlan.map(p=>({key:p.tile.key,lod:p.tile.lod,tx:p.tile.tx,ty:p.tile.ty,bounds:p.bounds,uv:p.uv,gradient:!!p.tile.heightOptions?.gradient,spacing:p.tile.heightOptions?.metresPerSample})),cachedLods:[...new Set([...tiles.values()].map(t=>t.lod))].sort((a,b)=>a-b)}));}
async function capture(name,lod){
  await page.evaluate(m=>{mpp=m;draw();schedule();},pixel(lod));
  await ready(lod);const state=await snapshot();
  const record={name,...state,files:{}};
  for(const grid of [false,true]){
    await page.evaluate(async grid=>{$('grid').checked=grid;draw();await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));},grid);
    const file=path.join(OUT,name+(grid?'_grid':'_clean')+'.png');await page.locator('#viewport').screenshot({path:file});record.files[grid?'grid':'clean']=path.relative(ROOT,file).replaceAll('\\','/');
  }
  summary.views.push(record);save();console.log(JSON.stringify({capture:name,lods:state.snapshot.rendered.lods,patches:state.patches.length,cache:state.cachedLods}));
  if(process.env.E2E_EXPECT_UNIFORM&&lod===-8)assert(state.patches.every(p=>p.lod===-8),'ready -8 view contains cached finer/coarser patches');
  return record;
}
async function open(layer,mode,lod){
  context=await browser.newContext({viewport:{width:1280,height:900},deviceScaleFactor:1});page=await context.newPage();
  page.on('pageerror',e=>summary.errors.push({label,message:e.message}));page.on('console',m=>{if(m.type()==='error'&&!m.location()?.url?.endsWith('/favicon.ico')&&!/409/.test(m.text()))summary.errors.push({label,message:m.text()});});
  page.on('response',async r=>{if(!/\/(height|tiles)\//.test(r.url()))return;if(r.status()>=400&&r.status()!==409)summary.tileErrors.push({label,url:r.url(),status:r.status()});if(r.ok()){const h=await r.allHeaders();summary.responses.push({label,url:r.url(),gradient:h['x-terrain-gradient-format']||null,interpolation:h['x-terrain-relief-interpolation']||null});}});
  await page.goto(BASE+'/api/status');const info=await page.evaluate(async()=>{const a=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});return {vendor:a?.info.vendor,architecture:a?.info.architecture};});assert.equal(info.architecture,'blackwell','Browser must be pinned to RTX 5090 before terrain requests');summary.browserGPU=info;
  await page.addInitScript(()=>{localStorage.setItem('neural-earth-preferences-v1',JSON.stringify({gpuAcceleration:true,erosionDefaultsVersion:1,rendering:{gpuRender:true,coarseGpu:false}}));localStorage.setItem('neural-earth-amplification-v2',JSON.stringify({mode:'off',intensity:1,before:false}));});
  await page.goto(BASE+'/?seed=42&profile=orogen&generation=null&gpu=1&mode='+layer+'&hydro=0&coarse_gpu=0&coarse_prepare=0');
  await page.waitForFunction(()=>window.terrainDebug&&terrainDebug.snapshot().world&&terrainDebug.snapshot().rendererReady,null,{timeout:180000});
  await page.evaluate(({centre,layer,mode,mpp})=>{
    setSharedSize([960,640]);$('prefetch').checked=false;$('refinementDepth').value='0';$('cacheLodGap').value='3';$('renderSea').checked=true;
    hydrology?.set({...TerrainHydrology.defaults(),enabled:false});styleRendering.set({enabled:false,automatic:false,interval:10});
    $('mapMode').value=layer;$('mapMode').onchange();TerrainAmplification.set({mode,intensity:1,before:false},true);
    $('forcedLod').value='';forcedView=null;cx=centre[0];cy=centre[1];globalThis.mpp=mpp;
    // mpp is a lexical application binding, not a property of window.
    window.__historyCamera={cx:centre[0],cy:centre[1],mpp};
  },{centre,layer,mode,mpp:pixel(lod)});
  await page.evaluate(()=>{({cx,cy,mpp}=window.__historyCamera);clampCamera();draw();schedule();});
  await page.addStyleTag({content:'#hud,#mini,#legend,#cursorAltitude{visibility:hidden!important}'});await ready(lod);
  if(!summary.world)summary.world=await page.evaluate(()=>({seed:world.seed,identity:world.world_identity,profile:world.cache_profile}));
}
async function wheelTo(lod,inward){
  const box=await page.locator('#viewport').boundingBox();await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.wheel(0,(inward?-1:1)*Math.log(2)/.0015);await ready(lod);
  summary.steps.push({label,lod,...await snapshot()});save();
}
async function pan(dx,dy){
  const box=await page.locator('#viewport').boundingBox(),x=box.x+box.width/2,y=box.y+box.height/2;
  await page.mouse.move(x,y);await page.mouse.down();await page.mouse.move(x+dx,y+dy,{steps:8});await page.mouse.up();await ready(-8);
}
async function precisionProbe(){
  const source=fs.readFileSync('web/terrain_renderer.js','utf8'),point=source.slice(source.indexOf('    fn surfacePoint('),source.indexOf('    fn surfaceNorth('));
  summary.precisionProbe=await page.evaluate(async point=>{
    const device=renderer.device,shader=`const probeMeta=array<f32,7>(-20000000.,-10000000.,20000000.,10000000.,1.,0.,1234.);fn surfaceMeta(i:i32)->f32{return probeMeta[i];}\n${point}\n@group(0) @binding(0) var<storage,read_write> result:array<vec4<f32>>;@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<256u){result[id.x]=vec4<f32>(surfacePoint(vec2<f32>(f32(id.x)*.1,0.),vec2<f32>(-8232421.875,-400390.625),1.),1.);}}`;
    const module=await renderer.compileShader(shader,'actual surfacePoint precision probe'),pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
    const output=device.createBuffer({size:4096,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:4096,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]}));pass.dispatchWorkgroups(4);pass.end();encoder.copyBufferToBuffer(output,0,read,0,4096);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const values=new Float32Array(read.getMappedRange()).slice();read.unmap();output.destroy();read.destroy();
    const distances=[];for(let i=1;i<256;i++)distances.push(Math.hypot(...[0,1,2].map(k=>values[i*4+k]-values[(i-1)*4+k])));
    const tile=visiblePlan[0].tile,o=tile.heightOptions,width=304,L=33*33,climate=new Float32Array(o.climate.length);
    for(let i=0;i<50;i++)climate.fill(o.climate[i*L+16*33+16],i*L,(i+1)*L);climate.set(o.climate.subarray(45*L,46*L),45*L);
    const heights=new Float32Array(width*width).fill(800),r=30*2**-8,origin=o.origin,options={width,height:width,halo:24,metresPerSample:r,lod:-8,climate,climateWidth:33,climateHeight:33,gradient:new Float32Array(width*width*2)};renderer.setMode('render');
    renderer.uploadTile('__probeA',heights,{...options,origin});renderer.uploadTile('__probeB',heights,{...options,origin:[origin[0]+128*r,origin[1]]});
    async function pixels(key){const buffer=device.createBuffer({size:256*256*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e=device.createCommandEncoder();e.copyTextureToBuffer({texture:renderer.tiles.get(key).color},{buffer,bytesPerRow:1024},[256,256]);device.queue.submit([e.finish()]);await buffer.mapAsync(GPUMapMode.READ);const v=new Uint8Array(buffer.getMappedRange()).slice();buffer.unmap();buffer.destroy();return v;}
    const a=await pixels('__probeA'),b=await pixels('__probeB');let max=0,sum=0,changed=0;
    for(let y=0;y<256;y++)for(let x=0;x<128;x++){let different=false;for(let k=0;k<3;k++){const d=Math.abs(a[(y*256+x+128)*4+k]-b[(y*256+x)*4+k]);max=Math.max(max,d);sum+=d;if(d)different=true;}if(different)changed++;}
    renderer.deleteTile('__probeA');renderer.deleteTile('__probeB');
    return {coordinateStepMetres:.1,unchangedSteps:distances.filter(d=>d===0).length,totalSteps:255,maxStepMetres:Math.max(...distances),smallestMaterialNoiseWavelengthMetres:30,overlappingFlatMaterial:{maxRGB:max,meanRGB:sum/(256*128*3),changedPixelFraction:changed/(256*128)}};
  },point);save();
}
function compare(reference,actual,requireEqual=true){
  const result=JSON.parse(execFileSync(pythonExecutable(),['-c',String.raw`import sys,json,numpy as np
from PIL import Image
a=np.asarray(Image.open(sys.argv[1]).convert('RGB'),dtype=np.float32);b=np.asarray(Image.open(sys.argv[2]).convert('RGB'),dtype=np.float32);d=np.abs(a-b)
print(json.dumps(dict(meanAbsoluteRGB=float(d.mean()),maxRGB=float(d.max()),changedPixelFraction=float(np.any(d>0,axis=2).mean()))))`,reference.files.clean,actual.files.clean],{encoding:'utf8'}));
  summary.comparisons.push({reference:reference.name,actual:actual.name,...result});save();
  if(process.env.E2E_EXPECT_UNIFORM&&requireEqual)assert.equal(result.maxRGB,0,'same-view appearance changed after zoom history');
}
(async()=>{
  try{
    summary.server=await(await fetch(BASE+'/api/gpu')).json();assert.equal(summary.server.devices.length,1);assert.equal(summary.server.devices[0].name,'NVIDIA GeForce RTX 5090');
    browser=await loadPlaywright().chromium.launch({headless:true,executablePath:browserExecutable()||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',args:['--enable-unsafe-webgpu']});
    if(process.env.E2E_PROBE_ONLY){await open('render','off',-8);await precisionProbe();summary.passed=true;save();console.log(summary.precisionProbe);return;}
    for(const layer of ['relief','render'])for(const mode of ['off','hybrid']){
      label=layer+'_'+mode;
      await open(layer,mode,-8);const fresh=await capture(label+'_fresh8',-8);await context.close();
      await open(layer,mode,-4);await ready(-4);
      let reference;
      for(let lod=-5;lod>=-12;lod--){await wheelTo(lod,true);if(lod===-8)reference=await capture(label+'_initial8',-8);}
      for(let lod=-11;lod<=-8;lod++)await wheelTo(lod,false);
      // Align sub-ULP wheel roundoff, retaining the real cached history.
      await page.evaluate(({centre,m})=>{cx=centre[0];cy=centre[1];mpp=m;draw();schedule();},{centre,m:pixel(-8)});
      const returned=await capture(label+'_returned8',-8);assert(returned.cachedLods.includes(-12),'finer history should remain cached');compare(reference,returned);compare(fresh,returned,false);
      await pan(-96,-40);await capture(label+'_pan8',-8);await pan(96,40);
      await page.evaluate(({centre,m})=>{cx=centre[0];cy=centre[1];mpp=m;draw();schedule();},{centre,m:pixel(-8)});
      const restored=await capture(label+'_pan_back8',-8);compare(reference,restored);
      await context.close();save();
    }
    assert.equal(summary.errors.length,0);assert.equal(summary.tileErrors.length,0);summary.passed=true;save();
  }catch(e){summary.failure={label,message:e.message,stack:e.stack};save();console.error(e);process.exitCode=1;}finally{await browser?.close();}
})();
