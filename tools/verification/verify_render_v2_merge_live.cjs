// Live merge acceptance: production viewer/server, physical tiles and RTX 5090.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const {loadPlaywright,browserExecutable,pythonExecutable}=require('../platform.cjs');
const ROOT=path.resolve(__dirname,'../..');process.chdir(ROOT);
const OUT=path.resolve('tmp/e2e/merge-rv2'),BASE='http://[::1]:8765',sites=fs.existsSync(path.join(OUT,'sites.json'))?JSON.parse(fs.readFileSync(path.join(OUT,'sites.json'))):[{name:'high-mountains',centre:[-8232421.875,-400390.625]},{name:'coastal-mountains',centre:[-14515275,-5186565]}];
fs.mkdirSync(OUT,{recursive:true});
const summary={sites,views:[],responses:[],errors:[],history:[],rivers:[]};let browser,page,label='startup';
const save=()=>fs.writeFileSync(path.join(OUT,'live-summary.json'),JSON.stringify(summary,null,2));
const pixel=lod=>30*2**lod*1.01;
async function ready(lod,mode){
 await page.waitForFunction(({lod,mode})=>{
  if(!window.terrainDebug||!world||worldLoading||generationPending||!rendererReady||error)return false;
  const s=terrainDebug.snapshot();return s.backend==='webgpu'&&s.visible.wanted>0&&s.visible.pending===0&&s.scheduler.queued===0&&s.scheduler.inflight===0&&publishedEpoch===cameraEpoch&&viewMode===mode&&visiblePlan.length>0&&visiblePlan.every(p=>p.tile.lod===lod&&isCurrentTile(p.tile))&&(!['render3d','render-v2'].includes(mode)||renderer.getStats().render3d.ready);
 },{lod,mode},{timeout:300000});
 await page.evaluate(async()=>{draw();await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));await renderer.device.queue.onSubmittedWorkDone();});
}
async function configure(site,lod,amp,mode,season=.5){
 await page.evaluate(({site,lod,amp,mode,season})=>{
  cx=site.centre[0];cy=site.centre[1];mpp=30*2**lod*1.01;
  hydrology.set({...TerrainHydrology.defaults(),enabled:false});TerrainAmplification.set({mode:amp,intensity:1,before:false},true);
  renderV2.set({season});$('mapMode').value=mode;$('mapMode').onchange();$('forcedLod').value=String(lod);forceViewLod();clampCamera();draw();schedule();
 },{site,lod,amp,mode,season});
 await ready(lod,mode);
}
async function capture(site,lod,amp,mode,season=.5,suffix=''){
 label=`${site.name}_${amp}_lod${lod}_${mode}_${Math.round(season*12)+1}${suffix}`;const start=performance.now();
 await configure(site,lod,amp,mode,season);const file=path.join(OUT,label+'.png');await page.locator('#viewport').screenshot({path:file});
 const s=await page.evaluate(()=>({state:terrainDebug.snapshot(),tiles:visiblePlan.map(p=>({lod:p.tile.lod,mode:p.tile.amplificationMode,surface:!!p.tile.heightOptions?.surface,gradient:!!p.tile.heightOptions?.gradient})),floors:world.relief_floors,gpuTiles:[...renderer.tiles.values()].filter(t=>!t.native).map(t=>({surface:!!t.surface,gradient:!!t.gradient,materialMeta:!!t.materialMeta}))}));
 if(lod<0&&amp==='hybrid')assert(s.tiles.every(t=>t.surface&&t.mode==='hybrid'));
 if(lod<s.floors[amp==='off'?'classic':amp])assert(s.tiles.every(t=>t.gradient));
 const v={site:site.name,lod,amp,mode,season,path:path.relative(ROOT,file).replaceAll('\\','/'),seconds:(performance.now()-start)/1000,...s};summary.views.push(v);save();console.log(label);return v;
}
function compare(a,b){const r=JSON.parse(execFileSync(pythonExecutable(),['-c',`import sys,json,numpy as np\nfrom PIL import Image\na=np.asarray(Image.open(sys.argv[1]).convert('RGB'),dtype=np.int16);b=np.asarray(Image.open(sys.argv[2]).convert('RGB'),dtype=np.int16);d=np.abs(a-b);print(json.dumps(dict(maxRGB=int(d.max()),meanRGB=float(d.mean()),changedPixels=int(np.any(d,axis=2).sum()))))`,a,b],{encoding:'utf8'}));return r;}
(async()=>{
 try{
  summary.server=await(await fetch(BASE+'/api/gpu')).json();assert.equal(summary.server.devices.length,1);assert.equal(summary.server.devices[0].name,'NVIDIA GeForce RTX 5090');
  browser=await loadPlaywright().chromium.launch({headless:true,executablePath:browserExecutable(),args:['--enable-unsafe-webgpu']});page=await browser.newPage({viewport:{width:1280,height:900},deviceScaleFactor:1});
  page.on('pageerror',e=>{summary.errors.push({label,message:e.message});save();});page.on('console',m=>{if((m.type()==='error'||/unavailable|validation|WGSL/i.test(m.text()))&&!/409/.test(m.text())&&!m.location()?.url?.endsWith('/favicon.ico')){summary.errors.push({label,message:m.text()});save();}});
  page.on('response',async r=>{if(!/\/(height|tiles|coarse)\//.test(r.url()))return;const h=await r.allHeaders();summary.responses.push({label,url:r.url(),status:r.status(),compute:h['x-terrain-compute-seconds'],amp:h['x-terrain-amplification-mode'],surface:h['x-terrain-surface-format'],gradient:h['x-terrain-gradient-format']});if(r.status()>=400&&r.status()!==409)summary.errors.push({label,status:r.status(),url:r.url()});});
  await page.goto(BASE+'/api/status');summary.browserGPU=await page.evaluate(async()=>{const a=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});return {vendor:a.info.vendor,architecture:a.info.architecture};});assert.equal(summary.browserGPU.architecture,'blackwell');
  await page.addInitScript(()=>{localStorage.setItem('neural-earth-preferences-v1',JSON.stringify({gpuAcceleration:true,erosionDefaultsVersion:1,rendering:{gpuRender:true,coarseGpu:false}}));localStorage.setItem('neural-earth-amplification-v2',JSON.stringify({mode:'off',intensity:1,before:false}));});
  await page.goto(BASE+'/?seed=42&profile=orogen&generation=null&mode=relief&gpu=1&hydro=0&coarse_gpu=0&coarse_prepare=0');
  await page.waitForFunction(()=>window.terrainDebug&&world&&rendererReady,null,{timeout:180000});
  await page.evaluate(()=>{setSharedSize([640,480]);$('prefetch').checked=false;$('refinementDepth').value='0';$('cacheLodGap').value='3';$('renderSea').checked=true;$('grid').checked=false;styleRendering.set({enabled:false,automatic:false,interval:10});});
  await page.addStyleTag({content:'#hud,#mini,#legend,#cursorAltitude{visibility:hidden!important}'});
  for(const site of sites)for(const lod of [4,0,-4,-8])for(const amp of ['off','hybrid']){
   for(const mode of ['relief','render','render3d'])await capture(site,lod,amp,mode);
   for(let month=0;month<12;month++)await capture(site,lod,amp,'render-v2',month/12);
  }
  for(const site of sites)for(const amp of ['off','hybrid']){
   await configure(site,-8,amp,'render');await page.evaluate(()=>{$('forcedLod').value='';forcedView=null;draw();schedule();});
   const box=await page.locator('#viewport').boundingBox();await page.mouse.move(box.x+box.width/2,box.y+box.height/2);
   for(let lod=-9;lod>=-12;lod--){await page.mouse.wheel(0,-Math.log(2)/.0015);await ready(lod,'render');}
   for(let lod=-11;lod<=-8;lod++){await page.mouse.wheel(0,Math.log(2)/.0015);await ready(lod,'render');}
   await page.evaluate(({site,m})=>{cx=site.centre[0];cy=site.centre[1];mpp=m;draw();schedule();},{site,m:pixel(-8)});await ready(-8,'render');
   for(const mode of ['render','relief','render3d','render-v2']){
    await page.evaluate(mode=>{renderV2.set({season:.5});$('mapMode').value=mode;$('mapMode').onchange();draw();schedule();},mode);await ready(-8,mode);
    const file=path.join(OUT,`${site.name}_${amp}_${mode}_history.png`);await page.locator('#viewport').screenshot({path:file});
    const original=summary.views.find(v=>v.site===site.name&&v.lod===-8&&v.amp===amp&&v.mode===mode&&v.season===.5),difference=compare(original.path,file),state=await page.evaluate(()=>({forced:forcedView,lods:[...new Set(visiblePlan.map(p=>p.tile.lod))],cached:[...new Set([...tiles.values()].map(t=>t.lod))]}));
    assert.equal(state.forced,null);assert.deepEqual(state.lods,[-8]);assert(state.cached.includes(-12));assert.equal(difference.maxRGB,0);summary.history.push({site:site.name,amp,mode,...difference,...state,path:path.relative(ROOT,file)});save();
   }
  }
  // Regional masks are active at LOD <=3; coarser views retain vectors.
  // Choose an authoritative perennial reach near the coastal test region.
  const [rx,ry]=sites[1].centre,bounds=[rx-250000,ry-190000,rx+250000,ry+190000];
  let network;
  for(let i=0;i<90;i++){
   network=await(await fetch(BASE+'/api/hydrology/42?world_profile=orogen&bounds='+bounds.join(',')+'&lod=4&month=7')).json();
   if(network.status==='ready'&&network.reaches?.length)break;
   await new Promise(r=>setTimeout(r,1000));
  }
  const reach=network.reaches.filter(r=>r.xy?.length>1&&r.flow_m3s>0).sort((a,b)=>b.flow_m3s-a.flow_m3s)[0];assert(reach,'No source river in coastal region');
  const riverSite={name:'coastal-river',centre:reach.xy[Math.floor(reach.xy.length/2)]};summary.riverSite={...riverSite,reachId:reach.id,width:reach.width_m};
  await configure(riverSite,0,'hybrid','render-v2',.5);
  await page.evaluate(()=>{hydrology.set({...TerrainHydrology.defaults(),enabled:true,density:2,intermittent:true});draw();schedule();});
  await ready(0,'render-v2');await page.waitForFunction(()=>{const h=hydrology.snapshot();return h.status==='ready'&&renderer.getStats().riverDisplay?.ready&&renderer.getStats().riverDisplay?.segments>0;},null,{timeout:180000});
  summary.rivers.push(await page.evaluate(()=>({hydrology:hydrology.snapshot(),renderer:renderer.getStats().riverDisplay,frame:TerrainHydrology.riverDisplay()})));
  await page.locator('#viewport').screenshot({path:path.join(OUT,'coastal_hybrid_render-v2_rivers.png')});
  assert(summary.rivers[0].renderer.segments>0,'No real regional rivers displayed');
  assert.equal(summary.errors.length,0);summary.passed=true;save();
 }catch(e){summary.failure={label,message:e.message,stack:e.stack};save();console.error(e);process.exitCode=1;}finally{await browser?.close();}
})();
