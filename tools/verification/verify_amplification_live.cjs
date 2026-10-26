// Live Neural Earth amplification verification. No mocked terrain/renderer/transport.
// CUDA isolation belongs to the server launch (CUDA_VISIBLE_DEVICES=1).
const {chromium}=require('playwright');
const {browserExecutable,pythonExecutable}=require('../platform.cjs');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process'),{performance}=require('node:perf_hooks');
const ROOT=path.resolve(__dirname,'../..');process.chdir(ROOT);
const OUT=path.resolve(process.env.E2E_OUT||'tmp/e2e'),BASE=process.env.BASE_URL||'http://127.0.0.1:8765';
fs.mkdirSync(OUT,{recursive:true});fs.mkdirSync(path.join(OUT,'layers'),{recursive:true});
const benchmark=JSON.parse(fs.readFileSync('tmp/amplification/timings.json','utf8'));
const sites=[...new Map(benchmark.timings.filter(t=>t.centre_metres).map(t=>[t.relief,{name:t.relief.replace('real-42-',''),centre:t.centre_metres}])).values()]
  .filter(s=>!process.env.E2E_SITES||process.env.E2E_SITES.split(',').includes(s.name));
const modes=(process.env.E2E_MODES||'off,hybrid,phasor,mse').split(',');
const lods=(process.env.E2E_LODS||'-2,-4,-6').split(',').map(Number),layers=(process.env.E2E_LAYERS||'render,relief').split(',');
const summary={schema:1,started:new Date().toISOString(),baseURL:BASE,sites,lods,layers,modes,views:[],responses:[],shortcuts:[],diagnostics:[],consoleErrors:[],pageErrors:[],tileErrors:[],protocolErrors:[],cancellations:[],contactSheets:[],
  methodology:{cold:'First delivery of this physical tile key in this run; parent/model caches are shared. Inspect cache flags to distinguish pre-existing hits.',warm:'Clear only browser/GPU tile caches, then request the same view again. Server physical RAM cache remains resident.',computeHeader:'X-Terrain-Compute-Seconds is the original generation receipt even on a RAM hit. executedComputeSeconds is zero on a cache hit; requestSeconds measures actual response latency.',seams:'Mean RGB absolute adjacent-pixel gradient at projected tile boundaries versus interior samples of the same direction. Ratio >2 is flagged for visual review, not treated as proof of a seam.'}};
let phase='startup',responseWork=new Set(),page,browser;
const save=()=>fs.writeFileSync(path.join(OUT,'summary.json'),JSON.stringify(summary,null,2));
const stat=v=>({n:v.length,mean:v.length?v.reduce((a,b)=>a+b,0)/v.length:null,median:v.length?[...v].sort((a,b)=>a-b)[Math.floor(v.length/2)]:null,min:v.length?Math.min(...v):null,max:v.length?Math.max(...v):null});
const py=(source,args=[])=>JSON.parse(execFileSync(pythonExecutable(),['-c',source,...args],{cwd:ROOT,encoding:'utf8',maxBuffer:8*1024**2}));
const imageStats=String.raw`
import sys,json
import numpy as np
from PIL import Image
im=np.asarray(Image.open(sys.argv[1]).convert('RGB'),dtype=np.float32)
geometry=json.loads(sys.argv[2]);h,w=im.shape[:2]
gx=np.abs(im[:,1:]-im[:,:-1]).mean(2);gy=np.abs(im[1:]-im[:-1]).mean(2)
xs=sorted(set(round(v) for v in geometry.get('xs',[]) if 8<v<w-8));ys=sorted(set(round(v) for v in geometry.get('ys',[]) if 8<v<h-8))
def direction(g,lines,other,vertical):
    if not vertical:g=g.T
    take=np.ones(g.shape[0],bool);take[:8]=False;take[-8:]=False
    for p in other:take[max(0,p-4):p+4]=False
    edge=[];inside=[]
    for p in lines:
        edge.extend(g[take,max(0,p-1)].tolist())
        for d in (-24,-12,12,24):
            q=p+d-1
            if 4<q<g.shape[1]-4:inside.extend(g[take,q].tolist())
    a=float(np.mean(edge)) if edge else None;b=float(np.mean(inside)) if inside else None
    return dict(boundaries=lines,boundaryMean=a,interiorMean=b,ratio=a/b if a is not None and b and b>1e-8 else None,samples=len(edge))
v=direction(gx,xs,ys,True);z=direction(gy,ys,xs,False)
vals=[d for d in (v,z) if d['ratio'] is not None];n=sum(d['samples'] for d in vals)
a=sum(d['boundaryMean']*d['samples'] for d in vals)/n if n else None
b=sum(d['interiorMean']*d['samples'] for d in vals)/n if n else None
print(json.dumps(dict(width=w,height=h,mean=float(im.mean()),std=float(im.std()),vertical=v,horizontal=z,ratio=a/b if b else None)))
`;
function checkHeaders(record,body){
  const h=record.headers,lod=record.lod,mode=record.mode,active=lod<0&&record.intensity>0;
  const fail=message=>summary.protocolErrors.push({url:record.url,message});
  if(active){
    const expected={hybrid:'hybrid-drainage-erosion',phasor:'phasor-noise',mse:'multiscale-erosion'}[mode];
    if(h['x-terrain-amplification']!==expected)fail('amplification kind');
    if(h['x-terrain-amplification-mode']!==mode)fail('amplification mode');
    if(!h['x-terrain-amplification-version'])fail('missing amplification version');
    if(Number(h['x-terrain-amplification-intensity'])!==record.intensity)fail('amplification intensity');
    if(!h['x-terrain-height-filter'])fail('missing height filter');
    if(mode==='mse'&&!h['x-terrain-amplification-stages'])fail('missing MSE stages');
    if(record.binary&&(h['x-terrain-surface-format']!=='rgba8unorm'||h['x-terrain-surface-channels']!=='flow,deposit,scree,rock'))fail('surface schema');
  }else if(h['x-terrain-amplification']||h['x-terrain-surface-format'])fail('legacy payload acquired amplification');
  if(lod<0&&!String(h['cache-control']).includes('no-store'))fail('negative LOD must be no-store');
  if(record.binary&&body){
    const width=Number(h['x-terrain-width']),cw=Number(h['x-terrain-climate-width']||0),ch=Number(h['x-terrain-climate-height']||0),cl=Number(h['x-terrain-climate-layers']||0),water=Number(h['x-terrain-hydro-layers']||0);
    const expected=width*width*4+cw*ch*cl*4+water*width*width*4+(active?width*width*4:0)+(h['x-terrain-gradient-format']?width*width*8:0);
    if(body.length!==expected)fail(`payload bytes ${body.length} != ${expected}`);
    record.payloadBytes=body.length;
  }
  if(h['x-terrain-device']&&!h['x-terrain-device'].includes('5090'))fail('server tile used a GPU other than RTX 5090');
}
async function collectResponse(response){
  const url=new URL(response.url()),match=url.pathname.match(/\/(height|tiles)\/[^/]+\/42\/(-?\d+)\/(-?\d+)\/(-?\d+)\.(bin|png)$/);if(!match)return;
  const record={phase,url:response.url(),lod:Number(match[2]),tx:Number(match[3]),ty:Number(match[4]),binary:match[5]==='bin',intensity:Number(url.searchParams.get('amplify')||0),mode:url.searchParams.get('amplify_mode')||'hybrid',status:response.status()};
  if(record.status===409){summary.cancellations.push(record);return;}
  if(!response.ok()){summary.tileErrors.push(record);return;}
  record.headers=await response.allHeaders();await response.finished();
  record.requestSeconds=response.request().timing().responseEnd/1000;
  record.headerComputeSeconds=Number(record.headers['x-terrain-compute-seconds']||0);record.cacheHit=record.headers['x-terrain-cache']==='hit';record.executedComputeSeconds=record.cacheHit?0:record.headerComputeSeconds;
  checkHeaders(record,record.binary?await response.body():null);summary.responses.push(record);
}
async function drain(){while(responseWork.size)await Promise.allSettled([...responseWork]);}
async function waitReady(expected){
  await page.waitForFunction(({lod,mode,layer})=>{
    if(!window.terrainDebug||!world||!rendererReady||worldLoading||generationPending||error)return false;
    const s=terrainDebug.snapshot();return s.backend==='webgpu'&&s.visible.wanted>0&&s.visible.pending===0&&s.scheduler.queued===0&&s.scheduler.inflight===0&&publishedEpoch===cameraEpoch&&forcedView?.lod===lod&&viewMode===layer&&visiblePlan.length>0&&visiblePlan.every(p=>p.tile.lod===lod&&(p.tile.amplify||0)===(mode==='off'?0:TerrainAmplification.get().intensity)&&(!p.tile.amplify||p.tile.amplifyMode===mode));
  },expected,{timeout:Number(process.env.E2E_TIMEOUT||300000)});
  await page.evaluate(async()=>{draw();await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));await renderer.device.queue.onSubmittedWorkDone();});await drain();
}
async function configure(site,lod,layer,mode,intensity){
  await page.evaluate(({site,lod,layer,mode,intensity})=>{
    for(const f of inflight.values())f.controller.abort();inflight.clear();queue=[];
    cx=site.centre[0];cy=site.centre[1];mpp=30*2**lod;
    $('prefetch').checked=false;$('refinementDepth').value='0';$('renderSea').checked=true;$('grid').checked=false;
    hydrology?.set({...TerrainHydrology.defaults(),enabled:false});styleRendering.set({enabled:false,automatic:false,interval:10});
    $('mapMode').value=layer;$('mapMode').onchange();TerrainAmplification.set({mode,intensity,before:false},true);
    $('forcedLod').value=String(lod);forceViewLod();if(forcedView?.lod!==lod)throw Error('Forced LOD selection failed: '+lod);clampCamera();draw();schedule();
  },{site,lod,layer,mode,intensity});
}
async function geometry(){return page.evaluate(()=>{
  const xs=[],ys=[];for(const p of visiblePlan){const b=p.tile.b;for(const x of [b[0],b[2]])xs.push((x-cx)/mpp+W/2);for(const y of [b[1],b[3]])ys.push((y-cy)/mpp+H/2);}
  return {xs,ys,W,H,cx,cy,mpp,tiles:visiblePlan.map(p=>({key:p.tile.key,lod:p.tile.lod,tx:p.tile.tx,ty:p.tile.ty,amplify:p.tile.amplify,mode:p.tile.amplifyMode}))};
});}
async function capture(site,lod,layer,mode,intensity=1,withWarm=true){
  const label=`${site.name}_${layer}_lod${lod}_${mode}_i${intensity}`,file=path.join(OUT,label+'.png');phase=label+'/cold';const start=performance.now(),first=summary.responses.length;
  await configure(site,lod,layer,mode,intensity);await waitReady({lod,mode,layer});const seconds=(performance.now()-start)/1000,geo=await geometry();const state=await page.evaluate(()=>{const s=terrainDebug.snapshot();return {camera:s.camera,refinement:s.refinement,rendered:s.rendered}});
  await page.locator('#gpuMap').screenshot({path:file});const image=py(imageStats,[file,JSON.stringify(geo)]);
  const response=summary.responses.slice(first).filter(r=>r.lod===lod&&(r.intensity>0?r.mode:'off')===mode);
  const entry={site:site.name,lod,layer,mode,intensity,path:path.relative(ROOT,file).replaceAll('\\','/'),completeSeconds:seconds,geometry:geo,state,image,
    cold:{tiles:response.length,cacheHits:response.filter(r=>r.cacheHit).length,headerComputeSeconds:stat(response.map(r=>r.headerComputeSeconds)),executedComputeSeconds:stat(response.map(r=>r.executedComputeSeconds)),requestSeconds:stat(response.map(r=>r.requestSeconds))}};
  summary.views.push(entry);save();console.log(JSON.stringify({capture:label,seconds,tiles:response.length,seamRatio:image.ratio}));
  if(process.env.E2E_GRID_DETECTOR&&layer==='relief'){
    const span=256*30*2**lod,tx=Math.floor(site.centre[0]/span),ty=Math.floor(site.centre[1]/span),record=response.find(r=>r.tx===tx&&r.ty===ty);
    assert(record,'centre tile missing');const url=new URL(record.url);url.searchParams.delete('session');url.searchParams.delete('epoch');const r=await page.request.get(url.toString());assert(r.ok());const bytes=await r.body(),file=path.join(OUT,label+'.bin'),headers=r.headers(),width=Number(headers['x-terrain-width']);fs.writeFileSync(file,bytes);
    (summary.gridRecords??=[]).push({site:site.name,mode,lod,height:file,width,halo:Number(headers['x-terrain-halo']||24),resolution:30*2**lod,tx,ty,floor:Number(headers['x-terrain-relief-floor-lod']||-6),gradientOffset:headers['x-terrain-gradient-format']?bytes.length-width*width*8:null});save();
  }
  if(withWarm){phase=label+'/warm';const index=summary.responses.length,t=performance.now();await page.evaluate(()=>resetBackend({preserveLod:true}));await waitReady({lod,mode,layer});
    const warm=summary.responses.slice(index).filter(r=>r.lod===lod&&(r.intensity>0?r.mode:'off')===mode);entry.warm={completeSeconds:(performance.now()-t)/1000,tiles:warm.length,cacheHits:warm.filter(r=>r.cacheHit).length,headerComputeSeconds:stat(warm.map(r=>r.headerComputeSeconds)),executedComputeSeconds:stat(warm.map(r=>r.executedComputeSeconds)),requestSeconds:stat(warm.map(r=>r.requestSeconds))};save();
  }
}
async function shortcuts(site){
  const lod=-4,layer='render';phase=site.name+'/shortcuts';await configure(site,lod,layer,'off',1);await page.evaluate(()=>resetBackend({preserveLod:true}));await waitReady({lod,mode:'off',layer});
  await page.evaluate(()=>{TerrainAmplification.set({mode:'hybrid',intensity:1,before:true},true);window.__e2eObserve=true;window.__e2eFrames=[];const tick=()=>{if(!window.__e2eObserve)return;__e2eFrames.push({patches:visiblePlan.length,pending:terrainDebug.snapshot().visible.pending});requestAnimationFrame(tick)};requestAnimationFrame(tick);});
  for(const [key,mode] of [['b','hybrid'],['b','off'],['Shift+B','hybrid'],['Shift+B','phasor'],['Shift+B','mse'],['Shift+B','off']]){
    phase=site.name+'/shortcuts/'+key+'->'+mode;console.log(JSON.stringify({shortcut:key,expected:mode,site:site.name}));await page.locator('#viewport').focus();const t=performance.now();await page.keyboard.press(key);await waitReady({lod,mode,layer});
    assert.equal(await page.evaluate(()=>TerrainAmplification.effectiveMode()),mode);summary.shortcuts.push({site:site.name,key,mode,seconds:(performance.now()-t)/1000});save();
  }
  const observation=await page.evaluate(()=>{__e2eObserve=false;return {blankFrames:__e2eFrames.filter(f=>f.patches===0).length,totalFrames:__e2eFrames.length,cacheSets:[...new Set([...tiles.values()].filter(t=>t.lod===-4).map(t=>t.amplify?t.amplifyMode:'off'))]}});
  summary.shortcuts.push({site:site.name,observation});assert.equal(observation.blankFrames,0,'blank frame during A/B');for(const mode of ['off','hybrid','phasor','mse'])assert(observation.cacheSets.includes(mode));save();
}
async function diagnostics(site){
  phase='diagnostics';const context=await page.evaluate(()=>({version:world.version,profile:world.cache_profile,worldProfile:generationProfile(),identity:world.world_identity})),lod=-4,span=256*30*2**lod,tx=Math.floor(site.centre[0]/span),ty=Math.floor(site.centre[1]/span);
  const cases=[['biomes','orogen-biomes'],['koppen','orogen-koppen'],['soil','soil'],['temperature','temperature'],['precipitation','precipitation'],['contours','relief']];
  for(const [name,layer] of cases){
    const paths=[];for(const mode of ['off','hybrid','phasor','mse']){
      const q=new URLSearchParams({profile:context.profile,world_profile:context.worldProfile,world_identity:context.identity,mode:layer});if(mode!=='off'){q.set('amplify','1');q.set('amplify_mode',mode);}if(name==='contours')q.set('contours',JSON.stringify({enabled:true,automatic:false,interval:5,width:1}));
      const url=`${BASE}/tiles/${context.version}/42/${lod}/${tx}/${ty}.png?${q}`,t=performance.now(),r=await page.request.get(url,{timeout:300000});assert(r.ok(),`${name}/${mode}: ${r.status()}`);const body=await r.body(),file=path.join(OUT,'layers',`${name}_${mode}.png`);fs.writeFileSync(file,body);
      const record={phase,url,lod,mode:mode==='off'?'hybrid':mode,intensity:mode==='off'?0:1,binary:false,status:r.status(),headers:r.headers()};checkHeaders(record);
      paths.push(file);summary.diagnostics.push({layer:name,mode,path:path.relative(ROOT,file).replaceAll('\\','/'),seconds:(performance.now()-t)/1000,headers:r.headers(),sha256:crypto.createHash('sha256').update(body).digest('hex')});
    }
    const diff=py(String.raw`import sys,json,numpy as np
from PIL import Image
a=np.asarray(Image.open(sys.argv[1]).convert('RGB'),dtype=np.float32)
result=[]
for file in sys.argv[2:]:
 b=np.asarray(Image.open(file).convert('RGB'),dtype=np.float32);d=np.abs(a-b);result.append(dict(meanAbsoluteRGB=float(d.mean()),changedPixelFraction=float(np.any(d>0,axis=2).mean()),maxRGB=float(d.max())))
print(json.dumps(result))`,paths);
    summary.diagnostics.filter(r=>r.layer===name&&r.mode!=='off').forEach((r,i)=>r.differenceFromOff=diff[i]);save();console.log(JSON.stringify({diagnostic:name,diff}));
  }
  // Same physical climate planes, distinct fine heights; climate inputs are not re-simulated.
  const physical=[];for(const mode of ['off','hybrid','phasor','mse']){const q=new URLSearchParams({profile:context.profile,world_profile:context.worldProfile,climate:'1'});if(mode!=='off'){q.set('amplify','1');q.set('amplify_mode',mode);}const r=await page.request.get(`${BASE}/height/${context.version}/42/${lod}/${tx}/${ty}.bin?${q}`,{timeout:300000});assert(r.ok());const h=r.headers(),b=await r.body(),n=Number(h['x-terrain-width'])**2*4,c=Number(h['x-terrain-climate-width'])*Number(h['x-terrain-climate-height'])*Number(h['x-terrain-climate-layers'])*4;physical.push({mode,heightHash:crypto.createHash('sha256').update(b.subarray(0,n)).digest('hex'),climateHash:crypto.createHash('sha256').update(b.subarray(n,n+c)).digest('hex')});}
  summary.physicalClimateCheck=physical;assert.equal(new Set(physical.map(r=>r.climateHash)).size,1);for(const record of physical.slice(1))assert.notEqual(record.heightHash,physical[0].heightHash,record.mode+' must change fine heights');save();
}
function finishArtifacts(){
  summary.timings=[];for(const mode of modes)for(const lod of lods){const records=summary.responses.filter(r=>r.lod===lod&&(r.intensity>0?r.mode:'off')===mode&&r.url.includes('mode=render')&&(mode==='off'||r.intensity===1)),cold=records.filter(r=>r.phase.endsWith('/cold')&&!r.cacheHit),warm=records.filter(r=>r.phase.endsWith('/warm'));summary.timings.push({mode,lod,coldMissComputeSeconds:stat(cold.map(r=>r.headerComputeSeconds)),warmHeaderReceiptSeconds:stat(warm.map(r=>r.headerComputeSeconds)),warmExecutedComputeSeconds:stat(warm.map(r=>r.executedComputeSeconds)),warmRequestSeconds:stat(warm.map(r=>r.requestSeconds)),warmCacheHits:warm.filter(r=>r.cacheHit).length,warmTiles:warm.length});}
  for(const timing of summary.timings){const views=summary.views.filter(v=>v.layer==='render'&&v.mode===timing.mode&&v.lod===timing.lod&&v.intensity===1);timing.coldCompleteViewSeconds=stat(views.map(v=>v.completeSeconds));timing.warmCompleteViewSeconds=stat(views.filter(v=>v.warm).map(v=>v.warm.completeSeconds));}
  summary.observedVersions=[...new Set(summary.responses.filter(r=>r.intensity>0).map(r=>r.mode+':'+r.headers['x-terrain-amplification-version']))];
  summary.sourceHashesEnd=Object.fromEntries(Object.keys(summary.sourceHashes).map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')]));summary.sourcesChangedDuringRun=Object.keys(summary.sourceHashes).filter(p=>summary.sourceHashes[p]!==summary.sourceHashesEnd[p]);
  summary.seamFlags=summary.views.filter(v=>v.image.ratio>2).map(v=>({path:v.path,ratio:v.image.ratio}));save();
  const made=py(String.raw`import json,sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
s=json.loads(Path(sys.argv[1]).read_text());root=Path(sys.argv[2]);out=Path(sys.argv[1]).parent
try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
except OSError:font=ImageFont.load_default()
made=[];modes=s.get('modes',['off','hybrid','phasor','mse']);columns=' / '.join(modes);ncols=len(modes)
for site in s['sites']:
 for layer in s['layers']:
  rows=s['lods'];canvas=Image.new('RGB',(ncols*640,60+len(rows)*510),'#15212b');d=ImageDraw.Draw(canvas);d.text((12,12),site['name']+' | '+layer+' | columns: '+columns,fill='white',font=font)
  for i,lod in enumerate(rows):
   for j,mode in enumerate(modes):
    r=next((v for v in s['views'] if v['site']==site['name'] and v['layer']==layer and v['lod']==lod and v['mode']==mode and v['intensity']==1),None)
    if r:
     im=Image.open(root/r['path']).convert('RGB');im.thumbnail((640,480));canvas.paste(im,(j*640,90+i*510));d.text((j*640+8,64+i*510),'LOD '+str(lod)+' | '+mode,fill='white',font=font)
  dest=out/(site['name']+'_'+layer+'_contact.png');canvas.save(dest);made.append(str(dest))
 for_intensity=[v for v in s['views'] if v['site']==site['name'] and v['layer']=='render' and v['lod']==-4]
 if any(v['intensity']==.5 for v in for_intensity):
  canvas=Image.new('RGB',(ncols*640,60+3*510),'#15212b');d=ImageDraw.Draw(canvas);d.text((12,12),site['name']+' | Render LOD -4 | '+columns,fill='white',font=font)
  for i,intensity in enumerate((.5,1,2)):
   for j,mode in enumerate(modes):
    r=next(v for v in for_intensity if v['mode']==mode and v['intensity']==(1 if mode=='off' else intensity));im=Image.open(root/r['path']).convert('RGB');canvas.paste(im,(j*640,90+i*510));d.text((j*640+8,64+i*510),mode+' | intensity '+str(intensity),fill='white',font=font)
  dest=out/(site['name']+'_intensity_contact.png');canvas.save(dest);made.append(str(dest))
if s['diagnostics']:
 canvas=Image.new('RGB',(ncols*256,50+6*286),'#15212b');d=ImageDraw.Draw(canvas);d.text((8,8),'LOD -4 | '+columns,fill='white',font=font)
 for i,name in enumerate(('biomes','koppen','soil','temperature','precipitation','contours')):
  for j,mode in enumerate(modes):
   r=next(v for v in s['diagnostics'] if v['layer']==name and v['mode']==mode);canvas.paste(Image.open(root/r['path']).convert('RGB'),(j*256,80+i*286));d.text((j*256+5,52+i*286),name+' | '+mode,fill='white',font=font)
 dest=out/'layers/contact.png';canvas.save(dest);made.append(str(dest))
print(json.dumps(made))`,[path.join(OUT,'summary.json'),ROOT]);
  summary.contactSheets=made.map(p=>path.relative(ROOT,p).replaceAll('\\','/'));save();
  if(summary.gridRecords?.length){const manifest=path.join(OUT,'height-manifest.json');fs.writeFileSync(manifest,JSON.stringify(summary.gridRecords));fs.writeFileSync(path.join(OUT,'grid-detector.log'),execFileSync(pythonExecutable(),['tools/verification/terrain_grid_detector.py',manifest],{cwd:ROOT,encoding:'utf8'}));summary.gridDetector=path.relative(ROOT,path.join(OUT,'grid-detector.json')).replaceAll('\\','/');save();}
}
(async()=>{
 try{
  summary.server=await(await fetch(BASE+'/api/status')).json();const preloadDeadline=Date.now()+240000;while(summary.server.model_preload?.state==='loading'&&Date.now()<preloadDeadline){await new Promise(r=>setTimeout(r,500));summary.server=await(await fetch(BASE+'/api/status')).json();}assert.equal(summary.server.gpu,'NVIDIA GeForce RTX 5090');assert.equal(summary.server.device_selection.visible_devices.length,1,'server must see only GPU 1');
  summary.serverControl=await(await fetch(BASE+'/api/gpu')).json();
  summary.sourceHashes=Object.fromEntries(['backend/terrain_amplification.py','backend/terrain_phasor.py','backend/terrain_hybrid.py','backend/terrain_server.py','web/index.html'].map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')]));
  summary.browserArguments=['--enable-unsafe-webgpu',...JSON.parse(process.env.TERRAIN_BROWSER_ARGS||'[]')];
  browser=await chromium.launch({executablePath:process.env.CHROME_PATH||browserExecutable()||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:summary.browserArguments});
  page=await browser.newPage({viewport:{width:1000,height:720},deviceScaleFactor:1});page.on('pageerror',e=>summary.pageErrors.push({phase,message:e.message,stack:e.stack}));page.on('console',m=>{if(m.type()==='error')summary.consoleErrors.push({phase,message:m.text(),location:m.location()});});page.on('response',r=>{const work=collectResponse(r).catch(e=>summary.protocolErrors.push({phase,message:e.message,url:r.url()})).finally(()=>responseWork.delete(work));responseWork.add(work);});
  if(process.env.E2E_NO_GRADIENTS)await page.route('**/height/**',route=>{const url=new URL(route.request().url());url.searchParams.delete('gradients');return route.continue({url:url.toString()});});
  await page.goto(BASE+'/api/status');const architecture=await page.evaluate(async()=>{const a=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});return a?.info.architecture;});assert.equal(architecture,'blackwell','Pin the browser to GPU 1 RTX 5090 with --use-adapter-luid before requesting terrain');
  await page.addInitScript(()=>{try{localStorage.setItem('neural-earth-preferences-v1',JSON.stringify({gpuAcceleration:true,erosionDefaultsVersion:1,rendering:{gpuRender:true,coarseGpu:false}}));}catch(_){}});
  await page.goto(BASE+'/?seed=42&profile=orogen&generation=null&gpu=1&mode=render&hydro=0&coarse_gpu=0&coarse_prepare=0',{timeout:180000});
  await page.waitForFunction(()=>window.terrainDebug&&terrainDebug.snapshot().world&&terrainDebug.snapshot().rendererReady,null,{timeout:180000});
  await page.evaluate(()=>{setSharedSize([640,480]);$('prefetch').checked=false;$('refinementDepth').value='0';hydrology?.set({...TerrainHydrology.defaults(),enabled:false});});
  await page.addStyleTag({content:'#hud,#mini,#legend,#annotations,#cursorAltitude{visibility:hidden!important}'});
  summary.world=await page.evaluate(()=>({seed:world.seed,profile:generationProfile(),cacheProfile:world.cache_profile,identity:world.world_identity,amplification:world.amplification}));
  for(const mode of modes.filter(m=>m!=='off'))assert(summary.world.amplification?.modes?.[mode]?.available,mode+' unavailable: '+(summary.world.amplification?.modes?.[mode]?.reason||'no capability'));
  summary.browserGPU=await page.evaluate(async()=>{const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'}),i=adapter.info;return {vendor:i.vendor,architecture:i.architecture,device:i.device,description:i.description,isFallbackAdapter:adapter.isFallbackAdapter};});save();
  for(const site of sites){for(const layer of layers)for(const lod of lods)for(const mode of modes)await capture(site,lod,layer,mode);
    if(!process.env.E2E_ONLY_MAIN){for(const mode of ['hybrid','phasor','mse'])for(const intensity of [.5,2])await capture(site,-4,'render',mode,intensity,false);await shortcuts(site);}}
  if(!process.env.E2E_ONLY_MAIN)await diagnostics(sites.find(s=>s.name==='coastal-mountains')||sites[0]);
  summary.finished=new Date().toISOString();summary.unexpectedConsoleErrors=summary.consoleErrors.filter(e=>!e.location?.url?.endsWith('/favicon.ico')&&!/status of 409/.test(e.message));summary.passed=summary.pageErrors.length===0&&summary.tileErrors.length===0&&summary.protocolErrors.length===0&&summary.unexpectedConsoleErrors.length===0;finishArtifacts();
  assert.equal(summary.unexpectedConsoleErrors.length,0,'console errors');assert.equal(summary.pageErrors.length,0,'page errors');assert.equal(summary.tileErrors.length,0,'tile errors');assert.equal(summary.protocolErrors.length,0,'protocol errors');console.log(JSON.stringify({passed:summary.passed,views:summary.views.length,tileResponses:summary.responses.length,seamFlags:summary.seamFlags,contactSheets:summary.contactSheets},null,2));
 }catch(e){summary.failure={message:e.message,stack:e.stack,phase};if(page){summary.failure.snapshot=await page.evaluate(()=>window.terrainDebug?.snapshot()).catch(()=>null);await page.screenshot({path:path.join(OUT,'failure.png')}).catch(()=>{});}save();console.error(e);process.exitCode=1;
 }finally{await browser?.close();}
})();
