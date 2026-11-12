// Full viewer against the running application: no routes, fixtures or mocked
// generation. Captures the default view, native terrain, and fine A/B relief.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const {loadPlaywright,browserExecutable,pythonExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..'),out=path.resolve(root,process.env.MAGNA_LIVE_OUTPUT||'output/magna-live');
const base=process.env.BASE_URL||'http://127.0.0.1:8765',seed=process.env.MAGNA_LIVE_SEED||'17470623790244073500';
fs.mkdirSync(out,{recursive:true});
const report={base,seed,started:new Date().toISOString(),frames:[],responses:[],errors:[],consoleErrors:[]};
const save=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));
let browser,page,phase='startup';
const summary=()=>page.evaluate(()=>({world:world?.seed,loading:worldLoading,generation:generationPending,error,
 backend:terrainDebug?.snapshot().backend,ready:rendererReady,mode:viewMode,overview:!!overview,overviewPending:overviewWanted,
 visible:visibleCounts(),rendered:terrainDebug?.snapshot().rendered,inflight:inflight.size,queued:queue.length,
 r3:renderer?.getStats().render3d,amplification:window.TerrainAmplification.get(),cx,cy,mpp,lod:displayLod()}));
async function waitFor(label,predicate,timeout=120000){
 const began=Date.now();let logged=0;
 while(Date.now()-began<timeout){
  if(await page.evaluate(predicate))return (Date.now()-began)/1000;
  await page.waitForTimeout(1000);
  if(Date.now()-logged>10000){console.log(JSON.stringify({waiting:label,seconds:Math.round((Date.now()-began)/1000),state:await summary()}));logged=Date.now();}
 }
 throw Error('Timed out: '+label);
}
async function capture(name){
 phase=name;
 await page.evaluate(async()=>{drawFrame();if(renderer?.available)await renderer.device.queue.onSubmittedWorkDone();});
 const file=path.join(out,name+'.png');await page.locator('#view').screenshot({path:file});
 const stats=JSON.parse(execFileSync(pythonExecutable(),['-c',String.raw`
import sys,json
import numpy as np
from PIL import Image
a=np.asarray(Image.open(sys.argv[1]).convert('RGB'))
crop=a[40:-80,60:-220] if a.shape[1]>500 else a
print(json.dumps(dict(width=a.shape[1],height=a.shape[0],std=float(crop.astype(float).std(axis=(0,1)).mean()),colors=int(len(np.unique(crop.reshape(-1,3),axis=0))))))
`,file],{encoding:'utf8'}));
 const frame={name,file:path.relative(root,file),stats,state:await summary()};report.frames.push(frame);save();
 assert(stats.colors>200&&stats.std>4,'No visible terrain in '+name+': '+JSON.stringify(stats));
 console.log(JSON.stringify({capture:name,stats,state:frame.state}));
}
async function landPoint(){return page.evaluate(()=>{
 let best=null;
 for(const tile of tiles.values()){
  if(!tile.heights||!tile.heightOptions||!isCurrentTile(tile))continue;
  const o=tile.heightOptions,n=o.width||304,halo=o.halo||24,r=o.metresPerSample,origin=o.origin;
  for(let y=halo+4;y<n-halo-4;y+=4)for(let x=halo+4;x<n-halo-4;x+=4){
   let h=tile.heights[y*n+x];if(o.encoding==='signed-sqrt')h=Math.sign(h)*h*h;
   const wx=origin[0]+(x+.5)*r,wy=origin[1]+(y+.5)*r;
   if(h<200||h>2200||Math.abs(wy)>worldBounds()[3]*.7)continue;
   const score=Math.abs(h-900)+Math.abs(wx)*.000001;
   if(!best||score<best.score)best={x:wx,y:wy,height:h,score};
  }
 }
 if(!best)throw Error('No physical land sample available');return best;
});}
async function setCamera(point,lod){
 await page.evaluate(({point,lod})=>{
  cx=point.x;cy=point.y;mpp=30*2**lod;$('prefetch').checked=false;$('refinementDepth').value='0';
  $('renderSea').checked=true;$('forcedLod').value=String(lod);forceViewLod();clampCamera();draw();schedule();
 },{point,lod});
 await waitFor('complete LOD '+lod,()=>world&&!worldLoading&&!generationPending&&rendererReady&&
  publishedEpoch===cameraEpoch&&visibleCounts().wanted>0&&visibleCounts().pending===0&&inflight.size===0&&
  visiblePlan.length>0&&visiblePlan.every(p=>p.tile.lod===forcedView?.lod&&isCurrentTile(p.tile)));
}
(async()=>{
 browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
 page=await browser.newPage({viewport:{width:960,height:720}});
 page.on('pageerror',e=>{report.errors.push({phase,message:e.message,stack:e.stack});save();});
 page.on('console',m=>{if(m.type()==='error'){report.consoleErrors.push({phase,message:m.text(),location:m.location()});save();}});
 page.on('response',async response=>{
  if(!/\/(height|coarse)\//.test(response.url()))return;
  const item={phase,url:response.url(),status:response.status(),headers:await response.allHeaders()};
  report.responses.push(item);save();
 });
 const began=Date.now();
 await page.goto(base+'/?seed='+seed+'&profile=orogen&gpu=1&amplify_mode=magna&amplify=1');
 await waitFor('first visible map',()=>window.terrainDebug&&world&&!worldLoading&&(!!overview||visiblePlan.length>0));
 report.firstVisibleSeconds=(Date.now()-began)/1000;await capture('01-default-satellite');
 assert.equal(await page.evaluate(()=>viewMode),'render-v2','Satellite must be the actual default');
 await waitFor('WebGPU terrain tiles',()=>rendererReady&&renderer?.available&&visiblePlan.length>0);
 report.firstTileSeconds=(Date.now()-began)/1000;
 const point=await landPoint();report.macroLand=point;save();
 phase='native';await setCamera(point,0);await capture('02-native-30m');
 const fine=await landPoint();report.nativeLand=fine;save();
 phase='magna';await setCamera(fine,-4);await capture('03-magna-1m875');
 report.magnaHeights=await page.evaluate(()=>{
  const t=visiblePlan.find(p=>isCurrentTile(p.tile)).tile;
  return {key:t.key,lod:t.lod,tx:t.tx,ty:t.ty,amplification:t.amplificationMode,min:Math.min(...t.heights),max:Math.max(...t.heights)};
 });
 assert.equal(report.magnaHeights.amplification,'magna');
 phase='classic';await page.locator('#view').click({position:{x:100,y:100}});await page.keyboard.press('b');
 assert.equal(await page.evaluate(()=>TerrainAmplification.effectiveMode()),'off');
 await waitFor('classic before view',()=>visibleCounts().pending===0&&inflight.size===0&&visiblePlan.length>0&&visiblePlan.every(p=>p.tile.lod===-4&&isCurrentTile(p.tile)));
 await capture('04-classic-before');
 phase='magna-after';await page.keyboard.press('b');
 assert.equal(await page.evaluate(()=>TerrainAmplification.effectiveMode()),'magna');
 await waitFor('Magna after view',()=>visibleCounts().pending===0&&inflight.size===0&&visiblePlan.length>0&&visiblePlan.every(p=>p.tile.lod===-4&&isCurrentTile(p.tile)));
 await capture('05-magna-after');
 phase='pan';await page.evaluate(()=>{cx+=mpp*120;cy+=mpp*60;draw();schedule();});
 await waitFor('pan',()=>visibleCounts().pending===0&&inflight.size===0&&visiblePlan.length>0&&visiblePlan.every(p=>isCurrentTile(p.tile)));
 await capture('06-pan');
 phase='globe';await page.locator('#globeView').click();
 await waitFor('globe',()=>sceneView==='globe'&&globe?.snapshot().ready);
 await capture('07-globe');
 assert.deepEqual(report.errors,[],'Uncaught browser errors');
 const failed=report.responses.filter(r=>r.status>=400&&r.status!==409);
 assert.deepEqual(failed,[],'Terrain requests must succeed');
 report.passed=true;report.finished=new Date().toISOString();save();
 console.log(JSON.stringify({passed:true,frames:report.frames.length,firstVisibleSeconds:report.firstVisibleSeconds,firstTileSeconds:report.firstTileSeconds,terrainResponses:report.responses.length}));
})().catch(async error=>{
 report.passed=false;report.failure={phase,message:error.message,stack:error.stack};
 if(page){report.failure.state=await summary().catch(()=>null);await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});}
 save();console.error(error);process.exitCode=1;
}).finally(async()=>{await browser?.close();});
