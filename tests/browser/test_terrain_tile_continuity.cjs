const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {readRepositoryFile} = require(NEURAL_EARTH_ROOT + '/tools/repository-files.cjs');
// Real client/coverage/renderer, held network packets during coarse promotion.
const assert=require('node:assert/strict'),fs=require('node:fs');
const {chromium}=require('E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');
const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM0LJrHwMDAxAAGAA7JAUW48M0QAAAAAElFTkSuQmCC','base64');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',args:['--enable-unsafe-webgpu']});
 try{
  const page=await browser.newPage({viewport:{width:1000,height:720}}),errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));
  let holdCoarse=true,releaseCoarse=[],holdOverview=true,releaseOverview=[],preparations=[];
  await page.route('https://continuity.test/**',async route=>{
   const u=new URL(route.request().url());
   if(u.pathname==='/')return route.fulfill({contentType:'text/html',body:readRepositoryFile('index.html','utf8')});
   if(u.pathname==='/terrain_styles.js')return route.fulfill({contentType:'application/javascript',body:'window.TerrainStyles='+readRepositoryFile('terrain_styles.json','utf8')+';\n'+readRepositoryFile('terrain_style_rendering.js','utf8')});
   if(u.pathname.endsWith('.js'))return route.fulfill({contentType:'application/javascript',body:readRepositoryFile(u.pathname.slice(1),'utf8')});
   if(u.pathname==='/api/inference/streams')return route.fulfill({json:{coarse:4,options:[1,2,4,8,16]}});
   if(u.pathname==='/api/world')return route.fulfill({json:{seed:'42',version:'natural-v1',world_profile:'natural',generation_profile:'natural',cache_profile:'test',world_identity:'test',world_bounds:[-20e6,-10e6,20e6,10e6],overview_bounds:[-20e6,-10e6,20e6,10e6],overview:'/overview.png',gpu:'GPU fixture'}});
   if(u.pathname==='/api/view')return route.fulfill({json:{accepted:true}});
   if(u.pathname==='/api/status')return route.fulfill({json:{scheduler:{queued:0,computing:0,encoding:0}}});
   if(u.pathname==='/api/coarse/prepare'){
    preparations.push(await page.evaluate(()=>({overview:!!overview,presented:presentedOverview===overview})));
    return route.fulfill({json:{state:'running'}});
   }
   if(u.pathname==='/overview.png'){
    if(holdOverview)await new Promise(resolve=>releaseOverview.push(resolve));
    return route.fulfill({contentType:'image/png',body:png}).catch(()=>{});
   }
   if(u.pathname.startsWith('/height/')||u.pathname.startsWith('/coarse/')){
    const native=u.pathname.startsWith('/coarse/'),lod=native?7:Number(u.pathname.split('/')[4]);
    requests.push({native,lod});
    if(native&&holdCoarse)await new Promise(resolve=>releaseCoarse.push(resolve));
    const width=native?160:304,halo=native?16:24,cw=native?41:33;
    const values=new Float32Array(width*width+5*cw*cw);values.fill(native?25:625,0,width*width);values.fill(20,width*width);
    return route.fulfill({contentType:'application/octet-stream',body:Buffer.from(values.buffer),headers:{
     'X-Terrain-Width':String(width),'X-Terrain-Halo':String(halo),'X-Terrain-Encoding':native?'signed-sqrt':'metres',
     'X-Terrain-Resolution':String(native?7680:30*2**lod),'X-Terrain-Climate-Width':String(cw),'X-Terrain-Climate-Height':String(cw),
     'X-Terrain-Climate-Layers':'5','X-Terrain-Stage':'coarse','X-Terrain-Elevation-Min':'625','X-Terrain-Elevation-Max':'625'
    }}).catch(()=>{});
   }
   return route.fulfill({contentType:'image/png',body:png});
  });
  await page.goto('https://continuity.test/?seed=42&profile=natural&mode=relief&prepare_world=1');
  await page.waitForFunction(()=>window.terrainDebug?.snapshot().world&&terrainDebug.snapshot().rendererReady);
  while(!releaseOverview.length)await page.waitForTimeout(20);
  await page.waitForTimeout(150);
  assert.equal(requests.length,0,'Layer overview loads before tile/NN work');
  assert.equal(preparations.length,0,'Background coarse waits for displayed layer');
  holdOverview=false;for(const release of releaseOverview.splice(0))release();
  // Macro view starts with complete parent geometry, rather than hundreds of
  // tiny native-source blocks. Hold those blocks so a regression cannot hide.
  await page.waitForFunction(()=>terrainDebug.snapshot().rendered.lods.some(l=>l>=9));
  assert(requests.every(r=>!r.native),'Macro startup must use parent LOD tiles');
  await page.waitForFunction(()=>terrainDebug.snapshot().overview);
  while(!preparations.length)await page.waitForTimeout(20);
  assert(preparations.every(p=>p.overview&&p.presented),'Coarse starts only after the layer was drawn');
  const baseline=await page.evaluate(()=>({signature:overviewSignature(),interval:styleRendering.get().interval}));
  await page.evaluate(()=>{mpp=960;cx=0;cy=0;draw();refresh()});
  await page.waitForFunction(()=>terrainDebug.snapshot().scheduler.inflight>0);
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  const waiting=await page.evaluate(()=>({snapshot:terrainDebug.snapshot(),signature:overviewSignature(),interval:styleRendering.get().interval}));
  assert.notEqual(waiting.interval,baseline.interval);
  assert.equal(waiting.signature,baseline.signature,'Disabled contour spacing must not invalidate overview');
  assert(waiting.snapshot.overview,'World relief remains visible while coarse loads');
  assert(waiting.snapshot.rendered.lods.some(l=>l>=9),'Ready parent stays on screen while coarse loads');
  assert(waiting.snapshot.visible.pending>0);
  // Release a single coarse block; the remaining area still uses its parent.
  await page.waitForFunction(()=>terrainDebug.snapshot().scheduler.inflight>0);
  while(!releaseCoarse.length)await page.waitForTimeout(20);
  releaseCoarse.shift()();
  await page.waitForFunction(()=>terrainDebug.snapshot().rendered.lods.includes(7));
  assert((await page.evaluate(()=>terrainDebug.snapshot())).rendered.lods.some(l=>l>=9));
  // A new enabled-contour overview must preserve the old overview until ready.
  holdOverview=true;
  await page.evaluate(()=>{styleRendering.set({enabled:true});restoreOverview();draw();schedule()});
  await page.waitForFunction(()=>!!overviewFlight);
  assert(await page.evaluate(()=>!!overview&&overviewWanted),'Overview refresh keeps a visible same-layer fallback');
  holdOverview=false;for(const release of releaseOverview.splice(0))release();
  await page.waitForFunction(()=>!overviewWanted);
  // Zoom out before the remaining coarse requests finish; parent is reusable.
  await page.evaluate(()=>{fit();refresh()});
  await page.waitForFunction(()=>terrainDebug.snapshot().rendered.lods.some(l=>l>=9));
  assert((await page.evaluate(()=>terrainDebug.snapshot())).overview);
  holdCoarse=false;for(const release of releaseCoarse)release();
  assert.deepEqual(errors,[]);
  console.log('Macro relief, parent LOD retention, partial coarse replacement, overview refresh and zoom-out continuity: OK');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
