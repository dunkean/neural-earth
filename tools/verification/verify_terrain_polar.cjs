const {loadPlaywright, browserExecutable} = require('../platform.cjs');
const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
// Real browser rendering and subscriptions, with distinct chart fixtures.
const {chromium}=loadPlaywright();
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:browserExecutable(),headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1000,height:720}}),errors=[],views=[],requests=[];
  await page.addInitScript(()=>Object.defineProperty(navigator,'gpu',{value:undefined}));
  page.on('pageerror',e=>errors.push(e.message));
  const png=await page.evaluate(()=>['#ff0000','#00ff00','#0000ff'].map(color=>{
   const c=document.createElement('canvas');c.width=c.height=256;const g=c.getContext('2d');g.fillStyle=color;g.fillRect(0,0,256,256);return c.toDataURL().split(',')[1];
  }));
  await page.route('**/api/world?**',r=>r.fulfill({json:{seed:'42',world_profile:'natural',generation_profile:'natural',gpu:'fixture',version:'test-v1',cache_profile:'test',world_topology:'sphere',world_diameter_km:12732.4,world_bounds:[-20e6,-10e6,20e6,10e6],overview_bounds:[-20e6,-10e6,20e6,10e6],overview:'/overview/test.png'}}));
  await page.route('**/api/view',r=>{const data=r.request().postDataJSON();views.push(data);return r.fulfill({json:{accepted:true,epoch:data.epoch}})});
  await page.route('**/api/status',r=>r.fulfill({json:{scheduler:{queued:0,computing:0,encoding:0}}}));
  await page.route('**/api/view/release',r=>r.fulfill({json:{released:true}}));
  await page.route('**/overview/test.png?**',r=>r.fulfill({contentType:'image/png',body:Buffer.from(png[2],'base64')}));
  await page.route('**/tiles/test-v1/**',r=>{
   const url=new URL(r.request().url()),polar=url.searchParams.get('neural_chart')==='polar';requests.push(url.href);
   return r.fulfill({contentType:'image/png',body:Buffer.from(png[polar?1:0],'base64'),headers:{'X-Terrain-Stage':'decoder','X-Terrain-Elevation-Min':'100','X-Terrain-Elevation-Max':'1000'}});
  });
  await page.goto('http://127.0.0.1:8765/?seed=42&profile=natural&view=globe&mode=relief&coarse_gpu=0&lighting='+encodeURIComponent(JSON.stringify({global:{strength:0}})));
  await page.waitForFunction(()=>window.terrainDebug?.snapshot().globe?.detailReady);
  await page.evaluate(()=>{globe.orbit(.4,Math.PI);globe.zoom(2000)});
  await page.waitForFunction(()=>globeCamera?.neuralChart==='polar'&&window.terrainDebug.snapshot().visible.pending===0&&visiblePlan.length>0&&visiblePlan.every(p=>p.tile.neuralChart==='polar'));
  const north=await page.evaluate(()=>({lod:requestedLod(),wanted:wanted.size,bounds:globeCamera.bounds}));
  assert(north.lod<=2&&north.wanted<100,'pole keeps a local fine LOD');
  const pixel=async()=>page.evaluate(()=>{globe.draw(W,H);const c=$('globeMap'),g=c.getContext('webgl'),p=new Uint8Array(4);g.readPixels(c.width/2,c.height/2,1,1,g.RGBA,g.UNSIGNED_BYTE,p);return [...p]});
  const northPixel=await pixel();assert(northPixel[1]>200&&northPixel[0]<20&&northPixel[2]<60,'north pole displays rotated chart detail');
  await page.evaluate(()=>globe.orbit(0,-2*Math.PI*2.1/globe.snapshot().altitude));
  await page.waitForFunction(()=>globeCamera?.neuralChart==='polar'&&window.terrainDebug.snapshot().visible.pending===0&&globeCamera.cx>0&&visiblePlan.length>0);
  const southPixel=await pixel();assert(southPixel[1]>200&&southPixel[0]<20&&southPixel[2]<60,'south pole displays rotated chart detail');
  assert(views.some(v=>v.neural_chart==='polar'&&v.tiles.some(t=>t.lod<=2)));
  assert(requests.some(u=>u.includes('neural_chart=polar')));
  await page.locator('#mapView').click();
  await page.waitForFunction(()=>window.terrainDebug.snapshot().view==='map'&&window.terrainDebug.snapshot().visible.pending===0&&visiblePlan.length>0&&visiblePlan.every(p=>!p.tile.neuralChart));
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({result:'OK',north,polarRequests:requests.filter(u=>u.includes('neural_chart=polar')).length,mapCacheIsolated:true}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
