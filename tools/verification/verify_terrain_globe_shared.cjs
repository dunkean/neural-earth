const {loadPlaywright, browserExecutable} = require('../platform.cjs');
const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
// Shared physical tiles and shaded textures across map/WebGPU and globe/WebGL.
const {chromium}=loadPlaywright();
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
 try{
  const page=await browser.newPage({viewport:{width:640,height:540}}),errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));
  const overview=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=c.height=64;const g=c.getContext('2d');g.fillStyle='#0044ff';g.fillRect(0,0,64,64);return c.toDataURL().split(',')[1]});
  await page.route('**/api/world?**',route=>route.fulfill({json:{seed:'42',world_profile:'natural',generation_profile:'natural',gpu:'fixture',version:'test-v1',cache_profile:'test',world_topology:'sphere',world_diameter_km:1273.24,world_bounds:[-2e6,-1e6,2e6,1e6],overview_bounds:[-2e6,-1e6,2e6,1e6],overview:'/overview/test.png'}}));
  await page.route('**/api/view',route=>route.fulfill({json:{accepted:true,epoch:route.request().postDataJSON().epoch}}));
  await page.route('**/api/status',route=>route.fulfill({json:{scheduler:{queued:0,computing:0,encoding:0}}}));
  await page.route('**/overview/test.png?**',route=>route.fulfill({contentType:'image/png',body:Buffer.from(overview,'base64')}));
  for(const endpoint of ['coarse','height'])await page.route(`**/${endpoint}/test-v1/**`,route=>{
   requests.push(route.request().url());const native=endpoint==='coarse',width=native?160:304,climateWidth=native?41:33;
   const address=new URL(route.request().url()).pathname.split('/'),tx=Number(address.at(-2)),ty=parseInt(address.at(-1));
   const elevation=native?1000:tx<0?(ty<0?100:-1000):(ty<0?3000:-4000);
   const values=new Float32Array(width*width+5*climateWidth*climateWidth);values.fill(native?Math.sqrt(elevation):elevation,0,width*width);
   values.fill(20,width*width,width*width+climateWidth*climateWidth);values.fill(1000,width*width+2*climateWidth*climateWidth,width*width+3*climateWidth*climateWidth);
   return route.fulfill({contentType:'application/octet-stream',body:Buffer.from(values.buffer),headers:{'X-Terrain-Width':String(width),'X-Terrain-Halo':native?'16':'24','X-Terrain-Climate-Width':String(climateWidth),'X-Terrain-Climate-Height':String(climateWidth),'X-Terrain-Encoding':native?'signed-sqrt':'float32','X-Terrain-Stage':native?'coarse':'decoder','X-Terrain-Resolution':native?'7680':'30','X-Terrain-Elevation-Min':'1000','X-Terrain-Elevation-Max':'1000'}});
  });
  await page.goto('http://127.0.0.1:8765/?seed=42&profile=natural&view=globe&lighting='+encodeURIComponent(JSON.stringify({global:{strength:0}})));
  await page.waitForFunction(()=>window.terrainDebug?.snapshot().rendererReady);
  assert(await page.evaluate(()=>renderer?.available),'WebGPU adapter is required for shared-render QA');
  await page.waitForFunction(()=>window.terrainDebug.snapshot().globe.detailReady);
  await page.evaluate(()=>{$('renderSea').checked=true;globe.zoom(mpp/45)});
  await page.waitForFunction(()=>{const s=window.terrainDebug.snapshot();return s.visible.pending===0&&s.rendered.sources.includes('decoder')},{},{timeout:60000});
  const globePixels=await page.locator('#globeMap').evaluate(c=>{globe.draw(c.clientWidth,c.clientHeight);const gl=c.getContext('webgl');return [[.3,.3],[.7,.3],[.3,.7],[.7,.7]].map(([x,y])=>{const p=new Uint8Array(4);gl.readPixels(Math.floor(c.width*x),c.height-1-Math.floor(c.height*y),1,1,gl.RGBA,gl.UNSIGNED_BYTE,p);return [...p]})});
  assert(new Set(globePixels.map(p=>p.join(','))).size===4,'four distinct landmarks are visible');
  const before=requests.length;
  await page.evaluate(()=>{window.testSharedTiles=new Map(tiles);window.testSharedTextures=new Map(renderer.tiles);window.testSharedUploads=renderer.uploads;mapCamera=null});
  await page.locator('#mapView').click();
  await page.waitForFunction(()=>{const s=window.terrainDebug.snapshot();return s.view==='map'&&s.visible.pending===0&&s.rendered.patches>0});
  assert(await page.evaluate(()=>[...testSharedTiles].every(([k,t])=>tiles.get(k)===t)),'map retains globe height tiles');
  assert(await page.evaluate(()=>[...testSharedTextures].every(([k,t])=>renderer.tiles.get(k)===t)),'map retains shaded GPU textures');
  const mapPixels=await page.evaluate(()=>{drawFrame();const c=document.createElement('canvas');c.width=$('gpuMap').width;c.height=$('gpuMap').height;const g=c.getContext('2d');g.drawImage($('gpuMap'),0,0);return [[.3,.3],[.7,.3],[.3,.7],[.7,.7]].map(([x,y])=>[...g.getImageData(Math.floor(c.width*x),Math.floor(c.height*y),1,1).data])});
  for(let i=0;i<4;i++)for(let channel=0;channel<4;channel++)assert(Math.abs(globePixels[i][channel]-mapPixels[i][channel])<=3,`same north/south and east/west orientation: ${JSON.stringify({globePixels,mapPixels})}`);
  await page.locator('#globeView').click();
  await page.waitForFunction(()=>window.terrainDebug.snapshot().view==='globe'&&window.terrainDebug.snapshot().visible.pending===0);
  await page.waitForTimeout(150);
  assert.equal(requests.length,before,'same region and LOD need no new transfers');
  assert(await page.evaluate(()=>renderer.uploads===testSharedUploads),'switching views does not shade or upload tiles again');
  assert.equal(await page.locator('#globeMap').evaluate(c=>c.getContext('webgl').getError()),0);
  assert.deepEqual(errors,[]);console.log(JSON.stringify({result:'OK',sharedRequests:before,globePixels,mapPixels,noNewTransfers:true,noNewUploads:true}));
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
