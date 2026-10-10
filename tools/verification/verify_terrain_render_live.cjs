const {loadPlaywright, browserExecutable} = require('../platform.cjs');
const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
// Smoke the real server, map / globe, PNG fallback and persisted material UI.
const {chromium}=loadPlaywright();
const assert=require('node:assert/strict'),path=require('node:path');
(async()=>{const browser=await chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});try{
  const page=await browser.newPage({viewport:{width:1280,height:850}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  page.setDefaultTimeout(180000);
  await page.goto('http://127.0.0.1:8765/?seed=42&profile=orogen&mode=render');
  await page.waitForFunction(()=>window.terrainDebug?.snapshot().world==='42',{timeout:180000});
  await page.waitForFunction(()=>terrainDebug.snapshot().renderer?.status==='webgpu'&&terrainDebug.snapshot().rendered.patches>0,{timeout:120000});
  await page.waitForFunction(()=>terrainDebug.snapshot().overview);
  await page.screenshot({path:path.join(NEURAL_EARTH_ROOT,'output/render-world.png')});
  const before=await page.evaluate(()=>terrainDebug.snapshot());assert.equal(before.mode,'render');assert.equal(before.backend,'webgpu');
  assert.equal(await page.locator('button[data-mode="render"]').count(),1);assert.equal(await page.locator('button[data-mode="soil"]').count(),1);
  await page.evaluate(()=>{document.getElementById('mapMode').value='soil';document.getElementById('mapMode').dispatchEvent(new Event('change'));});
  await page.waitForFunction(()=>terrainDebug.snapshot().mode==='soil'&&terrainDebug.snapshot().rendered.patches>0);
  await page.waitForFunction(()=>terrainDebug.snapshot().overview);
  await page.screenshot({path:path.join(NEURAL_EARTH_ROOT,'output/soil-world.png')});
  await page.evaluate(()=>{document.getElementById('mapMode').value='render';document.getElementById('mapMode').dispatchEvent(new Event('change'));});
  const region=await page.evaluate(()=>{
    for(const t of tiles.values())if(t.heights){const o=t.heightOptions,r=o.metresPerSample;for(let i=0;i<t.heights.length;i++){const raw=t.heights[i],h=o.encoding==='signed-sqrt'?Math.sign(raw)*raw*raw:raw;if(h>1500&&h<2500){const x=o.origin[0]+(i%o.width+.5)*r,y=o.origin[1]+(Math.floor(i/o.width)+.5)*r;return{x,y};}}}return null;
  });
  if(region){
    await page.evaluate(({x,y})=>{fitBounds([x-120000,y-75000,x+120000,y+75000]);},region);
    await page.waitForFunction(()=>{const d=terrainDebug.snapshot();return d.visible.wanted>0&&d.visible.lod===d.camera.lod&&d.rendered.lods.includes(d.camera.lod)&&d.visible.pending===0;},{timeout:120000});
    await page.screenshot({path:path.join(NEURAL_EARTH_ROOT,'output/render-region.png')});
  }
  await page.evaluate(()=>{const input=document.getElementById('render_snow_color');input.value='#ee1122';input.dispatchEvent(new Event('input'));});
  assert.equal(await page.evaluate(()=>TerrainRender.get().snow_color[0]),238/255);
  await page.evaluate(()=>document.getElementById('renderMaterialReset').click());assert.deepEqual(await page.evaluate(()=>TerrainRender.get().snow_color),[.94,.96,.97]);
  await page.locator('#globeView').click();await page.waitForFunction(()=>terrainDebug.snapshot().globe?.ready&&terrainDebug.snapshot().globe?.detailReady,{timeout:60000});
  await page.screenshot({path:path.join(NEURAL_EARTH_ROOT,'output/render-globe.png')});
  await page.locator('#mapView').click();
  // One small physical PNG tile verifies the independent renderer, not the
  // user's persisted GPU checkbox or an expensive worldwide NN regeneration.
  const png=await page.evaluate(async()=>{const d=terrainDebug.snapshot();const u=new URL('/api/world',location.origin);u.searchParams.set('seed',d.world);u.searchParams.set('world_profile',d.generationProfile);const w=await(await fetch(u)).json();const r=await fetch(`/tiles/natural-v1/${w.seed}/9/0/0.png?profile=${w.cache_profile}&world_profile=${w.generation_profile}&mode=render&render_settings=${encodeURIComponent(TerrainRender.serialize())}`);return{status:r.status,type:r.headers.get('content-type'),bytes:(await r.arrayBuffer()).byteLength};});
  assert.equal(png.status,200);assert(png.bytes>1000);assert.deepEqual(errors,[]);
  console.log(JSON.stringify({ok:true,backend:before.backend,worldPatches:before.rendered.patches,region,png,screenshots:['render-world','soil-world','render-region','render-globe']}));
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
