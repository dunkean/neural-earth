const {loadPlaywright, browserExecutable} = require('../platform.cjs');
const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
// Real source provinces on map/globe and southern seasonal material review.
const {chromium}=loadPlaywright();
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
  const out=path.join(NEURAL_EARTH_ROOT,'output/render-review');fs.mkdirSync(out,{recursive:true});
  const browser=await chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
  try{
    const page=await browser.newPage({viewport:{width:1280,height:850}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));page.setDefaultTimeout(180000);
    await page.goto('http://127.0.0.1:8765/?seed=42&profile=orogen&mode=pedology&prefetch=0');
    await page.waitForFunction(()=>terrainDebug?.snapshot().world==='42'&&terrainDebug.snapshot().overview&&terrainDebug.snapshot().rendered.patches>0);
    await page.waitForFunction(()=>terrainDebug.snapshot().visible.pending===0);
    assert.equal(await page.evaluate(()=>terrainDebug.snapshot().mode),'pedology');
    const planes=await page.evaluate(()=>[...tiles.values()].filter(t=>isCurrentTile(t)&&t.heights).map(t=>t.heightOptions.climate?.length/(t.heightOptions.climateWidth*t.heightOptions.climateHeight)));
    assert(planes.length>0&&planes.every(n=>n===50),'GPU Pedology receives source compositions, not bare diagnostic heights');
    assert(await page.locator('#pedologyLegend').isVisible());
    await page.screenshot({path:path.join(out,'pedology-world.png')});
    const source=await page.evaluate(async()=>{
      const d=terrainDebug.snapshot();const w=await(await fetch(`/api/world?seed=42&world_profile=${d.generationProfile}`)).json();
      const png=await fetch(`/tiles/natural-v1/42/9/0/0.png?profile=${w.cache_profile}&world_profile=${w.generation_profile}&mode=pedology`);
      return {status:png.status,bytes:(await png.arrayBuffer()).byteLength,stage:png.headers.get('X-Terrain-Stage')};
    });
    assert.equal(source.status,200);assert.equal(source.stage,'orogen-diagnostic');assert(source.bytes>1000);
    const southern=await page.evaluate(()=>{
      for(const t of tiles.values())if(t.heights){const o=t.heightOptions,r=o.metresPerSample;
        for(let i=0;i<t.heights.length;i++){
          const v=t.heights[i],h=o.encoding==='signed-sqrt'?Math.sign(v)*v*v:v;
          const x=o.origin[0]+(i%o.width+.5)*r,y=o.origin[1]+(Math.floor(i/o.width)+.5)*r;
          if(Math.abs(x)<14e6&&y>4.5e6&&y<8e6&&h>400&&h<1700)return{x,y,h};
        }
      }return null;
    });
    await page.locator('#globeView').click();
    await page.waitForFunction(()=>terrainDebug.snapshot().globe?.ready&&terrainDebug.snapshot().globe?.detailReady,{timeout:90000});
    await page.screenshot({path:path.join(out,'pedology-globe.png')});
    await page.locator('#mapView').click();
    assert(southern,'Southern land with medium mountains is present');
    await page.evaluate(({x,y})=>{
      const prefetch=document.getElementById('prefetch');prefetch.checked=false;
      const mode=document.getElementById('mapMode');mode.value='render';mode.dispatchEvent(new Event('change'));
      fitBounds([x-240000,y-150000,x+240000,y+150000]);
      const input=document.getElementById('render_season');input.value='0';input.dispatchEvent(new Event('input'));
    },southern);
    await page.waitForFunction(()=>{
      const d=terrainDebug.snapshot();
      return d.mode==='render'&&d.visible.wanted>0&&d.visible.lod===d.camera.lod&&
        d.visible.pending===0&&d.rendered.lods.includes(d.camera.lod)&&d.scheduler.inflight===0;
    },{timeout:180000});
    await page.evaluate(async()=>{draw();await renderer.device.queue.onSubmittedWorkDone()});
    await page.screenshot({path:path.join(out,'southern-summer-live.png')});
    await page.evaluate(()=>{const input=document.getElementById('render_season');input.value='.5';input.dispatchEvent(new Event('input'));});
    await page.evaluate(async()=>{draw();await renderer.device.queue.onSubmittedWorkDone()});
    await page.screenshot({path:path.join(out,'southern-winter-live.png')});
    await page.evaluate(()=>document.getElementById('renderMaterialReset').click());
    assert.deepEqual(errors,[]);
    const report={ok:true,source,southern,backend:await page.evaluate(()=>terrainDebug.snapshot().backend),errors};
    fs.writeFileSync(path.join(out,'live-report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
