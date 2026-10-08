const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const {sampleHeight, distanceLabel} = require('./terrain_map_tools.js');
const {chromium} = require(process.env.PLAYWRIGHT_PATH || 'E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');

// Bilinear sampling respects pixel centres, halo, and negative map coordinates.
const grid = new Float32Array(16);
grid[5] = -100; grid[6] = 100; grid[9] = 300; grid[10] = 500;
const tile = {heights:grid, heightOptions:{width:4,halo:1}, b:[-20,-20,0,0]};
assert.equal(sampleHeight(tile,{x:-15,y:-15}),-100);
assert.equal(sampleHeight(tile,{x:-10,y:-10}),200);
assert.equal(sampleHeight(tile,{x:-5,y:-5}),500);
assert.equal(sampleHeight({}, {x:0,y:0}),null);
assert.equal(distanceLabel(1500),'1.50 km');

(async () => {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',args:['--disable-gpu']});
  try {
    const page = await browser.newPage({viewport:{width:1280,height:900}});
    const errors = [], requests = [];
    page.on('pageerror',e=>errors.push(e.message));
    const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM0LJrHwMDAxAAGAA7JAUW48M0QAAAAAElFTkSuQmCC','base64');
    const physical = new Float32Array(304*304+5*33*33); physical.fill(-125.5,0,304*304);
    const gpuStub = `window.createTerrainRenderer=async()=>{const tiles=new Set();return{available:true,uploadTile(k){tiles.add(k)},hasTile(k){return tiles.has(k)},deleteTile(k){tiles.delete(k)},clear(){tiles.clear()},draw(){},getStats(){return{status:'webgpu'}}}}`;
    await page.route('https://map-tools.test/**',async route => {
      const u = new URL(route.request().url()); requests.push(u.pathname);
      if(u.pathname==='/')return route.fulfill({contentType:'text/html',body:fs.readFileSync('index.html','utf8')});
      if(u.pathname==='/terrain_renderer.js')return route.fulfill({contentType:'application/javascript',body:gpuStub});
      if(['/terrain_lod.js','/terrain_generation_controls.js','/terrain_map_tools.js'].includes(u.pathname))return route.fulfill({contentType:'application/javascript',body:fs.readFileSync(u.pathname.slice(1),'utf8')});
      if(u.pathname==='/api/world')return route.fulfill({json:{version:'natural-v1',cache_profile:'mock',world_profile:'natural',generation_profile:'natural',seed:u.searchParams.get('seed'),world_bounds:[-500000,-250000,500000,250000],overview_bounds:[-500000,-250000,500000,250000],overview:'/overview.png',gpu:'Mock GPU'}});
      if(u.pathname==='/api/status')return route.fulfill({json:{scheduler:{}}});
      if(u.pathname==='/api/view')return route.fulfill({json:{accepted:true}});
      if(u.pathname.startsWith('/height/')){
        const source=u.searchParams.get('source_lod'),lod=Number(u.pathname.split('/')[4]),width=source?256/2**(3-lod)+48:304;
        const values=new Float32Array(width*width+(u.searchParams.has('climate')?5*33*33:0));values.fill(-125.5,0,width*width);
        return route.fulfill({contentType:'application/octet-stream',body:Buffer.from(values.buffer),headers:{'X-Terrain-Width':String(width),'X-Terrain-Halo':'24','X-Terrain-Climate-Width':u.searchParams.has('climate')?'33':'0','X-Terrain-Climate-Height':u.searchParams.has('climate')?'33':'0','X-Terrain-Stage':source?'latent':'decoder','X-Terrain-Source-LOD':source||String(lod),'X-Terrain-Source-Resolution':source?'240':'30'}});
      }
      return route.fulfill({contentType:'image/png',body:png,headers:u.pathname.startsWith('/tiles/')?{'X-Terrain-Source-LOD':u.searchParams.get('source_lod')||u.pathname.split('/')[4],'X-Terrain-Stage':u.searchParams.has('source_lod')?'latent':'decoder'}:{}});
    });
    await page.goto('https://map-tools.test/?seed=42&profile=natural');
    const settle = () => page.waitForFunction(()=>terrainDebug.snapshot().visible.ready>0 && terrainDebug.snapshot().visible.pending===0);
    await settle();
    await page.locator('#generationPanel summary').click();
    await page.locator('#generationPreset').selectOption('mountains');
    assert.deepEqual(await page.evaluate(()=>generationControls.read().snr_altitude_gain),[4,1,1,1,1]);
    await page.locator('#generationPreset').selectOption('glacial');
    assert.deepEqual(await page.evaluate(()=>generationControls.read().snr_driver_gain),[3,1,1,1,1]);
    assert.deepEqual(await page.evaluate(()=>generationControls.read().snr_driver_range),[5,-10]);
    await page.locator('#generationPreset').selectOption('current');
    await page.locator('#generationPanel summary').click();
    await page.locator('#plus').click(); await page.locator('#plus').click(); await settle();
    const box = await page.locator('#viewport').boundingBox(), x=box.x+box.width/2, y=box.y+box.height/2;
    await page.mouse.move(x,y);
    await page.waitForFunction(()=>document.getElementById('cursorAltitude').textContent.includes('-125.5 m'));
    assert.match(await page.locator('#visibleExtent').textContent(),/width × height/);
    assert.match(await page.locator('#visibleExtent').textContent(),/km · \d+ × \d+ px/);
    const before = await page.evaluate(()=>terrainDebug.snapshot().camera);
    await page.locator('#measureDistance').click();
    await page.mouse.move(x,y); await page.mouse.down(); await page.mouse.move(x+120,y+90,{steps:4});
    await page.mouse.up();
    await page.waitForFunction(()=>document.getElementById('measureDistance').getAttribute('aria-pressed')==='false');
    assert.equal(await page.locator('#measuredDistance').textContent(),`Distance on map: ${distanceLabel(150*before.mpp)}`);
    const after = await page.evaluate(()=>terrainDebug.snapshot().camera);
    assert.equal(after.cx,before.cx); assert.equal(after.cy,before.cy);
    await page.mouse.move(x,y); await page.mouse.down(); await page.mouse.move(x+20,y); await page.mouse.up();
    assert.notEqual((await page.evaluate(()=>terrainDebug.snapshot().camera)).cx,before.cx,'next drag pans normally');
    await page.locator('#measureDistance').click(); await page.mouse.move(x,y); await page.mouse.down();
    await page.mouse.move(x+30,y+30); await page.keyboard.press('Escape'); await page.mouse.up();
    await page.waitForFunction(()=>document.getElementById('measuredDistance').textContent==='Distance: —');
    assert.equal(await page.locator('#measureDistance').getAttribute('aria-pressed'),'false');
    await page.locator('#gpuRender').uncheck(); await settle();
    await page.mouse.move(x,y);
    await page.waitForFunction(()=>document.getElementById('cursorAltitude').textContent.includes('-125.5 m'));
    assert.ok(requests.some(p=>p.startsWith('/height/')),'PNG fallback retrieves physical heights');
    await page.mouse.move(2,2);
    await page.waitForFunction(()=>document.getElementById('cursorAltitude').textContent==='Cursor elevation: —');
    await page.locator('#measureDistance').click(); await page.mouse.move(x,y); await page.mouse.down(); await page.mouse.move(x+120,y+90); await page.mouse.up();
    const screenshot = path.join(os.tmpdir(),'infinite-map-tools.png');
    await page.screenshot({path:screenshot});
    await page.setViewportSize({width:390,height:844});
    await page.waitForTimeout(100);
    assert.ok(await page.locator('#hud').evaluate(el=>el.getBoundingClientRect().right <= innerWidth));
    assert.deepEqual(errors,[]);
    console.log('Map tools passed: bilinear altitude, GPU/PNG, extent, distance, pan restoration, Escape, mobile HUD.');
    console.log(screenshot);
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
