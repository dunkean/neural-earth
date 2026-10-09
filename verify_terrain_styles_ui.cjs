// Complete viewer with synthetic physical tiles, real WebGPU/WebGL and PNG fallback.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
  const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--enable-unsafe-webgpu']});
  try{
    for(const gpu of [true,false]){
      const page=await browser.newPage({viewport:{width:1200,height:800}}),errors=[],requests=[],views=[],physicalRequests=[];
      let allowPhysical;const sourceGate=new Promise(resolve=>allowPhysical=resolve);
      if(!gpu)await page.addInitScript(()=>Object.defineProperty(navigator,'gpu',{value:undefined}));
      page.on('pageerror',e=>errors.push(e.message));
      const png=Buffer.from(await page.evaluate(()=>{const c=document.createElement('canvas');c.width=c.height=256;const ctx=c.getContext('2d');ctx.fillStyle='#bbd9ed';ctx.fillRect(0,0,256,256);return c.toDataURL().split(',')[1]}),'base64');
      await page.route('https://styles.test/**',async route=>{
        const url=new URL(route.request().url()),name=url.pathname;
        if(name==='/')return route.fulfill({contentType:'text/html',body:fs.readFileSync('index.html')});
        if(name==='/terrain_styles.js')return route.fulfill({contentType:'application/javascript',body:'window.TerrainStyles='+fs.readFileSync('terrain_styles.json','utf8')+';\n'+fs.readFileSync('terrain_style_rendering.js','utf8')});
        if(name.endsWith('.js'))return route.fulfill({contentType:'application/javascript',body:fs.readFileSync(path.basename(name))});
        if(name==='/api/world')return route.fulfill({json:{seed:'42',world_profile:'natural',generation_profile:'natural',gpu:'test',version:'test-v1',cache_profile:'test',world_topology:'sphere',world_diameter_km:12732.4,world_bounds:[-20e6,-10e6,20e6,10e6],overview_bounds:[-20e6,-10e6,20e6,10e6],overview:'/overview/test.png'}});
        if(name==='/api/view'){const body=route.request().postDataJSON();views.push(body);return route.fulfill({json:{accepted:true,epoch:body.epoch}});}
        if(name==='/api/status')return route.fulfill({json:{scheduler:{queued:0,computing:0,encoding:0}}});
        if(name.startsWith('/height/')||name.startsWith('/coarse/')){
          physicalRequests.push(url);
          if(gpu)await sourceGate;
          const native=name.startsWith('/coarse/'),width=native?160:304,halo=native?16:24,climateWidth=native?41:33;
          const values=new Float32Array(width*width+5*climateWidth*climateWidth);values.fill(native?Math.sqrt(1000):1000,0,width*width);
          return route.fulfill({contentType:'application/octet-stream',body:Buffer.from(values.buffer),headers:{'X-Terrain-Width':String(width),'X-Terrain-Halo':String(halo),'X-Terrain-Climate-Width':String(climateWidth),'X-Terrain-Climate-Height':String(climateWidth),'X-Terrain-Resolution':String(native?7680:30*2**Number(name.split('/')[4])),'X-Terrain-Encoding':native?'signed-sqrt':'metres','X-Terrain-Stage':'coarse','X-Terrain-Elevation-Min':'1000','X-Terrain-Elevation-Max':'1000'}});
        }
        if(name.startsWith('/overview/')||name.startsWith('/tiles/')){requests.push(url);return route.fulfill({contentType:'image/png',body:png});}
        return route.fulfill({json:{released:true}});
      });
      await page.goto('https://styles.test/?seed=42&profile=natural&mode=topographic&contours=1&contour_interval=50');
      if(gpu){
        await page.waitForFunction(()=>terrainDebug.snapshot().scheduler.inflight>0);
        const cancelled=await page.evaluate(()=>terrainDebug.snapshot().counters.aborts);
        await page.locator('#mapMode').selectOption('atlas',{force:true});
        assert.equal(await page.evaluate(()=>terrainDebug.snapshot().counters.aborts),cancelled,'changing style preserves source transfers in flight');
        allowPhysical();
      }
      await page.waitForFunction(()=>terrainDebug.snapshot().overview&&terrainDebug.snapshot().visible.pending===0);
      if(gpu)assert.equal(new Set(physicalRequests.map(u=>u.pathname)).size,physicalRequests.length,'pending source requests are not duplicated for a new style');
      assert.equal(await page.locator('#layerPanel button[data-mode]').evaluateAll(nodes=>nodes.filter(n=>Object.hasOwn(TerrainStyles,n.dataset.mode)).length),10);
      assert.equal(await page.locator('#showContours').isChecked(),true);
      assert.equal(await page.locator('#contourInterval').inputValue(),'50');
      assert.equal(await page.locator('#contourWidth').inputValue(),'0.9');
      assert.equal(await page.locator('#contourDensity').inputValue(),'3');
      assert.equal(await page.locator('#contourDensityControl').isVisible(),false,'manual links keep automatic density hidden');
      assert.equal(await page.evaluate(()=>terrainDebug.snapshot().backend),gpu?'webgpu':'png');
      const loaded=physicalRequests.length,uploads=await page.evaluate(()=>terrainDebug.snapshot().renderer?.uploads),aborts=await page.evaluate(()=>terrainDebug.snapshot().counters.aborts);
      for(const mode of ['blueprint','night','copernicus','topographic']){
        await page.locator('#mapMode').selectOption(mode,{force:true});
        await page.waitForFunction(mode=>terrainDebug.snapshot().mode===mode&&terrainDebug.snapshot().visible.pending===0&&terrainDebug.snapshot().overview,mode);
      }
      if(gpu){assert.equal(physicalRequests.length,loaded,'style switches must not request source data');assert.equal(await page.evaluate(()=>terrainDebug.snapshot().renderer.uploads),uploads,'style switches must not upload physical textures');assert.equal(await page.evaluate(()=>terrainDebug.snapshot().counters.aborts),aborts,'style switches must not cancel terrain work');}
      await page.locator('#renderPanel').evaluate(e=>e.open=true);
      await page.locator('#contourInterval').fill('20');await page.locator('#contourInterval').dispatchEvent('change');
      await page.waitForFunction(()=>new URLSearchParams(location.search).get('contour_interval')==='20'&&terrainDebug.snapshot().overview);
      await page.locator('#contourWidth').fill('2');
      await page.locator('#contourWidth').dispatchEvent('input');
      await page.waitForFunction(()=>new URLSearchParams(location.search).get('contour_width')==='2'&&terrainDebug.snapshot().overview&&terrainDebug.snapshot().visible.pending===0);
      assert.equal(await page.locator('#contourWidthValue').textContent(),'2.0 px');
      assert(views.some(v=>v.overview_contours?.width===2));
      assert(requests.some(u=>u.searchParams.has('contours')&&JSON.parse(u.searchParams.get('contours')).width===2));
      await page.locator('#contourAuto').check();
      assert.equal(await page.locator('#contourDensityControl').isVisible(),true);
      assert.equal(await page.locator('#contourInterval').isDisabled(),true);
      const defaultDensityInterval=await page.evaluate(()=>terrainDebug.snapshot().contours.interval);
      await page.locator('#contourDensity').fill('5');await page.locator('#contourDensity').dispatchEvent('input');
      await page.waitForFunction(()=>new URLSearchParams(location.search).get('contour_density')==='100'&&terrainDebug.snapshot().overview&&terrainDebug.snapshot().visible.pending===0);
      assert.equal(await page.evaluate(()=>terrainDebug.snapshot().contours.interval),defaultDensityInterval/4);
      if(gpu){assert.equal(physicalRequests.length,loaded,'width and density reuse physical terrain');assert.equal(await page.evaluate(()=>terrainDebug.snapshot().renderer.uploads),uploads);}
      const sharedURL=page.url();await page.goto(sharedURL);
      await page.waitForFunction(()=>terrainDebug.snapshot().overview&&terrainDebug.snapshot().visible.pending===0);
      assert.equal(await page.locator('#contourWidth').inputValue(),'2');
      assert.equal(await page.locator('#contourDensity').inputValue(),'5');
      assert.equal(await page.locator('#contourAuto').isChecked(),true);
      await page.locator('#renderPanel').evaluate(e=>e.open=true);
      await page.locator('#contourAuto').uncheck();
      assert.equal(await page.locator('#contourDensity').isDisabled(),true);
      await page.locator('#contourInterval').fill('20');await page.locator('#contourInterval').dispatchEvent('change');
      await page.waitForFunction(()=>terrainDebug.snapshot().overview&&terrainDebug.snapshot().contours.interval===20);
      await page.locator('#renderPanel').evaluate(e=>e.open=false);
      await page.locator('#globeView').click();
      await page.waitForFunction(()=>terrainDebug.snapshot().view==='globe'&&terrainDebug.snapshot().globe.detailReady&&terrainDebug.snapshot().visible.pending===0);
      assert(views.some(v=>v.mode==='topographic'&&v.overview_contours?.interval===20));
      assert(requests.some(u=>u.searchParams.get('mode')==='topographic'&&JSON.parse(u.searchParams.get('contours')).interval===20));
      await page.locator('#mapView').click();await page.locator('#layerPanel').evaluate(e=>e.open=true);
      await page.screenshot({path:`tmp/terrain-styles-ui-${gpu?'gpu':'png'}.png`});
      await page.goto('https://styles.test/?seed=42&profile=natural&mode=topographic&contours=1');
      await page.waitForFunction(()=>terrainDebug.snapshot().overview&&terrainDebug.snapshot().visible.pending===0);
      assert.equal(await page.locator('#contourAuto').isChecked(),true);
      assert.equal(await page.locator('#contourDensity').inputValue(),'3');
      assert.equal(await page.evaluate(()=>terrainDebug.snapshot().contours.interval===4*TerrainStyleRendering.autoInterval(terrainDebug.snapshot().camera.mpp,100)),true,'new views start at the previous minimum density');
      await page.locator('#renderPanel').evaluate(e=>e.open=true);
      assert.equal(await page.locator('#contourDensityValue').textContent(),'25 %');
      const centreInterval=await page.evaluate(()=>terrainDebug.snapshot().contours.interval);
      await page.locator('#contourDensity').fill('0');await page.locator('#contourDensity').dispatchEvent('input');
      await page.waitForFunction(()=>new URLSearchParams(location.search).get('contour_density')==='3.125'&&terrainDebug.snapshot().overview&&terrainDebug.snapshot().visible.pending===0);
      assert.equal(await page.evaluate(()=>terrainDebug.snapshot().contours.interval),Math.min(10000,centreInterval*8));
      const sparseURL=page.url();await page.goto(sparseURL);
      await page.waitForFunction(()=>terrainDebug.snapshot().overview&&terrainDebug.snapshot().visible.pending===0);
      assert.equal(await page.locator('#contourDensity').inputValue(),'0');
      assert.equal(await page.evaluate(()=>terrainDebug.snapshot().contours.density),3.125);
      await page.locator('#renderPanel').evaluate(e=>e.open=true);
      await page.screenshot({path:`tmp/terrain-contour-settings-${gpu?'gpu':'png'}.png`});
      assert.deepEqual(errors,[]);await page.close();
      console.log(`Styles, menus, physical tiles, curves, shared links and globe (${gpu?'GPU':'PNG'}): OK`);
    }
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
