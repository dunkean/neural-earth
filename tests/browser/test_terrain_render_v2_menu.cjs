// Full viewer controls with PNG fixtures: exercise the visible menu, not the
// hidden legacy select. No server, models or GPU required.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {loadPlaywright,browserExecutable}=require('../../tools/platform.cjs');
const root=path.resolve(__dirname,'../..');
(async()=>{
 const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--disable-gpu']});
 try{
  const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];page.on('pageerror',e=>errors.push(e.stack));
  await page.addInitScript(()=>{Object.defineProperty(navigator,'gpu',{value:undefined});localStorage.setItem('neural-earth-locale-v1','en');});
  const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM0LJrHwMDAxAAGAA7JAUW48M0QAAAAAElFTkSuQmCC','base64');
  const modes=['render-v2','render-v2-substrate','render-v2-moisture','render-v2-vegetation','render-v2-snow'];
  await page.route('https://v2-menu.test/**',async route=>{
   const url=new URL(route.request().url()),name=url.pathname;
   if(name==='/')return route.fulfill({contentType:'text/html',body:fs.readFileSync(path.join(root,'web/index.html'),'utf8')});
   if(name==='/terrain_styles.js')return route.fulfill({contentType:'application/javascript',body:'window.TerrainStyles='+fs.readFileSync(path.join(root,'web/terrain_styles.json'),'utf8')+';\n'+fs.readFileSync(path.join(root,'web/terrain_style_rendering.js'),'utf8')});
   if(name.endsWith('.js'))return route.fulfill({contentType:'application/javascript',body:fs.readFileSync(path.join(root,'web',path.basename(name)),'utf8')});
   if(name==='/api/world')return route.fulfill({json:{seed:'42',world_profile:'orogen',generation_profile:'orogen',gpu:'fixture',version:'natural-v1',cache_profile:'fixture',world_topology:'sphere',world_diameter_km:12732.4,world_bounds:[-20e6,-10e6,20e6,10e6],overview_bounds:[-20e6,-10e6,20e6,10e6],overview:'/overview.png',orogen_layers:{render:'Render',...Object.fromEntries(modes.map(m=>[m,m]))}}});
   if(name==='/api/status')return route.fulfill({json:{scheduler:{queued:0,computing:0,encoding:0}}});
   // Sharing persistence has its own regressions; don't create bogus UIDs.
   if(name==='/api/share')return route.fulfill({status:503,json:{error:'Sharing persistence unavailable in this menu fixture'}});
   if(name==='/api/gpu')return route.fulfill({status:503,json:{error:'GPU controls unavailable in this menu fixture'}});
   if(name.startsWith('/api/'))return route.fulfill({json:{accepted:true}});
   return route.fulfill({contentType:'image/png',body:png});
  });
  await page.goto('https://v2-menu.test/?seed=42&profile=orogen&mode=relief&gpu=0&coarse_gpu=0&coarse_prepare=0');
  await page.waitForFunction(()=>window.terrainDebug?.snapshot().world==='42');
  const layer=page.locator('#layerPanel button[data-mode="render-v2"]');
  await page.locator('#layerPanel > summary').click();
  assert.equal(await page.locator('#layerPanel button[data-mode^="render-v2"]').count(),1,'one visible layer, diagnostics excluded from all layer groups');
  assert.equal(await layer.isVisible(),true);
  assert.equal(await layer.evaluate(e=>e.closest('fieldset').querySelector('legend').textContent),'Main maps');
  await layer.click();
  await page.waitForFunction(()=>terrainDebug.snapshot().mode==='render-v2');
  assert.equal(await page.locator('#layerPanel .selectedLayer').textContent(),'Render v2');
  await page.locator('#renderPanel > summary').click();await page.locator('#tabRendering').click();
  assert.equal(await page.locator('#renderV2Season').isVisible(),true);
  assert.equal(await page.locator('#renderV2Diagnostic').isVisible(),true);
  await page.locator('#renderV2Season').fill('0.25');await page.locator('#renderV2Season').dispatchEvent('input');
  await page.locator('#renderV2Diagnostic').selectOption('render-v2-moisture');
  await page.waitForFunction(()=>terrainDebug.snapshot().mode==='render-v2-moisture'&&new URL(location.href).searchParams.get('v2season')==='0.25');
  await page.evaluate(()=>hydrology.set({month:4,density:.6,lakes:false}));
  await page.evaluate(()=>render3d.set({enabled:false,density:1.35,clearings:.4}));
  for(const [locale,seasonLabel,moisture,density,area,riverDensity,forestDensity] of [['fr','Saison','Humidité locale','Densité','Zone visible','Densité des cours d’eau','Densité (%)'],['en','Season','Local moisture','Density','Visible area','Stream density','Density (%)'],['zh','季节','局部湿度','密度','可见区域','河流密度','密度（%）'],['es','Estación','Humedad local','Densidad','Área visible','Densidad de cursos de agua','Densidad (%)'],['pt','Estação','Umidade local','Densidade','Área visível','Densidade de cursos d’água','Densidade (%)']]){
   await page.locator('#ui-language').selectOption(locale);
   assert.equal(await page.locator('#renderV2Season').getAttribute('aria-label'),seasonLabel);
   assert.equal(await page.locator('#renderV2Diagnostic option[value="render-v2-moisture"]').textContent(),moisture);
   assert.equal(await page.locator('#renderV2Season').inputValue(),'0.25');
   assert.equal(await page.locator('#renderV2Diagnostic').inputValue(),'render-v2-moisture');
   assert.equal(await page.locator('#brush_density').getAttribute('aria-label'),density);
   assert.equal(await page.locator('#brush_density').inputValue(),'1');
   await page.waitForFunction(label=>document.getElementById('visibleExtent').textContent.startsWith(label+':'),area);
   assert.equal(await page.locator('#hydrologySettings [data-hydro="density"]').evaluate(input=>input.closest('label').firstChild.textContent.trim()),riverDensity);
   assert.equal(await page.locator('#hydrologySettings [data-hydro="month"]').inputValue(),'4');
   assert.equal(await page.locator('#hydrologySettings [data-hydro="month"] option[value="4"]').textContent(),await page.evaluate(lang=>new Intl.DateTimeFormat(lang,{month:'long',timeZone:'UTC'}).format(new Date(Date.UTC(2000,3,1))),locale));
   assert.equal(await page.locator('#hydrologySettings [data-hydro="density"]').inputValue(),'0.6');
   assert.equal(await page.locator('#hydrologySettings [data-hydro="lakes"]').isChecked(),false);
   assert.equal(await page.locator('#forestDensity').getAttribute('aria-label'),forestDensity);
   assert.equal(await page.locator('#forestDensity_range').getAttribute('aria-label'),forestDensity);
   assert.equal(await page.locator('#forestDensity').inputValue(),'135');
   assert.equal(await page.locator('#forestClearings').inputValue(),'40');
   assert.equal(await page.locator('#forestEnabled').isChecked(),false);
   assert.equal(await page.locator('#renderV2Date').textContent(),await page.evaluate(lang=>new Intl.DateTimeFormat(lang,{month:'long',timeZone:'UTC'}).format(new Date(Date.UTC(2024,0,1+Math.floor(.25*365)))),locale));
  }
  await page.locator('#ui-language').selectOption('en');
  await page.evaluate(async()=>{
   const parent=document.createElement('div');document.body.append(parent);
   const state={seed:42,profile:'natural',worldIdentity:'fixture',worldBounds:[0,0,100,100],bounds:[0,0,100,100],lod:0,mpp:1,width:100,height:100,view:'map',project:(x,y)=>[x,y]};
   const network={status:'ready',reaches:[{id:'river',xy:[[0,50],[90,50]],flow_m3s:500,area_m2:2e10,width_m:5,basin_id:3}],lakes:[{id:'lake',rings:[[[20,20],[30,20],[30,30],[20,30]]],area_m2:2e6,level_m:200,max_depth_m:12,closed:true}]};
   const viewer=TerrainHydrology.create({getState:()=>state,initial:{enabled:true,lakes:true},fetchJSON:async()=>({ok:true,json:async()=>network})});
   viewer.mount(parent);const ctx=document.createElement('canvas').getContext('2d');viewer.draw(ctx,state);await new Promise(resolve=>setTimeout(resolve,0));viewer.draw(ctx,state);
   window.__localeHydroFixture={viewer,parent};
  });
  assert.equal(await page.evaluate(()=>window.__localeHydroFixture.viewer.inspect(25,25)),true);
  for(const [locale,lake] of [['fr','Lac'],['en','Lake'],['zh','湖泊'],['es','Lago'],['pt','Lago']]){
   await page.locator('#ui-language').selectOption(locale);
   assert((await page.evaluate(()=>window.__localeHydroFixture.parent.querySelector('[data-hydro-inspector]').textContent)).startsWith(lake+' ·'));
   assert.equal(await page.evaluate(()=>window.__localeHydroFixture.viewer.snapshot().selected),'lake');
  }
  assert.equal(await page.evaluate(()=>window.__localeHydroFixture.viewer.inspect(45,50)),true);
  for(const [locale,river] of [['fr','Fleuve principal'],['en','Main river'],['zh','主要河流'],['es','Río principal'],['pt','Rio principal']]){
   await page.locator('#ui-language').selectOption(locale);
   assert((await page.evaluate(()=>window.__localeHydroFixture.parent.querySelector('[data-hydro-inspector]').textContent)).startsWith(river+' ·'));
   assert.equal(await page.evaluate(()=>window.__localeHydroFixture.viewer.snapshot().selected),'river');
  }
  await page.evaluate(()=>{window.__localeHydroFixture.viewer.dispose();window.__localeHydroFixture.parent.remove();delete window.__localeHydroFixture});
  await page.locator('#ui-language').selectOption('en');
  assert.equal(await page.locator('#layerPanel .selectedLayer').textContent(),'Render v2');
  assert.equal(await layer.getAttribute('aria-pressed'),'true');
  // Existing diagnostic deep links remain valid after removing their buttons.
  const legacy=new URL(page.url());legacy.searchParams.set('mode','render-v2-moisture');await page.goto(legacy.href);
  await page.waitForFunction(()=>window.terrainDebug?.snapshot().world==='42');
  assert.equal(await page.evaluate(()=>terrainDebug.snapshot().mode),'render-v2-moisture',JSON.stringify({url:page.url(),errors,status:await page.locator('#status').textContent()}));
  await page.locator('#renderPanel > summary').click();await page.locator('#tabRendering').click();
  assert.equal(await page.locator('#renderV2Diagnostic').inputValue(),'render-v2-moisture');
  assert.equal(await page.locator('#renderV2Season').inputValue(),'0.25');
  await page.locator('#renderV2Diagnostic').selectOption('render-v2');
  await page.waitForFunction(()=>terrainDebug.snapshot().mode==='render-v2');
  assert.deepEqual(errors,[]);console.log('Single Render v2 layer, visible controls, diagnostics and deep-link restoration: OK');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
