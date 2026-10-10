const {loadPlaywright, browserExecutable, pythonExecutable} = require('../platform.cjs');
const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {readRepositoryFile} = require(NEURAL_EARTH_ROOT + '/tools/repository-files.cjs');
const {chromium}=loadPlaywright();

const fs=require('node:fs'),assert=require('node:assert/strict');

const schema=JSON.parse(require('node:child_process').execFileSync(pythonExecutable(),['-c','import json; from terrain_generation import generator_schema; print(json.dumps(generator_schema()))'],{encoding:'utf8'}));
const root=process.cwd(),png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM0LJrHwMDAxAAGAA7JAUW48M0QAAAAAElFTkSuQmCC','base64');

(async()=>{const browser=await chromium.launch({executablePath:browserExecutable(),headless:true});try{

 const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[],worldRequests=[],generationRequests=[];let appliedSettings=null,failNextGeneration=false;page.setDefaultTimeout(10000);page.on('pageerror',e=>errors.push(e.message));

 await page.route('https://lod.test/**',async route=>{const u=new URL(route.request().url());

 if(u.pathname==='/')return route.fulfill({contentType:'text/html',body:readRepositoryFile(root+'/index.html','utf8')});

 if(u.pathname==='/terrain_renderer.js')return route.fulfill({contentType:'application/javascript',body:'window.createTerrainRenderer=async()=>({available:false,clear(){},draw(){},deleteTile(){},getStats(){return{}}})'});

 if(u.pathname==='/terrain_styles.js')return route.fulfill({contentType:'application/javascript',body:'window.TerrainStyles='+readRepositoryFile('terrain_styles.json','utf8')+';\n'+readRepositoryFile('terrain_style_rendering.js','utf8')});
 if(u.pathname.endsWith('.js'))return route.fulfill({contentType:'application/javascript',body:readRepositoryFile(root+u.pathname,'utf8')});

 if(u.pathname==='/api/generation/run'){
  const data=route.request().postDataJSON();generationRequests.push(data);
  if(failNextGeneration){failNextGeneration=false;return route.fulfill({status:400,json:{error:'Generation test failure'}});}
  if(data.stage==='all'||data.stage==='restore')appliedSettings={...schema.defaults_by_profile[data.profile],...data.settings};
  else{const changed=Object.fromEntries(schema.stage_groups[data.stage].map(key=>[key,data.settings[key]]));appliedSettings={...appliedSettings,...changed};}
  for(const name of ['relief','erosion','climate'])appliedSettings['orogen_'+name+'_stage']||={relief:'a',erosion:'b',climate:'c'}[name].repeat(64);
  return route.fulfill({json:{world_profile:data.profile,generation_settings:appliedSettings}});
 }

 if(u.pathname==='/api/world'){worldRequests.push(Object.fromEntries(u.searchParams));return route.fulfill({json:{seed:'42',version:'mock',cache_profile:'mock',world_profile:u.searchParams.get('world_profile'),generation_profile:u.searchParams.get('world_profile'),generation_schema:schema,generation_settings:{...schema.defaults_by_profile[u.searchParams.get('world_profile')],...(u.searchParams.has('generation')?JSON.parse(u.searchParams.get('generation')):{})},orogen_layers:{'orogen-biomes':'Orogen biomes','orogen-height':'Orogen initial relief'},world_bounds:[-20e6,-10e6,20e6,10e6],overview_bounds:[-20e6,-10e6,20e6,10e6],overview:'/overview.png',gpu:'Mock'}});}

 if(u.pathname==='/api/view')return route.fulfill({json:{accepted:true}});

 if(u.pathname==='/api/status')return route.fulfill({json:{scheduler:{queued:0,computing:0,encoding:0}}});

 return route.fulfill({contentType:'image/png',body:png,headers:{'X-Terrain-Stage':'final-dem-mip'}});

 });

 await page.goto('https://lod.test/?seed=42&profile=natural');await page.waitForFunction(()=>terrainDebug.snapshot().world==='42'&&terrainDebug.snapshot().visible.pending===0);

 assert.equal(await page.locator('#renderPanel #refinementDepth').count(),1);
 assert.equal(await page.locator('#renderPanel #prefetch').count(),1);
 assert.equal(await page.locator('#renderPanel #mapMode').count(),0);
 assert.equal(await page.locator('#controls #mapMode').count(),1);
 assert.equal(await page.locator('#noisePanel #snr0').count(),1);
 assert.equal(await page.locator('#climatePanel #orogen_temperature_equator').count(),1);
 assert.equal(await page.locator('#generationPanel #orogen_temperature_equator').count(),1);
 assert.equal(await page.locator('#renderPanel #snr0').count(),1);
 assert.equal(await page.locator('#controls > #climatePanel, #controls > #noisePanel').count(),0);
 const before=worldRequests.length;

 await page.getByRole('button',{name:'Tectonic',exact:true}).click();

 assert.equal(await page.inputValue('#generatorType'),'orogen');

 await page.locator('#generationPanel > summary').click();
 for(const key of ['relief','propagation','post','climate','raster'])
  assert.equal(await page.isChecked('#orogen_gpu_'+key),false,'historical CPU default');
 await page.click('#tabClimate');
 await page.check('#orogen_gpu_climate');
 await page.click('#tabRelief');
 await page.check('#orogen_gpu_relief');

 await page.selectOption('#generatorType','custom');

 assert.equal(await page.inputValue('#worldGenerator'),'custom');

 assert.equal(await page.inputValue('#climateSource'),'orogen');
 assert.equal(await page.locator('#climateSource option').count(),1);
 await page.click('#tabErosion');
 assert.deepEqual(await page.locator('#reliefPipeline option').allTextContents(),['None','Orogen · GPU','Orogen · CPU','City · GPU']);
 await page.selectOption('#reliefPipeline','orogen-gpu');
 assert.equal(await page.evaluate(()=>generationControls.read().orogen_gpu_erosion),true);
 assert.equal(await page.evaluate(()=>generationControls.read().relief_pipeline),'orogen');
 await page.evaluate(()=>{const draft=generationControls.draftSnapshot();generationControls.sync(draft.profile,draft.settings);});
 assert.equal(await page.inputValue('#reliefPipeline'),'orogen-gpu','saved GPU choice reloads');
 await page.selectOption('#reliefPipeline','orogen');
 assert.equal(await page.evaluate(()=>generationControls.read().orogen_gpu_erosion),false);
 await page.click('#tabRelief');await page.selectOption('#generatorType','natural');await page.click('#tabErosion');
 assert.equal(await page.locator('#orogen_hydraulic').isEnabled(),true);
 await page.selectOption('#reliefPipeline','original');
 assert.equal(await page.locator('#orogen_hydraulic').isEnabled(),false);
 await page.selectOption('#reliefPipeline','city-gpu');
 assert.equal(await page.locator('#cityErosionParameters').isVisible(),true);
 assert.equal(await page.locator('#orogen_hydraulic').isEnabled(),false);
 await page.fill('#city_erosion_strength','0.75');
 await page.fill('#city_erosion_iterations','8');
 assert.equal(await page.evaluate(()=>generationControls.read().relief_pipeline),'city-gpu');
 assert.equal(await page.evaluate(()=>generationControls.read().city_erosion_strength),.75);
 await page.selectOption('#reliefPipeline','original');
 assert.equal(await page.locator('#cityErosionParameters').isVisible(),false);
 await page.click('#tabRelief');
 await page.selectOption('#continentalStyle','archipelago');
 assert.equal(await page.inputValue('#frequency0'),'1.6');
 await page.click('#tabClimate');
 await page.selectOption('#orogenClimateSection','temperature');
 await page.fill('#orogen_temperature_equator','33');

 assert.equal(await page.inputValue('#worldGenerator'),'natural');

 assert.equal(worldRequests.length,before,'selector changes must stay drafts');

 await page.locator('#renderPanel > summary').click();await page.click('#tabSnr');
 await page.fill('#snr0','0.123');
 await page.fill('#snrAltitude0','2');
 await page.fill('#snrDriver0','3');
 await page.fill('#snrLatitudeGain','4');
 await page.selectOption('#snrDetailMode','per-lod');
 await page.locator('#noisePanel .advancedSettings > summary').filter({hasText:'Per-LOD detail amplitude'}).click();
 await page.fill('#snrLod5','0.25');
 await page.screenshot({path:'tmp/settings-noise-desktop.png'});

 await page.locator('#generationPanel > summary').click();

 await page.click('#apply');

 await page.waitForFunction(()=>terrainDebug.snapshot().generationSettings?.cond_snr[0]===0.123);

 assert.equal(worldRequests.at(-1).world_profile,'natural');

 assert.equal(JSON.parse(worldRequests.at(-1).generation).cond_snr[0],0.123,'menubar Generate must apply the shared Settings draft');

 assert.equal(await page.inputValue('#worldGenerator'),await page.inputValue('#generatorType'));
 const applied=JSON.parse(worldRequests.at(-1).generation);
 assert.equal(applied.climate_source,'orogen');assert.equal(applied.orogen_temperature_equator,33);
 assert.equal(applied.snr_latitude_gain,4);assert.equal(applied.snr_lod[5],.25);
 assert.equal(applied.snr_altitude_gain[0],2);assert.equal(applied.snr_driver_gain[0],3);
 assert.equal(applied.relief_pipeline,'original');
 assert.equal(applied.city_erosion_strength,.75);
 assert.equal(applied.city_erosion_iterations,8);
 assert.equal(applied.orogen_gpu_climate,true);
 assert.equal(applied.orogen_gpu_relief,true);
 assert.equal(applied.orogen_gpu_erosion,false);
 assert.equal(applied.snr_detail_mode,'per-lod');
 assert.equal(generationRequests.at(-1).stage,'all','menubar Generate runs all three stages');
 await page.click('#tabClimate');await page.fill('#orogen_temperature_equator','32');
 await page.click('#generateClimate');
 await page.waitForFunction(()=>terrainDebug.snapshot().generationSettings?.orogen_temperature_equator===32);
 assert.equal(generationRequests.at(-1).stage,'climate');
 assert.equal(JSON.parse(worldRequests.at(-1).generation).relief_pipeline,'original');
 await page.click('#tabErosion');await page.selectOption('#reliefPipeline','orogen-gpu');
 await page.click('#generateErosion');
 await page.waitForFunction(()=>terrainDebug.snapshot().generationSettings?.orogen_gpu_erosion===true);
 assert.equal(generationRequests.at(-1).stage,'erosion');
 assert.equal(await page.inputValue('#reliefPipeline'),'orogen-gpu');
 await page.click('#tabRelief');await page.click('#generateRelief');
 await page.waitForFunction(()=>document.getElementById('generationDraftStatus').textContent.startsWith('Relief generated'));
 assert.equal(generationRequests.at(-1).stage,'relief');
 await page.click('#applyGeneration');
 await page.waitForFunction(()=>document.getElementById('generationDraftStatus').textContent==='Relief, erosion and climate generated.');
 assert.equal(generationRequests.at(-1).stage,'all','Settings general button runs all three stages');
 await page.click('#saveGenerationA');
 await page.locator('#renderPanel > summary').click();await page.click('#tabSnr');
 await page.uncheck('#snrAdaptive');
 assert.equal(await page.evaluate(()=>generationControls.read().snr_adaptive_enabled),false);
 assert.equal(await page.locator('#snrLatitudeGain').isEnabled(),false);
 assert.equal(await page.evaluate(()=>generationControls.read().snr_latitude_gain),4,'turning adaptive off preserves its values');
 const beforeSnr=generationRequests.length;
 await page.click('#applySnr');
 await page.waitForFunction(()=>terrainDebug.snapshot().generationSettings?.snr_adaptive_enabled===false);
 assert.equal(generationRequests.length,beforeSnr+1);
 assert.equal(generationRequests.at(-1).stage,'settings','SNR applies without regenerating source stages');
 await page.locator('#generationPanel > summary').click();
 const beforeRestore=generationRequests.length;
 await page.click('#switchGenerationAB');
 await page.waitForFunction(()=>terrainDebug.snapshot().generationSettings?.snr_adaptive_enabled===true);
 assert.equal(generationRequests.length,beforeRestore+1);assert.equal(generationRequests.at(-1).stage,'restore','A restores retained stages');
 await page.click('#switchGenerationAB');
 await page.waitForFunction(()=>terrainDebug.snapshot().generationSettings?.snr_adaptive_enabled===false);
 assert.equal(generationRequests.length,beforeRestore+2);assert.equal(generationRequests.at(-1).stage,'restore','B restores retained stages');
 await page.locator('#generationPanel > summary').click();
 // Every generation field, including exact RGB arrays, resets in the form
 // and in the actual generation request. Invalid drafts must not block reset.
 const resetProfile=await page.evaluate(()=>generationControls.draftSnapshot().profile);
 const baseline=await page.evaluate(profile=>generationControls.defaults(profile),resetProfile);
 await page.evaluate(schema=>{
   for(const [id,spec] of Object.entries(schema.properties)){
     const input=document.getElementById(id);if(!input)continue;
     if(spec.color){input.value='#123456';input.dispatchEvent(new Event('input',{bubbles:true}));}
     else if(input.type==='checkbox')input.checked=!input.checked;
     else if(spec.minimum!==undefined)input.value=spec.default===spec.minimum?spec.maximum:spec.minimum;
   }
   document.getElementById('macroScale').value='invalid';
   for(const key of ['vegetation_tint','rock_tint','snow_color']){const input=document.getElementById('render_'+key);input.value='#123456';input.dispatchEvent(new Event('input'));}
   const season=document.getElementById('render_season');season.value='0';season.dispatchEvent(new Event('input'));
 },schema);
 const assertReset=async()=>{
   assert.deepEqual(await page.evaluate(()=>TerrainRender.get()),await page.evaluate(()=>TerrainRender.defaults()),'Generation reset includes all Render settings');
   for(const key of ['vegetation_tint','rock_tint','snow_color'])assert.deepEqual(JSON.parse(await page.getAttribute('#render_'+key,'data-rgb')),await page.evaluate(k=>TerrainRender.defaults()[k],key),'Render exact RGB reset');
   const draft=await page.evaluate(()=>generationControls.read());
   for(const [key,value] of Object.entries(draft)){
     if(key.endsWith('_stage'))continue;
     assert.deepEqual(value,baseline[key],key+' returns to its default');
   }
   for(const [key,spec] of Object.entries(schema.properties).filter(([,spec])=>spec.color)){
     const hex='#'+baseline[key].map(v=>Math.round(v*255).toString(16).padStart(2,'0')).join('');
     assert.equal(await page.inputValue('#'+key),hex,key+' visible color resets');
     assert.deepEqual(JSON.parse(await page.getAttribute('#'+key,'data-rgb')),baseline[key],key+' RGB precision resets');
   }
 };
 await page.locator('#renderPanel').evaluate(el=>el.open=false);
 await page.locator('#generationPanel').evaluate(el=>el.open=true);
 failNextGeneration=true;await page.click('#resetGeneration');
 await page.waitForFunction(()=>document.getElementById('generationDraftStatus').textContent==='Settings reset; generation was not applied.');
 await assertReset();
 await page.click('#resetGeneration');
 await page.waitForFunction(()=>document.getElementById('generationDraftStatus').textContent==='All generation settings have been reset.');
 await assertReset();
 const resetRequest=generationRequests.at(-1);assert.equal(resetRequest.stage,'all');
 assert.deepEqual(resetRequest.settings,baseline,'Reset sends every canonical default');
 await page.reload();await page.waitForFunction(()=>terrainDebug.snapshot().world==='42'&&terrainDebug.snapshot().visible.pending===0);
 await assertReset();
 await page.locator('#generationPanel > summary').click();
 await page.locator('#layerPanel > summary').click();
 await page.getByRole('button',{name:'Biomes',exact:true}).click();
 await page.waitForFunction(()=>terrainDebug.snapshot().mode==='orogen-biomes');
 await page.setViewportSize({width:390,height:844});
 await page.locator('#generationPanel > summary').click();await page.click('#tabClimate');
 await page.screenshot({path:'tmp/settings-climate-mobile.png'});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 assert.equal(await page.locator('#climatePanel .generationBody').evaluate(el=>el.getBoundingClientRect().right<=innerWidth),true);
 const beforeRandom=generationRequests.length;
 await page.click('#random');
 await page.waitForFunction(()=>document.getElementById('loadingIndicator').hidden);
 assert.equal(generationRequests.length,beforeRandom+1);
 assert.equal(generationRequests.at(-1).stage,'all','Random directly generates all three stages');

 assert.deepEqual(errors,[]);console.log('Generator selectors share drafts, Orogen-only climate and Generate parameters: OK');

 }finally{await browser.close()}})().catch(e=>{console.error(e);process.exitCode=1});
