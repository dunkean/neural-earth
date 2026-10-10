const path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');process.chdir(root);
const {readRepositoryFile}=require(root+'/tools/repository-files.cjs');
const {loadPlaywright,browserExecutable,pythonExecutable}=require(root+'/tools/platform.cjs');
const {chromium}=loadPlaywright();
const schema=JSON.parse(require('node:child_process').execFileSync(pythonExecutable(),['-c','import json; from terrain_generation import generator_schema; print(json.dumps(generator_schema()))'],{encoding:'utf8'}));
const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM0LJrHwMDAxAAGAA7JAUW48M0QAAAAAElFTkSuQmCC','base64');
const gpuKeys=['relief','propagation','post','climate','erosion','raster'].map(stage=>'orogen_gpu_'+stage);
const storageKey='neural-earth-preferences-v1';

(async()=>{
 const browser=await chromium.launch({executablePath:browserExecutable(),headless:true});
 try{
  for(const enabled of [true,false]){
   const context=await browser.newContext(),page=await context.newPage(),errors=[],worldRequests=[];
   let holdFirstWorld=true,releaseFirstWorld;
   const firstWorldGate=new Promise(resolve=>{releaseFirstWorld=resolve});
   page.on('pageerror',error=>errors.push(error.message));
   await page.route('https://startup.test/**',async route=>{
    const u=new URL(route.request().url());
    if(u.pathname==='/')return route.fulfill({contentType:'text/html',body:readRepositoryFile('index.html','utf8')});
    if(u.pathname==='/terrain_renderer.js')return route.fulfill({contentType:'application/javascript',body:'window.createTerrainRenderer=async()=>({available:true,clear(){},draw(){return true},deleteTile(){},getStats(){return{}}})'});
    if(u.pathname==='/terrain_styles.js')return route.fulfill({contentType:'application/javascript',body:'window.TerrainStyles='+readRepositoryFile('terrain_styles.json','utf8')+';\n'+readRepositoryFile('terrain_style_rendering.js','utf8')});
    if(u.pathname.endsWith('.js'))return route.fulfill({contentType:'application/javascript',body:readRepositoryFile(u.pathname.slice(1),'utf8')});
    if(u.pathname==='/api/world'){
     const profile=u.searchParams.get('world_profile'),settings=JSON.parse(u.searchParams.get('generation'));
     worldRequests.push({profile,settings});
     if(holdFirstWorld)await firstWorldGate;
     return route.fulfill({json:{seed:'42',version:'test',cache_profile:'test',world_profile:profile,generation_profile:profile,generation_schema:schema,generation_settings:settings,world_bounds:[-20e6,-10e6,20e6,10e6],overview_bounds:[-20e6,-10e6,20e6,10e6],overview:'/overview.png',gpu:'Mock'}});
    }
    if(u.pathname==='/api/generation/run')return route.fulfill({json:{world_profile:route.request().postDataJSON().profile,generation_settings:route.request().postDataJSON().settings}});
    if(u.pathname==='/api/view')return route.fulfill({json:{accepted:true}});
    if(u.pathname==='/api/status')return route.fulfill({json:{scheduler:{queued:0,computing:0,encoding:0}}});
    return route.fulfill({contentType:'image/png',body:png});
   });
   await page.goto('https://startup.test/');
   await page.locator('#gpuWelcome').waitFor({state:'visible'});
   assert.equal(worldRequests.length,0,'No world generation before the first GPU choice');
   assert.equal(await page.evaluate(()=>renderer),null,'Renderer also waits for the choice');
   const orogenButton=page.getByRole('button',{name:'Orogen',exact:true});
   assert.equal(await page.inputValue('#worldGenerator'),'orogen','Orogen is selected before onboarding');
   assert.equal(await page.inputValue('#generatorType'),'orogen');
   assert.equal(await orogenButton.getAttribute('aria-pressed'),'true','Menubar highlights Orogen immediately');
   assert.equal(await page.evaluate(()=>/orogent/i.test(document.documentElement.outerHTML)),false);
   const firstWorldRequest=page.waitForRequest(request=>new URL(request.url()).pathname==='/api/world');
   await page.locator('#gpuWelcome button[value='+(enabled?'yes':'no')+']').click();
   await firstWorldRequest;
   assert.equal(await page.inputValue('#worldGenerator'),'orogen','Menubar selection does not wait for the world response');
   assert.equal(await orogenButton.getAttribute('aria-pressed'),'true');
   assert.equal(await page.evaluate(()=>world),null,'Initial world request is still held');
   assert.equal(await page.inputValue('#reliefPipeline'),enabled?'orogen-gpu':'orogen','GPU choice is reflected while generating');
   holdFirstWorld=false;releaseFirstWorld();
   await page.waitForFunction(()=>world?.seed==='42'&&rendererReady);
   const initial=worldRequests[0];
   assert.equal(initial.profile,'orogen');
   assert.equal(initial.settings.orogen_detail,100000);
   assert.equal(initial.settings.orogen_continent_count,3);
   assert.equal(initial.settings.orogen_continent_variety,.85);
   assert.equal(initial.settings.relief_pipeline,'orogen');
   for(const key of gpuKeys)assert.equal(initial.settings[key],enabled,key+' is applied before generation');
   for(const id of ['gpuRender','coarseGpu'])assert.equal(await page.isChecked('#'+id),enabled);
   assert.equal(await page.inputValue('#generatorType'),'orogen');
   assert.equal(await page.inputValue('#reliefPipeline'),enabled?'orogen-gpu':'orogen');
   for(const key of gpuKeys.filter(key=>key!=='orogen_gpu_erosion'))assert.equal(await page.isChecked('#'+key),enabled);
   assert.equal(await page.getByRole('button',{name:'Orogen',exact:true}).getAttribute('aria-pressed'),'true');
   assert.equal(await page.evaluate(()=>document.body.textContent.includes('Tectonic')),false);

   // The prompt is one-time, and the complete settings are restored on a new visit.
   await page.goto('https://startup.test/');
   await page.waitForFunction(()=>world?.seed==='42');
   assert.equal(await page.locator('#gpuWelcome').count(),0);
   for(const key of gpuKeys)assert.equal(worldRequests.at(-1).settings[key],enabled);
   await page.locator('#generationPanel > summary').click();
   await page.locator('#orogen_detail').fill('123000');
   await page.locator('#orogen_gpu_relief').setChecked(!enabled);
   await page.locator('#tabErosion').click();
   await page.selectOption('#reliefPipeline',enabled?'orogen':'orogen-gpu');
   const stored=await page.evaluate(key=>JSON.parse(localStorage.getItem(key)),storageKey);
   assert.equal(stored.generation.settings.orogen_detail,123000);
   assert.equal(stored.generation.settings.orogen_gpu_relief,!enabled);
   assert.equal(stored.generation.settings.orogen_gpu_erosion,!enabled);
   assert.equal('orogen_relief_stage' in stored.generation.settings,false);
   await page.reload();
   await page.waitForFunction(()=>world?.seed==='42');
   assert.equal(worldRequests.at(-1).settings.orogen_detail,123000,'Reload preserves local drafts over the URL produced by this tab');
   assert.equal(worldRequests.at(-1).settings.orogen_gpu_relief,!enabled);
   assert.equal(worldRequests.at(-1).settings.orogen_gpu_erosion,!enabled);
   await page.goto('https://startup.test/');
   await page.waitForFunction(()=>world?.seed==='42');
   assert.equal(worldRequests.at(-1).settings.orogen_detail,123000);
   assert.equal(worldRequests.at(-1).settings.orogen_gpu_relief,!enabled);
   assert.equal(worldRequests.at(-1).settings.orogen_gpu_erosion,!enabled);
   assert.equal(await page.locator('#gpuWelcome').count(),0);

   // An explicit generation link still controls the imported world after onboarding.
   const linked={...schema.defaults_by_profile.natural,orogen_detail:20000};
   await page.goto('https://startup.test/?profile=natural&generation='+encodeURIComponent(JSON.stringify(linked)));
   await page.waitForFunction(()=>world?.seed==='42');
   assert.equal(worldRequests.at(-1).profile,'natural');
   assert.equal(worldRequests.at(-1).settings.orogen_detail,20000);
   assert.deepEqual(errors,[]);
   await context.close();
  }
  console.log('First-launch GPU choice, Orogen defaults and local preferences: OK');
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
