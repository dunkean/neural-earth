const {loadPlaywright,browserExecutable}=require('../../tools/platform.cjs');
const path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');process.chdir(root);
const {readRepositoryFile}=require(root+'/tools/repository-files.cjs');

(async()=>{
  const {chromium}=loadPlaywright();
  const browser=await chromium.launch({executablePath:browserExecutable(),headless:true,args:['--disable-gpu']});
  try{
    const page=await browser.newPage({viewport:{width:1280,height:900}}),changes=[],errors=[],transfers=[];
    page.on('pageerror',e=>errors.push(e.message));
    const options={coarse:[{id:'teacher',label:'Original'},{id:'solver8',label:'Coarse8 · 25k'},{id:'solver2',label:'Coarse2 · 200k'}],
      base:[{id:'teacher',label:'Original'},{id:'native3_base99m_v1',label:'Native3 · 99 Mo'}],decoder:[{id:'teacher',label:'Original'},{id:'distilled200k',label:'Décodeur64 · 200k'},{id:'compact10k',label:'Décodeur16 · 10k'}]};
    let state={options,selection:{coarse:'solver8',base:'teacher',decoder:'distilled200k'},revision:'r0',switching:false};
    const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM0LJrHwMDAxAAGAA7JAUW48M0QAAAAAElFTkSuQmCC','base64');
    await page.route('https://models.test/**',async route=>{
      const url=new URL(route.request().url());
      if(url.pathname==='/')return route.fulfill({contentType:'text/html',body:readRepositoryFile('index.html','utf8')});
      if(url.pathname==='/browser/')return route.fulfill({contentType:'text/html',body:'<p>Browser engine destination</p>'});
      if(url.pathname==='/api/runtime')return route.fulfill({json:{default:'python',options:[
        {id:'python',label:'Python / CUDA',url:'/',available:true},
        {id:'wasm',label:'Rust / WASM / WebGPU',url:'/browser/',available:true}]}});
      if(url.pathname==='/api/share'&&route.request().method()==='POST'){
        const snapshot=route.request().postDataJSON();transfers.push(snapshot);
        return route.fulfill({json:{generation:'gABCDEFGHIJKLMNOPQR',rendering:'rABCDEFGHIJKLMNOPQR',recipe:'portable'}});
      }
      if(url.pathname==='/terrain_styles.js')return route.fulfill({contentType:'application/javascript',body:'window.TerrainStyles='+readRepositoryFile('terrain_styles.json','utf8')+';\n'+readRepositoryFile('terrain_style_rendering.js','utf8')});
      if(url.pathname.endsWith('.js'))return route.fulfill({contentType:'application/javascript',body:readRepositoryFile(url.pathname.slice(1),'utf8')});
      if(url.pathname==='/api/models'){
        if(route.request().method()==='POST'){
          const data=route.request().postDataJSON();changes.push(data);state={...state,selection:data.selection,revision:'r'+changes.length};
          return route.fulfill({status:202,json:{restarting:true,revision:state.revision}});
        }
        return route.fulfill({json:state});
      }
      if(url.pathname==='/api/world')return route.fulfill({json:{version:'natural-v1',cache_profile:state.revision,world_identity:state.revision,
        world_profile:'natural',generation_profile:'natural',seed:url.searchParams.get('seed'),world_bounds:[-2e6,-1e6,2e6,1e6],
        world_topology:'sphere',world_diameter_km:40000/Math.PI,
        overview_bounds:[-2e6,-1e6,2e6,1e6],overview:'/overview.png',gpu:'GPU fixture'}});
      if(url.pathname==='/api/status')return route.fulfill({json:{scheduler:{queued:0,computing:0,encoding:0}}});
      if(url.pathname==='/api/gpu')return route.fulfill({json:{settings:{mode:'single',devices:'auto'},devices:[],
        plan:{mode:'single',devices:[0],requested_mode:'single'},environment:[],restart_required:false}});
      if(url.pathname==='/api/inference/streams')return route.fulfill({json:{coarse:4}});
      if(url.pathname.startsWith('/api/'))return route.fulfill({json:{accepted:true}});
      return route.fulfill({contentType:'image/png',body:png});
    });
    await page.goto('https://models.test/?seed=42&profile=natural&coarse_gpu=0&gpu=0');
    await page.waitForFunction(()=>terrainDebug.snapshot().world&&document.querySelectorAll('#distillCoarse option').length===3);
    await page.locator('#distillPanel').evaluate(el=>el.open=true);
    assert.deepEqual(await page.locator('#distillCoarse option').evaluateAll(els=>els.map(e=>e.value)),['teacher','solver8','solver2']);
    assert.deepEqual(await page.locator('#distillBase option').evaluateAll(els=>els.map(e=>e.value)),['teacher','native3_base99m_v1']);
    assert.equal(await page.inputValue('#distillBase'),'teacher');
    assert.equal(await page.inputValue('#inferenceRuntime'),'python');
    assert.deepEqual(await page.locator('#distillDecoder option').evaluateAll(els=>els.map(e=>e.value)),['teacher','distilled200k','compact10k']);
    assert.equal(await page.locator('#distillModels a').getAttribute('href'),'/tile40/');
    await page.evaluate(()=>{cx=123;cy=456;mpp=960;draw()});
    await page.selectOption('#distillCoarse','solver2');await page.selectOption('#distillDecoder','compact10k');
    await page.click('#distillApply');
    await page.waitForFunction(()=>document.getElementById('distillStatus').textContent.startsWith('NN appliqués'));
    assert.deepEqual(changes[0],{selection:{coarse:'solver2',base:'teacher',decoder:'compact10k'},reset:false});
    assert.deepEqual(await page.evaluate(()=>({cx,cy,mpp,seed:world.seed})),{cx:123,cy:456,mpp:960,seed:'42'});
    await page.click('#distillReset');
    await page.waitForFunction(()=>document.getElementById('distillStatus').textContent.startsWith('Cache NN réinitialisé'));
    assert.equal(changes[1].reset,true);
    assert.deepEqual(changes[1].selection,changes[0].selection);
    assert.deepEqual(errors,[]);
    await page.screenshot({path:'tmp/selected-models-ui.png'});
    await page.selectOption('#distillBase','native3_base99m_v1');
    await page.click('#distillApply');
    await page.waitForFunction(()=>document.getElementById('distillStatus').textContent.startsWith('NN appliqués'));
    assert.equal(changes[2].selection.base,'native3_base99m_v1');
    // Reject unsupported worlds without navigating or dropping their settings.
    await page.selectOption('#inferenceRuntime','wasm');await page.click('#applyRuntime');
    assert.match(await page.locator('#runtimeStatus').textContent(),/planètes Orogen/);
    await page.evaluate(()=>{
      world.world_profile='orogen';world.generation_profile='orogen';
      world.generation_settings={...generationControls.defaults('orogen'),world_topology:'sphere',height_source:'orogen',climate_source:'orogen',relief_pipeline:'orogen'};
      viewMode='relief';cx=123;cy=456;mpp=960;
    });
    await page.click('#applyRuntime');await page.waitForURL('https://models.test/browser/**');
    const applied=transfers.at(-1);
    assert.equal(applied.generation.seed,'42');
    assert.equal(applied.generation.settings.height_source,'orogen');
    assert.equal(applied.rendering.gpuRender,false);
    const destination=new URL(page.url());
    assert.equal(destination.searchParams.get('position'),'123,456');
    assert.equal(destination.searchParams.get('zoom'),'960');
    assert.equal(destination.hash,'#recipe=portable');
    assert.deepEqual(errors,[]);
    console.log('Curated model menus, finite-tile link, apply/restart and reset preserve seed/camera: OK');
  }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
