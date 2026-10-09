const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {readRepositoryFile} = require(NEURAL_EARTH_ROOT + '/tools/repository-files.cjs');
// Browser UX tests use mocked geography/transport and a fake renderer.
// No neural inference or hardware GPU work takes place.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto'),path=require('node:path');
const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM0LJrHwMDAxAAGAA7JAUW48M0QAAAAAElFTkSuQmCC','base64');
const physical=new Float32Array(304*304+5*33*33);
physical.fill(1000,0,304*304);
for(let plane=0;plane<5;plane++)physical.fill([20,1500,1000,50,-.0065][plane],304*304+plane*33*33,304*304+(plane+1)*33*33);
const heightBody=Buffer.from(physical.buffer),html=readRepositoryFile('index.html','utf8'),lodScript=readRepositoryFile('terrain_lod.js','utf8'),controlsScript=readRepositoryFile('terrain_generation_controls.js','utf8');
const gpuStub=`window.fakeUploads=[];window.createTerrainRenderer=async(canvas,options)=>{await new Promise(r=>window.resolveGPUStub=r);const tiles=new Map();options.onStatus?.('webgpu');return{available:true,status:'webgpu',uploadTile(k,h,o){if(o.climate?.length!==5*33*33)throw Error('Climate packet missing');window.fakeUploads.push({key:k,mode:o.mode,climateWidth:o.climateWidth,climateLength:o.climate.length});tiles.set(k,h);return true},hasTile(k){return tiles.has(k)},deleteTile(k){tiles.delete(k)},clear(){tiles.clear()},draw(rects){window.fakeDraw=rects;this.lastDrawnRects=rects.filter(r=>tiles.has(r.key));return true},getStats(){return{status:'webgpu',tiles:tiles.size,bytes:tiles.size*653588}}}}`;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function settled(page,after=-1){
  await page.waitForFunction(epoch=>{const s=terrainDebug.snapshot();return s.cameraEpoch>epoch&&s.visible.pending===0&&s.visible.lod===s.camera.lod},after);
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
}
(async()=>{
  const reportPath=process.env.NAVIGATION_MOCK_REPORT||'E:/TerrainDiffusionRuntime/terrestrial-bootstrap-runtime/generation-controls-browser.json';
  fs.mkdirSync(path.dirname(reportPath),{recursive:true});
  const harness=readRepositoryFile(__filename),harnessSnapshot=reportPath.replace(/\.json$/, '-harness.cjs');
  fs.writeFileSync(harnessSnapshot,harness);
  const sha=value=>crypto.createHash('sha256').update(value).digest('hex');
  const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--disable-gpu']});
  try{
    const page=await browser.newPage({viewport:{width:5760,height:3240},locale:'fr-FR'}),errors=[],requests=[],epochs=[];
    let heightDelay=0,rejectNextWorld=false;
    page.on('pageerror',e=>errors.push(e.message));
    await page.route('https://navigation.test/**',async route=>{
      const u=new URL(route.request().url());requests.push({path:u.pathname,query:Object.fromEntries(u.searchParams),method:route.request().method(),time:Date.now()});
      if(u.pathname==='/')return route.fulfill({contentType:'text/html',body:html});
      if(u.pathname==='/terrain_renderer.js')return route.fulfill({contentType:'application/javascript',body:gpuStub});
      if(u.pathname==='/terrain_lod.js')return route.fulfill({contentType:'application/javascript',body:lodScript});
      if(u.pathname==='/terrain_map_tools.js')return route.fulfill({contentType:'application/javascript',body:readRepositoryFile('terrain_map_tools.js','utf8')});
      if(u.pathname==='/terrain_generation_controls.js')return route.fulfill({contentType:'application/javascript',body:controlsScript});
      if(u.pathname==='/api/world'){
        if(rejectNextWorld){rejectNextWorld=false;return route.fulfill({status:400,json:{error:'Settings rejected for this test'}});}
        const profile=u.searchParams.get('world_profile'),settings=u.searchParams.has('generation')?JSON.parse(u.searchParams.get('generation')):null,
          internal=settings?profile+'--g'+crypto.createHash('sha256').update(JSON.stringify(settings)).digest('hex').slice(0,24):profile,
          b=profile==='terrestrial-earthlike'||settings?[-20e6,-10e6,20e6,10e6]:[-500000,-250000,500000,250000];
        return route.fulfill({json:{version:'natural-v1',cache_profile:'mock',generation_profile:internal,world_profile:profile,generation_settings:settings,world_identity:crypto.createHash('sha256').update(internal+u.searchParams.get('seed')).digest('hex'),world_version:'continental-test-v1',seed:u.searchParams.get('seed'),tile_size:256,native_resolution:30,world_bounds:b,initial_bounds:profile==='terrestrial-earthlike'||settings?b:[-15000,-10000,15000,10000],initial_image:profile==='natural'&&!settings?'/generated/terrain.png':null,overview_bounds:b,overview:'/api/overview/mock.png',preview_min_lod:7,gpu:'Fake GPU',navigation_protocol:1}});
      }
      if(u.pathname==='/api/view'){const body=route.request().postDataJSON();requests.at(-1).body=body;epochs.push(body.epoch);assert(['terrestrial-earthlike','natural'].includes(body.world_profile.split('--g')[0]));return route.fulfill({json:{accepted:true,epoch:body.epoch}})}
      if(u.pathname==='/api/view/release')return route.fulfill({json:{released:true}});
      if(u.pathname==='/api/coarse/prepare'){
        assert.equal(route.request().method(),'POST');
        requests.at(-1).body=route.request().postDataJSON();
        return route.fulfill({json:{state:'running',seed:requests.at(-1).body.seed,world_profile:requests.at(-1).body.world_profile,prepared:0,total:8192}});
      }
      if(u.pathname==='/api/status')return route.fulfill({json:{scheduler:{queued:0,computing:0,encoding:0}}});
      if(u.pathname.startsWith('/height/')){
        const lod=Number(u.pathname.split('/')[4]),stage=u.searchParams.get('world_profile')==='terrestrial-earthlike'&&lod>=7?'conditioning-preview':lod>=4?'coarse':lod===3?'latent':'decoder';
        if(heightDelay)await sleep(heightDelay);
        return route.fulfill({contentType:'application/octet-stream',body:heightBody,headers:{'X-Terrain-Width':'304','X-Terrain-Halo':'24','X-Terrain-Climate-Width':'33','X-Terrain-Climate-Height':'33','X-Terrain-Resolution':String(30*2**lod),'X-Terrain-Source-Resolution':String(stage==='coarse'?7680:stage==='latent'?240:stage==='decoder'?30:30*2**lod),'X-Terrain-Stage':stage,'X-Terrain-Cache':'hit'}}).catch(()=>{});
      }
      return route.fulfill({contentType:'image/png',body:png});
    });
    const preparations=()=>requests.filter(r=>r.path==='/api/coarse/prepare');
    // Exercise real Chrome startup/random/button event flow with mocked API IO.
    await page.goto('https://navigation.test/?seed=42&profile=terrestrial-earthlike&mode=biomes');
    await page.waitForFunction(()=>window.terrainDebug?.snapshot().world==='42'&&terrainDebug.snapshot().overview);
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    assert.equal(preparations().length,0,'default startup must not enqueue the entire learned planet');
    await page.locator('#random').click();await page.locator('#apply').click();
    await page.waitForFunction(()=>terrainDebug.snapshot().world!=='42'&&terrainDebug.snapshot().overview);
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    const randomSeed=await page.evaluate(()=>terrainDebug.snapshot().world);
    assert.equal(preparations().length,0,'default random-seed navigation must not enqueue an entire planet');
    await page.locator('#runtimePanel').evaluate(el=>el.open=true);
    await Promise.all([page.waitForResponse(r=>new URL(r.url()).pathname==='/api/coarse/prepare'),page.locator('#prepareWorld').click()]);
    assert.equal(preparations().length,1,'explicit preparation button must submit one request');
    assert.deepEqual(preparations()[0].body,{seed:randomSeed,world_profile:'terrestrial-earthlike',max_windows:8192});
    const preparationPolicy={defaultStartupPosts:0,defaultRandomSeedPosts:0,randomSeed,explicitButtonRequest:preparations()[0].body};
    // The existing navigation scenario explicitly opts into automatic work.
    epochs.length=0;
    const automaticResponse=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/coarse/prepare');
    await page.goto('https://navigation.test/?seed=42&profile=terrestrial-earthlike&mode=biomes&prepare=1');
    await page.waitForFunction(()=>window.terrainDebug?.snapshot().world==='42'&&terrainDebug.snapshot().overview);
    await automaticResponse;
    assert.equal(preparations()[1].body.seed,'42');
    preparationPolicy.explicitQueryStartsAutomaticPreparation=true;
    assert(!requests.some(r=>r.path.startsWith('/height/')),'NN format requests must wait for renderer boot');
    await page.evaluate(()=>window.resolveGPUStub());
    await page.waitForFunction(()=>window.terrainDebug?.snapshot().rendererReady&&terrainDebug.snapshot().world==='42');await settled(page);
    let state=await page.evaluate(()=>terrainDebug.snapshot());
    assert.equal(state.profile,'terrestrial-earthlike');assert.equal(state.mode,'biomes');assert.deepEqual(state.worldBounds,[-20e6,-10e6,20e6,10e6]);assert.equal(state.camera.latitude,0);
    assert.equal(state.backend,'webgpu');assert(state.camera.mpp>6000,'default fit should show the global extent');assert.equal(state.rendered.hasFineHistory,false);
    assert.equal(await page.locator('#legend').isVisible(),false,'preview descriptions must not cover the map');
    assert.equal(await page.locator('#lodIndicator').textContent(),'LOD '+state.camera.lod);
    assert((await page.locator('header').innerText()).includes('Procedural World'));assert(!html.includes('Ã')&&!html.includes('Â'),'UTF-8 source must remain readable');
    assert(requests.find(r=>r.path.startsWith('/api/overview/')).time<requests.find(r=>r.path.startsWith('/height/')).time,'macro preview must not wait for GPU initialization');
    await page.locator('#renderPanel').evaluate(el=>el.open=true);await page.locator('#prefetch').uncheck();let previous=state.cameraEpoch;await page.locator('#native').click();await settled(page,previous);
    state=await page.evaluate(()=>terrainDebug.snapshot());const nativeState=state;
    assert.equal(state.camera.mpp,30);assert(state.camera.lod>state.camera.nominalLod,'wide viewport must adapt to cache budget');assert(state.cache.bytes<=state.cache.budget);
    const before=state;await page.evaluate(()=>{const view=document.getElementById('viewport');for(let i=0;i<30;i++)view.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}))});
    const instant=await page.evaluate(()=>terrainDebug.snapshot());assert.equal(instant.camera.cx,before.camera.cx+90000);assert(instant.counters.draws-before.counters.draws<=2);await settled(page,before.cameraEpoch);
    previous=(await page.evaluate(()=>terrainDebug.snapshot())).cameraEpoch;heightDelay=200;await page.locator('#fit').click();
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    state=await page.evaluate(()=>terrainDebug.snapshot());assert.equal(state.rendered.hasFineHistory,false,'cached native tiles must immediately disappear on zoom out');heightDelay=0;await settled(page,previous);
    for(const mode of ['relief','temperature','precipitation','biomes']){
      previous=(await page.evaluate(()=>terrainDebug.snapshot())).cameraEpoch;await page.locator('#renderPanel').evaluate(el=>el.open=true);await page.locator('#mapMode').selectOption(mode);await settled(page,previous);
      state=await page.evaluate(()=>terrainDebug.snapshot());assert.equal(state.mode,mode);assert(state.cache.keys.every(k=>k.includes('/'+mode+'/')));assert.equal(new URL(page.url()).searchParams.get('mode'),mode);
    }
    const heightBefore=requests.filter(r=>r.path.startsWith('/height/')).length;
    previous=state.cameraEpoch;await page.locator('#renderPanel').evaluate(el=>el.open=true);await page.locator('#gpuRender').uncheck();await settled(page,previous);assert(requests.some(r=>r.path.startsWith('/tiles/')));
    previous=(await page.evaluate(()=>terrainDebug.snapshot())).cameraEpoch;await page.locator('#renderPanel').evaluate(el=>el.open=true);await page.locator('#gpuRender').check();await settled(page,previous);
    assert(requests.filter(r=>r.path.startsWith('/height/')).length>heightBefore);
    previous=(await page.evaluate(()=>terrainDebug.snapshot())).cameraEpoch;await page.locator('#worldGenerator').selectOption('natural');await page.locator('#apply').click();await page.waitForFunction(()=>terrainDebug.snapshot().profile==='natural');await settled(page,previous);
    state=await page.evaluate(()=>terrainDebug.snapshot());assert.equal(state.world,'42');assert.equal(state.camera.latitude,null);assert(state.cache.keys.every(k=>k.startsWith('exact/natural/')));assert.equal(new URL(page.url()).searchParams.get('profile'),'natural');
    previous=state.cameraEpoch;await page.locator('#worldGenerator').selectOption('custom');await page.locator('#apply').click();await page.waitForFunction(()=>terrainDebug.snapshot().profile==='terrestrial-earthlike');await settled(page,previous);
    state=await page.evaluate(()=>terrainDebug.snapshot());assert(state.cache.keys.every(k=>k.startsWith('exact/terrestrial-earthlike/')));assert.equal(state.camera.latitude,0);
    await page.locator('#native').click();await settled(page);await page.locator('#mini').click({position:{x:110,y:1}});await page.evaluate(()=>new Promise(r=>requestAnimationFrame(r)));state=await page.evaluate(()=>terrainDebug.snapshot());assert(state.camera.latitude>85,'global mini map must navigate to the north pole');
    previous=state.cameraEpoch;await page.locator('#seed').fill('98765');await page.locator('#apply').click();await page.waitForFunction(()=>terrainDebug.snapshot().world==='98765');await settled(page,previous);
    state=await page.evaluate(()=>terrainDebug.snapshot());assert(state.cache.bytes<=state.cache.budget);assert.deepEqual(errors,[]);assert(epochs.every((e,i)=>i===0||e>=epochs[i-1]));
    assert(requests.filter(r=>r.path.startsWith('/height/')).every(r=>r.query.climate==='1'&&['terrestrial-earthlike','natural'].includes(r.query.world_profile.split('--g')[0])));
    // Parameters are drafts until Apply; identity transport follows the admitted token.
    await page.goto('https://navigation.test/?seed=42&profile=terrestrial-earthlike&coarse_prepare=0');
    await page.waitForFunction(()=>terrainDebug.snapshot().overview);
    await page.evaluate(()=>window.resolveGPUStub());await settled(page);
    await page.locator('#generationPanel > summary').click();await settled(page);
    assert(await page.locator('#frequency0').isDisabled());assert(await page.locator('#octaves0').isDisabled());assert(await page.locator('#dropWater').isDisabled());
    assert(await page.locator('#continentalStrength').isDisabled());assert(await page.locator('#macroScale').isDisabled());assert(await page.locator('#continentalStyle').isEnabled());
    for(let i=1;i<5;i++){assert(await page.locator('#frequency'+i).isEnabled());assert(await page.locator('#octaves'+i).isEnabled());}
    await page.locator('#native').click();await settled(page);
    const cameraBefore=await page.evaluate(()=>terrainDebug.snapshot().camera);
    const worldsBefore=requests.filter(r=>r.path==='/api/world').length,prepBefore=preparations().length;
    await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#generationPreset').selectOption('continental');
    assert(await page.locator('#frequency0').isEnabled());assert(await page.locator('#octaves0').isEnabled());assert(await page.locator('#dropWater').isEnabled());
    assert(await page.locator('#continentalStrength').isEnabled());assert(await page.locator('#macroScale').isEnabled());
    assert.equal(await page.locator('html').getAttribute('lang'),'en');
    assert.equal(await page.locator('#dropWater').inputValue(),'0.5');
    assert((await page.locator('#worldDescription').innerText()).includes('40,000 × 20,000'));
    await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#sourceChannel').selectOption('0');await page.locator('#frequency0').fill('25.125');await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#sourceChannel').selectOption('1');await page.locator('#frequency1').fill('1e-6');
    await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#sourceChannel').selectOption('2');await page.locator('#frequency2').fill('0');await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#sourceChannel').selectOption('3');await page.locator('#frequency3').fill('-2.5');
    await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#sourceChannel').selectOption('0');await page.locator('#octaves0').fill('12');await page.locator('#macroScale').fill('75.25');
    await page.locator('#continentalStrength').fill('1.23456789');
    await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#generationChannel').selectOption('1');await page.locator('#snr1').fill('4.5');await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#generationChannel').selectOption('2');await page.locator('#snr2').fill('0.0001');
    assert.equal(requests.filter(r=>r.path==='/api/world').length,worldsBefore,'draft edits must not create worlds');
    assert.equal(preparations().length,prepBefore,'draft edits must not prepare an entire planet');
    for(const invalid of ['', '0,5', 'Infinity', 'NaN', '0x10']){
      await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#sourceChannel').selectOption('0');await page.locator('#frequency0').fill(invalid);await page.locator('#applyGeneration').click();
      assert((await page.locator('#generationDraftStatus').innerText()).startsWith('Frequency · Elevation:'));
      assert.equal(requests.filter(r=>r.path==='/api/world').length,worldsBefore,'invalid drafts must not reach the API');
    }
    await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#sourceChannel').selectOption('0');await page.locator('#frequency0').fill('25.125');
    const customStart=requests.length;
    await page.locator('#applyGeneration').click();
    await page.waitForFunction(()=>terrainDebug.snapshot().generationProfile?.includes('--g'));await settled(page);
    let custom=await page.evaluate(()=>terrainDebug.snapshot()),settings=custom.generationSettings;
    assert.equal(custom.profile,'natural');assert.equal(settings.height_source,'natural-continental');assert.equal(settings.continental_strength,1.23456789);
    assert.deepEqual(settings.cond_snr,[.5,4.5,.0001,.5,.5]);assert.equal(settings.macro_scale_km,75.25);
    assert.equal(settings.frequency_mult[0],25.125);assert.equal(settings.frequency_mult[1],1e-6);assert.equal(settings.octaves[0],12);
    assert.equal(settings.frequency_mult[2],0);assert.equal(settings.frequency_mult[3],-2.5);
    assert.equal(await page.locator('#continentalStrength').inputValue(),'1.23456789');
    assert.equal(await page.locator('#frequency0').inputValue(),'25.125');
    assert.deepEqual({cx:custom.camera.cx,cy:custom.camera.cy,mpp:custom.camera.mpp},{cx:cameraBefore.cx,cy:cameraBefore.cy,mpp:cameraBefore.mpp});
    assert.deepEqual(JSON.parse(new URL(page.url()).searchParams.get('generation')),settings);
    const tokenA=custom.generationProfile;
    for(const selector of ['#worldDescription','#status','#info']){const label=await page.locator(selector).textContent();assert(label.includes('Natural continental'));assert(label.includes('experimental'));assert(!label.includes('Original natural profile'));assert(!label.includes('NN reference'));}
    assert(requests.slice(customStart).filter(r=>r.path.startsWith('/height/')).every(r=>r.query.world_profile===tokenA));
    assert(requests.slice(customStart).filter(r=>r.path.startsWith('/api/overview/')).every(r=>r.query.world_profile===tokenA));
    assert(requests.slice(customStart).filter(r=>r.path==='/api/view').every(r=>r.body.world_profile===tokenA));
    await page.locator('#runtimePanel').evaluate(el=>el.open=true);
    await Promise.all([page.waitForResponse(r=>new URL(r.url()).pathname==='/api/coarse/prepare'),page.locator('#prepareWorld').click()]);
    assert.equal(preparations().at(-1).body.world_profile,tokenA);
    await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#saveGenerationA').click();
    await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#generationChannel').selectOption('0');await page.locator('#snr0').fill('.1');await page.locator('#applyGeneration').click();
    await page.waitForFunction(old=>terrainDebug.snapshot().generationProfile!==old,tokenA);await settled(page);
    const tokenB=await page.evaluate(()=>terrainDebug.snapshot().generationProfile);assert.notEqual(tokenB,tokenA);
    rejectNextWorld=true;await page.locator('#switchGenerationAB').click();
    await page.waitForFunction(()=>document.getElementById('generationDraftStatus').textContent==='Settings rejected for this test');
    assert.equal(await page.locator('#legend').isVisible(),true);
    assert.equal(await page.locator('#legend').textContent(),'Settings rejected for this test');
    assert.equal(await page.locator('#switchGenerationAB').innerText(),'Show A','failed switch must not advance A/B state');
    assert.equal(await page.evaluate(()=>terrainDebug.snapshot().generationSettings.cond_snr[0]),.1,'failed admission must retain applied settings');
    await page.locator('#switchGenerationAB').click();await page.waitForFunction(old=>terrainDebug.snapshot().generationProfile===old,tokenA);await settled(page);
    await page.locator('#switchGenerationAB').click();await page.waitForFunction(old=>terrainDebug.snapshot().generationProfile===old,tokenB);await settled(page);
    const shareURL=page.url();settings=await page.evaluate(()=>terrainDebug.snapshot().generationSettings);
    await page.reload();await page.waitForFunction(()=>terrainDebug.snapshot().overview);await page.evaluate(()=>window.resolveGPUStub());await settled(page);
    assert.equal(await page.evaluate(()=>terrainDebug.snapshot().generationProfile),tokenB);
    assert.deepEqual(await page.evaluate(()=>terrainDebug.snapshot().generationSettings),settings);
    const requestsBeforeRandom=requests.length;
    await page.locator('#random').click();await page.locator('#apply').click();await page.waitForFunction(()=>terrainDebug.snapshot().world!=='42');await settled(page);
    assert.deepEqual(await page.evaluate(()=>terrainDebug.snapshot().generationSettings),settings);
    assert.equal(await page.evaluate(()=>terrainDebug.snapshot().generationProfile),tokenB);
    assert(requests.slice(requestsBeforeRandom).filter(r=>r.path==='/api/world').every(r=>JSON.parse(r.query.generation).cond_snr[0]===.1));
    await page.locator('#generationPanel > summary').click();
    await page.locator('#resetGeneration').click();await page.waitForFunction(()=>terrainDebug.snapshot().generationProfile==='natural');await settled(page);
    assert.equal(new URL(page.url()).searchParams.get('generation'),null);
    assert.equal(await page.locator('#heightSource').inputValue(),'natural');
    assert((await page.locator('#worldDescription').innerText()).includes('Natural · NN reference'));
    assert((await page.locator('#info').textContent()).includes('Original natural profile'));
    assert(await page.locator('#continentalStyle').isDisabled());assert(await page.locator('#macroScale').isDisabled());assert(await page.locator('#frequency0').isEnabled());
    assert.deepEqual(errors,[]);
    await page.setViewportSize({width:1400,height:950});await settled(page);
    await page.locator('.generationBody').evaluate(element=>{element.scrollTop=0});
    const controlsScreenshot=reportPath.replace(/\.json$/,'-ui.png');await page.screenshot({path:controlsScreenshot});
    const generationControls={draftDoesNotGenerate:true,inactiveControlsDisabled:true,failedApplyShowsError:true,failedCompareKeepsState:true,sourcePresetTransport:true,customLabelsDistinguishReference:true,allDownstreamUseInternalIdentity:true,cameraPreserved:true,settingsChangeIdentity:true,compareAB:true,urlReloadReproducesSettings:true,seedRetainsSettings:true,resetRestoresBaseline:true,shareURL};
    const report={passed:true,browser:browser.version(),gpuUsed:false,nnInferencePerformed:false,api:'fully mocked',preparationPolicy,
      generationControls,controlsScreenshot,controlsSourceSha256:sha(readRepositoryFile('terrain_generation_controls.js')),
      htmlSha256:sha(readRepositoryFile('index.html')),harnessSha256:sha(harness),harnessSnapshot,
      htmlBeforePolicySha256:sha(readRepositoryFile('E:/TerrainDiffusionRuntime/terrestrial-bootstrap-runtime/index-before-manual-world-preparation.html')),
      harnessUnchangedDuringRun:sha(readRepositoryFile(__filename))===sha(harness),
      wideViewport:[5760,3240],native:{camera:nativeState.camera,cacheBytes:nativeState.cache.bytes},final:{camera:state.camera,profile:state.profile,mode:state.mode,rendered:state.rendered},requests:requests.length,cameraPosts:epochs.length};
    fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report,null,2));
  }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exit(1)});
