// Real local HTTP server and renderer; stays at world scale to bound NN work.
const {chromium}=require('E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
(async()=>{
 const root='E:/TerrainDiffusionRuntime/terrestrial-bootstrap-runtime/',harness=fs.readFileSync(__filename);
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const errors=[],requests=[],worldResponses=[];
 try{
  const page=await browser.newPage({viewport:{width:1400,height:950}});
  page.on('pageerror',error=>errors.push(error.message));
  page.on('request',request=>requests.push(request.url()));
  page.on('response',async response=>{if(new URL(response.url()).pathname==='/api/world')worldResponses.push({status:response.status(),body:await response.json()});});
  const snap=()=>page.evaluate(()=>terrainDebug.snapshot());
  const settled=()=>page.waitForFunction(()=>{const s=terrainDebug.snapshot();return !!s.world&&s.visible.wanted>0&&s.visible.pending===0;},null,{timeout:120000});
  await page.goto('http://127.0.0.1:8765/?seed=42&profile=natural');await settled();
  await page.locator('#generationPanel > summary').click();await page.waitForTimeout(300);
  const before=await snap(),worldCount=worldResponses.length;
  await page.locator('#generationPanel').evaluate(el=>el.open=true);await page.locator('#generationChannel').selectOption('0');await page.locator('#snr0').fill('0.2');await page.waitForTimeout(300);
  assert.equal(worldResponses.length,worldCount);
  await page.locator('#applyGeneration').click();
  await page.waitForFunction(()=>terrainDebug.snapshot().generationSettings?.cond_snr[0]===.2);await settled();
  const noise=await snap();assert.match(noise.generationProfile,/^natural--g[0-9a-f]{24}$/);
  assert.deepEqual(noise.camera,before.camera);
  await page.locator('#saveGenerationA').click();
  await page.locator('#generationPreset').selectOption('continental');
  await page.locator('#applyGeneration').click();
  await page.waitForFunction(()=>terrainDebug.snapshot().generationSettings?.height_source==='natural-continental');await settled();
  const continental=await snap();assert.notEqual(continental.generationProfile,noise.generationProfile);
  assert.deepEqual(continental.camera,before.camera);
  assert(!(await page.locator('#info').textContent()).includes('Original natural profile'));
  assert((await page.locator('#worldDescription').innerText()).includes('experimental'));
  assert.equal(await page.locator('#macroScale').isDisabled(),false);
  const shareURL=page.url();
  await page.screenshot({path:root+'generation-controls-live-ui.png'});
  await page.locator('#switchGenerationAB').click();
  await page.waitForFunction(token=>terrainDebug.snapshot().generationProfile===token,noise.generationProfile);await settled();
  await page.locator('#switchGenerationAB').click();
  await page.waitForFunction(token=>terrainDebug.snapshot().generationProfile===token,continental.generationProfile);await settled();
  await page.reload();
  await page.waitForFunction(token=>terrainDebug.snapshot().generationProfile===token,continental.generationProfile);await settled();
  assert.deepEqual((await snap()).generationSettings,continental.generationSettings);
  await page.locator('#generationPanel > summary').click();
  await page.locator('#resetGeneration').click();
  await page.waitForFunction(()=>terrainDebug.snapshot().generationProfile==='natural');await settled();
  assert.deepEqual(errors,[]);assert(worldResponses.every(row=>row.status===200));
  assert(requests.some(url=>new URL(url).pathname.startsWith('/height/')&&new URL(url).searchParams.get('world_profile')===continental.generationProfile));
  const checks={draftDoesNotRequestWorld:true,realSettingsApplied:true,cameraPreserved:true,accurateCustomLabels:true,compareAB:true,reload:true,reset:true,noPageErrors:true,parameterizedHeightRequest:true};
  const report={passed:true,checks,browser:browser.version(),transport:'actual local HTTP server',nnScope:'world-scale source previews; separate CUDA runtime proof covers LOD3/LOD2',
    backend:continental.backend,shareURL,noise,continental,errors,worldResponses:worldResponses.map(row=>({status:row.status,generationProfile:row.body.generation_profile,worldIdentity:row.body.world_identity})),
    harnessSha256:crypto.createHash('sha256').update(harness).digest('hex'),harnessUnchanged:fs.readFileSync(__filename).equals(harness)};
  fs.writeFileSync(root+'generation-controls-live-browser.json',JSON.stringify(report,null,2));
  fs.writeFileSync(root+'generation-controls-live-harness.cjs',harness);
  console.log(JSON.stringify({passed:true,checks,backend:report.backend,report:root+'generation-controls-live-browser.json'}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
