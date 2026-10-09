const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const assert=require('node:assert/strict'),fs=require('node:fs');
const {chromium}=require('E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',args:['--disable-gpu']});
 try{
  const page=await browser.newPage(),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.route('http://stream-test.local/**',route=>route.fulfill({contentType:'text/html',body:'<div id="paneRendering"></div>'}));
  await page.goto('http://stream-test.local/?coarse_streams=8');
  await page.addScriptTag({content:fs.readFileSync('terrain_inference_controls.js','utf8')});
  await page.evaluate(()=>{
   window.calls=[];window.setting=4;window.rejectSetting=false;
   window.controls=TerrainInferenceControls.mount({fetchSettings:async(url,options)=>{
    calls.push({url,options});
    if(options&&!rejectSetting)setting=JSON.parse(options.body).coarse;
    return {ok:!rejectSetting,json:async()=>rejectSetting?{error:'Essai refusé'}:{coarse:setting}};
   },onChange:count=>{const url=new URL(location.href);url.searchParams.set('coarse_streams',count);history.replaceState(null,'',url);}});
  });
  await page.waitForFunction(()=>!document.querySelector('select').disabled);
  assert.equal(await page.locator('select').inputValue(),'8');
  assert.deepEqual(await page.locator('option').allTextContents(),['1','2','4','8','16']);
  await page.selectOption('select','16');
  await page.waitForFunction(()=>controls.get()===16);
  assert.equal(new URL(page.url()).searchParams.get('coarse_streams'),'16');
  await page.evaluate(()=>rejectSetting=true);await page.selectOption('select','2');
  await page.waitForFunction(()=>document.querySelector('[role=status]').textContent==='Essai refusé');
  assert.equal(await page.locator('select').inputValue(),'16');
  await page.evaluate(async()=>{rejectSetting=false;setting=1;await controls.sync();});
  assert.equal(await page.locator('select').inputValue(),'1');
  await page.evaluate(()=>controls.observe(4));
  assert.equal(await page.locator('select').inputValue(),'4');
  assert.deepEqual(errors,[]);
  // Load the real settings panel; intercept generation to keep this check read-only.
  await page.route('http://127.0.0.1:8765/api/world**',route=>route.fulfill({status:503,contentType:'application/json',body:'{"error":"UI inspection"}'}));
  await page.goto('http://127.0.0.1:8765/?coarse_prepare=0');
  await page.waitForSelector('#coarseStreams',{state:'attached'});
  await page.waitForFunction(()=>!document.querySelector('#coarseStreams').disabled);
  assert.equal(await page.locator('#coarseStreams').inputValue(),'4');
  console.log('Streams UI: query restoration, POST, rejection, backend sync, telemetry and live panel OK');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
