const {loadPlaywright, browserExecutable} = require('../platform.cjs');
const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {chromium}=loadPlaywright();
const assert=require('node:assert/strict');
const fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({executablePath:browserExecutable(),headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1100,height:760}}),errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.url().includes('/tiles/'))requests.push(r.url())});
  await page.goto('http://127.0.0.1:8765/?seed=42');
  await page.waitForFunction(()=>document.getElementById('viewport').dataset.loaded>0,{},{timeout:120000});
  await page.waitForFunction(()=>document.getElementById('viewport').dataset.pending==='0',{},{timeout:120000});
  await page.screenshot({path:'generated/island-v2/overview-ui.png'});
  const initialLOD=Number(await page.locator('#viewport').getAttribute('data-lod'));
  await page.locator('#plus').click();
  await page.waitForFunction(lod=>Number(document.getElementById('viewport').dataset.lod)<lod,initialLOD);
  await page.waitForFunction(()=>document.getElementById('viewport').dataset.pending==='0',{},{timeout:120000});
  assert(requests.some(url=>url.includes('/42/4/')));
  console.log('PASS: island overview and zoom automatically request finer tiles');
  // Small screen for intermediate resolution to keep the integration test focused.
  await page.setViewportSize({width:520,height:440});
  await page.locator('#plus').click();
  await page.waitForFunction(()=>document.getElementById('viewport').dataset.pending==='0'&&Number(document.getElementById('viewport').dataset.lod)===3,{},{timeout:180000});
  await page.screenshot({path:'generated/island-v2/latent-ui.png'});
  await page.locator('#native').click();
  await page.waitForFunction(()=>document.getElementById('viewport').dataset.pending==='0'&&Number(document.getElementById('viewport').dataset.lod)===0,{},{timeout:120000});
  await page.screenshot({path:'generated/island-v2/native-ui.png'});
  console.log('PASS: intermediate model and native 30m model load visible tiles');
  const before=requests.length;
  await page.mouse.move(250,300);await page.mouse.down();await page.mouse.move(480,300,{steps:10});await page.mouse.up();
  for(let i=0;i<6;i++)await page.locator('#viewport').press('ArrowLeft');
  await page.waitForFunction(()=>document.getElementById('viewport').dataset.pending==='0',{},{timeout:120000});
  assert(requests.length>before,'Pan should query new coordinates');
  await page.mouse.move(220,300);await page.mouse.wheel(0,-800);
  await page.waitForFunction(()=>document.getElementById('legend').textContent.includes('agrandissement'));
  assert.match(await page.locator('#info').innerText(),/Terrain 30 m/);
  await page.locator('#fit').click();
  await page.waitForFunction(()=>document.getElementById('viewport').dataset.pending==='0',{},{timeout:120000});
  await page.locator('#seed').fill('43');await page.locator('#apply').click();
  await page.waitForFunction(()=>document.getElementById('status').textContent.includes('Seed 43')&&document.getElementById('viewport').dataset.pending==='0'&&document.getElementById('viewport').dataset.loaded>0,{},{timeout:120000});
  await page.locator('#seed').fill('42');await page.locator('#apply').click();
  await page.waitForFunction(()=>document.getElementById('status').textContent.includes('Seed 42')&&document.getElementById('viewport').dataset.pending==='0'&&document.getElementById('viewport').dataset.loaded>0,{},{timeout:120000});
  const cached=await page.request.get('http://127.0.0.1:8765/tiles/42/0/0/0.png');
  assert.equal(cached.headers()['x-terrain-cache'],'hit');
  // Use a fresh interior coordinate to verify CUDA even when the UI revisits cached tiles.
  let probeX=13;while(fs.existsSync(`generated/island-v2/42/0/${probeX}_5.png`))probeX++;
  const probe=await page.request.get(`http://127.0.0.1:8765/tiles/42/0/${probeX}/5.png`,{timeout:120000});
  assert.equal(probe.status(),200);assert.equal(probe.headers()['x-terrain-stage'],'decoder');
  const status=await(await page.request.get('http://127.0.0.1:8765/api/status')).json();
  assert(Object.values(status.cuda_forward_calls).every(v=>v>0));
  assert.deepEqual(errors,[]);
  fs.writeFileSync('generated/island-v2/verification.json',JSON.stringify({passed:true,checks:['overview','zoom finer tiles','latent GPU','decoder GPU','native30m','pan new coordinates','cache reuse','seed switching','no JS errors'],requests:requests.length,status},null,2));
  console.log('PASS: pan, native resolution, cache, all CUDA models; no JavaScript errors',JSON.stringify(status));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
