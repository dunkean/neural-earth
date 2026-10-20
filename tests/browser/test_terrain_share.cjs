const path=require('node:path'),fs=require('node:fs'),os=require('node:os'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const root=path.resolve(__dirname,'../..');process.chdir(root);
const {readRepositoryFile}=require(root+'/tools/repository-files.cjs');
const {loadPlaywright,browserExecutable,pythonExecutable}=require(root+'/tools/platform.cjs');
const {chromium}=loadPlaywright();
const schema=JSON.parse(execFileSync(pythonExecutable(),['-c','import json; from terrain_generation import generator_schema; print(json.dumps(generator_schema()))'],{encoding:'utf8'}));
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'terrain-share-'));
const api=`import sys,json
from flask import Flask
from terrain_share import register_share_routes
app=Flask(__name__)
register_share_routes(app,sys.argv[1])
with app.test_client() as client:
 response=client.open(sys.argv[3],method=sys.argv[2],data=sys.stdin.read(),content_type='application/json')
 print(json.dumps(dict(status=response.status_code,body=response.json)))`;
const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM0LJrHwMDAxAAGAA7JAUW48M0QAAAAAElFTkSuQmCC','base64');
(async()=>{
 const browser=await chromium.launch({executablePath:browserExecutable(),headless:true});
 try{
  const errors=[],requests=[];
  async function create(origin,size,gpu){
   const context=await browser.newContext({viewport:size}),page=await context.newPage();
   page.on('pageerror',error=>errors.push(error.message));
   await context.addInitScript(enabled=>localStorage.setItem('neural-earth-preferences-v1',JSON.stringify({gpuAcceleration:enabled,erosionDefaultsVersion:1,rendering:{gpuRender:enabled,coarseGpu:enabled},generation:{profile:'orogen',settings:{orogen_detail:99999}}})),gpu);
   await page.route(origin+'/**',async route=>{
    const u=new URL(route.request().url());
    if(u.pathname.startsWith('/api/share')){
     const result=JSON.parse(execFileSync(pythonExecutable(),['-c',api,path.join(temporary,u.hostname),route.request().method(),u.pathname],{encoding:'utf8',input:route.request().postData()||''}));
     return route.fulfill({status:result.status,json:result.body});
    }
    if(u.pathname==='/')return route.fulfill({contentType:'text/html',body:readRepositoryFile('index.html','utf8')});
    if(u.pathname==='/terrain_renderer.js')return route.fulfill({contentType:'application/javascript',body:'window.createTerrainRenderer=async()=>({available:true,clear(){},draw(){return false},deleteTile(){},getStats(){return{}}})'});
    if(u.pathname==='/terrain_styles.js')return route.fulfill({contentType:'application/javascript',body:'window.TerrainStyles='+readRepositoryFile('terrain_styles.json','utf8')+';\n'+readRepositoryFile('terrain_style_rendering.js','utf8')});
    if(u.pathname.endsWith('.js'))return route.fulfill({contentType:'application/javascript',body:readRepositoryFile(u.pathname.slice(1),'utf8')});
    if(u.pathname==='/api/world'){
     const settings=JSON.parse(u.searchParams.get('generation')),seed=u.searchParams.get('seed'),profile=u.searchParams.get('world_profile');requests.push({origin,settings,seed,profile});
     return route.fulfill({json:{seed,version:'natural-v1',cache_profile:'test',world_identity:'same-world',world_profile:profile,generation_profile:profile,generation_schema:schema,generation_settings:settings,world_topology:'sphere',world_bounds:[-20e6,-10e6,20e6,10e6],overview_bounds:[-20e6,-10e6,20e6,10e6],overview:'/overview.png',gpu:'Mock',orogen_layers:{render:'Materials'}}});
    }
    if(u.pathname==='/api/view')return route.fulfill({json:{accepted:true}});
    if(u.pathname==='/api/status')return route.fulfill({json:{scheduler:{queued:0,computing:0,encoding:0}}});
    if(u.pathname==='/api/inference/streams')return route.fulfill({json:{coarse:4}});
    if(u.pathname==='/api/gpu')return route.fulfill({json:{settings:{mode:'single'},devices:[],plan:{mode:'single'}}});
    return route.fulfill({contentType:'image/png',body:png});
   });
   return {context,page};
  }
  const sender=await create('https://sender.test',{width:1280,height:900},false);
  await sender.page.goto('https://sender.test/?seed=18446744073709551615&profile=natural');
  await sender.page.waitForFunction(()=>world&&!worldLoading&&rendererReady);
  await sender.page.evaluate(()=>{
   cx=123456.125;cy=-78901.25;mpp=1600;viewMode='render';$('mapMode').value='render';
   renderMaterials.set({season:.23,snow:.75,vegetation_tint:[.123456,.78,.9]});
   TerrainLighting.set({global:{azimuth:47,exaggeration:8},lods:{5:{strength:.15}}});
   styleRendering.set({enabled:true,automatic:false,interval:333,width:1.7,density:50});syncContourControls();
   $('grid').checked=true;$('renderSea').checked=true;$('seaMaxLod').value=2;$('prefetch').checked=false;$('cacheLodGap').value=5;
   $('refinementDepth').value=2;forcedView={lod:6,signature:viewSignature()};$('forcedLod').value=6;draw();schedule();
  });
  const viewport=await sender.page.locator('#viewport').boundingBox();
  const click={x:viewport.width/2+60,y:viewport.height/2-40};
  await sender.page.locator('#viewport').click({position:click,modifiers:['Alt']});
  await sender.page.locator('#viewport').click({position:{x:click.x+100,y:click.y+40},modifiers:['Alt']});
  assert.equal(await sender.page.locator('.mapPin').count(),2,'Alt+click creates coordinate-only pins');
  const pins=await sender.page.evaluate(()=>terrainPins.get());
  assert(Math.abs(pins[0][0]-(123456.125+60*1600))<1e-5);
  assert(Math.abs(pins[0][1]-(-78901.25-40*1600))<1600,'mouse events may round a fractional CSS pixel, but pin coordinates stay within one pixel of the click');
  assert.equal(new URL(sender.page.url()).searchParams.get('pins'),pins.map(pin=>pin.join(',')).join(';'),'pin coordinates update the URL immediately');
  await sender.page.locator('.mapPin').first().click();
  assert.equal(await sender.page.locator('.mapPin').count(),1,'clicking a pin removes it');
  assert.equal(new URL(sender.page.url()).searchParams.get('pins'),pins[1].join(','));
  const expected=await sender.page.evaluate(()=>captureShare());
  const state=await sender.page.evaluate(()=>shareControls.save());
  const code=await sender.page.evaluate(state=>TerrainShare.format(state),state);
  const originalURL=JSON.stringify(expected.generation.settings)+JSON.stringify(expected.rendering);
  assert(code.length<encodeURIComponent(originalURL).length,'portable compressed code is smaller than the old encoded JSON');
  assert(code.includes('layer=render')&&code.includes('zoom=1600')&&code.includes('position=123456.125,-78901.25'));
  assert.equal(state.generation.length,19);
  const receiver=await create('https://receiver.test',{width:800,height:600},true);
  const relative=await sender.page.evaluate(code=>TerrainShare.url(TerrainShare.parse(code)),code);
  await receiver.page.goto('https://receiver.test'+relative);
  await receiver.page.waitForFunction(()=>world&&!worldLoading&&rendererReady&&!restoringShare);
  const actual=await receiver.page.evaluate(()=>captureShare());
  assert.deepEqual(actual.generation.settings,expected.generation.settings,'explicit imported generation wins over saved drafts and GPU preferences');
  assert.deepEqual(actual.rendering,expected.rendering,'all rendering and tile controls survive cross-installation import');
  assert.deepEqual(actual.camera,expected.camera,'logical viewport, zoom, position, layer and forced LOD match on a smaller screen');
  assert.equal(requests.at(-1).seed,'18446744073709551615','64-bit seeds remain exact');
  assert.equal(await receiver.page.locator('#gpuRender').isChecked(),false);
  assert.equal(await receiver.page.locator('.mapPin').count(),1,'pins import across installations');
  assert.equal(new URL(receiver.page.url()).searchParams.get('pins'),expected.camera.pins);
  await receiver.page.locator('.mapPin').click();
  assert.equal(await receiver.page.locator('.mapPin').count(),0);
  assert.equal(new URL(receiver.page.url()).searchParams.has('pins'),false,'removing the last pin removes its URL field');
  await receiver.page.reload();
  await receiver.page.waitForFunction(()=>world&&!worldLoading&&rendererReady&&!restoringShare);
  assert.equal(await receiver.page.locator('.mapPin').count(),0,'removed pins stay removed after reload');
  assert.equal(await receiver.page.locator('#unlockShareSize').isVisible(),false,'size control is inside the closed panel');
  assert(fs.readdirSync(path.join(temporary,'receiver.test')).length>=2,'portable import populates a separate empty store');
  // Globe camera is retained independently of the projected map camera.
  await sender.page.evaluate(()=>{setSceneView('globe');globe.restore({yaw:2.4,pitch:.6,altitude:.08});});
  const globeBounds=await sender.page.locator('#viewport').boundingBox();
  await sender.page.locator('#globeMap').click({position:{x:globeBounds.width/2,y:globeBounds.height/2},modifiers:['Alt']});
  assert.equal(await sender.page.locator('.mapPin').count(),2,'Alt+click also pins the globe surface');
  const globeExpected=await sender.page.evaluate(()=>captureShare().camera);
  const globeState=await sender.page.evaluate(()=>shareControls.save());
  const globeURL=await sender.page.evaluate(state=>TerrainShare.url(state),globeState);
  await receiver.page.goto('https://receiver.test'+globeURL);
  await receiver.page.waitForFunction(()=>world&&!worldLoading&&rendererReady&&!restoringShare);
  assert.deepEqual(await receiver.page.evaluate(()=>captureShare().camera),globeExpected,'globe orientation and altitude round trip');
  const visiblePin=receiver.page.locator('.mapPin:visible');
  assert(await visiblePin.count()>=1,'front-facing globe pins are displayed');
  await visiblePin.last().click();
  assert.equal(await receiver.page.locator('.mapPin').count(),1,'globe pins can be removed with a click');
  // Invalid camera data never starts a world with guessed settings.
  await receiver.page.goto('https://receiver.test'+relative.replace('zoom=1600','zoom=NaN'));
  await receiver.page.waitForFunction(()=>$('status').textContent.includes('Invalid camera number'));
  assert.equal(await receiver.page.evaluate(()=>world),null);
  assert.deepEqual(errors,[]);
  await sender.context.close();await receiver.context.close();
  console.log(`Portable sharing and coordinate-only pins across empty installations, conflicting preferences, smaller viewports and globe cameras: OK (${code.length} characters versus ${encodeURIComponent(originalURL).length} in encoded JSON)`);
 }finally{await browser.close();assert.equal(path.dirname(temporary),os.tmpdir());assert(path.basename(temporary).startsWith('terrain-share-'));fs.rmSync(temporary,{recursive:true,force:true});}
})().catch(error=>{console.error(error);process.exitCode=1});
