// Join the right and left longitude strips of the actual viewer, at LOD 7.
// node tools/verification/capture_resources_wrap.cjs (port 8775 only)
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/resources-gpu/wrap');
const report={seed:42,lod:7,frames:[],errors:[]},modes=['orogen-crust','render-v2','resources-dep-potential-copper','resources-dep-stock-copper'];
(async()=>{
 fs.mkdirSync(out,{recursive:true});const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu','--ignore-gpu-blocklist']});
 try{
  const page=await browser.newPage({viewport:{width:1000,height:760}});page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto('http://127.0.0.1:8775/?seed=42&profile=orogen&gpu=1');
  await page.waitForFunction(()=>typeof world!=='undefined'&&world&&renderer?.status==='webgpu',null,{timeout:180000});
  report.world=await page.evaluate(()=>({diameter:world.world_diameter_km,bounds:worldBounds(),profile:world.generation_profile}));assert.equal(report.world.diameter,4000);
  await page.addStyleTag({content:'#hud,#mini,#legend,#annotations{display:none!important}'});
  await page.evaluate(()=>{setSharedSize([410,260]);$('prefetch').checked=false;hydrology.set({enabled:false});});
  for(const mode of modes)for(const side of ['right','left']){
   await page.evaluate(({mode,side})=>{const b=worldBounds();mpp=3840;cx=side==='right'?b[2]-410*mpp/2:b[0]+410*mpp/2;cy=2300000;clampCamera();$('mapMode').value=mode;$('mapMode').dispatchEvent(new Event('change',{bubbles:true}));$('forcedLod').value='7';forcedView={lod:7,signature:viewSignature()};draw();schedule();},{mode,side});
   if(mode.startsWith('resources'))await page.waitForFunction(m=>TerrainResources.state().current===m,mode,{timeout:120000});
   await page.waitForFunction(()=>terrainDebug.snapshot().visible.pending===0,null,{timeout:180000});
   await page.evaluate(()=>new Promise(ok=>requestAnimationFrame(()=>requestAnimationFrame(ok))));
   const file=mode+'-'+side+'.png';await page.locator('#viewport').screenshot({path:path.join(out,file)});
   report.frames.push({mode,side,file,state:await page.evaluate(()=>terrainDebug.snapshot())});
  }
  const sheet=await browser.newPage({viewport:{width:820,height:1200}});
  const img=(mode,side)=>`<img src="data:image/png;base64,${fs.readFileSync(path.join(out,mode+'-'+side+'.png')).toString('base64')}">`;
  await sheet.setContent(`<meta charset="utf-8"><style>body{margin:0;background:#14212b;color:white;font:13px system-ui}p{margin:8px}.row{display:flex}img{width:410px;height:260px;display:block}</style><p>4 000 km · seed 42 · LOD 7 · longitude wrap at the central join</p>${modes.map(m=>`<p>${m}</p><div class="row">${img(m,'right')}${img(m,'left')}</div>`).join('')}`);
  await sheet.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth));
  await sheet.locator('body').screenshot({path:path.join(root,'docs/images/resources-gpu-wrap.jpg'),type:'jpeg',quality:90});
  assert.equal(report.errors.length,0);report.ok=true;
 }finally{fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
