// Join the right and left longitude strips of the actual viewer, at LOD 7.
// node tools/verification/capture_resources_wrap.cjs (port 8775 only)
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/resources-gpu/wrap');
const lod=Number(process.argv.includes('--lod')?process.argv[process.argv.indexOf('--lod')+1]:7);
const width=lod===10?190:410;
const report={seed:42,lod,frames:[],errors:[]},modes=['orogen-crust','render-v2','resources-dep-potential-copper','resources-dep-stock-copper'];
(async()=>{
 fs.mkdirSync(out,{recursive:true});const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu','--ignore-gpu-blocklist']});
 try{
  const page=await browser.newPage({viewport:{width:1000,height:760}});page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto('http://127.0.0.1:8775/?seed=42&profile=orogen&gpu=1');
  await page.waitForFunction(()=>typeof world!=='undefined'&&world&&renderer?.status==='webgpu',null,{timeout:180000});
  report.world=await page.evaluate(()=>({diameter:world.world_diameter_km,bounds:worldBounds(),profile:world.generation_profile}));assert.equal(report.world.diameter,4000);
  await page.addStyleTag({content:'#hud,#mini,#legend,#annotations{display:none!important}'});
  await page.evaluate(width=>{setSharedSize([width,260]);$('prefetch').checked=false;hydrology.set({enabled:false});},width);
  for(const mode of modes)for(const side of ['right','left']){
   await page.evaluate(({mode,side,lod,width})=>{const b=worldBounds();mpp=30*2**lod;cx=side==='right'?b[2]-width*mpp/2:b[0]+width*mpp/2;cy=2300000;clampCamera();$('mapMode').value=mode;$('mapMode').dispatchEvent(new Event('change',{bubbles:true}));$('forcedLod').value=String(lod);forcedView={lod,signature:viewSignature()};draw();schedule();},{mode,side,lod,width});
   if(mode.startsWith('resources'))await page.waitForFunction(m=>TerrainResources.state().current===m,mode,{timeout:120000});
   await page.waitForFunction(()=>terrainDebug.snapshot().visible.pending===0,null,{timeout:180000});
   await page.evaluate(()=>new Promise(ok=>requestAnimationFrame(()=>requestAnimationFrame(ok))));
   const file=mode+'-'+side+'-'+lod+'.png';await page.locator('#viewport').screenshot({path:path.join(out,file)});
   report.frames.push({mode,side,file,state:await page.evaluate(()=>terrainDebug.snapshot())});
  }
  const sheet=await browser.newPage({viewport:{width:width*2,height:1200}});
  const img=(mode,side)=>`<img src="data:image/png;base64,${fs.readFileSync(path.join(out,mode+'-'+side+'-'+lod+'.png')).toString('base64')}">`;
  await sheet.setContent(`<meta charset="utf-8"><style>body{margin:0;background:#14212b;color:white;font:13px system-ui}p{margin:8px}.row{display:flex}img{width:${width}px;height:260px;display:block}</style><p>4 000 km · seed 42 · LOD ${lod} · longitude wrap at the central join</p>${modes.map(m=>`<p>${m}</p><div class="row">${img(m,'right')}${img(m,'left')}</div>`).join('')}`);
  await sheet.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth));
  await sheet.locator('body').screenshot({path:path.join(root,`docs/images/resources-gpu-wrap${lod===7?'':'-lod'+lod}.jpg`),type:'jpeg',quality:90});
  // Actual rendered pixel regression, independent of the resource field mirror.
  report.seam=await sheet.evaluate(width=>{
    const result=[];
    for(const row of document.querySelectorAll('.row')){
      const [a,b]=row.querySelectorAll('img'),c=document.createElement('canvas');c.width=width*2;c.height=260;
      const ctx=c.getContext('2d');ctx.drawImage(a,0,0,width,260);ctx.drawImage(b,410,0,width,260);
      const p=ctx.getImageData(0,0,width*2,260).data;let jump=0,reference=0,n=0;
      for(let y=30;y<250;y++)for(let k=0;k<3;k++){
        const at=x=>p[(y*width*2+x)*4+k];jump+=Math.abs(at(width-1)-at(width));
        reference+=(Math.abs(at(width-2)-at(width-1))+Math.abs(at(width)-at(width+1)))/2;n++;
      }
      result.push({jump:jump/n,reference:reference/n});
    }return result;
  },width);
  for(let i=0;i<modes.length;i++)assert(report.seam[i].jump<Math.max(5,3*report.seam[i].reference),modes[i]+' has a visible longitude seam');
  assert.equal(report.errors.length,0);report.ok=true;console.log(JSON.stringify({lod,seam:report.seam}));
 }finally{fs.writeFileSync(path.join(out,`report-${lod}.json`),JSON.stringify(report,null,2));await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
