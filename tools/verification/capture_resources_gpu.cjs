// Real viewer captures on the 4,000 km reference world. Uses the same
// Playwright/platform and fixed-viewport workflow as the Render v2 reviews.
// node tools/verification/capture_resources_gpu.cjs [--port 8775]
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/resources-gpu/r3g-captures');
const arg=(n,d)=>process.argv.includes(n)?process.argv[process.argv.indexOf(n)+1]:d;
const base=`http://127.0.0.1:${arg('--port',8775)}`;
const width=410,height=260,lods=[10,7,4,2,0];
require(path.join(root,'web/terrain_resources_gpu.js'));
const report={seed:'42',worldDiameterKm:4000,frames:[],errors:[],responses:[]};
const save=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));
(async()=>{
  fs.mkdirSync(out,{recursive:true});
  const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu','--ignore-gpu-blocklist']});
  try{
    const page=await browser.newPage({viewport:{width:1000,height:760}});
    page.on('pageerror',e=>report.errors.push(e.message));
    page.on('response',r=>{if(r.status()>=400&&!/favicon|illustration\/index/.test(r.url()))report.responses.push({status:r.status(),url:r.url()});});
    await page.goto(base+'/?seed=42&profile=orogen&gpu=1');
    await page.waitForFunction(()=>typeof world!=='undefined'&&world&&renderer?.status==='webgpu',null,{timeout:180000});
    report.world=await page.evaluate(()=>({diameter:world.world_diameter_km,profile:world.generation_profile,bounds:worldBounds(),identity:world.world_identity}));
    assert.equal(report.world.diameter,4000);
    await page.addStyleTag({content:'#hud,#mini,#legend,#annotations{display:none!important}'});
    await page.evaluate(([w,h])=>{setSharedSize([w,h]);document.getElementById('prefetch').checked=false;hydrology.set({enabled:false});},[width,height]);
    async function pkg(mode){
      const url=base+`/api/resources/42/gpu?mode=${mode}&world_profile=${report.world.profile}`;
      for(;;){const r=await fetch(url);if(r.status===202){await new Promise(ok=>setTimeout(ok,1000));continue;}assert(r.ok);const b=await r.arrayBuffer(),n=new DataView(b).getUint32(0,true),h=JSON.parse(new TextDecoder().decode(new Uint8Array(b,4,n)));return {header:h,data:TerrainResourcesGPU.parse(h,b.slice(4+n))};}
    }
    async function layer(mode){
      await page.evaluate(m=>{const s=document.getElementById('mapMode');s.value=m;s.dispatchEvent(new Event('change',{bubbles:true}));},mode);
      if(mode.startsWith('resources-'))await page.waitForFunction(m=>TerrainResources.state().current===m,mode,{timeout:180000});
    }
    async function capture(metal,mode,lod,center,tag){
      await layer(mode);
      await page.waitForFunction(()=>{const s=terrainDebug.snapshot();return s.visible.pending===0&&s.rendered.patches>0;},null,{timeout:600000});
      await page.evaluate(()=>new Promise(ok=>requestAnimationFrame(()=>requestAnimationFrame(ok))));
      const state=await page.evaluate(()=>terrainDebug.snapshot());
      assert.equal(state.camera.mpp,30*2**lod);assert.equal(state.refinement.forced,lod);
      if(lod<=2)assert(state.rendered.sources.every(s=>s==='decoder'),'Close captures require real decoder tiles');
      const file=`${metal}-${tag}-${lod}.png`;
      await page.locator('#gpuMap').screenshot({path:path.join(out,file)});
      report.frames.push({metal,mode,lod,center,file,state});save();
      console.log(JSON.stringify({metal,mode,lod,sources:state.rendered.sources,pending:state.visible.pending}));
    }
    for(const metal of ['copper','iron','tin']){
      const p=await pkg('resources-dep-stock-'+metal),words=new Float32Array(p.data.deposits.buffer),ints=new Int32Array(p.data.deposits.buffer);
      let best=-1,bestMass=-1;
      for(let k=0;k<p.header.deposits;k++){const y=(ints[k*12+1]+.5)*30;if(Math.abs(y)<2.5e6&&words[k*12+11]>bestMass){best=k;bestMass=words[k*12+11];}}
      assert(best>=0);const center=[(ints[best*12]+.5)*30,(ints[best*12+1]+.5)*30];
      report[metal]={center,mass:bestMass,footprint:[words[best*12+4],words[best*12+5]]};
      for(const lod of lods){
        await page.evaluate(({lod,center})=>{[cx,cy]=lod===10?[0,0]:center;mpp=30*2**lod;clampCamera();$('forcedLod').value=String(lod);forcedView={lod,signature:viewSignature()};draw();schedule();},{lod,center});
        await capture(metal,'render-v2',lod,center,'render');
        await capture(metal,'resources-dep-potential-'+metal,lod,center,'potential');
        if(metal!=='tin')await capture(metal,'resources-dep-stock-'+metal,lod,center,'stock');
      }
    }
    // Assemble actual screenshots, using the existing HTML contact-sheet method.
    const sheet=await browser.newPage({viewport:{width:width*5,height:height*2+100}});
    for(const metal of ['copper','iron','tin'])for(const kind of (metal==='tin'?['potential']:['potential','stock'])){
      const unit='t/kmÃ‚²',title=`${kind==='potential'?'Potentiel (zones)':'Stock contenu (gisements)'} Ã‚· ${metal==='copper'?'cuivre':metal==='iron'?'fer':'étain'} Ã‚· ${unit}`;
      const img=(tag,lod)=>`<img src="data:image/png;base64,${fs.readFileSync(path.join(out,`${metal}-${tag}-${lod}.png`)).toString('base64')}">`;
      await sheet.setContent(`<meta charset="utf-8"><style>body{margin:0;background:#14212b;color:#dce6eb;font:13px system-ui}h3,p{margin:8px}main{display:grid;grid-template-columns:repeat(5,${width}px)}img{display:block;width:${width}px;height:${height}px}small{display:block;padding:3px}</style><h3>${title}</h3><p>Orogen Ã‚· graine 42 Ã‚· diamètre 4 000 km Ã‚· Render v2 en haut, ressource en bas Ã‚· juillet</p><main>${lods.map(l=>`<div><small>LOD ${l} Ã‚· ${30*2**l} m/pixel</small>${img('render',l)}${img(kind,l)}</div>`).join('')}</main>`);
      await sheet.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth));
      await sheet.locator('body').screenshot({path:path.join(root,`docs/images/resources-gpu-${kind}-${metal}.jpg`),type:'jpeg',quality:90});
    }
    report.ok=report.errors.length===0&&report.frames.length===40;save();
  }finally{await browser.close();}
})().catch(error=>{report.failure=error.stack;save();console.error(error);process.exitCode=1;});
