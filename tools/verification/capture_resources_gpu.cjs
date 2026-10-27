// Real viewer captures on the 4,000 km reference world. Uses the same
// Playwright/platform and fixed-viewport workflow as the Render v2 reviews.
// node tools/verification/capture_resources_gpu.cjs [--port 8775]
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const arg=(n,d)=>process.argv.includes(n)?process.argv[process.argv.indexOf(n)+1]:d;
const locale=arg('--locale','fr');assert(['fr','en'].includes(locale));
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/resources-gpu/resources-fix-captures',locale);
const base=`http://127.0.0.1:${arg('--port',8775)}`;
const width=410,height=260,lods=[10,7,4,2,0];
const metals=arg('--resources','copper,gypsum,points').split(',');
require(path.join(root,'web/terrain_resources_gpu.js'));
const report=process.argv.includes('--append')&&fs.existsSync(path.join(out,'report.json'))
  ?JSON.parse(fs.readFileSync(path.join(out,'report.json'),'utf8'))
  :{seed:'42',locale,worldDiameterKm:4000,frames:[],errors:[],responses:[]};
report.frames=report.frames.filter(f=>!metals.includes(f.metal));
const save=()=>fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));
(async()=>{
  fs.mkdirSync(out,{recursive:true});
  const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu','--ignore-gpu-blocklist']});
  try{
    const page=await browser.newPage({viewport:{width:1000,height:760}});
    await page.addInitScript(language=>localStorage.setItem('neural-earth-locale-v1',language),locale);
    page.on('pageerror',e=>report.errors.push(e.message));
    page.on('response',r=>{if(r.status()>=400&&!/favicon|illustration\/index/.test(r.url()))report.responses.push({status:r.status(),url:r.url()});});
    await page.goto(base+'/?seed=42&profile=orogen&gpu=1');
    await page.waitForFunction(()=>typeof world!=='undefined'&&world&&renderer?.status==='webgpu',null,{timeout:180000});
    assert(await page.evaluate(()=>typeof TerrainResourcesGPU.aggregate==='function'),'8775 is serving stale viewer sources');
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
      const display=await page.evaluate(()=>{const a=renderer?.res?.aggregation;return {label:TerrainResources.label(TerrainResources.specs()[$('mapMode').value]?.label||'Render v2'),aggregation:a?{input:a.input,symbols:a.symbols,clusters:a.clusters,ms:a.ms}:null};});
      if(mode.startsWith('resources-')){
        const inspection=await page.evaluate(async()=>{
          const a=renderer?.res?.aggregation;if(!a)return null;
          const g=a.groups.filter(g=>g.members.length>1&&Math.abs(g.x-cx)<205*mpp&&Math.abs(g.y-cy)<130*mpp).sort((a,b)=>Math.hypot(a.x-cx,a.y-cy)-Math.hypot(b.x-cx,b.y-cy))[0];
          if(!g)return null;
          await TerrainResources.inspect({x:g.x,y:g.y,mpp,seed:42,profile:world.generation_profile});
          return {expected:g.members.length,...TerrainResources.state().inspection,text:document.querySelector('[data-resource-inspector]').textContent};
        });
        if(inspection){assert.equal(inspection.members,inspection.expected,'Inspector must list every cluster member');display.inspection=inspection;}
      }
      assert.equal(state.camera.mpp,30*2**lod);assert.equal(state.refinement.forced,lod);
      if(lod<=2)assert(state.rendered.sources.every(s=>s==='decoder'),'Close captures require real decoder tiles');
      const file=`${metal}-${tag}-${lod}.png`;
      await page.locator('#gpuMap').screenshot({path:path.join(out,file)});
      report.frames.push({metal,mode,lod,center,file,state,display});save();
      console.log(JSON.stringify({metal,mode,lod,sources:state.rendered.sources,pending:state.visible.pending}));
    }
    for(const metal of metals){
      const p=await pkg('resources-dep-'+metal),words=new Float32Array(p.data.deposits.buffer),ints=new Int32Array(p.data.deposits.buffer);
      let best=-1,bestMass=-1;
      for(let k=0;k<p.header.deposits;k++){const y=(ints[k*12+1]+.5)*30;if(Math.abs(y)<2.5e6&&words[k*12+11]>bestMass&&words[k*12+4]<2500){best=k;bestMass=words[k*12+11];}}
      assert(best>=0);const center=[(ints[best*12]+.5)*30,(ints[best*12+1]+.5)*30];
      report[metal]={center,mass:bestMass,footprint:[words[best*12+4],words[best*12+5]]};
      for(const lod of lods){
        await page.evaluate(({lod,center})=>{[cx,cy]=lod===10?[0,0]:center;mpp=30*2**lod;clampCamera();$('forcedLod').value=String(lod);forcedView={lod,signature:viewSignature()};draw();schedule();},{lod,center});
        await capture(metal,'render-v2',lod,center,'render');
        await capture(metal,'resources-dep-'+metal,lod,center,'sites');
      }
    }
    // Assemble actual screenshots, using the existing HTML contact-sheet method.
    const sheet=await browser.newPage({viewport:{width:width*5,height:height*2+100}});
    for(const metal of metals){
      const kind='sites',title=report.frames.find(f=>f.metal===metal&&f.mode.startsWith('resources-')).display.label;
      const img=(tag,lod)=>`<img src="data:image/png;base64,${fs.readFileSync(path.join(out,`${metal}-${tag}-${lod}.png`)).toString('base64')}">`;
      await sheet.setContent(`<meta charset="utf-8"><style>body{margin:0;background:#14212b;color:#dce6eb;font:13px system-ui}h3,p{margin:8px}main{display:grid;grid-template-columns:repeat(5,${width}px)}img{display:block;width:${width}px;height:${height}px}small{display:block;padding:3px}</style><h3>${title} · ${locale}</h3><p>Orogen · ${locale==='fr'?'graine 42 · diamètre 4 000 km · Render v2 en haut, ressource en bas · juillet':'seed 42 · diameter 4,000 km · Render v2 above, resource below · July'}</p><main>${lods.map(l=>`<div><small>LOD ${l} · ${30*2**l} m/pixel</small>${img('render',l)}${img(kind,l)}</div>`).join('')}</main>`);
      await sheet.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth));
      await sheet.locator('body').screenshot({path:path.join(root,`docs/images/resources-gpu-${kind}-${metal}${locale==='fr'?'':'-en'}.jpg`),type:'jpeg',quality:90});
    }
    report.resources=[...new Set([...(report.resources||[]),...metals])];
    report.ok=report.errors.length===0&&report.frames.length===report.resources.length*10;save();
  }finally{await browser.close();}
})().catch(error=>{report.failure=error.stack;save();console.error(error);process.exitCode=1;});
