// Real viewer integration: resource menu, Diagnostic group and symbol inspection.
// node tools/verification/verify_resources_inspector.cjs [--port 8775]
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const port=process.argv.includes('--port')?process.argv[process.argv.indexOf('--port')+1]:8775;
const root=path.resolve(__dirname,'../..'),base=`http://127.0.0.1:${port}`;
(async()=>{
  const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu','--ignore-gpu-blocklist']});
  try{
    const page=await browser.newPage({viewport:{width:1200,height:800}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(base+'/?seed=42&profile=orogen&gpu=1');
    await page.waitForFunction(()=>typeof world!=='undefined'&&world&&renderer?.status==='webgpu',null,{timeout:180000});
    const menu=await page.evaluate(()=>({
      resources:[...document.querySelectorAll('#mapMode option')].filter(o=>o.value.startsWith('resources-dep-')&&!o.hidden&&!o.disabled).map(o=>o.value),
      groups:[...document.querySelectorAll('#layerPanel fieldset>legend')].map(e=>e.textContent),
      diagnostics:[...document.querySelectorAll('#layerPanel fieldset')].find(e=>e.querySelector('legend')?.textContent==='Diagnostic')?.textContent
    }));
    assert(!menu.resources.some(m=>/-stock-|-recoverable-/.test(m)));
    assert(!menu.resources.includes('resources-dep-star-metal'));
    assert.equal(menu.groups.at(-1),'Diagnostic');
    assert(menu.diagnostics.includes('Prospectivité')&&menu.diagnostics.includes('Potentiel'));
    const mode='resources-dep-gypsum';
    await page.evaluate(m=>{const s=document.getElementById('mapMode');s.value=m;s.dispatchEvent(new Event('change',{bubbles:true}));},mode);
    await page.waitForFunction(m=>TerrainResources.state().current===m,mode,{timeout:180000});
    const result=await page.evaluate(async()=>{
      const u=`/api/resources/42/gpu?mode=resources-dep-gypsum&world_profile=${encodeURIComponent(world.generation_profile)}`;
      const bytes=await (await fetch(u)).arrayBuffer(),n=new DataView(bytes).getUint32(0,true),h=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,4,n)));
      const d=TerrainResourcesGPU.parse(h,bytes.slice(4+n)),v=new Int32Array(d.deposits.buffer),f=new Float32Array(d.deposits.buffer);
      let k=0;for(let i=1;i<h.deposits;i++)if(f[12*i+11]>f[12*k+11])k=i;
      await TerrainResources.inspect({x:(v[12*k]+.5)*30,y:(v[12*k+1]+.5)*30,seed:42,profile:world.generation_profile,mpp:30});
      return {text:document.querySelector('[data-resource-inspector]').textContent,legend:document.querySelector('[data-resource-body]').textContent,count:h.deposits};
    });
    assert(result.text.includes('récupérables')&&result.text.includes('teneur')&&result.text.includes('profondeur')&&result.text.includes('emprise'),result.text);
    assert(result.text.includes('Gypse'));
    assert(!result.text.includes('Stock contenu'));
    assert(result.legend.includes('6 px')&&result.legend.includes('log'));
    assert.equal(errors.length,0);
    const report={ok:true,menu,...result,errors};
    fs.writeFileSync(path.join(root,'output/resources-delivery/inspector.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify(report));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
