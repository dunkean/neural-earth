// Time to first pixel of resource layers, as a user meets it: (optionally) a
// cold server, the viewer opens an Orogen world, then each layer is chosen in
// the layer menu. A layer counts as shown when its package is bound on the
// GPU (TerrainResources.state().current) and a frame has been drawn.
//
// usage: node tools/benchmarks/benchmark_resources_first_pixel.cjs [--spawn PYTHON] [--port 8775]
//        [--seed 7985403534069300490] [--clear-cache DIR] [--json OUT] [--layers a,b,...]
// --spawn starts `PYTHON -u backend/terrain_server.py --port PORT` (TERRAIN_PRELOAD_MODELS=0)
// and stops it at the end (--log FILE keeps its output); --clear-cache empties the resource cache first.
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const fs=require('node:fs'),path=require('node:path'),{spawn}=require('node:child_process');
const root=path.resolve(__dirname,'../..');
const arg=(name,fallback)=>{const i=process.argv.indexOf(name);return i>0?process.argv[i+1]:fallback;};
const port=Number(arg('--port',8775)),seed=arg('--seed','7985403534069300490'),python=arg('--spawn',null),clear=arg('--clear-cache',null);
const layers=(arg('--layers','resources-bio-npp,resources-geo-rock,resources-geo-context,resources-agri-cereal-suitability,resources-mat-stone,'+
  'resources-season-snow,resources-bio-wood-stock,resources-agri-harvest-month,resources-dep-points,resources-dep-stock-copper')).split(',');
const base=`http://127.0.0.1:${port}`;
const sleep=ms=>new Promise(ok=>setTimeout(ok,ms));
async function up(timeout=180000){const t=Date.now();while(Date.now()-t<timeout){try{if((await fetch(base+'/')).ok)return (Date.now()-t)/1000;}catch(_){}await sleep(500);}throw Error('server not up');}
(async()=>{
  if(clear&&fs.existsSync(clear))for(const f of fs.readdirSync(clear))fs.rmSync(path.join(clear,f),{force:true,recursive:true});
  let server=null;const report={port,seed,cold:!!python,clearedCache:!!clear,layers:[]};
  if(python){
    const log=arg('--log',null),fd=log?fs.openSync(log,'w'):'ignore';
    server=spawn(python,['-u','backend/terrain_server.py','--port',String(port)],{cwd:root,env:{...process.env,TERRAIN_PRELOAD_MODELS:'0',PYTHONUNBUFFERED:'1'},stdio:['ignore',fd,fd]});
    report.serverStartS=await up();
  }
  const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu','--ignore-gpu-blocklist']});
  try{
    const page=await browser.newPage({viewport:{width:1280,height:760}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    const t0=Date.now();
    await page.goto(`${base}/?seed=${seed}&profile=orogen&gpu=1`);
    await page.waitForFunction(()=>window.TerrainResources&&Object.keys(window.TerrainResources.specs()).length>0&&typeof renderer!=='undefined'&&renderer?.status==='webgpu',null,{timeout:300000});
    report.worldOpenS=(Date.now()-t0)/1000;
    for(const mode of layers){
      const start=Date.now();let sawProgress=null;
      await page.evaluate(m=>{const s=document.getElementById('mapMode');s.value=m;s.dispatchEvent(new Event('change',{bubbles:true}));},mode);
      let ok=false;
      while(Date.now()-start<600000){
        const st=await page.evaluate(()=>window.TerrainResources.state());
        if(st.progress&&(!sawProgress||st.progress.includes('Calcul')))sawProgress=st.progress;
        if(st.current===mode){ok=true;break;}
        if(st.progress&&st.progress.startsWith('Couche indisponible'))break;
        await sleep(100);
      }
      await page.evaluate(()=>new Promise(ok=>requestAnimationFrame(()=>requestAnimationFrame(ok))));
      const row={mode,ok,firstPixelS:(Date.now()-start)/1000,sinceWorldOpenS:(Date.now()-t0)/1000-report.worldOpenS,progressShown:sawProgress};
      report.layers.push(row);console.log(JSON.stringify(row));
    }
    report.errors=errors;
  }finally{await browser.close();if(server)server.kill();}
  const out=arg('--json',null);if(out){fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify(report,null,2));}
  console.log(JSON.stringify({worldOpenS:report.worldOpenS,serverStartS:report.serverStartS}));
})().catch(e=>{console.error(e);process.exitCode=1;});
