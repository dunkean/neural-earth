// Times uncached 30 m tiles in the real browser for GEMM tile candidates.
// Requires npm run serve:wasm. NEURAL_EARTH_TILES="bm,bn,bk,tm,tn;..." ('' = default).
const {chromium}=require('playwright');
const {browserExecutable,browserArguments}=require('../platform.cjs');
(async()=>{
  const browser=await chromium.launch({executablePath:browserExecutable(),headless:true,args:browserArguments()});
  try {
    const page=await (await browser.newContext({viewport:{width:400,height:300}})).newPage();
    page.on('pageerror',e=>console.log('pageerror:',e.message));
    await page.goto(`${process.env.NEURAL_EARTH_WASM_URL||'http://127.0.0.1:8780'}/?seed=42&gpu=0&mode=relief`,{waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>window.terrainDebug?.snapshot().overview,null,{timeout:180000});
    const tiles=(process.env.NEURAL_EARTH_TILES??'').split(';');
    for(const tile of tiles) {
      const result=await page.evaluate(async tile=>{
        const post=(path,body)=>TerrainTransport.fetch(path,{method:'POST',body:JSON.stringify(body)});
        await post('/api/debug/conv-tile',{tile});
        const stats=async()=>await (await TerrainTransport.fetch('/api/debug/stats')).json();
        const time=async(x,y)=>{const a=await stats(),t=performance.now();const r=await TerrainTransport.fetch(`/height/wasm-v1/42/0/${x}/${y}.bin?mode=relief`);if(!r.ok)throw Error(await r.text());await r.arrayBuffer();const ms=Math.round(performance.now()-t),b=await stats();return {ms,gpuWait:Math.round(b[0]-a[0]),dispatches:b[1]-a[1],tflops:+((b[2]-a[2])/ms).toFixed(2)};};
        const warm=await time(1188,-1076);
        await post('/api/debug/conv-tile',{tile});
        return {warm,cold:[await time(1188,-1076),await time(1300,-1000)]};
      },tile);
      console.log(`tile ${tile||'default'}:`,JSON.stringify(result));
    }
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
