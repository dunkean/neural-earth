const {loadPlaywright,browserExecutable}=require('../../tools/platform.cjs');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
(async()=>{
  const {chromium}=loadPlaywright();
  const browser=await chromium.launch({executablePath:browserExecutable(),headless:true,args:['--disable-gpu']});
  try{
    const page=await browser.newPage(),calls=[],errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    const manifest={models:{decoder:{file:'decoder.pt',training_step:10000}},total_model_bytes:30000000,
      selected_models:{coarse:'coarse2',base:'base_residual8k',decoder:'decoder16'}};
    const result={pixels:2,sideM:40000,spacingM:20000,timings:{totalMs:100},requestMs:110};
    await page.addInitScript(({result})=>{
      window.Worker=class{
        postMessage(payload){window.workerPayload=payload;setTimeout(()=>this.onmessage({data:{...result,
          height:new Float32Array([100,200,300,400]),provider:'wasm'}}),0)}
        terminate(){}
      };
    },{result});
    let reject=false;
    await page.route('https://tile.test/**',async route=>{
      const url=new URL(route.request().url());calls.push(url.pathname);
      if(url.pathname==='/tile40/')return route.fulfill({contentType:'text/html',body:fs.readFileSync(root+'/webgpu/tile40_demo.html','utf8')});
      if(url.pathname.endsWith('.mjs'))return route.fulfill({contentType:'text/javascript',body:fs.readFileSync(root+'/webgpu/'+url.pathname.split('/').pop().replace('demo','tile40_demo'),'utf8')});
      if(url.pathname==='/tile40/python/catalog.json')return route.fulfill({json:manifest});
      if(url.pathname==='/tile40/python/generate'){
        assert.equal(route.request().postDataJSON().input.sketch_m[7][8],3150);
        if(reject)return route.fulfill({status:409,json:{error:'Génération déjà en cours.'}});
        return route.fulfill({contentType:'application/octet-stream',body:Buffer.from(new Float32Array([100,200,300,400]).buffer),
          headers:{'X-Terrain-Metadata':JSON.stringify({...result,manifest,provider:'python-cpu'})}});
      }
      if(url.pathname==='/tile40/catalog.json')return route.fulfill({json:{status:'ready',default_decoder:'decoder16',decoders:[{id:'decoder16',label:'Compact'},{id:'decoder64',label:'Qualité'}]}});
      if(url.pathname.endsWith('/manifest.json'))return route.fulfill({json:manifest});
      throw Error('Unexpected request '+url.pathname);
    });
    await page.goto('https://tile.test/tile40/');
    await page.waitForFunction(()=>!document.getElementById('generate').disabled);
    assert.equal(await page.inputValue('#engine'),'python');
    assert(!calls.includes('/tile40/catalog.json'),'Python must work without ONNX preparation');
    await page.fill('#row','8');await page.fill('#column','9');await page.fill('#altitude','3150');await page.click('#edit');
    await page.selectOption('#provider','cpu');await page.click('#generate');
    await page.waitForFunction(()=>document.getElementById('status').textContent.includes('serveur Python'));
    assert.equal(JSON.parse(await page.textContent('#metadata')).provider,'python-cpu');
    reject=true;await page.click('#generate');
    await page.waitForFunction(()=>document.getElementById('status').textContent.includes('déjà en cours'));
    assert.equal(await page.isEnabled('#engine'),true);
    await page.selectOption('#decoder','decoder64');
    await page.waitForFunction(()=>!document.getElementById('generate').disabled);
    await page.selectOption('#engine','browser');
    await page.waitForFunction(()=>!document.getElementById('generate').disabled);
    assert.equal(await page.inputValue('#decoder'),'decoder64');
    await page.selectOption('#provider','wasm');await page.click('#generate');
    await page.waitForFunction(()=>document.getElementById('status').textContent.includes('navigateur · CPU'));
    assert.equal(await page.evaluate(()=>workerPayload.input.sketch_m[7][8]),3150);
    assert.equal(JSON.parse(await page.textContent('#metadata')).provider,'wasm');
    assert.deepEqual(errors,[]);
    console.log('Python default without ONNX, binary result/provenance, server error recovery and browser switch preserve sketch: OK');
  }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
