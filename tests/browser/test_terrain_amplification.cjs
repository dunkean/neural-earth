const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),http=require('node:http');
const {test}=require('node:test'),{loadPlaywright,browserExecutable}=require('../../tools/platform.cjs');
const root=require('node:path').resolve(__dirname,'../..'),read=name=>fs.readFileSync(root+'/web/'+name,'utf8');
const capability={available:true,modes:{hybrid:{available:true,version:'same-v1'},phasor:{available:true,version:'same-v1'},mse:{available:true,version:'same-v1'}}};
function controls(search='',saved=null,kind='python',legacy=true){
  const storage=new Map(saved?[[legacy?'neural-earth-amplification-v1':'neural-earth-amplification-v2',JSON.stringify(saved)]]:[]);
  const context={window:{NeuralEarthRuntime:{kind}},location:{search},URLSearchParams,storage,localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)}};
  vm.createContext(context);vm.runInContext(read('terrain_render_controls.js'),context);return context;
}
test('mode defaults, old URLs, v1 migration, shared state and WASM capability',()=>{
  const c=controls(),a=c.window.TerrainAmplification;
  assert.equal(a.get().mode,'off');assert.equal(a.get().intensity,1);a.setSupport(capability);
  a.set({mode:'phasor',intensity:1.35});assert.equal(a.effective(),1.35);assert.equal(a.effectiveMode(),'phasor');
  const p=new URLSearchParams();a.writeParams(p);assert.equal(p.get('amplify'),'1.35');assert.equal(p.get('amplify_mode'),'phasor');
  a.compare();assert.equal(a.effective(),0);assert.equal(a.get().intensity,1.35);a.writeParams(p);assert.equal(p.get('amplify_before'),'1');
  a.compare();a.writeParams(p);assert.equal(p.has('amplify_before'),false);
  const b=controls('?'+p).window.TerrainAmplification;b.setSupport(capability);assert.equal(b.effectiveMode(),'phasor');
  const zero=controls('?amplify=0&amplify_mode=phasor').window.TerrainAmplification;zero.setSupport(capability);assert.equal(zero.get().mode,'phasor');assert.equal(zero.get().intensity,0);
  const old=controls('?amplify=1.4').window.TerrainAmplification;old.setSupport(capability);assert.equal(old.effectiveMode(),'hybrid');
  const migrated=controls('',{enabled:true,intensity:1.7,before:true}),m=migrated.window.TerrainAmplification;m.setSupport(capability);
  assert.equal(m.get().mode,'hybrid');assert.equal(m.effective(),0);m.compare();assert.equal(m.effective(),1.7);
  assert.equal(JSON.parse(migrated.storage.get('neural-earth-amplification-v2')).mode,'hybrid');
  const restored=controls('',a.get(),'python',false).window.TerrainAmplification;restored.setSupport(capability);assert.equal(restored.effectiveMode(),'phasor');
  for(const legacy of [true,false])for(const mode of ['mse','phasor','hybrid']){const explicit=controls('',{mode,intensity:.8},'python',legacy).window.TerrainAmplification;explicit.setSupport(capability);assert.equal(explicit.effectiveMode(),mode);}
  const legacyShare=controls().window.TerrainAmplification;legacyShare.setSupport(capability);legacyShare.set({enabled:true,intensity:.5});assert.equal(legacyShare.effectiveMode(),'hybrid');
  legacyShare.writeParams(p);assert.equal(p.get('amplify_mode'),'hybrid');
  const wasm=controls('?amplify=1&amplify_mode=phasor',null,'wasm').window.TerrainAmplification;wasm.setSupport(capability);assert.equal(wasm.effective(),0);assert.equal(wasm.effectiveMode(),'off');
  a.set({mode:'off',intensity:1.35});a.writeParams(p);assert.equal(p.has('amplify'),false);assert.equal(p.has('amplify_mode'),false);
});
test('four-way cycling preserves intensity and ignores unavailable modes',()=>{
  const a=controls().window.TerrainAmplification;a.setSupport(capability);a.set({mode:'off',intensity:.65});
  for(const mode of ['hybrid','phasor','mse','off','hybrid']){a.cycle();assert.equal(a.effectiveMode(),mode);assert.equal(a.get().intensity,.65);}
  a.setSupport({available:true,modes:{mse:{available:false},phasor:{available:true}}});
  assert.equal(a.effectiveMode(),'off');a.cycle();assert.equal(a.effectiveMode(),'phasor');a.compare();assert.equal(a.effectiveMode(),'off');
});

test('unavailable hybrid falls back to off and retains the explicit choice for recovery',()=>{
  const a=controls('?amplify=1.25&amplify_mode=hybrid').window.TerrainAmplification;
  a.setSupport({available:true,modes:{hybrid:{available:false,reason:'hybrid dependency missing'},phasor:{available:true}}});
  assert.equal(a.effectiveMode(),'off');assert.equal(a.effective(),0);assert.equal(a.get().mode,'hybrid');assert.equal(a.get().intensity,1.25);
  const p=new URLSearchParams();a.writeParams(p);assert.equal(p.get('amplify_mode'),'hybrid');assert.equal(p.get('amplify'),'1.25');
  a.setSupport(capability);assert.equal(a.effectiveMode(),'hybrid');assert.equal(a.effective(),1.25);
});
test('optional RGBA8 surface follows heights/climate/water; legacy and malformed payloads',()=>{
  const c={window:{},navigator:{},fetch:()=>{}};vm.createContext(c);vm.runInContext(read('terrain_transport.js'),c);
  const parse=c.window.TerrainPhysicalPayload.surface,headers=new Map(),floatCount=9+5*4+4*9;
  assert.equal(parse(new ArrayBuffer(floatCount*4),headers, floatCount,3),null);
  headers.set('X-Terrain-Surface-Format','rgba8unorm');headers.set('X-Terrain-Surface-Channels','flow,deposit,scree,rock');
  const buffer=new ArrayBuffer(floatCount*4+3*3*4);new Uint8Array(buffer).set([64,128,191,255],floatCount*4);
  const surface=parse(buffer,headers,floatCount,3);assert.deepEqual([...surface.slice(0,4)],[64,128,191,255]);assert.equal(surface.byteOffset,floatCount*4);
  assert.throws(()=>parse(buffer.slice(0,-1),headers,floatCount,3),/Truncated/);
  headers.set('X-Terrain-Surface-Format','float');assert.throws(()=>parse(buffer,headers,floatCount,3),/format/);
  headers.set('X-Terrain-Surface-Format','rgba8unorm');headers.set('X-Terrain-Gradient-Format','rg32float');
  const extended=new ArrayBuffer(buffer.byteLength+9*8);new Uint8Array(extended).set(new Uint8Array(buffer));
  new Float32Array(extended,buffer.byteLength,18).set([.125,-.25]);
  assert.deepEqual([...parse(extended,headers,floatCount,3).slice(0,4)],[64,128,191,255]);
  assert.deepEqual([...c.window.TerrainPhysicalPayload.gradient(extended,headers,floatCount,3).slice(0,2)],[.125,-.25]);
  assert.throws(()=>parse(extended.slice(0,-4),headers,floatCount,3),/Truncated/);
});

test('server viewer supports -12 while WASM keeps its own floor and parent coverage',()=>{
  const LOD=require('../../web/terrain_lod.js');assert.equal(LOD.MIN_LOD,-12);
  const source=read('index.html'),line=source.match(/^const MIN_LOD=.*$/m)[0];
  const scale={};vm.createContext(scale);vm.runInContext(source.match(/^function scaleDistance\(.*$/m)[0],scale);assert.equal(scale.scaleDistance(.5),'50 cm');assert.equal(scale.scaleDistance(.005),'5 mm');assert.equal(scale.scaleDistance(2),'2 m');
  for(const minLOD of [undefined,0]){const c={window:{TerrainLOD:LOD,NeuralEarthRuntime:minLOD===undefined?undefined:{kind:'wasm',minLOD}}};vm.createContext(c);vm.runInContext(line+'\nglobalThis.minimum=MIN_LOD;',c);assert.equal(c.minimum,minLOD??-12);}
  const span=256*30*2**-12,parent={key:'parent',lod:-6,tx:-1,ty:-1},fine={key:'fine',lod:-12,tx:-1,ty:-1};
  assert.equal(LOD.plan([parent],-12,[-span,-span,0,0])[0].tile.key,'parent');
  assert.equal(LOD.plan([parent,fine],-12,[-span,-span,0,0])[0].tile.key,'fine');
  const c=viewer();c.world.relief_floors={hybrid:-3};c.window.TerrainAmplification.set({mode:'hybrid',intensity:1});assert.match(c.taskFor(-12,-1,-1,0).key,/amp-hybrid-same-v1-1.00\/interp-from-3-v1/);
});
function viewer(){
  const c=controls();Object.assign(c,{nnEngine:'exact',cacheIdentity:()=> 'world',world:{seed:'42',amplification:capability},coarseGpuEnabled:()=>false,neuralChart:()=>null,coarseInterpolation:'monotone',hydroTileIdentity:()=>'',generationProfile:()=> 'orogen',hydrology:null,renderV2:{isStyle:()=>false},gpuEnabled:()=>true,fantasy:{isStyle:()=>false},viewMode:'render',requestMode:()=> 'render',fixedClimateLod:()=>false,epoch:1,amplificationFallbacks:new Set(),tiles:new Map()});
  const source=read('index.html');vm.runInContext(['taskFor','isCompatibleTile','isCurrentTile','terrainTiles','drawableTerrainTiles','minimumDisplayLod'].map(name=>source.match(new RegExp('^function '+name+'\\(.*$','m'))[0]).join('\n'),c);
  c.TerrainLOD=require('../../web/terrain_lod.js');const start=source.indexOf('function planTerrainRegion('),end=source.indexOf('\nfunction ',start+10);vm.runInContext(source.slice(start,end),c);
  c.window.TerrainAmplification.setSupport(capability);return c;
}
test('before/Hybrid/Phasor/MSE keys separate even with identical module versions',()=>{
  const c=viewer(),a=c.window.TerrainAmplification;a.set({mode:'mse',intensity:1});
  const mse=c.taskFor(-2,0,0,0),coarse=c.taskFor(0,0,0,0);a.set({mode:'phasor',intensity:1});const phasor=c.taskFor(-2,0,0,0);
  assert.notEqual(mse.key,phasor.key);assert.match(mse.key,/amp-mse-same-v1-1.00/);assert.match(phasor.key,/amp-phasor-same-v1-1.00/);
  assert.equal(c.isCurrentTile({...mse,gpu:true}),false);assert.equal(c.isCurrentTile({...phasor,gpu:true}),true);
  a.compare();const before=c.taskFor(-2,0,0,0);assert.notEqual(before.key,phasor.key);assert.equal(c.taskFor(0,0,0,0).key,coarse.key);
  a.compare();assert.equal(c.taskFor(-2,0,0,0).key,phasor.key);
  a.set({mode:'hybrid',intensity:1});const hybrid=c.taskFor(-2,0,0,0);
  assert.match(hybrid.key,/amp-hybrid-same-v1-1.00/);assert.equal(new Set([before.key,mse.key,phasor.key,hybrid.key]).size,4);
  assert.equal(c.isCurrentTile({...hybrid,gpu:true}),true);assert.equal(c.isCurrentTile({...phasor,gpu:true}),false);
});
test('delayed negative-LOD comparisons keep previous coverage until replacements arrive (PNG and GPU)',()=>{
  const LOD=require('../../web/terrain_lod.js'),span=256*7.5;
  for(const gpu of [true,false]){
    const c=viewer(),a=c.window.TerrainAmplification;c.gpuEnabled=()=>gpu;a.set({mode:'mse',intensity:1});
    const tile=tx=>({...c.taskFor(-2,tx,0,0),gpu,...(gpu?{heights:new Float32Array(1)}:{image:{}})});
    const old=[tile(0),tile(1)];for(const t of old){c.tiles.set(t.key,t);c.amplificationFallbacks.add(t.key);}
    a.set({mode:'phasor',intensity:1});
    const plan=()=>c.planTerrainRegion(c.drawableTerrainTiles(-2),-2,[0,0,span*2,span],{minLod:-2});
    assert.deepEqual(Array.from(plan(),p=>p.tile.key),old.map(t=>t.key));assert.equal(c.terrainTiles().length,0,'fallbacks must not satisfy scheduling coverage');
    const first=tile(0);c.tiles.set(first.key,first);assert.deepEqual(Array.from(plan(),p=>p.tile.key),[first.key,old[1].key]);
    const second=tile(1);c.tiles.set(second.key,second);assert.deepEqual(Array.from(plan(),p=>p.tile.key),[first.key,second.key]);
    assert.equal(c.tiles.size,4,'the previous set stays cached');
  }
});
test('previous finer tiles survive a reset LOD limit and yield to current parents',()=>{
  const c=viewer(),a=c.window.TerrainAmplification;a.set({mode:'mse',intensity:1});
  const old={...c.taskFor(-6,0,0,0),gpu:true,heights:new Float32Array(1)};c.tiles.set(old.key,old);c.amplificationFallbacks.add(old.key);
  a.set({mode:'phasor',intensity:1});const region=[0,0,120,120],options={minLod:-2};
  let plan=c.planTerrainRegion(c.drawableTerrainTiles(-2),-2,region,options);assert.equal(plan.length,1);assert.equal(plan[0].tile.key,old.key);
  const parent={...c.taskFor(0,0,0,0),gpu:true,heights:new Float32Array(1)};c.tiles.set(parent.key,parent);
  plan=c.planTerrainRegion(c.drawableTerrainTiles(-2),-2,region,options);assert.equal(plan.length,1);assert.equal(plan[0].tile.key,parent.key);
});
test('a new finer tile clips previous coverage with matching UVs and no holes or overlap',()=>{
  const c=viewer(),a=c.window.TerrainAmplification;a.set({mode:'mse',intensity:1});
  const old={...c.taskFor(-2,0,0,0),gpu:true,heights:new Float32Array(1)};c.tiles.set(old.key,old);c.amplificationFallbacks.add(old.key);
  a.set({mode:'phasor',intensity:1});const child={...c.taskFor(-3,0,0,0),gpu:true,heights:new Float32Array(1)};c.tiles.set(child.key,child);
  const span=256*7.5,plan=c.planTerrainRegion(c.drawableTerrainTiles(-3),-2,[0,0,span,span],{minLod:-3});
  assert.equal(plan.length,3);assert.equal(plan[0].tile.key,child.key);
  const area=plan.reduce((n,p)=>n+(p.bounds[2]-p.bounds[0])*(p.bounds[3]-p.bounds[1]),0);assert.equal(area,span*span);
  assert.deepEqual(Array.from(plan[1].uv),[.5,0,.5,1]);assert.deepEqual(Array.from(plan[2].uv),[0,.5,.5,.5]);
  for(let i=0;i<plan.length;i++)for(let j=i+1;j<plan.length;j++){const a=plan[i].bounds,b=plan[j].bounds;assert(Math.min(a[2],b[2])<=Math.max(a[0],b[0])||Math.min(a[3],b[3])<=Math.max(a[1],b[1]));}
});
test('a forced Render LOD excludes finer cached history',()=>{
  const c=viewer(),a=c.window.TerrainAmplification;a.set({mode:'phasor',intensity:1});c.forcedView={lod:-4};
  const tile=lod=>({...c.taskFor(lod,0,0,0),gpu:true,heights:new Float32Array(1)}),fine=tile(-6),target=tile(-4);c.tiles.set(fine.key,fine);c.tiles.set(target.key,target);
  const min=c.minimumDisplayLod(-4);assert.equal(min,-4);const plan=c.planTerrainRegion(c.drawableTerrainTiles(min),-4,[0,0,120,120],{minLod:min});assert.equal(plan.length,1);assert.equal(plan[0].tile.key,target.key);
  Object.assign(c,{forcedView:null,MIN_LOD:-6,MAX_LOD:11,continuousRefinement:()=>false,cacheLodGap:()=>3});assert.equal(c.minimumDisplayLod(-4),-6,'camera-driven views still retain configured finer history');
});

test('camera views at/below each relief floor ignore finer history and retain ready parents',()=>{
  for(const mode of ['off','hybrid','phasor','mse'])for(const gap of [0,3,4])for(const automatic of [false,true]){
    const c=viewer(),a=c.window.TerrainAmplification;
    Object.assign(c,{forcedView:null,MIN_LOD:-12,MAX_LOD:11,continuousRefinement:()=>automatic,refinementPolicy:()=>({target:-12}),cacheLodGap:()=>gap});
    c.world.relief_floors={classic:-6,hybrid:-4,phasor:-5,mse:-6};a.set({mode,intensity:1});
    const floor=c.world.relief_floors[mode==='off'?'classic':mode];
    assert.equal(c.minimumDisplayLod(floor),floor);assert.equal(c.minimumDisplayLod(-8),-8);
    const tile=(lod,tx=-1)=>({...c.taskFor(lod,tx,-1,0),gpu:true,heights:new Float32Array(1)});
    const parent=tile(-4),target=tile(-8),history=[-9,-10,-11,-12].map(lod=>tile(lod));
    for(const t of [parent,target,...history])c.tiles.set(t.key,t);
    const region=[-30,-30,0,0],min=c.minimumDisplayLod(-8),plan=()=>c.planTerrainRegion(c.drawableTerrainTiles(min),-8,region,{minLod:min});
    assert.deepEqual(Array.from(plan(),p=>p.tile.key),[target.key]);
    c.tiles.delete(target.key);assert.deepEqual(Array.from(plan(),p=>p.tile.key),[parent.key]);
    assert.equal(c.tiles.size,5,'finer tiles stay cached for the next zoom-in');
    a.set({mode:'hybrid',intensity:0});assert.equal(c.minimumDisplayLod(-5),automatic?-12:Math.max(-12,-5-gap),'zero intensity follows the classic floor');
    c.forcedView={lod:-3};assert.equal(c.minimumDisplayLod(-3),-3,'forced LOD remains exact above the floor');
  }
});
test('controls keyboard, badge and amplified Render/Render3D shaders',async t=>{
  const html='<meta charset="utf-8"><header></header><div id="controls"><div id="pane"></div></div><canvas id="map" width="256" height="256" style="width:256px;height:256px"></canvas>'+['terrain_render_controls.js','terrain_forest_brushes.js','terrain_render3d.js','terrain_renderer.js','terrain_locale.js'].map(n=>`<script src="/${n}"></script>`).join('');
  const server=http.createServer((req,res)=>{const name=new URL(req.url,'http://x').pathname.slice(1);res.setHeader('Content-Type',name.endsWith('.js')?'application/javascript':'text/html');res.end(name.endsWith('.js')?read(name):html);});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
  try{
    browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
    const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));await page.addInitScript(()=>localStorage.setItem('neural-earth-locale-v1','fr'));await page.goto(`http://127.0.0.1:${server.address().port}`);
    const forced=await page.evaluate(html=>{const doc=new DOMParser().parseFromString(html,'text/html'),select=doc.getElementById('forcedLod');return [-2,-4,-6,-7,-8,-9,-10,-11,-12].map(lod=>{select.value=String(lod);return select.value;});},read('index.html'));
    assert.deepEqual(forced,['-2','-4','-6','-7','-8','-9','-10','-11','-12'],'fine LOD choices must remain selectable');
    await page.evaluate(()=>{TerrainAmplification.mount(document.getElementById('pane'),()=>{});});
    assert.equal(await page.locator('#amplificationSettings').isVisible(),false);
    await page.evaluate(capability=>TerrainAmplification.setSupport(capability),capability);
    assert.deepEqual(await page.locator('#amplificationMode option').evaluateAll(options=>options.map(o=>o.value)),['off','hybrid','phasor','mse']);
    assert.equal(await page.locator('#amplificationMode option[value=off]').textContent(),'Classique (raffinement actuel)');
    await page.selectOption('#amplificationMode','phasor');
    await page.locator('#amplificationIntensity').fill('1.4');await page.locator('body').click();await page.keyboard.press('b');
    assert.equal(await page.evaluate(()=>TerrainAmplification.effective()),0);await page.keyboard.press('b');assert.equal(await page.evaluate(()=>TerrainAmplification.effective()),1.4);
    for(const [language,label] of [['en','Terrain amplification (experimental)'],['fr','Amplification du terrain (expérimental)'],['zh','地形增强（实验性）'],['es','Amplificación del terreno (experimental)'],['pt','Amplificação do terreno (experimental)']]){
      await page.selectOption('#ui-language',language);
      assert.equal(await page.locator('#amplificationSettings legend').textContent(),label);
      assert.equal(await page.locator('#amplificationMode').inputValue(),'phasor');
      assert.equal(await page.evaluate(()=>TerrainAmplification.effective()),1.4);
    }
    await page.selectOption('#ui-language','fr');
    await page.evaluate(()=>TerrainAmplification.set({mode:'off',intensity:1.4}));
    for(const mode of ['hybrid','phasor','mse','off']){await page.keyboard.press('Shift+B');assert.equal(await page.evaluate(()=>TerrainAmplification.effectiveMode()),mode);}
    await page.click('#amplificationCycle');assert.equal(await page.evaluate(()=>TerrainAmplification.effectiveMode()),'hybrid');
    await page.evaluate(()=>TerrainAmplification.displayed(['hybrid']));assert.match(await page.locator('#amplificationBadge').textContent(),/Hybride/);
    await page.evaluate(()=>TerrainAmplification.setSupport({available:false,modes:{hybrid:{available:false,reason:'hybrid dependency missing'},phasor:{available:false},mse:{available:false}}}));
    assert.equal(await page.evaluate(()=>TerrainAmplification.effectiveMode()),'off');
    assert.equal(await page.locator('#amplificationMode').inputValue(),'off');
    assert.match(await page.locator('#amplificationUnavailable').textContent(),/Amplification désactivée.*hybrid dependency missing/);
    assert.equal(await page.locator('#amplificationSettings').isVisible(),true);
    await page.evaluate(capability=>{window.NeuralEarthRuntime={kind:'wasm'};TerrainAmplification.setSupport(capability);},capability);
    assert.equal(await page.locator('#amplificationSettings').isVisible(),false);assert.equal(await page.evaluate(()=>TerrainAmplification.effective()),0);
    await page.evaluate(()=>{window.NeuralEarthRuntime={kind:'python'};});
    await page.evaluate(capability=>TerrainAmplification.setSupport(capability),capability);assert.equal(await page.evaluate(()=>TerrainAmplification.effectiveMode()),'hybrid');
    await page.evaluate(()=>TerrainAmplification.displayed(['phasor']));assert.match(await page.locator('#amplificationBadge').textContent(),/Phasor/);
    await page.evaluate(()=>TerrainAmplification.setSupport({available:true,modes:{mse:{available:true},phasor:{available:false}}}));
    assert.equal(await page.locator('#amplificationMode option[value=phasor]').count(),0);
    assert.equal(await page.locator('#amplificationCycle').isVisible(),false);
    await page.evaluate(()=>TerrainAmplification.setSupport({available:true,modes:{mse:{available:true},phasor:{available:true}}}));
    const result=await page.evaluate(async()=>{
      const renderer=await createTerrainRenderer(document.getElementById('map'));if(!renderer)return {skipped:true};
      const width=304,L=33*33,h=new Float32Array(width*width).fill(800),climate=new Float32Array(50*L),fill=(i,v)=>climate.fill(v,i*L,(i+1)*L);
      [22,500,.0065,2,500,.0065].forEach((v,i)=>fill(21+i,v));for(const i of [36,37,38])fill(i,.4);for(const i of [39,40,41])fill(i,.5);fill(42,.3);for(const i of [46,47,48])fill(i,.4);fill(49,1);climate.set([-2e7,-1e7,2e7,1e7,0,0,1234],45*L);
      renderer.setMode('render');const options={climate,climateWidth:33,climateHeight:33,metresPerSample:7.5,lod:-2,origin:[0,0]};
      const surface=new Uint8Array(width*width*4);for(let i=0;i<width*width;i++)surface.set([200,120,180,200],i*4);
      const gradient=new Float32Array(width*width*2).fill(.125);
      renderer.uploadTile('before',h,options);renderer.uploadTile('after',h,{...options,surface,gradient});
      const read=async key=>{const texture=renderer.tiles.get(key).color,buffer=renderer.device.createBuffer({size:256*256*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),encoder=renderer.device.createCommandEncoder();encoder.copyTextureToBuffer({texture},{buffer,bytesPerRow:1024},[256,256]);renderer.device.queue.submit([encoder.finish()]);await buffer.mapAsync(GPUMapMode.READ);const data=new Uint8Array(buffer.getMappedRange()).slice();buffer.unmap();buffer.destroy();return data;};
      const before=await read('before'),after=await read('after');const different=before.some((v,i)=>v!==after[i]);
      renderer.setMode('render3d');for(let i=0;i<400&&!renderer.getStats().render3d.ready&&!renderer.getStats().render3d.error;i++)await new Promise(r=>setTimeout(r,20));
      renderer.draw([{key:'after',lod:-2,x:0,y:0,width:256,height:256,uv:[0,0,1,1]}],{width:256,height:256,dpr:1});await renderer.device.queue.onSubmittedWorkDone();
      const stats=renderer.getStats().render3d;renderer.dispose();return {different,stats};
    });
    if(result.skipped)t.diagnostic('WebGPU adapter unavailable');else{assert.equal(result.different,true);assert.equal(result.stats.error,null);assert.equal(result.stats.ready,true);}
    assert.deepEqual(errors,[]);
  }finally{await browser?.close();await new Promise(resolve=>server.close(resolve));}
});
