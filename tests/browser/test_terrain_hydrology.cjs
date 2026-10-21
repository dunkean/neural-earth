// Viewer overlay assertions (request cache, settings, hierarchy, lakes,
// inspector) and one real WebGPU probe of water colours and authority.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const root=path.resolve(__dirname,'../..');
const hydro=require(path.join(root,'web/terrain_hydrology.js'));
const {loadPlaywright,browserExecutable}=require('../../tools/platform.cjs');
// Records strokes (with their style) and fills in drawing order, and whether a
// stroke falls inside the ready-patch clip (nonzero rule).
function recorder(){const ops=[],strokes=[],fills=[];return {ops,strokes,fills,save(){},restore(){this.rule=null;},beginPath(){},rect(){},clip(rule='nonzero'){this.rule=rule;},
  moveTo(){},lineTo(){},quadraticCurveTo(){this.curved=true;},closePath(){},setLineDash(d){this.dash=d;},
  stroke(){const s={width:this.lineWidth,alpha:this.globalAlpha,colour:this.strokeStyle,dash:this.dash||[],ready:this.rule==='nonzero'};strokes.push(s);ops.push(['stroke',s]);},
  fill(rule){const f={colour:this.fillStyle,rule,curved:!!this.curved};this.curved=false;fills.push(f);ops.push(['fill',f]);}};}
const settle=ms=>new Promise(resolve=>setTimeout(resolve,ms));

(async()=>{
  let state={seed:42,profile:'natural',worldIdentity:'fixture',worldBounds:[-1e6,-1e6,1e6,1e6],bounds:[0,0,1000,1000],lod:7};
  const calls=[];
  const viewer=hydro.create({getState:()=>state,fetchJSON:async url=>{calls.push(url);return {ok:true,status:200,json:async()=>({status:'ready',reaches:[],lakes:[]})};}});
  const ctx={save(){},restore(){},setLineDash(){}};
  viewer.draw(ctx,state);await settle(0);
  viewer.draw(ctx,state);await settle(0);assert.equal(calls.length,1);
  state={...state,lod:0};viewer.draw(ctx,state);await settle(0);assert.equal(calls.length,2,'Finer geometry must be fetched inside the same bounds');
  assert.equal(new URL(calls[1],'http://fixture').searchParams.get('lod'),'0');viewer.dispose();

  // Server work stays off by default. Generated meanders were removed: old
  // URLs and recipes asking for them must not reach the server or a cache key.
  const defaults=hydro.defaults();
  for(const name of ['terrainDetail','valleyFit','smoothBanks','intermittent','smoothShores'])assert.equal(defaults[name],false,name);
  assert.deepEqual([defaults.lakes,defaults.density,defaults.minArea,defaults.widthScale,defaults.minLakeArea,defaults.minLakeDepth],[true,1,0,1,1,0]);
  assert(!('meanders' in defaults)&&!('nestedMeanders' in defaults));
  assert.deepEqual(hydro.requestParams(defaults),{});assert.equal(hydro.optionsIdentity(defaults),'');
  const legacy=hydro.normalize({meanders:true,nestedMeanders:true});
  assert(!('meanders' in legacy)&&!('nestedMeanders' in legacy));assert.deepEqual(hydro.requestParams(legacy),{});
  const old=hydro.fromParams(new URLSearchParams('hydroCosmetics=meanders,nestedMeanders,valleyFit'));
  assert.deepEqual(hydro.requestParams(old),{hydro_cosmetics:'valleyFit'});assert.equal(hydro.optionsIdentity(old),'/hydro_cosmetics-valleyFit');
  // Display settings round-trip through the URL, never change the server
  // identity, and malformed values fall back or clamp without NaN.
  const chosen=hydro.normalize({terrainDetail:true,smoothBanks:true,density:.5,minArea:20,widthScale:1.5,intermittent:true,lakes:false,minLakeArea:.2,minLakeDepth:2,smoothShores:true}),url=new URLSearchParams();
  hydro.writeParams(url,chosen);assert.deepEqual(hydro.fromParams(url),chosen);
  assert.equal(hydro.optionsIdentity(chosen),hydro.optionsIdentity({terrainDetail:true,smoothBanks:true}));
  const empty=new URLSearchParams();hydro.writeParams(empty,defaults);assert.equal(empty.toString(),'','Defaults add no URL parameter');
  const bad=hydro.fromParams(new URLSearchParams('hydroDensity=abc&hydroWidth=99&hydroMinArea=-4&hydroLakeArea=&hydroMonth=x'));
  assert.deepEqual([bad.density,bad.widthScale,bad.minArea,bad.minLakeArea,bad.month],[1,2,0,1,0]);
  assert(Object.values(hydro.normalize({density:NaN,widthScale:'x'})).every(v=>typeof v!=='number'||Number.isFinite(v)));

  const urls=[],reaches=[{id:1,xy:[[0,0],[2e5,0]],flow_m3s:20,width_m:30,strahler:4},{id:2,xy:[[0,9e4],[2e5,9e4]],flow_m3s:5000,width_m:3000,strahler:7}];
  const options=hydro.create({getState:()=>null,initial:{meanders:true,valleyFit:true},fetchJSON:async url=>{urls.push(new URL(url,'http://fixture').searchParams);return {ok:true,status:200,json:async()=>({status:'ready',reaches,lakes:[]})};}});
  const atlas={...state,lod:5,mpp:960,width:400,height:300,view:'map',bounds:[0,-1e5,3e5,2e5],project:(x,y)=>[x/960,y/960+100]};
  options.draw(recorder(),atlas);await settle(0);
  assert.equal(urls[0].get('hydro_cosmetics'),'valleyFit');assert.equal(urls[0].get('hydro_detail'),null);
  assert.deepEqual(options.tileParams(),{hydro:'1',hydro_month:'0',hydro_cosmetics:'valleyFit'});
  options.set({density:2,widthScale:1.5,minLakeArea:0,intermittent:true});options.draw(recorder(),atlas);await settle(0);
  assert.equal(urls.length,1,'Display settings restyle the stored network without a fetch');
  options.set({density:1,widthScale:1,terrainDetail:true});options.draw(recorder(),atlas);await settle(0);
  assert.equal(urls.length,2,'A changed server option must refetch, not just restyle');assert.equal(urls[1].get('hydro_detail'),'1');

  // Ready physical patches leave resolved rivers to the raster and keep the
  // narrow one at its cartographic width; native coarse/preview patches draw all.
  const ready=recorder();options.draw(ready,{...atlas,physicalWater:true,physicalRects:[{x:0,y:0,width:400,height:300}]});
  const onPatch=ready.strokes.filter(s=>s.ready);
  assert.equal(onPatch.length,1,'Only the wide river is left to the physical raster');
  assert(onPatch[0].width>=.7&&onPatch[0].alpha>=.6,`Narrow river too faint ${JSON.stringify(onPatch)}`);
  const coarse=recorder();options.draw(coarse,{...atlas,physicalWater:false,physicalRects:[]});
  assert.equal(coarse.strokes.length,2);assert(Math.abs(coarse.strokes[1].width-3000/960)<1e-9,'Wide rivers keep their hydraulic width, drawn last');
  options.dispose();

  // Hierarchy: visibility by flow and drainage at the view's threshold, with
  // downstream closure; width and opacity grow with flow; regimes are styled.
  const months=(q,dry=[])=>Array.from({length:12},(_,i)=>dry.includes(i+1)?0:q);
  const network={status:'ready',reaches:[
    {id:'trunk',xy:[[0,0],[1e5,0]],flow_m3s:500,area_m2:2e10,width_m:150,downstream_id:'lower',basin_id:3,monthly_flow_m3s:months(500).map((q,i)=>q*(.5+i/11))},
    {id:'lower',xy:[[1e5,0],[1.5e5,0]],flow_m3s:1,area_m2:2.1e10,width_m:5,downstream_id:null,basin_id:3},
    {id:'mid',xy:[[5e4,4e4],[5e4,0]],flow_m3s:5,area_m2:4e8,width_m:12,downstream_id:'trunk',basin_id:3},
    {id:'rill',xy:[[2e4,3e4],[2e4,0]],flow_m3s:.5,area_m2:1.2e7,width_m:2,downstream_id:'trunk',basin_id:3},
    {id:'wadi',xy:[[7e4,3e4],[7e4,0]],flow_m3s:8,area_m2:6e8,width_m:10,downstream_id:'trunk',basin_id:3,monthly_flow_m3s:months(12,[6,7,8,9])},
    {id:'tiny',xy:[[9e4,3e4],[9e4,0]],flow_m3s:1e-6,area_m2:1.1e7,width_m:.5,downstream_id:'trunk',basin_id:3},
  ],lakes:[
    {id:'big',rings:[[[3e4,-1e4],[4e4,-1e4],[4e4,1e4],[3e4,1e4]]],area_m2:2e8,max_depth_m:12,level_m:200,closed:false},
    {id:'pit',rings:[[[6e4,-5e3],[6.1e4,-5e3],[6.1e4,5e3]]],area_m2:2e5,max_depth_m:20,level_m:210,closed:true},
  ]};
  const fetches=[];let scene=null;
  const layered=hydro.create({getState:()=>scene,initial:{},fetchJSON:async url=>{fetches.push(url);return {ok:true,status:200,json:async()=>network};}});
  const at=mpp=>({...state,lod:2,mpp,width:1600,height:1000,view:'map',bounds:[-2e4,-5e4,2e5,5e4],project:(x,y)=>[(x+2e4)/mpp,(y+5e4)/mpp]});
  const drawAt=mpp=>{const r=recorder();scene=at(mpp);layered.draw(r,scene);return {r,snap:layered.snapshot()};};
  drawAt(120);await settle(0);
  let {r:frame,snap}=drawAt(120);
  assert.equal(snap.shownReaches,3,'Threshold at 120 m/px: trunk, mid and the downstream lower reach');
  const [lower,mid,trunk]=frame.strokes;
  assert(lower&&mid&&trunk,'Lower reach kept by downstream closure despite its low flow');
  assert(mid.width<trunk.width&&mid.alpha<trunk.alpha&&lower.width<=mid.width,'Higher flow draws wider and more opaque');
  assert(trunk.width<=4.5&&lower.width>=.7);
  ({r:frame,snap}=drawAt(15));
  assert.equal(snap.shownReaches,4,'Zooming in reveals the rill, never the dry 1e-6 m³/s path');
  assert(Math.abs(frame.strokes[frame.strokes.length-1].width-150/15)<1e-9,'Resolved trunk keeps its physical width');
  // Intermittent streams are off by default; on, a seasonal one is dashed water.
  layered.set({intermittent:true});({r:frame,snap}=drawAt(120));
  assert.equal(fetches.length,1);assert.equal(snap.shownReaches,4);
  const dashed=frame.strokes.find(s=>s.dash.length);assert(dashed&&dashed.colour!=='#9b7a50','Seasonal stream: dashed water in the annual view');
  layered.set({month:7});drawAt(120);await settle(0);assert.equal(fetches.length,2,'Month changes refetch');
  ({r:frame}=drawAt(120));
  assert(frame.strokes.some(s=>s.colour==='#9b7a50'&&s.dash.length),'Dry in the selected month: dashed brown');
  // Density 0 keeps only the main river and its continuation.
  layered.set({month:0,intermittent:false,density:0});drawAt(120);await settle(0);({snap}=drawAt(120));assert.equal(snap.shownReaches,2);
  // A fixed minimum catchment overrides the zoom-dependent threshold.
  layered.set({density:1});({snap}=drawAt(960));assert.equal(snap.shownReaches,2);
  layered.set({minArea:5});({snap}=drawAt(960));assert.equal(snap.shownReaches,4);
  layered.set({minArea:0});
  // Lakes: filled after the rivers so strokes beneath are covered; the 1 km²
  // default hides pits; threshold and shore smoothing are display only.
  ({r:frame,snap}=drawAt(120));
  assert.equal(snap.shownLakes,1);assert.equal(frame.fills.length,1);
  const kinds=frame.ops.map(o=>o[0]),firstFill=kinds.indexOf('fill');
  assert(firstFill>kinds.indexOf('stroke')&&frame.ops[firstFill][1].rule==='evenodd'&&frame.ops[firstFill][1].colour!==undefined);
  assert(kinds.lastIndexOf('stroke')>firstFill,'Shore outline follows the fill');
  layered.set({minLakeArea:0,smoothShores:true});({r:frame,snap}=drawAt(120));
  assert.equal(snap.shownLakes,2);assert(frame.fills.every(f=>f.curved));
  network.lakes[0].resolution='source';({r:frame}=drawAt(120));
  assert(frame.ops.some(([k,o])=>k==='stroke'&&o.dash.length&&o.colour==='#2d7894'),'Coarse source shoreline at Base zoom: dashed outline');
  assert.equal(frame.fills.length,1,'An approximate source lake must not paint water over detailed terrain; only the small Base lake is filled');
  delete network.lakes[0].resolution;
  layered.set({lakes:false});({snap}=drawAt(120));assert.equal(snap.shownLakes,0);
  layered.set({lakes:true,minLakeArea:1,smoothShores:false});assert.equal(fetches.length,3,'Lake settings never refetch');
  // Inspector hit tests: a lake interior, then a river line.
  drawAt(120);
  assert.equal(layered.inspect((3.5e4+2e4)/120,(4.8e3+5e4)/120),true);assert.equal(layered.snapshot().selected,'big');
  assert.equal(layered.inspect((5e4+2e4)/120,(2e4+5e4)/120),true);assert.equal(layered.snapshot().selected,'mid');
  layered.dispose();
  assert.equal(hydro.riverClass(500),'Main river');assert.equal(hydro.riverClass(.05),'Rill');
  assert.equal(hydro.regime({flow_m3s:12,monthly_flow_m3s:months(12,[6,7,8])}).regime,'seasonal');
  assert.equal(hydro.regime({flow_m3s:12,monthly_flow_m3s:months(12)}).regime,'unresolved','Flat months prove nothing');
  assert.equal(hydro.regime({flow_m3s:12,regime:'ephemeral'}).regime,'ephemeral','Backend dry regime used without a series');
  assert.deepEqual(hydro.regime({flow_m3s:700,dry_months:3,monthly_flow_m3s:months(700).map((q,i)=>i<3?5:q)}).dry,[],'5 m³/s under a 700 m³/s mean still flows');
  const counted=hydro.regime({flow_m3s:3,dry_months:4});assert.deepEqual([counted.regime,counted.dry,counted.count],['unresolved',[],4],'A count never becomes invented months');
  assert.equal(hydro.regime({flow_m3s:10,dry_months:0,low_flow_months:3}).count,3,'Low-flow periods remain distinct from dry months');
  assert.deepEqual(hydro.regime({flow_m3s:12,monthly_flow_m3s:months(12,[2,9])}).dry,[2,9]);
  assert(hydro.thresholdArea(defaults,120)>4*hydro.thresholdArea(defaults,15),'Tributaries appear gradually when zooming in');

  // A lake inflow can merge into a wetter, smaller-catchment tributary whose
  // own display threshold fails. Keep that receiver continuation as well.
  const merged=[
    {id:'in',xy:[[0,0],[1000,0]],flow_m3s:100,area_m2:2e8,downstream_id:'out'},
    {id:'owner',xy:[[1000,0],[2000,0]],flow_m3s:200,area_m2:1e6,downstream_id:'out'},
    {id:'out',xy:[[2000,0],[3000,0]],flow_m3s:300,area_m2:3e8},
    {id:'in',xy:[[1000,0],[1500,0]],flow_m3s:100,area_m2:2e8,downstream_id:'out',display_downstream_id:'owner',morphology:'connector'}
  ];
  const mergerView={...atlas,mpp:120,lod:2};
  const merger=hydro.create({getState:()=>mergerView,fetchJSON:async()=>({ok:true,status:200,json:async()=>({status:'ready',reaches:merged,lakes:[]})})});
  merger.draw(recorder(),mergerView);await settle(0);merger.draw(recorder(),mergerView);
  assert.equal(merger.snapshot().shownReaches,4,'A selected inflow retains the lake connector it merges into');
  merger.dispose();

  const continental=[
    {id:'small-head',flow_m3s:1000,area_m2:1e11,width_m:228,strahler:1,strahler_scale:'source'},
    {id:'main',flow_m3s:2000,area_m2:2e11,width_m:350,strahler:4,strahler_scale:'source',display_downstream_id:'owner'},
    {id:'large-head',flow_m3s:5000,area_m2:3e11,width_m:600,strahler:1,strahler_scale:'source'},
    {id:'owner',flow_m3s:500,area_m2:1e6,width_m:160,strahler:1,strahler_scale:'source'}
  ].map((r,i)=>({...r,xy:[[i*1000,0],[i*1000+500,0]]}));
  const continentView={...atlas,mpp:3840,lod:7};
  const continent=hydro.create({getState:()=>continentView,fetchJSON:async()=>({ok:true,status:200,json:async()=>({status:'ready',reaches:continental,lakes:[]})})});
  continent.draw(recorder(),continentView);await settle(0);continent.draw(recorder(),continentView);
  assert.equal(continent.snapshot().shownReaches,3,'Continental headwaters thin out while a downstream merge owner stays visible');
  continent.set({minArea:1});continent.draw(recorder(),continentView);
  assert.equal(continent.snapshot().shownReaches,4,'An explicit catchment threshold can reveal continental headwaters');
  continent.dispose();

  // Default networks refine as terrain arrives: a new terrain revision refetches
  // a covered view, an identical one does not, and bursts coalesce into one
  // deferred request that still sees the last revision, then stops.
  const revisions=[];let frames=0,view={...atlas,terrainRevision:1};
  const local=hydro.create({getState:()=>view,requestFrame:()=>{frames++;local.draw(recorder(),view);},fetchJSON:async url=>{revisions.push(view.terrainRevision);return {ok:true,status:200,json:async()=>({status:'ready',reaches,lakes:[]})};}});
  local.draw(recorder(),view);await settle(0);local.draw(recorder(),view);await settle(0);
  assert.deepEqual(revisions,[1],'An unchanged terrain revision reuses the network');
  view={...view,terrainRevision:2};local.draw(recorder(),view);view={...view,terrainRevision:3};local.draw(recorder(),view);
  assert.deepEqual(revisions,[1],'Terrain arrivals are coalesced, not fetched per frame');
  await settle(700);assert.deepEqual(revisions,[1,3],'The deferred frame fetches the latest revision');
  const settled=frames;await settle(700);assert.deepEqual(revisions,[1,3]);assert(frames-settled<=1,'No redraw loop once current');
  local.dispose();

  // Physical readiness needs real river water: ocean-only or dry RGBA fields
  // (e.g. a fast tile whose source rivers stayed dry) keep the vectors on.
  const fields=new Float32Array(16);fields.set([0,1,1,1],0);assert.equal(hydro.hasRiver(fields),false);
  fields.set([5,.4,3,1],4);assert.equal(hydro.hasRiver(fields),true);

  const html='<canvas id="map"></canvas><script src="/terrain_renderer.js"></script>';
  const server=http.createServer((req,res)=>{res.setHeader('Content-Type',req.url.endsWith('.js')?'application/javascript':'text/html');res.end(req.url==='/terrain_renderer.js'?fs.readFileSync(path.join(root,'web/terrain_renderer.js')):html);});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
  try{
    const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const report=await page.evaluate(async()=>{
      let failure=null;const r=await createTerrainRenderer(document.getElementById('map'),{onStatus:(s,e)=>{if(s==='fallback')failure=e;}});
      if(!r)return {skipped:true};
      r.context.configure({device:r.device,format:r.format,alphaMode:'premultiplied',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
      const N=80,h=new Float32Array(N*N).fill(10),water=new Float32Array(N*N*4);
      for(let i=0;i<N*N;i++)water.set([20,1,2,1],i*4);
      const put=(key,elevation,hydrology)=>r.uploadTile(key,elevation,{width:N,height:N,halo:8,metresPerSample:30,lod:0,hydrology});
      const sample=async(key,size=64)=>{
        r.draw([{key,lod:0,x:0,y:0,width:size,height:size}],{width:size,height:size,dpr:1});
        const buffer=r.device.createBuffer({size:256*size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const encoder=r.device.createCommandEncoder();encoder.copyTextureToBuffer({texture:r.context.getCurrentTexture()},{buffer,bytesPerRow:256},[size,size]);r.device.queue.submit([encoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);const bytes=new Uint8Array(buffer.getMappedRange()),at=Math.floor(size/2)*256+Math.floor(size/2)*4;
        const rgb=[0,1,2].map(c=>bytes[at+(r.format.startsWith('bgra')?2-c:c)]);buffer.unmap();buffer.destroy();return rgb;
      };
      put('lake',h,water);const lake=await sample('lake');
      put('legacy',h);const legacy=await sample('legacy');
      const absent=water.slice();for(let i=3;i<absent.length;i+=4)absent[i]=0;put('absent',h,absent);const noAuthority=await sample('absent');
      const below=h.slice().fill(-20),dry=water.slice();for(let i=0;i<N*N;i++)dry.set([0,0,0,1],i*4);
      put('dry',below,dry);put('sea',below);const dryRgb=await sample('dry'),seaRgb=await sample('sea');
      const stripes=water.slice();for(let y=0;y<N;y++)for(let x=0;x<N;x++)if(x%4)stripes.set([0,0,0,1],(y*N+x)*4);
      put('stripes',h,stripes);const minified=await sample('stripes',16);
      r.dispose();return {lake,legacy,noAuthority,dryRgb,seaRgb,minified,failure};
    });
    if(report.skipped){console.log('SKIP: WebGPU adapter unavailable');return;}
    assert.equal(report.failure,null);assert.deepEqual(errors,[]);assert.deepEqual(report.noAuthority,report.legacy);
    const t=Math.min(1,Math.max(0,(10-.15)/(12-.15))),blend=t*t*(3-2*t);
    const expected=[.1019608,.3019608,.3607843].map((v,i)=>Math.round(255*(v+([.0431373,.1607843,.2705882][i]-v)*blend)));
    report.lake.forEach((v,i)=>assert(Math.abs(v-expected[i])<=1,`Water colour ${report.lake} vs ${expected}`));
    assert.notDeepEqual(report.dryRgb,report.seaRgb,'An owned dry basin below sea level must render as land');
    report.minified.forEach((v,i)=>assert(Math.abs(v-(.25*report.lake[i]+.75*report.legacy[i]))<=2,
                                         'Minified subpixel water must retain its area coverage'));
    console.log(JSON.stringify(report));
  }finally{await browser.close();server.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
