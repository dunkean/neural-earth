// Resource UI only; no learned terrain, compile jobs or GPU allocation.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {execFileSync}=require('node:child_process');
const {loadPlaywright,browserExecutable,pythonExecutable}=require('../../tools/platform.cjs');
const root=path.resolve(__dirname,'../..');
const specs=JSON.parse(execFileSync(pythonExecutable(),['-c',"import sys,json;sys.path.insert(0,'backend');from terrain_resource_layers import legend_specs;print(json.dumps(legend_specs()))"],{cwd:root,encoding:'utf8',env:{...process.env,TERRAIN_RESOURCES_FANTASY:'1',PYTHONDONTWRITEBYTECODE:'1'}}));
(async()=>{
 const server=http.createServer((req,res)=>{
  if(req.url==='/terrain_resources.js'){res.setHeader('Content-Type','application/javascript');res.end(fs.readFileSync(path.join(root,'web/terrain_resources.js')));return;}
  res.setHeader('Content-Type','text/html');res.end('<meta charset="utf-8"><div id="hud"></div><select id="mapMode"></select><div id="layerPanel"></div><script>window.fetch=async()=>({status:202,json:async()=>({})});</script><script src="/terrain_resources.js"></script>');
 });
 await new Promise(ok=>server.listen(0,'127.0.0.1',ok));
 const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true});
 try{
  const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const report=await page.evaluate(async specs=>{
   const select=document.getElementById('mapMode'),layerPanel=document.getElementById('layerPanel');
   for(const [mode,spec] of Object.entries(specs)){
    select.add(new Option(spec.label,mode));
    let group=layerPanel.querySelector(`[data-group="${spec.group}"]`);
    if(!group){group=document.createElement('fieldset');group.dataset.group=spec.group;group.innerHTML='<legend>Original</legend>';layerPanel.append(group);}
    const button=document.createElement('button');button.dataset.mode=mode;group.append(button);
   }
   TerrainResources.setWorld({seed:42,generation_profile:'fixture',resource_layers:specs});
   const failures=[];
   for(const locale of ['fr','en','zh','es','pt']){
    localStorage.setItem('neural-earth-locale-v1',locale);window.dispatchEvent(new Event('neural-locale-change'));
    for(const [mode,spec] of Object.entries(specs)){
     select.value=mode;TerrainResources.sync(mode);
     const expected=TerrainResources.label(spec.label),text=document.querySelector('[data-resource-body]').textContent;
     if(select.selectedOptions[0].textContent!==expected||!text.includes(expected)||!text.includes(TerrainResources.label(spec.status)))failures.push([locale,mode]);
    }
    if(layerPanel.querySelector('[data-group="diagnostic"] legend').textContent!==TerrainResources.label('Diagnostic'))failures.push([locale,'diagnostic group']);
   }
   localStorage.setItem('neural-earth-locale-v1','en');select.value='resources-dep-copper';TerrainResources.sync(select.value);
   const source=new Uint32Array(24),values=new Float32Array(source.buffer);values[10]=41;values[22]=42;
   TerrainResources.setRenderer({res:{header:{site_indices:true,world_bounds:[-100,-100,100,100]},source:{deposits:source},aggregation:{groups:[{x:15,y:15,mass:30,members:[0,1]}]}}});
   const deposits=[41,42,43].map((site_index,i)=>({site_index,symbol_cell:[0,0],label:'Petits filons Pb–Ag',distance_m:0,medieval_recoverable:{copper:10*(i+1)},contained:{copper:100},contained_units:{copper:'t'},ore_t:10000,depth_top_m:2,footprint_m:[20,10]}));
   await TerrainResources.inspect({x:15,y:15,seed:42,profile:'fixture',mpp:30,fetchJSON:async()=>({ok:true,status:200,json:async()=>({cell:{lod:10,i:0,j:0,area_km2:1},dominant:{context:'Craton',rock:'Calcaire'},values:{},deposits})})});
   return {failures,inspection:TerrainResources.state().inspection,text:document.querySelector('[data-resource-inspector]').textContent,specs:Object.keys(specs).length};
  },specs);
  assert.deepEqual(errors,[]);assert.deepEqual(report.failures,[]);
  assert.equal(report.inspection.members,2,'coincident unrelated sites must not enter the cluster inspector');
  assert.equal(report.inspection.cluster.count,2);assert(report.text.includes('Cluster')&&report.text.includes('Copper'));
  console.log(JSON.stringify({specs:report.specs,locales:5,inspection:report.inspection}));
 }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
