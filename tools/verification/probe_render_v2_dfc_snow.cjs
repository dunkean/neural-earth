// Isolated real-world causal probes. Copies the reference manifest to a
// diagnostic directory, never changes its production protocol or main receipt.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module');
const root=path.resolve(__dirname,'../..'),variant=process.env.SNOW_PROBE||'snow-zero';
const env=[],draw=[],expected=[],rendererChanges=[];let lighting='';
const snow='return clamp(sstep(2.,-5.,temperature-.8*valley)*sstep(.12,.7,potential)*(1.-sstep(.45,1.25,slope)),0.,1.);';
if(variant==='production'){}
else if(variant==='marine-clear-water')env.push(['max(depth,0.)*.15*max(.08,cover*cover)','max(depth,0.)*select(.15,.035,kind<1.5)*max(.08,cover*cover)']);
else if(variant==='snow-zero')env.push([snow,'return 0.;']);
else if(variant==='snow-climate')env.push([snow,'return clamp(sstep(2.,-5.,temperature)*sstep(.12,.7,potential),0.,1.);']);
else if(variant==='snow-no-valley')env.push([snow,'return clamp(sstep(2.,-5.,temperature)*sstep(.12,.7,potential)*(1.-sstep(.45,1.25,slope)),0.,1.);']);
else if(variant==='light-off')lighting='const originalVectors=TerrainLighting.vectors;TerrainLighting.vectors=lod=>{const v=originalVectors(lod);v[0]=0;return v;};';
else if(variant==='macro-normal'){
 draw.push(['v2MacroNormal=normalize(vec3<f32>(-s.y*2.*params.lighting.w,1.,-s.z*2.*params.lighting.w));','v2MacroNormal=normalize(mix(macroNormal,normalize(vec3<f32>(-s.y*2.*params.lighting.w,1.,-s.z*2.*params.lighting.w)),1.-groundSmooth(8.,32.,fp)));']);
 draw.push(['m.normal=normalize(vec3<f32>(-s.y*2.*params.lighting.w-detail.y*gain,1.,-s.z*2.*params.lighting.w-detail.w*gain));','m.normal=normalize(mix(macroNormal,normalize(vec3<f32>(-s.y*2.*params.lighting.w-detail.y*gain,1.,-s.z*2.*params.lighting.w-detail.w*gain)),1.-groundSmooth(8.,32.,fp)));']);
}else if(variant==='coarse-constant-population')rendererChanges.push(['let population=r3ForestExpected(height,length(gradient));','let actualPopulation=r3ForestExpected(height,length(gradient));let population=vec4<f32>(.7,actualPopulation.gba);']);
else if(variant==='coarse-no-bog')expected.push(['let bog=valley*(1.-sstep(.02,.08,slope))*sstep(1.6,3.,humid)*(1.-sstep(11.,17.,grow))*(1.-trop);','let bog=0.;']);
else if(variant==='coarse-solid')rendererChanges.push(['vec4<f32>(groundToDisplay(response),canopy)','vec4<f32>(1.,0.,0.,1.)']);
else if(variant==='fine-constant-population')draw.push(['let population=r3Aux(p,2,meanLod);','let actualPopulation=r3Aux(p,2,meanLod);let population=vec4<f32>(.7,actualPopulation.gba);']);
else if(variant==='fine-no-bog')rendererChanges.push(['return v2ForestTarget(r3.shaders.bake).replace(legacy,',"return v2ForestTarget(r3.shaders.bake.replace('let bog=valley*(1.-sstep(.02,.08,slope))*sstep(1.6,3.,humid)*(1.-sstep(11.,17.,grow))*(1.-trop);','let bog=0.;')).replace(legacy,"]);
else if(['fine-no-disturbance','fine-no-stand','fine-no-eligibility'].includes(variant)){
 const replacements={
  'fine-no-disturbance':['let disturbed=sstep(1.05,1.5,r3Fbm(5,2,p+vec2<f32>(-wy,wx),seed+19u));','let disturbed=0.;'],
  'fine-no-stand':['let stand=r3Fbm(2,3,p+vec2<f32>(wx,wy),seed+17u);','let stand=0.;'],
  'fine-no-eligibility':['let eligible=rock*(1.-r3Perennial)*sstep(5.5,13.5,r3THot+.8*valley);','let eligible=1.;']
 };const [before,after]=replacements[variant];rendererChanges.push(['return v2ForestTarget(r3.shaders.bake).replace(legacy,',`return v2ForestTarget(r3.shaders.bake.replace(${JSON.stringify(before)},${JSON.stringify(after)})).replace(legacy,`]);
}
else if(variant==='fine-optical-coverage'){
 const before='cover=clamp(cover*(1.+mosaic*stand)+.05*stand*cover*(1.-cover),0.,1.)*(1.-.3*disturbed*(1.-.7*trop));';
 const after='cover=1.-pow(max(1.-cover,.001),exp(mosaic*stand+.05*stand*(1.-cover))*(1.-.3*disturbed*(1.-.7*trop)));';
 rendererChanges.push(['return v2ForestTarget(r3.shaders.bake).replace(legacy,',`return v2ForestTarget(r3.shaders.bake.replace(${JSON.stringify(before)},${JSON.stringify(after)})).replace(legacy,`]);
}
else if(variant==='snow-litter'){
 draw.push(['var<private> v2CanopyTone:f32;','var<private> v2CanopyTone:f32;var<private> v2GroundSnow:f32;']);
 draw.push(['v2CanopyTone=e.tone;','v2CanopyTone=e.tone;v2GroundSnow=e.snow;']);
 draw.push(['return mix(ground,litter,.65*(1.-.6*st.dry));','if(params.field.z<=-8.){litter=mix(litter,ground,v2GroundSnow);}return mix(ground,litter,.65*(1.-.6*st.dry));']);
}
else throw Error('Unknown SNOW_PROBE '+variant);
const out=path.resolve(root,process.env.SNOW_PROBE_OUT||'output/render-v2-floor/dfc-snow-causal/'+variant);fs.mkdirSync(out,{recursive:true});
const manifest=JSON.parse(fs.readFileSync(path.join(root,'output/render-v2-reference/manifest.json'),'utf8'));
const selected=process.env.REVIEW_CASES?.split(',')||['dfc-rolling'];manifest.sites=manifest.sites.filter(s=>selected.includes(s.id));
const zooms=process.env.SNOW_ZOOMS?.split(',')||['153.6km-lod3','40km'];manifest.zooms=manifest.zooms.filter(z=>zooms.includes(z.id));manifest.styles=[process.env.SNOW_CONTROL_R3==='1'?'render3d-july':'v2-july'];manifest.expectedFrames=manifest.sites.length*manifest.zooms.length;
fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2));process.env.REVIEW_REFERENCE_BASE=out;
const runner=path.join(__dirname,'review_render_v2_reference.cjs');let code=fs.readFileSync(runner,'utf8');
code=code.replace("assert.deepEqual(manifest.styles,['v2-july','render3d-july','v2-january']);",`assert.deepEqual(manifest.styles,${JSON.stringify(manifest.styles)});`);
for(const line of ["     await mode('render3d');await capture(z,'render3d-july');","     await mode('render-v2');await season(0);await capture(z,'v2-january');"]){if(!code.includes(line))throw Error('Probe calendar hook missing');code=code.replace(line,'');}
if(process.env.SNOW_CONTROL_R3==='1')code=code.replace("await mode('render-v2');await season(.5);await capture(z,'v2-july');","await mode('render3d');await season(.5);await capture(z,'render3d-july');");
code=code.replace("await page.waitForFunction(()=>window.terrainDebug?.snapshot().renderer?.render3d?.ready,null,{timeout:120000});","await page.waitForFunction(()=>{const s=window.terrainDebug?.snapshot();if(s?.rendererReady&&!s.renderer)throw Error(window.__renderProbeFailure||'Renderer fallback');return s?.renderer?.render3d?.ready;},null,{timeout:120000});");
code=code.replace('const state=await settle(z),dimensions=', 'const state=await settle(z);state.rendererProbe=await page.evaluate(()=>[...(window.__renderProbeRenderer?.tiles.values()||[])].map(t=>({native:t.native,params:t.params,format:t.albedoFormat,aux:t.auxFormat,bakedKey:t.bakedKey})));const dimensions=');
const extra=`\n;(()=>{const changes=${JSON.stringify(draw)},env=${JSON.stringify(env)};for(const [a,b]of ${JSON.stringify(expected)}){if(!TerrainRender3D.shaders.forestExpected.includes(a))throw Error('Coarse expected hook missing');TerrainRender3D.shaders.forestExpected=TerrainRender3D.shaders.forestExpected.replace(a,b);}for(const [a,b]of env){if(!TerrainRenderV2.environment.includes(a))throw Error('Snow environment hook missing');TerrainRenderV2.environment=TerrainRenderV2.environment.replace(a,b);}const production=TerrainRenderV2.drawShader;TerrainRenderV2.drawShader=(...args)=>{let s=production(...args);for(const [a,b]of [...env,...changes]){if(!s.includes(a))throw Error('Snow draw hook missing: '+a);s=s.replace(a,b);}return s;};${lighting}})();`;
const anchor="   page.on('pageerror',e=>entry.errors.push(e.stack));";
if(!code.includes(anchor))throw Error('Reference page hook missing');
code=code.replace(anchor,anchor+`\n   await page.route('**/terrain_render_v2.js*',async route=>{await route.fulfill({status:200,contentType:'application/javascript',body:fs.readFileSync(path.join(root,'web/terrain_render_v2.js'),'utf8')+${JSON.stringify(extra)}});});\n   await page.route('**/terrain_renderer.js*',async route=>{let source=fs.readFileSync(path.join(root,'web/terrain_renderer.js'),'utf8');for(const [a,b]of ${JSON.stringify(rendererChanges)}){if(!source.includes(a))throw Error('Coarse renderer hook missing');source=source.replace(a,b);}source+='\\n;{const original=createTerrainRenderer;window.createTerrainRenderer=(canvas,options={})=>original(canvas,{...options,onStatus:(s,d)=>{if(s===\\'fallback\\')window.__renderProbeFailure=d;options.onStatus?.(s,d);}}).then(r=>{window.__renderProbeRenderer=r;return r;});}';await route.fulfill({status:200,contentType:'application/javascript',body:source});});\n`);
fs.writeFileSync(path.join(out,'ABLATION.json'),JSON.stringify({kind:'Isolated technical causal ablation of REAL frozen-world sources; not production acceptance',variant,env,draw,expected,rendererChanges,lighting,mainProtocolUnchanged:true},null,2));
function run(){const child=new Module(runner,module);child.filename=runner;child.paths=Module._nodeModulePaths(__dirname);child._compile(code,runner);}
process.argv=process.argv.filter(a=>!['--run','--freeze'].includes(a));process.argv.push('--freeze');run();process.argv.pop();process.argv.push('--run');run();
