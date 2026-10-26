// Causal ablations on cached generated terrain. Never edits production files.
// Each output records its precise shader/source override; no visual acceptance
// may present these diagnostic images as unmodified production.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module');
const root=path.resolve(__dirname,'../..'),variant=process.env.PATCH_PROBE||'no-clearings';
const renderer=[],draw=[],environment=[];
if(variant==='no-clearings')environment.push(['return control*mix(.4,.08,sstep(.8,1.8,humid))*(1.-.85*trop);','return 0.;']);
else if(variant==='bounded-thinning')environment.push(['return 1./(1.+exp(clamp(-4.*(score-1.07*threshold),-20.,20.)));','let meanShare=1./(1.+exp(clamp(threshold*1.702/1.15,-20.,20.)));return min(.55,2.*meanShare/(1.+exp(clamp(-score/1.15,-20.,20.))));']);
else if(variant==='constant-population')draw.push(['let population=r3Aux(p,2,meanLod);','let rawPopulation=r3Aux(p,2,meanLod);let population=vec4<f32>(.47,rawPopulation.gba);']);
else if(variant==='no-outcrop')renderer.push(['let outcrop=surfaceCover(clamp(exposure,0.,1.),vec2<f32>(.75*crag.x+.25*grove.x*sqrt(grove.y),crag.y),.35);','let outcrop=select(surfaceCover(clamp(exposure,0.,1.),vec2<f32>(.75*crag.x+.25*grove.x*sqrt(grove.y),crag.y),.35),0.,surfaceDetailOverride.w>.5);']);
else if(variant==='continuous-herbs')renderer.push(['color=surfaceMix(color,grass,surfaceCover(herbs*(1.-outcrop),sward,.12*variation));','color=surfaceMix(color,grass,select(surfaceCover(herbs*(1.-outcrop),sward,.12*variation),herbs*(1.-outcrop),surfaceDetailOverride.w>.5));']);
else if(variant==='fields')draw.push(['let groundMean=v2GroundTint(macroColor,e);','if(fp>=0.){return vec4<f32>(r3Aux(p,2,meanLod).r,e.rock,e.grass,1.);}\nlet groundMean=v2GroundTint(macroColor,e);']);
else throw Error('Unknown PATCH_PROBE: '+variant);
process.env.REVIEW_OUT=process.env.REVIEW_OUT||'output/render-v2-floor/cfb-causal/'+variant;
process.env.REVIEW_CASES=process.env.REVIEW_CASES||'cfb-rolling';
process.env.REVIEW_SELECTION=process.env.REVIEW_SELECTION||'output/render-v2-reference/working-selection-bd.json';
process.env.REVIEW_ZOOMS=process.env.REVIEW_ZOOMS||'10km';
const runner=path.join(__dirname,'review_render_v2_real_matrix.cjs');let code=fs.readFileSync(runner,'utf8');
if(process.env.PATCH_JULY_ONLY==='1'){
 code=code.replace('const expectedFrames=zooms.length*3;','const expectedFrames=zooms.length;');
 for(const line of ["     await mode('render3d');await capture(zoom,extent,'render3d-july');","     await mode('render-v2');await season(0);await capture(zoom,extent,'v2-january');"]){if(!code.includes(line))throw Error('July-only runner hook missing');code=code.replace(line,'');}
}
code=code.replace("await page.waitForFunction(()=>window.terrainDebug?.snapshot().renderer?.render3d?.ready,null,{timeout:120000});","await page.waitForFunction(()=>{const s=window.terrainDebug?.snapshot();if(s?.rendererReady&&!s.renderer)throw Error(window.__renderProbeFailure||'Renderer fallback');return s?.renderer?.render3d?.ready;},null,{timeout:120000});");
const anchor="   page.on('pageerror',e=>entry.errors.push(e.stack));";
if(!code.includes(anchor))throw Error('Real runner probe hook missing');
const extra=`\n;(()=>{const changes=${JSON.stringify(draw)},env=${JSON.stringify(environment)};for(const [a,b]of env){if(!TerrainRenderV2.environment.includes(a))throw Error('Environment probe hook missing: '+a);TerrainRenderV2.environment=TerrainRenderV2.environment.replace(a,b);}const production=TerrainRenderV2.drawShader;TerrainRenderV2.drawShader=(...args)=>{let s=production(...args);for(const [a,b]of changes){if(!s.includes(a))throw Error('Draw probe hook missing: '+a);s=s.replace(a,b);}return s;};})();`;
code=code.replace(anchor,anchor+`\n   await page.route('**/terrain_render_v2.js*',async route=>{await route.fulfill({status:200,contentType:'application/javascript',body:fs.readFileSync(path.join(root,'web/terrain_render_v2.js'),'utf8')+${JSON.stringify(extra)}});});\n   await page.route('**/terrain_renderer.js*',async route=>{let source=fs.readFileSync(path.join(root,'web/terrain_renderer.js'),'utf8');for(const [a,b]of ${JSON.stringify(renderer)}){if(!source.includes(a))throw Error('Renderer probe hook missing: '+a);source=source.replace(a,b);}source+='\\n;{const original=createTerrainRenderer;window.createTerrainRenderer=(canvas,options={})=>original(canvas,{...options,onStatus:(state,detail)=>{if(state===\\'fallback\\')window.__renderProbeFailure=detail;options.onStatus?.(state,detail);}});}';await route.fulfill({status:200,contentType:'application/javascript',body:source});});\n`);
const out=path.resolve(root,process.env.REVIEW_OUT);fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,'ABLATION.json'),JSON.stringify({kind:'CAUSAL instrumented real cached terrain, production unchanged',variant,renderer,draw,environment},null,2));
const child=new Module(runner,module);child.filename=runner;child.paths=Module._nodeModulePaths(__dirname);child._compile(code,runner);
