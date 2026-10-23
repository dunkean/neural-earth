// Technical readback of real GPU-baked canopy fields. Not an appearance plate.
// R=legacy ecology coverage, G=actual candidate field mip0, B=field mip4.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module');
const root=path.resolve(__dirname,'../..');
process.env.REVIEW_OUT=process.env.REVIEW_OUT||'output/render-v2-floor/real-canopy-fields';
process.env.REVIEW_CASES=process.env.REVIEW_CASES||'af-rolling';process.env.REVIEW_ZOOMS='640m';
const runner=path.join(__dirname,'review_render_v2_real_matrix.cjs');
let code=fs.readFileSync(runner,'utf8');
const anchor="   page.on('pageerror',e=>entry.errors.push(e.stack));";
if(!code.includes(anchor))throw Error('Real runner changed; probe hook needs review');
const shaderAnchor='let groundMean=v2GroundTint(macroColor,e);';
const replacement='if(fp>=0.){return vec4<f32>(r3Aux(p,4,0.).r,r3Aux(p,2,0.).r,r3Aux(p,2,4.).r,1.);}\n'+shaderAnchor;
code=code.replace(anchor,anchor+`\n   await page.route('**/terrain_render_v2.js*',async route=>{const source=fs.readFileSync(path.join(root,'web/terrain_render_v2.js'),'utf8');const extra=${JSON.stringify(`\n;(()=>{const production=TerrainRenderV2.drawShader;TerrainRenderV2.drawShader=(...a)=>{const s=production(...a);if(!s.includes(${JSON.stringify(shaderAnchor)}))throw Error('Real canopy probe shader hook missing');return s.replace(${JSON.stringify(shaderAnchor)},${JSON.stringify(replacement)});};})();`)};await route.fulfill({status:200,contentType:'application/javascript',body:source+extra});});\n`);
fs.mkdirSync(path.resolve(root,process.env.REVIEW_OUT),{recursive:true});
fs.writeFileSync(path.resolve(root,process.env.REVIEW_OUT,'PROBE.json'),JSON.stringify({kind:'TECHNICAL actual cached GPU field probe, not appearance acceptance',channels:{R:'r3Aux(p,4,0).r ecology',G:'r3Aux(p,2,0).r actual candidate cover',B:'r3Aux(p,2,4).r candidate cover averaged16x16 DEM samples'},quantization:'RGB screenshot 8bit',sourceOverride:{shaderAnchor,replacement}},null,2));
const child=new Module(runner,module);child.filename=runner;child.paths=Module._nodeModulePaths(__dirname);child._compile(code,runner);
