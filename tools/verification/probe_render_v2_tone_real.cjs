// Real terrain ablation of the V2 aux2.a payload interpretation only.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module');
const root=path.resolve(__dirname,'../..');
process.env.REVIEW_OUT=process.env.REVIEW_OUT||'output/render-v2-floor/bd-tone-legacy';
process.env.REVIEW_CASES=process.env.REVIEW_CASES||'af-rolling,cfb-rolling';
process.env.REVIEW_SELECTION=process.env.REVIEW_SELECTION||'output/render-v2-reference/working-selection-bd.json';
const runner=path.join(__dirname,'review_render_v2_real_matrix.cjs');let code=fs.readFileSync(runner,'utf8');
const anchor="   page.on('pageerror',e=>entry.errors.push(e.stack));";
if(!code.includes(anchor))throw Error('Real runner changed; tone ablation hook needs review');
const before='st.tone=select(field.a,v2CanopyTone,params.field.z<=-8.);',after='st.tone=field.a;';
const extra=`\n;(()=>{const production=TerrainRenderV2.drawShader;TerrainRenderV2.drawShader=(...a)=>{const s=production(...a);if(!s.includes(${JSON.stringify(before)}))throw Error('Tone ablation shader hook missing');return s.replace(${JSON.stringify(before)},${JSON.stringify(after)});};})();`;
code=code.replace(anchor,anchor+`\n   await page.route('**/terrain_render_v2.js*',async route=>{await route.fulfill({status:200,contentType:'application/javascript',body:fs.readFileSync(path.join(root,'web/terrain_render_v2.js'),'utf8')+${JSON.stringify(extra)}});});\n`);
const out=path.resolve(root,process.env.REVIEW_OUT);fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,'ABLATION.json'),JSON.stringify({kind:'Real generated terrain, technical single-variable shader ablation: restore bank payload interpreted as legacy canopy tone',before,after,productionOtherwiseUnchanged:true},null,2));
const child=new Module(runner,module);child.filename=runner;child.paths=Module._nodeModulePaths(__dirname);child._compile(code,runner);
