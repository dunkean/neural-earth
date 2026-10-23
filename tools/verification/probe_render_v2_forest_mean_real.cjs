// Single-variable ablation: use the existing forest mean before crowns resolve.
// Production files stay unchanged; crown and patch footprint filters stay active.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module');
const root=path.resolve(__dirname,'../..');
process.env.REVIEW_OUT=process.env.REVIEW_OUT||'output/render-v2-floor/bd-forest-reveal-mean';
process.env.REVIEW_CASES=process.env.REVIEW_CASES||'af-rolling,cfb-rolling';
process.env.REVIEW_SELECTION=process.env.REVIEW_SELECTION||'output/render-v2-reference/working-selection-bd.json';
process.env.REVIEW_ZOOMS=process.env.REVIEW_ZOOMS||'10km,640m';
const runner=path.join(__dirname,'review_render_v2_real_matrix.cjs');let code=fs.readFileSync(runner,'utf8');
const anchor="   page.on('pageerror',e=>entry.errors.push(e.stack));";
if(!code.includes(anchor))throw Error('Real runner changed; forest mean ablation hook needs review');
const overrides=[
 ['var reveal=forestReveal(fp);','var reveal=select(forestReveal(fp),1.,params.field.z<=-8.);'],
 ['reveal*=mix(.06,1.,crownVis*crownVis);','reveal*=select(mix(.06,1.,crownVis*crownVis),1.,params.field.z<=-8.);']
];
const extra=`\n;(()=>{const production=TerrainRenderV2.drawShader;TerrainRenderV2.drawShader=(...a)=>{let s=production(...a);for(const [before,after]of ${JSON.stringify(overrides)}){if(!s.includes(before))throw Error('Forest mean ablation shader hook missing: '+before);s=s.replace(before,after);}return s;};})();`;
code=code.replace(anchor,anchor+`\n   await page.route('**/terrain_render_v2.js*',async route=>{await route.fulfill({status:200,contentType:'application/javascript',body:fs.readFileSync(path.join(root,'web/terrain_render_v2.js'),'utf8')+${JSON.stringify(extra)}});});\n`);
const out=path.resolve(root,process.env.REVIEW_OUT);fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,'ABLATION.json'),JSON.stringify({kind:'Real generated terrain, technical ablation of forest reveal gate only; existing material mean, filtered patch/crown reveal unchanged',overrides,productionOtherwiseUnchanged:true},null,2));
const child=new Module(runner,module);child.filename=runner;child.paths=Module._nodeModulePaths(__dirname);child._compile(code,runner);
