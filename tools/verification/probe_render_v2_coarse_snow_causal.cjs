// Real coarse captures with an explicit browser-only causal shader override.
// This is diagnostic evidence, never a final production image set.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),variant=process.env.COARSE_ABLATION||'no-population';
const overrides={
 'no-population':['let population=r3ForestExpected(height,length(gradient));','let population=vec3<f32>(0.);'],
 'full-snow':['let beach=clamp(r3Beach/.8,0.,1.);let eligible=select(0.,1.,height>=0.);','e.snow=1.;let beach=clamp(r3Beach/.8,0.,1.);let eligible=select(0.,1.,height>=0.);'],
};
assert(overrides[variant],'Unknown diagnostic variant');
const [before,after]=overrides[variant],source=fs.readFileSync(path.join(root,'web/terrain_renderer.js'),'utf8');
assert.equal(source.split(before).length,2,'Expected one precise coarse V2 hook');
const overridden=source.replace(before,after),sha=s=>crypto.createHash('sha256').update(s).digest('hex');
process.env.REVIEW_NATIVE_FREEZE=process.env.REVIEW_NATIVE_FREEZE||'output/render-v2-snow-annual-reference/freeze.json';
process.env.REVIEW_NATIVE_OUTPUT='output/render-v2-coarse-causal/'+variant;
process.env.REVIEW_CASES='dfc-rolling';
const base=path.join(root,process.env.REVIEW_NATIVE_OUTPUT);fs.mkdirSync(base,{recursive:true});
const evidence={kind:'Explicit diagnostic override on real generated coarse blocks, not a production appearance pass',variant,before,after,productionSourceSha256:sha(source),overriddenSourceSha256:sha(overridden)};
fs.writeFileSync(path.join(base,'ABLATION.json'),JSON.stringify(evidence,null,2));
const runner=path.join(__dirname,'review_render_v2_reference.cjs'),original=Module.prototype._compile;
Module.prototype._compile=function(code,filename){
 if(filename===runner){
  const anchor="   page.on('pageerror',e=>entry.errors.push(e.stack));";
  assert(code.includes(anchor),'Capture runner hook changed');
  code=code.replace(anchor,anchor+`\n   entry.shaderAblation=${JSON.stringify(evidence)};\n   await page.route('**/terrain_renderer.js*',async route=>{await route.fulfill({status:200,contentType:'application/javascript',body:${JSON.stringify(overridden)}});});`);
 }
 return original.call(this,code,filename);
};
require('./probe_render_v2_native_coarse.cjs');
