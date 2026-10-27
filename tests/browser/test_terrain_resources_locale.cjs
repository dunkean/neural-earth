// Enumerate the exact server serializer, including the optional fantasy specs.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const {pythonExecutable}=require('../../tools/platform.cjs');
test('every served resource label, unit, status and class has all five locales',()=>{
  const context={window:{addEventListener(){}},document:{getElementById:()=>null},navigator:{language:'en'},localStorage:{getItem:()=>null}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../../web/terrain_resources.js'),'utf8'),context);
  const specs=JSON.parse(execFileSync(pythonExecutable(),['-c',"import sys,json;sys.path.insert(0,'backend');from terrain_resource_layers import legend_specs;print(json.dumps(legend_specs()))"],{cwd:path.join(__dirname,'../..'),env:{...process.env,TERRAIN_RESOURCES_FANTASY:'1',PYTHONDONTWRITEBYTECODE:'1'},encoding:'utf8'}));
  const labels=new Set();
  for(const spec of Object.values(specs)){
    for(const key of ['label','unit','status'])labels.add(spec[key]);
    for(const [name] of spec.classes||[])labels.add(name);
  }
  labels.add('Diagnostic');
  const missing=[...labels].filter(v=>!context.window.TerrainResources.hasTranslation(v));
  assert.deepEqual(missing,[],'Missing translations from the server specs');
  for(const text of labels)for(const locale of ['fr','en','zh','es','pt'])assert(context.window.TerrainResources.label(text,locale)?.trim(),`${locale}: ${text}`);
  console.log(`${Object.keys(specs).length} server specs, ${labels.size} labels, five locales`);
});
