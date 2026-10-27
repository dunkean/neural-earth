import {readFile,writeFile,cp,mkdir,stat,access,rm} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createHash} from 'node:crypto';
import {spawn} from 'node:child_process';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {fileURLToPath} from 'node:url';
import {resolve,join,dirname} from 'node:path';
const root=fileURLToPath(new URL('../../',import.meta.url)),out=join(root,'wasm/dist');
const run=(command,args,cwd=root)=>new Promise((resolve,reject)=>{
  const process=spawn(command,args,{cwd,stdio:'inherit',shell:false});process.on('error',reject);process.on('exit',code=>code===0?resolve():reject(Error(`${command} exited ${code}`)));
});
const exists=path=>access(path).then(()=>true,()=>false);
if(!await exists(join(root,'native/magna-urbis/rust/crates/neural/src/pipeline.rs')))await run('git',['submodule','update','--init','native/magna-urbis']);
await import('./prepare-engine.mjs');
if(process.argv.includes('--refresh-config')) {
  const python=process.env.PYTHON||join(root,process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python');
  await run(python,['tools/wasm/export-config.py']);
}
if(!process.argv.includes('--skip-compile')) {
  for(const [crate,outDir]of [['neural-earth','pkg'],['neural-earth-render','render-pkg']])await run(process.platform==='win32'?'wasm-pack.exe':'wasm-pack',['build',`native/${crate}`,'--target','web','--release','--out-dir',`../../wasm/${outDir}`]);
}
const sibling=join(root,'../city_generator/rust/pkg/neural-weights');
const weights=resolve(process.env.NEURAL_EARTH_WASM_WEIGHTS||(await exists(join(sibling,'coarse.safetensors'))&&await exists(join(sibling,'decoder.safetensors'))?sibling:join(root,'wasm/models')));
if(!await exists(join(weights,'coarse.safetensors'))||!await exists(join(weights,'decoder.safetensors'))) {
  await mkdir(weights,{recursive:true});
  await run('cargo',['run','--release','--manifest-path','native/neural-earth-weights/Cargo.toml','--',
    join(root,'distill/weights/coarse4.pt'),join(root,'distill/weights/decoder.pt'),weights]);
}
await mkdir(join(out,'runtime/models'),{recursive:true});
const {preparePackedBase}=await import('./packed-base.mjs');
const packedBase=await preparePackedBase(root,join(out,'runtime/models/base.nf4'));
// Explicit generated artifact from the former original-base build.
await rm(join(root,'wasm/dist/runtime/models/base.safetensors'),{force:true});
await cp(join(root,'web'),out,{recursive:true});
// Keep the in-app method links available in static and offline WASM builds.
for(const directory of ['design','methods'])await cp(join(root,'docs',directory),join(out,'docs',directory),{recursive:true});
for(const name of ['client.mjs','runtime-worker.mjs','orogen-worker.mjs','models.mjs','physics.mjs','physical-tile.mjs','share.mjs','generation-schema.json','webgpu.mjs'])await cp(join(root,'wasm',name),join(out,'runtime',name));
await cp(join(root,'wasm/pkg'),join(out,'runtime/pkg'),{recursive:true});
await cp(join(root,'wasm/render-pkg'),join(out,'runtime/render-pkg'),{recursive:true});
const globeSource=(await readFile(join(root,'web/terrain_globe.js'),'utf8')).replace(/\r\n/g,'\n');
const camera=globeSource.slice(globeSource.indexOf('    function orbitBy'),globeSource.indexOf("    canvas.addEventListener('pointerdown'"));
const input=globeSource.slice(globeSource.indexOf("    canvas.addEventListener('pointerdown'"),globeSource.indexOf('    const api='));
const coordinates=globeSource.slice(globeSource.indexOf('      screenToWorld('),globeSource.indexOf('      setImage('));
if(!camera.includes('function camera')||!input.includes('pointermove')||!coordinates.includes('worldToScreen'))throw Error('Globe integration does not match shared source');
await writeFile(join(out,'runtime/globe.mjs'),(await readFile(join(root,'wasm/globe.mjs'),'utf8')).replace('/* SHARED_CAMERA */',camera).replace('/* SHARED_INPUT */',input).replace('/* SHARED_COORDINATES */',coordinates));
await cp(join(root,'native/orogen/vendor'),join(out,'runtime/orogen'),{recursive:true});
await cp(join(root,'native/orogen/climate-parameters.json'),join(out,'runtime/climate-parameters.json'));
let delaunator=(await readFile(join(out,'runtime/orogen/delaunator.cjs'),'utf8')).replace(/\r\n/g,'\n');
const start=delaunator.indexOf("const epsilon"),end=delaunator.lastIndexOf('return Delaunator;');
if(start<0||end<start)throw Error('Unrecognized Delaunator bundle');
await writeFile(join(out,'runtime/orogen/delaunator.js'),delaunator.slice(start,end)+'\nexport default Delaunator;\n');
const worker=join(out,'runtime/orogen/planet-worker.js');
await writeFile(worker,(await readFile(worker,'utf8')).replace("'./delaunator.cjs'","'./delaunator.js'"));
let html=(await readFile(join(root,'web/index.html'),'utf8')).replace(/\r\n/g,'\n');
// Parser-blocking ES modules ensure the local transport exists before the shared UI starts.
html=html.replace('<script src="/terrain_transport.js"></script>','<script>window.NeuralEarthRuntime={kind:"wasm",minLOD:0,renderLabel:"WASM",preferencesKey:"neural-earth-wasm-preferences-v1"};</script>');
html=html.replace('<script>\n\'use strict\';','<script type="module">\nimport "./runtime/client.mjs";\nwindow.installWasmPreferences();\n\'use strict\';');
// CRLF input is normalized by the shared source edits.
if(!html.includes('window.installWasmPreferences();'))throw Error('Shared viewer bootstrap not found');
html=html.replace('Connecting to generator…','Loading Neural Earth · Rust / WASM…');
html=html.replace('<meta charset="utf-8">','<link rel="icon" href="data:,">\n<meta charset="utf-8">');
html=html.replaceAll('src="/','src="./');
await writeFile(join(out,'index.html'),html);
const styles=JSON.parse(await readFile(join(root,'web/terrain_styles.json'),'utf8'));
await writeFile(join(out,'terrain_styles.js'),`window.TerrainStyles=${JSON.stringify(styles)};\n`+await readFile(join(root,'web/terrain_style_rendering.js'),'utf8'));
// This optional NN panel accesses the viewer's lexical state. Keep it in the
// same generated module, as Python's classic scripts share that state globally.
html=html.replace('<script src="./terrain_model_controls.js"></script>','');
const closing=html.lastIndexOf('</script>');
html=html.slice(0,closing)+'\n'+await readFile(join(root,'web/terrain_model_controls.js'),'utf8')+'\n'+html.slice(closing);
await writeFile(join(out,'index.html'),html);
const models={};
for(const name of ['coarse','base','decoder']) {
  const file=name==='base'?'base.nf4':`${name}.safetensors`;
  const source=name==='base'?join(out,'runtime/models',file):join(weights,file),digest=createHash('sha256');
  for await(const chunk of createReadStream(source))digest.update(chunk);
  models[name]={file,bytes:(await stat(source)).size,sha256:digest.digest('hex'),...(name==='base'?packedBase:{})};
  if(name!=='base')await cp(source,join(out,'runtime/models',models[name].file));
}
const {stdout:engineRevision}=await promisify(execFile)('git',['-C','native/magna-urbis','rev-parse','HEAD'],{cwd:root});
const runtimeHash=createHash('sha256');
for(const file of ['runtime/pkg/neural_earth_wasm_bg.wasm','runtime/render-pkg/neural_earth_render_bg.wasm','runtime/runtime-worker.mjs','runtime/physics.mjs','runtime/webgpu.mjs','runtime/generation-schema.json','runtime/climate-parameters.json'])runtimeHash.update(await readFile(join(out,file)));
await writeFile(join(out,'runtime/models/manifest.json'),JSON.stringify({version:1,engine:'magna-urbis-neural',engineRevision:engineRevision.trim(),runtimeSha256:runtimeHash.digest('hex'),precision:'adaptive-fp16-fp32',profile:'coarse4-nf4-b75-t2-distilled-decoder',models},null,2)+'\n');
console.log(`Static Neural Earth: ${out}\nRun npm run serve:wasm (no Python or inference service).`);
