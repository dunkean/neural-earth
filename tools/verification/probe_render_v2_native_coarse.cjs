// Separate REAL native-coarse receipt. Default is an offline plan only.
// Never edits the main reference manifest/protocol or counts toward its frames.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),Module=require('node:module');
const root=path.resolve(__dirname,'../..');
const main=path.join(root,'output/render-v2-reference');
const parentFreeze=path.resolve(root,process.env.REVIEW_NATIVE_FREEZE||path.join(main,'freeze.json'));
const frozen=JSON.parse(fs.readFileSync(parentFreeze,'utf8'));
const codeLabel=frozen.codeHash.slice(0,12);
const mainRun=[codeLabel,frozen.manifestHash.slice(0,8),frozen.protocolHash.slice(0,8)].join('-');
const base=path.resolve(root,process.env.REVIEW_NATIVE_OUTPUT||'output/render-v2-native-coarse-'+codeLabel);
const manifest=JSON.parse(fs.readFileSync(path.join(main,'manifest.json'),'utf8'));
assert.equal(manifest.worldIdentity,frozen.worldIdentity);
manifest.sites=manifest.sites.filter(s=>['arid-plain','dfc-rolling'].includes(s.id));
assert.equal(manifest.sites.length,2);
manifest.label='SEPARATE REAL native-coarse probe; not part of the main900/hydro90';
manifest.zooms=[4,6].map(lod=>({id:'native-lod'+lod,lod,extentMetres:1024*30*2**lod,
 sourceResolutionMetres:7680,geometryLod:7,sourceQualityLod:4,expectedStage:'coarse'}));
manifest.expectedFrames=12;
manifest.nativeCoarseScope={parentRun:mainRun,parentFreeze:path.relative(root,parentFreeze),productionCodeHash:frozen.codeHash,
 source:'Real learned coarse blocks; geometryLOD7 and physical samples7680m, source quality4 for BOTH forcedLOD4 and6',
 morphology:'No new30m morphology claim; same centres as main, larger contextual extents491.52/1966.08km',
 limitation:'Population is the documented deterministic expected proxy, not a resolved crown field or exact fine mean',
 acceptance:'Pending actual rendering and human image review; planning does not certify quality'};
fs.mkdirSync(base,{recursive:true});fs.writeFileSync(path.join(base,'manifest.json'),JSON.stringify(manifest,null,2));
process.env.REVIEW_REFERENCE_BASE=base;
const runner=path.join(__dirname,'review_render_v2_reference.cjs');let code=fs.readFileSync(runner,'utf8');
function replace(a,b){assert(code.includes(a),'Reference hook changed: '+a);code=code.replace(a,b);}
replace("const expectedStage=z.lod===3?'latent':'decoder';","const expectedStage=z.expectedStage;");
replace('s.rendered.lods.every(l=>l===z.lod)','s.rendered.lods.every(l=>l===z.geometryLod)');
replace('s.rendered.qualityLods.every(l=>l===z.lod)','s.rendered.qualityLods.every(l=>l===z.sourceQualityLod)');
replace("assert.equal(state.mode,tag==='render3d-july'?'render3d':'render-v2');",`assert.equal(state.mode,tag==='render3d-july'?'render3d':'render-v2');
     assert(state.coarseGpu&&state.cache.keys.some(k=>k.includes('/native-coarse/')),'Missing actual native-coarse blocks');
     assert.equal(state.renderer.render3d.tiles,0,'Unexpected fine r3 pipeline tiles in the native-coarse probe');
     state.nativeCoarseEvidence={forcedLod:z.lod,geometryLod:z.geometryLod,sourceQualityLod:z.sourceQualityLod,
      sourceResolutionMetres:z.sourceResolutionMetres,stage:z.expectedStage,proxyPopulation:true};`);
const child=new Module(runner,module);child.filename=runner;child.paths=Module._nodeModulePaths(__dirname);child._compile(code,runner);
