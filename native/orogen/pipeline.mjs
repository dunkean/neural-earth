import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {serialize,deserialize} from 'node:v8';
const request=JSON.parse(fs.readFileSync(process.argv[2],'utf8')), dir=process.argv[3];
if(request.gpuRuntime){const {install}=await import(pathToFileURL(request.gpuRuntime));install();}
globalThis.OROGEN_SKIP_POST=request.skipPost;
globalThis.OROGEN_THRESHOLD=request.threshold;
globalThis.OROGEN_MOTION=request.motion;
globalThis.OROGEN_SPREAD=request.spread;
globalThis.OROGEN_PHYSICAL_INPUT=!!request.inputFile;
globalThis.OROGEN_CLIMATE=request.climateSettings||{};
if(request.inputFile){const b=fs.readFileSync(request.inputFile);request.grayscale=new Float32Array(b.buffer,b.byteOffset,b.byteLength/4);request.cmd='importHeightmap';}else request.cmd='generate';
const {KOPPEN_CLASSES}=await import('./vendor/koppen.js');
const {biomeColor}=await import('./vendor/color-map.js');
let relief;
function save(result){
 const fields={xyz:result.r_xyz,triangles:result.triangles,plates:result.r_plate,elevation:result.r_elevation,convergence:result.r_stress,...result.debugLayers};
 for(const [key,value]of Object.entries(result))if(key.startsWith('r_')&&value?.length===result.numRegions)fields[key.slice(2)]=value;
 if(result.debugLayers.koppen){
  const colors=Array.from(result.r_elevation,(e,i)=>biomeColor(result.debugLayers.koppen[i],e));
  for(let c=0;c<3;c++){fields['biome_'+c]=Float32Array.from(colors,x=>x[c]);fields['koppen_color_'+c]=Float32Array.from(result.debugLayers.koppen,id=>KOPPEN_CLASSES[id].color[c]);}
 }
 const ids=new Map(result.plateSeeds.map((id,i)=>[id,i]));fields.plates=Float32Array.from(result.r_plate,id=>ids.get(id));
 const oceans=new Set(result.plateIsOcean);fields.crust=Float32Array.from(result.r_plate,id=>oceans.has(id)?1:0);
 const manifest={numRegions:result.numRegions,fields:{},timing:result._pipelineTiming,params:result._params,
  elevationTiming:result._timing,postTiming:result._postTiming};
 for(const [key,value]of Object.entries(fields)){if(!ArrayBuffer.isView(value))continue;const v=Float32Array.from(value);fs.writeFileSync(`${dir}/${key}.f32`,Buffer.from(v.buffer));manifest.fields[key]=v.length;}
 fs.writeFileSync(`${dir}/graph.json`,JSON.stringify(manifest));
}
globalThis.self={postMessage(result){
 if(result.type==='error')throw Error(result.stack||result.message);
 if(result.type==='done'){
  relief=request.resumeFile?{...relief,...result,debugLayers:{...relief.debugLayers,...result.debugLayers}}:result;
  save(relief);
 }
 if(result.type==='climateDone'){
  const t=result._climateTiming;
  const timing=[...relief._pipelineTiming,...['wind','ocean','precipitation','temperature','koppen'].map(stage=>({stage:'External relief climate: '+stage,ms:t[stage]}))];
  relief={...relief,...result,type:'done',debugLayers:{...relief.debugLayers,...result.climateDebugLayers},_pipelineTiming:timing};
  save(relief);
 }
}};
const worker=await import(request.gpuVendorDirectory?pathToFileURL(request.gpuVendorDirectory+'/planet-worker.js'):'./vendor/planet-worker.js');
const deferredClimate=request.externalErosion&&!request.skipClimate;
if(request.externalErosion)request.skipClimate=true;
if(request.resumeFile){
 const retained=deserialize(fs.readFileSync(request.resumeFile));
 if(request.legacyHeightConvention)worker.rebaseLegacyHeightConvention(retained.state,retained.result);
 worker.restoreRetainedState(retained.state);
 relief={...retained.result,_pipelineTiming:[],_postTiming:[]};
 if(request.resumeCommand==='erosion')self.postMessage(worker.applyRetainedErosion(request,relief.debugLayers.hotspot,request.imported));
 else if(request.resumeCommand==='climate')self.onmessage({data:{...request,cmd:'computeClimate'}});
 else throw Error('Invalid retained Orogen stage');
}else self.onmessage({data:request});
if(request.externalErosion){
 // Python owns the GPU; stdin resumes this retained world after relief.
 fs.writeFileSync(`${dir}/relief.ready`,'ready');
 const response=request.gpuRuntime?globalThis.OROGEN_GPU_EXTERNAL():JSON.parse(fs.readFileSync(0,'utf8'));
 const bytes=fs.readFileSync(response.elevationFile);
 const elevation=new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
 worker.setExternalElevation(elevation);
 const original=relief.r_elevation;
 relief.debugLayers.erosionDelta=Float32Array.from(elevation,(e,i)=>e-original[i]);
 relief.r_elevation=elevation;
 if(!deferredClimate)save(relief);
 else self.onmessage({data:{cmd:'computeClimate',temperatureOffset:request.temperatureOffset,
  precipitationOffset:request.precipitationOffset,landCoverage:request.landCoverage}});
}
if(request.snapshotOutput)fs.writeFileSync(request.snapshotOutput,serialize({state:worker.exportRetainedState(),result:relief}));
if(request.gpuRuntime)globalThis.OROGEN_GPU_FINISH();
