// RPC is the sole runtime boundary: no API request leaves the browser.
const worker=new Worker(new URL('./runtime-worker.mjs',import.meta.url),{type:'module'});
const pending=new Map();let sequence=0;
export function rpc(path,options={}) {
  const id=++sequence, signal=options.signal;
  return new Promise((resolve,reject)=>{
    const abort=()=>{pending.delete(id);worker.postMessage({cancel:id});reject(new DOMException('Request aborted','AbortError'));};
    if(signal?.aborted){abort();return;}
    signal?.addEventListener('abort',abort,{once:true});
    pending.set(id,{resolve,reject,cleanup:()=>signal?.removeEventListener('abort',abort)});
    worker.postMessage({id,path:String(path),method:options.method||'GET',body:options.body||null});
  });
}
worker.onmessage=({data})=>{
  if(data.progress){const status=document.getElementById('status');if(status)status.textContent=data.progress;return;}
  const request=pending.get(data.id);if(!request)return;pending.delete(data.id);request.cleanup();
  if(data.error){request.reject(Error(data.error));return;}
  request.resolve(new Response(data.body??null,{status:data.status||200,headers:data.headers}));
};
worker.onerror=event=>{for(const request of pending.values()){request.cleanup();request.reject(Error(event.message||'WASM worker stopped'));}pending.clear();};
window.TerrainTransport={fetch:rpc,release:(path,data={})=>worker.postMessage({id:0,path,method:'POST',body:JSON.stringify(data)})};
window.NeuralEarthRuntime={kind:'wasm',minLOD:0,renderLabel:'WASM',preferencesKey:'neural-earth-wasm-preferences-v1',basePath:new URL('../',import.meta.url).pathname};
// Preserve Python preferences: this runtime does not change the stored GPU choices.
window.createTerrainRenderer=async()=>null;
await import('./globe.mjs');
const restrict=()=>{
  for(const id of ['gpuRender','coarseGpu']){const input=document.getElementById(id);if(input){input.checked=false;input.disabled=true;}}
  for(const id of ['prefetch','continuousRefine']){const input=document.getElementById(id);if(input)input.checked=false;}
  const engine=document.getElementById('nnEngine');if(engine)engine.disabled=true;
  for(const id of ['generateRelief','generateErosion','generateClimate']){const button=document.getElementById(id);if(button){button.disabled=true;button.title='Use Generate full pipeline in this browser build';}}
  for(const id of ['coarseStreams','snrAdaptive','snrDetailMode','prepareEntireWorld','prepareWorld','stopPreparation']){const input=document.getElementById(id);if(input){input.disabled=true;if(input.type==='checkbox')input.checked=false;}}
  const noiseHelp=document.getElementById('generationSourceHelp');if(noiseHelp)noiseHelp.textContent='Browser runtime · Orogen sphere · complete generation pipeline · neural refinement to 30 m · render and inference in Rust/WASM.';
  const mode=document.getElementById('mapMode');if(mode)for(const option of mode.options)option.disabled=!['fantasy-ink','fantasy-color','render','relief','biomes','temperature','precipitation','orogen-height'].includes(option.value);
  for(const id of ['worldGenerator','generatorType','heightSource','climateSource','worldTopology']){
    const input=document.getElementById(id);if(input){const supported=id==='worldTopology'?'sphere':'orogen';for(const option of input.options)option.disabled=option.value!==supported;}
  }
  const erosion=document.getElementById('reliefPipeline');if(erosion)for(const option of erosion.options)option.disabled=!['orogen','original'].includes(option.value);
  for(const input of document.querySelectorAll('input[id^="snrAltitude"],input[id^="snrDriver"],input[id^="snrLatitude"],input[id^="snrLod"],#snrBins,#snrDriverChannel'))input.disabled=true;
  for(const input of document.querySelectorAll('input[id^="orogen_gpu_"],input[id^="city_erosion_"]'))input.disabled=true;
};
window.addEventListener('DOMContentLoaded',restrict);
window.installWasmPreferences=()=>{
  const control=window.TerrainGenerationControls;
  control.initializePreferences=async()=>({preferences:{...JSON.parse(localStorage.getItem(window.NeuralEarthRuntime.preferencesKey)||'{}'),gpuAcceleration:false,rendering:{gpuRender:false,coarseGpu:false}},firstLaunch:false,resetErosion:false});
  const defaults=control.defaults;
  control.defaults=profile=>({...defaults(profile),height_source:'orogen',climate_source:'orogen',world_topology:'sphere',snr_adaptive_enabled:false,...Object.fromEntries(['relief','propagation','post','climate','erosion','raster'].map(stage=>['orogen_gpu_'+stage,false]))});
  // The existing controls keep one WASM capability boundary rather than a second UI.
  const mount=control.mount;
  control.mount=options=>{
    const result=mount(options),sync=result.sync;
    result.sync=(...args)=>{sync(...args);restrict();};
    return result;
  };
};
