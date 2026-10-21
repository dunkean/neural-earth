// Inference stays off the UI thread;all noise and conditioning are generated here.
importScripts('./ort/ort.webgpu.min.js');
let sessions=null, manifest=null, selected=null, selectedDecoder=null, modelBase=null, busy=false;
const runtime=import('./runtime/tile40_runtime.mjs');

async function initialize(provider) {
  if (sessions && selected===provider) return;
  if (sessions) for (const session of Object.values(sessions)) await session.release();
  sessions=null;selected=null;
  ort.env.wasm.wasmPaths=new URL('./ort/',self.location).href;
  ort.env.wasm.numThreads=provider==='wasm' && self.crossOriginIsolated ? Math.min(8,navigator.hardwareConcurrency || 1) : 1;
  const created={};
  try {
    for (const [stage,model] of Object.entries(manifest.models)) created[stage]=
      await ort.InferenceSession.create(modelBase+model.file,{executionProviders:[provider]});
    sessions=created;selected=provider;
  } catch(error) {for (const session of Object.values(created)) await session.release();throw error;}
}

onmessage=async ({data}) => {
  if (busy) {postMessage({error:'Un calcul est déjà en cours.'});return;}
  busy=true;
  const started=performance.now();
  try {
    if (!['decoder16','decoder64'].includes(data.decoder)) throw Error('Décodeur inconnu.');
    if (selectedDecoder!==data.decoder) {
      if (sessions) for (const session of Object.values(sessions)) await session.release();
      sessions=null;selected=null;manifest=null;selectedDecoder=null;
      modelBase=new URL(`./model/${data.decoder}/`,self.location).href;
      const response=await fetch(modelBase+'manifest.json');
      if (!response.ok) throw Error('Modèles indisponibles.');
      manifest=await response.json();selectedDecoder=data.decoder;
    }
    let provider=data.provider, fallback=null;
    if (provider==='auto') {
      let adapter=null;
      try {adapter=navigator.gpu ? await navigator.gpu.requestAdapter({powerPreference:'high-performance'}) : null;}
      catch(error) {fallback=String(error);}
      const info=adapter ? adapter.info || {} : {};
      const identity=`${info.vendor || ''} ${info.architecture || ''} ${info.description || ''}`.toLowerCase();
      provider=adapter && !/swiftshader|llvmpipe|software|warp/.test(identity) ? 'webgpu' : 'wasm';
    }
    const {generateTile}=await runtime;
    let result;
    try {await initialize(provider);result=await generateTile(ort,sessions,manifest,data.input);}
    catch(error) {
      if (data.provider!=='auto' || provider!=='webgpu') throw error;
      fallback=String(error);provider='wasm';await initialize(provider);
      result=await generateTile(ort,sessions,manifest,data.input);
    }
    postMessage({...result,provider,fallback,requestMs:performance.now()-started},[result.height.buffer]);
  } catch(error) {postMessage({error:String(error)});}
  finally {busy=false;}
};
