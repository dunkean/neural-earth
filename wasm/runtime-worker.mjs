import {physicalTile} from './physical-tile.mjs';
import init,{NeuralEarth,render_tile} from './pkg/neural_earth_wasm.js';
import {loadModels} from './models.mjs';
import {physicalFields,orogenRequest,orogenSeed} from './physics.mjs';
import {share} from './share.mjs';
import {probeWebGPU} from './webgpu.mjs';
const progress=label=>self.postMessage({progress:label});
let engine,loading,world,worldKey,settings,schema,viewEpoch=-1,completed=0,computing=0;
let revision=1,warming;
let modelManifest;
let precision;
const cancelled=new Set(),tiles=new Map();let tileBytes=0;
const json=value=>({body:JSON.stringify(value),headers:{'Content-Type':'application/json'}});
const ready=async()=>{
  if(!loading)loading=(async()=>{
    await probeWebGPU();
    await init();
    schema=await (await fetch(new URL('./generation-schema.json',import.meta.url))).json();
    const loaded=await loadModels(progress);modelManifest=loaded.manifest;
    progress('Uploading neural weights to WebGPU…');
    engine=await NeuralEarth.create(...loaded.models);precision=engine.precision();
    progress(`Networks ready · Rust / WASM / WebGPU · ${precision.toUpperCase()}`);
  })();
  await loading;
};
const defaults=()=>({...schema.defaults_by_profile.orogen,world_topology:'sphere',height_source:'orogen',climate_source:'orogen',snr_adaptive_enabled:false,
  ...Object.fromEntries(['relief','propagation','post','climate','erosion','raster'].map(k=>['orogen_gpu_'+k,false]))});
function validate(input={}) {
  const s={...defaults(),...input};
  if(s.world_topology!=='sphere'||s.height_source!=='orogen'||s.climate_source!=='orogen')throw Error('This browser build supports spherical Orogen worlds.');
  if(!['orogen','original'].includes(s.relief_pipeline))throw Error('Use Orogen CPU erosion or no erosion in the browser build.');
  for(const [key,spec]of Object.entries(schema.properties)) {
    const value=s[key];if(value===undefined)continue;
    if(spec.length!==undefined&&(!Array.isArray(value)||value.length!==spec.length))throw Error(`Invalid ${key}`);
    for(const item of Array.isArray(value)?value:[value]) {
      if(spec.minimum!==undefined&&(!Number.isFinite(item)||item<spec.minimum))throw Error(`Invalid ${key}`);
      if(spec.maximum!==undefined&&(!Number.isFinite(item)||item>spec.maximum))throw Error(`Invalid ${key}`);
      if(spec.integer&&!Number.isInteger(item))throw Error(`Invalid ${key}`);
    }
  }
  if(!Array.isArray(s.cond_snr)||s.cond_snr.length!==5||s.cond_snr.some(v=>!Number.isFinite(v)||v<.001||v>2))throw Error('Conditioning noise must contain five ratios in [0.001, 2].');
  return s;
}
async function generate(seed,input) {
  await ready();settings=validate(input);
  if(!/^\d{1,20}$/.test(seed)||BigInt(seed)>0xffffffffffffffffn)throw Error('Invalid world seed');
  const key=JSON.stringify([seed,settings,modelManifest,precision]);
  if(key===worldKey)return world;
  const source=new Worker(new URL('./orogen-worker.mjs',import.meta.url),{type:'module'});
  // The GPU compiles network pipelines while Orogen runs in its own worker.
  warming??=engine.warm();
  const sparseSeed=await orogenSeed(seed);
  const result=await new Promise((resolve,reject)=>{
    source.onmessage=({data})=>{if(data.type==='ready')source.postMessage(orogenRequest(sparseSeed,settings));else if(data.type==='progress')progress(`Orogen · ${data.label}`);else if(data.type==='error')reject(Error(data.message));else if(data.type==='done')resolve(data);};
    source.onerror=e=>reject(Error(e.message));
  }).finally(()=>source.terminate());
  await warming;
  const circumference=settings.world_diameter_km*1000*Math.PI;
  progress('Projecting Orogen and climate in WASM…');
  engine.set_world(seed,result.r_xyz,physicalFields(result),circumference);
  tiles.clear();tileBytes=0;completed=0;
  const identity=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(key))),v=>v.toString(16).padStart(2,'0')).join('');
  const bounds=[-circumference/2,-circumference/4,circumference/2,circumference/4];
  world={version:'wasm-v1',world_version:'wasm-v1',cache_profile:`rust-${precision}-coarse4-nf4-b75-decoder`,generation_profile:'orogen',world_profile:'orogen',
    world_identity:identity,appearance_identity:'wasm-raster-v1',generation_settings:settings,generation_schema:schema,conditioning_noise:settings.cond_snr,
    world_topology:'sphere',world_diameter_km:settings.world_diameter_km,world_bounds:bounds,overview_bounds:bounds,initial_bounds:bounds,initial_image:null,
    seed,tile_size:256,halo:0,native_resolution:30,min_lod:0,overview:`/api/overview/wasm-v1/${seed}.png`,gpu:`Rust · WASM · WebGPU · ${precision.toUpperCase()}`,
    orogen_layers:{render:'WASM terrain materials','orogen-height':'Orogen source elevation'},generation_stages:{},
    stages:{coarse:7680,latent:240,decoder:30},preview_min_lod:8,navigation_protocol:1};
  worldKey=key;progress('World ready · neural tiles generated locally on demand');return world;
}
async function png(heights,fields,w,h,step,mode,url,halo=0) {
  const lighting=JSON.parse(url.searchParams.get('lighting')||'{}'),contours=JSON.parse(url.searchParams.get('contours')||'{}');
  const materials=JSON.parse(url.searchParams.get('render_settings')||'{}');
  const rgba=render_tile(heights,fields,w,h,step,mode,JSON.stringify({lighting,contours,materials}));
  const canvas=new OffscreenCanvas(w-2*halo,h-2*halo);canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba),w,h),-halo,-halo);
  const blob=await canvas.convertToBlob({type:'image/png'});
  return {body:await blob.arrayBuffer(),headers:{'Content-Type':'image/png'}};
}
async function dispatch(request) {
  const url=new URL(request.path,self.location.origin),path=url.pathname.replace(/^\/reference/,'');
  const body=request.body?JSON.parse(request.body):{};
  if(path==='/api/view'){viewEpoch=Math.max(viewEpoch,body.epoch??-1);return json({accepted:true});}
  if(path==='/api/view/release')return json({released:true});
  if(path==='/api/status')return json({gpu:'Rust · WASM · WebGPU',scheduler:{queued:0,computing,encoding:0},preload:{state:engine?'ready':'loading'},inference_streams:{coarse:1},inference:{coarse_streams:1}});
  // Development hooks of npm run bench:wasm.
  if(path==='/api/debug/stats'){await ready();return json(Array.from(engine.stats()));}
  if(path==='/api/debug/conv-tile'){await ready();engine.set_conv_tile(body.tile||'');engine.clear_neural_cache();tiles.clear();tileBytes=0;return json({tile:body.tile||''});}
  if(path==='/api/inference/streams')return json({coarse:1});
  if(path==='/api/gpu')return json({settings:{mode:'single',devices:'auto'},devices:[],environment:['mode'],plan:{mode:'single',devices:[0]},restart_required:false,restart_available:false});
  if(path==='/api/coarse/prepare')return json({state:'idle',reason:'Browser inference follows visible tiles'});
  if(path.startsWith('/api/share'))return json(await share(path,body));
  if(path==='/api/world')return json(await generate(url.searchParams.get('seed')||'42',JSON.parse(url.searchParams.get('generation')||'null')||undefined));
  if(path==='/api/generation/run') {
    if(body.stage!=='all'&&body.stage!=='restore'&&body.stage!=='settings')throw Error('The browser build currently regenerates the complete Orogen pipeline. Use Generate full pipeline.');
    await generate(String(body.seed),body.settings);return json({profile:'orogen',generation_profile:'orogen',settings,seed:world.seed,generation_settings:settings});
  }
  if(path==='/api/generation/schema'){await ready();return json(schema);}
  if(path==='/api/models'||path==='/api/distill/models') {
    const selection={coarse:'solver4',base:'nf4_b75',decoder:'distilled200k'};
    if(request.method==='POST') {
      if(body.selection&&Object.keys(selection).some(k=>body.selection[k]!==selection[k]))throw Error('This build includes coarse4, the distilled NF4 B75 base and the distilled decoder.');
      engine?.clear_neural_cache();tiles.clear();tileBytes=0;revision++;
    }
    return json({revision,switching:false,selection,options:{coarse:[{id:'solver4',label:'Rust coarse4'}],base:[{id:'nf4_b75',label:'Rust distilled · NF4 B75'}],decoder:[{id:'distilled200k',label:'Rust distilled'}]}});
  }
  if(!world)throw Error('Open a world before requesting terrain');
  const mode=url.searchParams.get('mode')||'relief';
  if(!['relief','render','biomes','temperature','precipitation','orogen-height'].includes(mode))throw Error(`Layer ${mode} is not supported by this browser build`);
  if(url.searchParams.get('world_identity')&&url.searchParams.get('world_identity')!==world.world_identity)return {status:409};
  if(path.startsWith('/api/overview/')) {
    const [x,y,x1]=world.world_bounds,step=(x1-x)/1024,fields=engine.source(x,y,step,1024,512);
    return png(fields.slice(0,1024*512),fields,1024,512,step,mode,url);
  }
  const match=path.match(/^\/(tiles|height)\/wasm-v1\/(\d+)\/(-?\d+)\/(-?\d+)\/(-?\d+)\.(png|bin)$/);
  if(!match)throw Error(`Unsupported local runtime route: ${path}`);
  const [,kind,seed,lodText,txText,tyText]=match,lod=Number(lodText),tx=Number(txText),ty=Number(tyText);
  if(seed!==world.seed)return {status:409};
  if(!Number.isInteger(lod)||lod<0||lod>11||![tx,ty].every(v=>Number.isSafeInteger(v)&&Math.abs(v)<10000000))throw Error('Invalid tile coordinates');
  if(url.searchParams.get('neural_chart')==='polar')throw Error('The rotated polar neural chart is not supported in this browser build');
  const step=30*2**lod,span=256*step,fields=engine.source(tx*span-24*step,ty*span-24*step,step,304,304);
  let heights,stage='conditioning-preview',hit=false;
  if(lod<=7&&mode!=='orogen-height') {
    const key=`${lod}/${tx}/${ty}`;heights=tiles.get(key);hit=!!heights;
    if(heights){tiles.delete(key);tiles.set(key,heights);}
    else {
      progress(`Neural tile LOD ${lod} · ${tx}, ${ty} · Rust / WASM`);computing++;
      try{heights=await engine.tile(lod,tx,ty,settings.cond_snr[0],settings.cond_snr.slice(1));}finally{computing--;}
      if(!heights.every(Number.isFinite))throw Error('Non-finite neural terrain');
      while(tileBytes+heights.byteLength>64*1024*1024&&tiles.size){const key=tiles.keys().next().value;tileBytes-=tiles.get(key).byteLength;tiles.delete(key);}
      tiles.set(key,heights);tileBytes+=heights.byteLength;completed++;
    }
    stage=lod>=4?'coarse':lod===3?'latent':'final-dem-mip';
  }else heights=fields.slice(0,304*304);
  let minimum=Infinity,maximum=-Infinity;
  for(const value of heights){minimum=Math.min(minimum,value);maximum=Math.max(maximum,value);}
  let response=kind==='height'?physicalTile(heights,fields,304,24,url.searchParams.get('climate')==='1'):await png(heights,fields,304,304,step,mode,url,24);
  response.headers={...response.headers,'X-Terrain-Stage':stage,'X-Terrain-Cache':hit?'hit':'miss','X-Terrain-Source-Resolution':String(lod>=8?world.world_bounds[2]*2/1024:lod>=4?7680:lod===3?240:30),
    'X-Terrain-Elevation-Min':String(minimum),'X-Terrain-Elevation-Max':String(maximum)};
  return response;
}
let queue=Promise.resolve();
self.onmessage=({data})=>{
  if(data.cancel){cancelled.add(data.cancel);return;}
  const run=async()=>{
    if(cancelled.delete(data.id))return;
    try {
      const result=await dispatch(data);
      if(cancelled.delete(data.id))return;
      self.postMessage({id:data.id,...result},result.body instanceof ArrayBuffer?[result.body]:[]);
    }catch(error){if(!cancelled.delete(data.id))self.postMessage({id:data.id,...json({error:error?.message||String(error)}),status:500});}
  };
  // Telemetry and camera updates must never wait behind GPU computation.
  if(/^\/(reference\/)?api\/(view|status|inference\/streams|gpu)/.test(new URL(data.path,self.location.origin).pathname))run();
  else queue=queue.then(run,run);
};
