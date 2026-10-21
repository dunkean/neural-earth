import {presets,sketchPreset,heightSummary,reliefPixels} from './demo_helpers.mjs';
const $=id => document.getElementById(id);
let worker=null, engineRevision=0;
let sketch=sketchPreset('Massif'), result=null, requested=null, manifest=null, requestedManifest=null, requestedAt=0;
for (const name of presets) $('preset').add(new Option(name,name,name==='Massif',name==='Massif'));
function drawSketch() {
  const context=$('sketch').getContext('2d');
  for (let y=0;y<16;y++) for (let x=0;x<16;x++) {
    const value=Math.round(40+Math.min(8000,Math.max(0,sketch[y][x]))/8000*210);
    context.fillStyle=`rgb(${value},${value},${value})`;context.fillRect(x*20,y*20,20,20);
  }
  const s=heightSummary(sketch.flat());
  $('sketchSummary').textContent=`Croquis : ${s.minM.toFixed(0)} à ${s.maxM.toFixed(0)} m. Grille de 16 × 16 altitudes.`;
}
function edit(y,x) {
  const value=Number($('altitude').value);
  if (!Number.isFinite(value) || value < -1000 || value > 8000 || !Number.isInteger(x) || !Number.isInteger(y) || x < 0 || x > 15 || y < 0 || y > 15) return;
  sketch[y][x]=value;$('row').value=y+1;$('column').value=x+1;drawSketch();
}
function pointer(event) {
  const box=$('sketch').getBoundingClientRect();
  edit(Math.max(0,Math.min(15,Math.floor((event.clientY-box.top)/box.height*16))),
       Math.max(0,Math.min(15,Math.floor((event.clientX-box.left)/box.width*16))));
}
$('sketch').onpointerdown=event => {$('sketch').setPointerCapture(event.pointerId);pointer(event);};
$('sketch').onpointermove=event => {if (event.buttons) pointer(event);};
$('edit').onclick=() => edit(Number($('row').value)-1,Number($('column').value)-1);
$('preset').onchange=() => {sketch=sketchPreset($('preset').value);drawSketch();};
drawSketch();
function download(blob,name) {
  const link=document.createElement('a'),url=URL.createObjectURL(blob);
  link.href=url;link.download=name;link.click();setTimeout(() => URL.revokeObjectURL(url),1000);
}
const metadata=() => ({input:result.input,side_m:result.sideM,pixels:result.pixels,spacing_m:result.spacingM,
  coordinates:'Rows and columns increase from the sketch top-left;sample centres are(i+0.5)*spacing_m.',
  height_file:'terrain40.f32',height_format:'Little-endian Float32,row-major,metres',
  summary:result.summary,provider:result.provider,timings:result.timings,request_ms:result.requestMs,preview_ready_ms:result.previewReadyMs,
  models:result.manifest.models,selected_models:result.manifest.selected_models,quality_accepted:false});
$('generate').onclick=async () => {
  const seed=Number($('seed').value);
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 4294950000) {$('status').textContent='La graine doit être un entier entre 0 et 4294950000.';return;}
  requested={sketch_m:sketch.map(row => [...row]),climate:$('climate').value,seed};
  requestedManifest=manifest;
  result=null;
  requestedAt=performance.now();
  $('generate').disabled=true;$('decoder').disabled=true;$('engine').disabled=true;$('provider').disabled=true;$('image').disabled=true;$('dem').disabled=true;$('parameters').disabled=true;
  $('status').textContent='Calcul en cours… Au premier calcul, le moteur et les modèles sont chargés.';
  const payload={provider:$('provider').value,decoder:$('decoder').value,input:requested};
  if ($('engine').value==='python') {
    try {
      const response=await fetch('./python/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      if (!response.ok) {const error=await response.json();throw Error(error.error||`Erreur serveur ${response.status}`);}
      const data=JSON.parse(response.headers.get('X-Terrain-Metadata'));
      const height=new Float32Array(await response.arrayBuffer());
      if (height.length!==data.pixels*data.pixels) throw Error('Réponse du serveur incomplète.');
      requestedManifest=data.manifest;
      receive({...data,height,requestMs:performance.now()-requestedAt});
    } catch(error) {receive({error:error.message});}
  } else {
    if (!worker) {
      worker=new Worker('./worker.js');
      worker.onmessage=({data})=>receive(data);
      worker.onerror=event=>receive({error:event.message});
    }
    worker.postMessage(payload);
  }
};
function receive(data) {
  $('generate').disabled=false;$('decoder').disabled=false;$('engine').disabled=false;$('provider').disabled=false;
  if (data.error) {$('status').textContent=`Calcul interrompu : ${data.error}`;return;}
  result={...data,input:requested,manifest:requestedManifest,summary:heightSummary(data.height)};
  const canvas=$('result');canvas.width=data.pixels;canvas.height=data.pixels;
  canvas.getContext('2d',{willReadFrequently:true}).putImageData(new ImageData(reliefPixels(data.height,data.pixels,data.spacingM),data.pixels,data.pixels),0,0);
  result.previewReadyMs=performance.now()-requestedAt;
  const location={'python-cuda':'le serveur Python · GPU CUDA','python-cpu':'le serveur Python · CPU',webgpu:'le navigateur · GPU',wasm:'le navigateur · CPU'}[data.provider];
  $('status').textContent=`Tuile calculée sur ${location} en ${(data.timings.totalMs/1000).toFixed(2)} s. Temps avec chargement : ${(data.requestMs/1000).toFixed(2)} s.${data.fallback ? ' Le GPU étant indisponible, le calcul a utilisé le CPU.' : ''}`;
  const s=result.summary;
  $('heightSummary').textContent=`40 × 40 km · ${data.pixels} × ${data.pixels} altitudes · min. ${s.minM.toFixed(1)} m · max. ${s.maxM.toFixed(1)} m · moyenne ${s.meanM.toFixed(1)} m · relief entre les percentiles 10 et 90 : ${s.reliefM.toFixed(1)} m.`;
  $('metadata').textContent=JSON.stringify(metadata(),null,2);
  $('dem').disabled=false;$('parameters').disabled=false;
  const rendered=result;
  // Encode ahead of the click so the download retains its user activation,
  // including after GPU inference on browsers with stricter download policies.
  canvas.toBlob(blob=>{if(result===rendered && blob){rendered.imageBlob=blob;$('image').disabled=false;}});
}
$('result').onpointermove=event => {
  if (!result) return;
  const box=$('result').getBoundingClientRect(), x=Math.min(result.pixels-1,Math.max(0,Math.floor((event.clientX-box.left)/box.width*result.pixels))),
    y=Math.min(result.pixels-1,Math.max(0,Math.floor((event.clientY-box.top)/box.height*result.pixels)));
  $('cursor').textContent=`Position : ${((x+.5)*result.spacingM/1000).toFixed(2)} km, ${((y+.5)*result.spacingM/1000).toFixed(2)} km · altitude ${result.height[y*result.pixels+x].toFixed(1)} m.`;
};
$('image').onclick=() => {if(result?.imageBlob)download(result.imageBlob,'terrain40.png');};
$('dem').onclick=() => {
  // Write explicit little-endian floats rather than relying on host endianness.
  const buffer=new ArrayBuffer(result.height.length*4),view=new DataView(buffer);
  result.height.forEach((v,i) => view.setFloat32(i*4,v,true));
  download(new Blob([buffer]),'terrain40.f32');
};
$('parameters').onclick=() => download(new Blob([JSON.stringify(metadata(),null,2)],{type:'application/json'}),'terrain40.json');
async function loadDecoder() {
  $('generate').disabled=true;
  const revision=++engineRevision;
  try {
    const python=$('engine').value==='python';
    const response=await fetch(python ? `./python/catalog.json?decoder=${$('decoder').value}` : `./model/${$('decoder').value}/manifest.json`);
    if (!response.ok) throw Error('Modèles indisponibles.');
    const loaded=await response.json();
    if(revision!==engineRevision)return false;
    manifest=loaded;
    $('bundleSize').textContent=`Modèles : ${(manifest.total_model_bytes/1e6).toFixed(1)} Mo. ${python ? 'Conservés sur le serveur ; aucune préparation ONNX nécessaire.' : 'Chargés dans le navigateur une seule fois par choix de décodeur et de calcul.'}`;
    $('generate').disabled=false;
    return true;
  } catch(error) {if(revision===engineRevision)$('status').textContent=`Impossible de charger les modèles : ${error.message}`;return false;}
}
$('decoder').onchange=loadDecoder;
async function prepare() {
  $('prepareRetry').hidden=true;
  $('generate').disabled=true;
  if ($('engine').value==='python') {
    $('decoder').disabled=false;
    if(await loadDecoder())$('status').textContent='Backend Python prêt. Choisis une forme, un climat et un décodeur.';
    return;
  }
  const revision=++engineRevision;
  try {
    let response=await fetch('./catalog.json'),state=await response.json();
    if (state.status!=='ready' && state.status!=='preparing') {
      response=await fetch('./prepare',{method:'POST'});state=await response.json();
    }
    while (state.status==='preparing') {
      if(revision!==engineRevision)return;
      $('status').textContent='Préparation des modèles pour le navigateur… Cette étape n’est nécessaire qu’une fois.';
      await new Promise(resolve=>setTimeout(resolve,1000));
      response=await fetch('./catalog.json');state=await response.json();
    }
    if(revision!==engineRevision)return;
    if (state.status!=='ready') throw Error(state.error||'Préparation interrompue.');
    const previousDecoder=$('decoder').value;
    $('decoder').replaceChildren(...state.decoders.map(item=>new Option(item.label,item.id)));
    $('decoder').value=state.decoders.some(item=>item.id===previousDecoder) ? previousDecoder : state.default_decoder;
    $('decoder').disabled=false;
    if(await loadDecoder())$('status').textContent='Choisis une forme et un climat, puis génère une tuile.';
  } catch(error) {if(revision===engineRevision){$('status').textContent=error.message;$('prepareRetry').hidden=false;}}
}
$('engine').onchange=()=>{
  ++engineRevision;
  const python=$('engine').value==='python';
  const choices=python ? [['auto','Automatique : CUDA, sinon CPU'],['cpu','CPU du serveur'],['cuda','GPU CUDA du serveur']] :
    [['auto','Automatique : GPU, sinon CPU'],['wasm','CPU du navigateur'],['webgpu','GPU du navigateur']];
  $('provider').replaceChildren(...choices.map(([value,label])=>new Option(label,value)));
  $('engineHelp').textContent=python ? 'Python utilise les modèles sur le serveur pour comparer rapidement la qualité.' : 'Le calcul et les modèles restent dans le navigateur, sur CPU ou GPU WebGPU.';
  if(worker){worker.terminate();worker=null;}
  prepare();
};
$('prepareRetry').onclick=prepare;
prepare();
