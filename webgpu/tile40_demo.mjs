import {presets,sketchPreset,heightSummary,reliefPixels} from './demo_helpers.mjs';
import {setupTileLocale,tileText as t,presetLabel} from './locale.mjs';
const $=id => document.getElementById(id);
const messages=new Map();
function message(id,render){messages.set(id,render);$(id).textContent=render();}
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
  message('sketchSummary',()=>t(`Croquis : ${s.minM.toFixed(0)} à ${s.maxM.toFixed(0)} m. Grille de 16 × 16 altitudes.`,`Sketch: ${s.minM.toFixed(0)} to ${s.maxM.toFixed(0)} m. Grid of 16 × 16 elevations.`,`草图：${s.minM.toFixed(0)} 至 ${s.maxM.toFixed(0)} 米。16 × 16 高程网格。`,`Boceto: ${s.minM.toFixed(0)} a ${s.maxM.toFixed(0)} m. Cuadrícula de 16 × 16 elevaciones.`,`Esboço: ${s.minM.toFixed(0)} a ${s.maxM.toFixed(0)} m. Grade de 16 × 16 altitudes.`));
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
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 4294950000) {message('status',()=>t('La graine doit être un entier entre 0 et 4294950000.','Seed must be an integer between 0 and 4294950000.','种子必须是 0 至 4294950000 之间的整数。','La semilla debe ser un entero entre 0 y 4294950000.','A semente deve ser um inteiro entre 0 e 4294950000.'));return;}
  requested={sketch_m:sketch.map(row => [...row]),climate:$('climate').value,seed};
  requestedManifest=manifest;
  result=null;
  requestedAt=performance.now();
  $('generate').disabled=true;$('decoder').disabled=true;$('engine').disabled=true;$('provider').disabled=true;$('image').disabled=true;$('dem').disabled=true;$('parameters').disabled=true;
  message('status',()=>t('Calcul en cours… Au premier calcul, le moteur et les modèles sont chargés.','Computing… The engine and models are loaded on the first run.','正在计算…首次运行时会加载引擎和模型。','Calculando… En el primer cálculo se cargan el motor y los modelos.','Calculando… No primeiro cálculo, o mecanismo e os modelos são carregados.'));
  const payload={provider:$('provider').value,decoder:$('decoder').value,input:requested};
  if ($('engine').value==='python') {
    try {
      const response=await fetch('./python/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      if (!response.ok) {const error=await response.json();throw Error(error.error||t(`Erreur serveur ${response.status}`,`Server error ${response.status}`,`服务器错误 ${response.status}`,`Error del servidor ${response.status}`,`Erro do servidor ${response.status}`));}
      const data=JSON.parse(response.headers.get('X-Terrain-Metadata'));
      const height=new Float32Array(await response.arrayBuffer());
      if (height.length!==data.pixels*data.pixels) throw Error(t('Réponse du serveur incomplète.','Incomplete server response.','服务器响应不完整。','Respuesta incompleta del servidor.','Resposta incompleta do servidor.'));
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
  if (data.error) {message('status',()=>t(`Calcul interrompu : ${data.error}`,`Calculation interrupted: ${data.error}`,`计算中断：${data.error}`,`Cálculo interrumpido: ${data.error}`,`Cálculo interrompido: ${data.error}`));return;}
  result={...data,input:requested,manifest:requestedManifest,summary:heightSummary(data.height)};
  const canvas=$('result');canvas.width=data.pixels;canvas.height=data.pixels;
  canvas.getContext('2d',{willReadFrequently:true}).putImageData(new ImageData(reliefPixels(data.height,data.pixels,data.spacingM),data.pixels,data.pixels),0,0);
  result.previewReadyMs=performance.now()-requestedAt;
  const location=()=>({'python-cuda':t('le serveur Python · GPU CUDA','Python server · CUDA GPU','Python 服务器 · CUDA GPU','servidor Python · GPU CUDA','servidor Python · GPU CUDA'),'python-cpu':t('le serveur Python · CPU','Python server · CPU','Python 服务器 · CPU','servidor Python · CPU','servidor Python · CPU'),webgpu:t('le navigateur · GPU','browser · GPU','浏览器 · GPU','navegador · GPU','navegador · GPU'),wasm:t('le navigateur · CPU','browser · CPU','浏览器 · CPU','navegador · CPU','navegador · CPU')})[data.provider] ?? data.provider;
  message('status',()=>{const total=(data.timings.totalMs/1000).toFixed(2),load=(data.requestMs/1000).toFixed(2),fallback=data.fallback?t(' Le GPU étant indisponible, le calcul a utilisé le CPU.',' GPU unavailable; computation used the CPU.',' GPU 不可用，使用 CPU 计算。',' GPU no disponible; se utilizó la CPU.',' GPU indisponível; o cálculo usou a CPU.') : '';return t(`Tuile calculée sur ${location()} en ${total} s. Temps avec chargement : ${load} s.`,`Tile computed on ${location()} in ${total} s. Time including loading: ${load} s.`,`地块在 ${location()} 上计算耗时 ${total} 秒。含加载时间：${load} 秒。`,`Tesela calculada en ${location()} en ${total} s. Tiempo con carga: ${load} s.`,`Bloco calculado em ${location()} em ${total} s. Tempo com carregamento: ${load} s.`)+fallback;});
  const s=result.summary;
  message('heightSummary',()=>t(`40 × 40 km · ${data.pixels} × ${data.pixels} altitudes · min. ${s.minM.toFixed(1)} m · max. ${s.maxM.toFixed(1)} m · moyenne ${s.meanM.toFixed(1)} m · relief entre les percentiles 10 et 90 : ${s.reliefM.toFixed(1)} m.`,`40 × 40 km · ${data.pixels} × ${data.pixels} elevations · min. ${s.minM.toFixed(1)} m · max. ${s.maxM.toFixed(1)} m · mean ${s.meanM.toFixed(1)} m · relief between the 10th and 90th percentiles: ${s.reliefM.toFixed(1)} m.`,`40 × 40 公里 · ${data.pixels} × ${data.pixels} 个高程 · 最低 ${s.minM.toFixed(1)} 米 · 最高 ${s.maxM.toFixed(1)} 米 · 平均 ${s.meanM.toFixed(1)} 米 · 第 10 至 90 百分位高差：${s.reliefM.toFixed(1)} 米。`,`40 × 40 km · ${data.pixels} × ${data.pixels} elevaciones · mín. ${s.minM.toFixed(1)} m · máx. ${s.maxM.toFixed(1)} m · media ${s.meanM.toFixed(1)} m · relieve entre percentiles 10 y 90: ${s.reliefM.toFixed(1)} m.`,`40 × 40 km · ${data.pixels} × ${data.pixels} altitudes · mín. ${s.minM.toFixed(1)} m · máx. ${s.maxM.toFixed(1)} m · média ${s.meanM.toFixed(1)} m · relevo entre os percentis 10 e 90: ${s.reliefM.toFixed(1)} m.`));
  messages.delete('metadata');$('metadata').textContent=JSON.stringify(metadata(),null,2);
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
  const px=((x+.5)*result.spacingM/1000).toFixed(2),py=((y+.5)*result.spacingM/1000).toFixed(2),height=result.height[y*result.pixels+x].toFixed(1);
  message('cursor',()=>t(`Position : ${px} km, ${py} km · altitude ${height} m.`,`Position: ${px} km, ${py} km · elevation ${height} m.`,`位置：${px} 公里，${py} 公里 · 高程 ${height} 米。`,`Posición: ${px} km, ${py} km · elevación ${height} m.`,`Posição: ${px} km, ${py} km · altitude ${height} m.`));
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
    if (!response.ok) throw Error(t('Modèles indisponibles.','Models unavailable.','模型不可用。','Modelos no disponibles.','Modelos indisponíveis.'));
    const loaded=await response.json();
    if(revision!==engineRevision)return false;
    manifest=loaded;
    const size=(manifest.total_model_bytes/1e6).toFixed(1);
    message('bundleSize',()=>t(`Modèles : ${size} Mo. ${python?'Conservés sur le serveur ; aucune préparation ONNX nécessaire.':'Chargés dans le navigateur une seule fois par choix de décodeur et de calcul.'}`,`Models: ${size} MB. ${python?'Stored on the server; no ONNX preparation required.':'Loaded in the browser once per decoder and compute choice.'}`,`模型：${size} MB。${python?'存储在服务器上，无需准备 ONNX。':'每种解码器和计算选项在浏览器中只加载一次。'}`,`Modelos: ${size} MB. ${python?'Guardados en el servidor; no se requiere preparación ONNX.':'Cargados en el navegador una vez por decodificador y cálculo.'}`,`Modelos: ${size} MB. ${python?'Mantidos no servidor; não é necessário preparar ONNX.':'Carregados no navegador uma vez por escolha de decodificador e cálculo.'}`));
    $('generate').disabled=false;
    return true;
  } catch(error) {if(revision===engineRevision)message('status',()=>t(`Impossible de charger les modèles : ${error.message}`,`Unable to load models: ${error.message}`,`无法加载模型：${error.message}`,`No se pueden cargar los modelos: ${error.message}`,`Não foi possível carregar os modelos: ${error.message}`));return false;}
}
$('decoder').onchange=loadDecoder;
async function prepare() {
  $('prepareRetry').hidden=true;
  $('generate').disabled=true;
  if ($('engine').value==='python') {
    $('decoder').disabled=false;
    if(await loadDecoder())message('status',()=>t('Backend Python prêt. Choisis une forme, un climat et un décodeur.','Python backend ready. Choose a shape, climate and decoder.','Python 后端已就绪。请选择形状、气候和解码器。','Backend Python listo. Elige una forma, un clima y un decodificador.','Backend Python pronto. Escolha uma forma, clima e decodificador.'));
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
      message('status',()=>t('Préparation des modèles pour le navigateur… Cette étape n’est nécessaire qu’une fois.','Preparing browser models… This step is only needed once.','正在为浏览器准备模型…此步骤只需执行一次。','Preparando modelos para el navegador… Este paso solo es necesario una vez.','Preparando modelos para o navegador… Esta etapa é necessária apenas uma vez.'));
      await new Promise(resolve=>setTimeout(resolve,1000));
      response=await fetch('./catalog.json');state=await response.json();
    }
    if(revision!==engineRevision)return;
    if (state.status!=='ready') throw Error(state.error||t('Préparation interrompue.','Preparation interrupted.','准备已中断。','Preparación interrumpida.','Preparação interrompida.'));
    const previousDecoder=$('decoder').value;
    $('decoder').replaceChildren(...state.decoders.map(item=>new Option(decoderLabel(item.id,item.label),item.id)));
    $('decoder').value=state.decoders.some(item=>item.id===previousDecoder) ? previousDecoder : state.default_decoder;
    $('decoder').disabled=false;
    if(await loadDecoder())message('status',()=>t('Choisis une forme et un climat, puis génère une tuile.','Choose a shape and climate, then generate a tile.','请选择形状和气候，然后生成地块。','Elige una forma y un clima y genera una tesela.','Escolha uma forma e um clima e gere um bloco.'));
  } catch(error) {if(revision===engineRevision){message('status',()=>error.message);$('prepareRetry').hidden=false;}}
}
function decoderLabel(id,fallback){return id==='decoder16'?t('Décodeur16 · 10k · compact','Decoder16 · 10k · compact','Decoder16 · 10k · 紧凑','Decodificador16 · 10k · compacto','Decodificador16 · 10k · compacto'):id==='decoder64'?t('Décodeur64 · 200k · qualité','Decoder64 · 200k · quality','Decoder64 · 200k · 高质量','Decodificador64 · 200k · calidad','Decodificador64 · 200k · qualidade'):fallback;}
function providerLabel(id,python){return id==='auto'?(python?t('Automatique : CUDA, sinon CPU','Automatic: CUDA, otherwise CPU','自动：CUDA，否则 CPU','Automático: CUDA o CPU','Automático: CUDA ou CPU'):t('Automatique : GPU, sinon CPU','Automatic: GPU, otherwise CPU','自动：GPU，否则 CPU','Automático: GPU o CPU','Automático: GPU ou CPU')):id==='cpu'?t('CPU du serveur','Server CPU','服务器 CPU','CPU del servidor','CPU do servidor'):id==='cuda'?t('GPU CUDA du serveur','Server CUDA GPU','服务器 CUDA GPU','GPU CUDA del servidor','GPU CUDA do servidor'):id==='wasm'?t('CPU du navigateur','Browser CPU','浏览器 CPU','CPU del navegador','CPU do navegador'):t('GPU du navigateur','Browser GPU','浏览器 GPU','GPU del navegador','GPU do navegador');}
function updateEngineHelp(){const python=$('engine').value==='python';$('engineHelp').textContent=python?t('Python utilise les modèles sur le serveur pour comparer rapidement la qualité.','Python uses server-side models for quick quality comparisons.','Python 使用服务器模型快速比较质量。','Python utiliza modelos del servidor para comparar rápidamente la calidad.','Python usa modelos no servidor para comparar a qualidade rapidamente.'):t('Le calcul et les modèles restent dans le navigateur, sur CPU ou GPU WebGPU.','Computation and models stay in the browser, on CPU or WebGPU.','计算和模型保留在浏览器中，使用 CPU 或 WebGPU。','El cálculo y los modelos permanecen en el navegador, con CPU o WebGPU.','O cálculo e os modelos permanecem no navegador, na CPU ou WebGPU.');}
$('engine').onchange=()=>{
  ++engineRevision;
  const python=$('engine').value==='python';
  const choices=python ? ['auto','cpu','cuda'] : ['auto','wasm','webgpu'];
  $('provider').replaceChildren(...choices.map(value=>new Option(providerLabel(value,python),value)));
  updateEngineHelp();
  if(worker){worker.terminate();worker=null;}
  prepare();
};
$('prepareRetry').onclick=prepare;
message('heightSummary',()=>t('Sortie : 1334 × 1334 altitudes en mètres, soit environ 30 m par pixel.','Output: 1334 × 1334 elevations in metres, about 30 m per pixel.','输出：1334 × 1334 个以米为单位的高程，约每像素 30 米。','Salida: 1334 × 1334 elevaciones en metros, unos 30 m por píxel.','Saída: 1334 × 1334 altitudes em metros, cerca de 30 m por pixel.'));
message('metadata',()=>t('Aucun résultat calculé.','No result computed.','尚无计算结果。','Ningún resultado calculado.','Nenhum resultado calculado.'));
setupTileLocale(()=>{for(const option of $('preset').options)option.textContent=presetLabel(option.value);for(const option of $('provider').options)option.textContent=providerLabel(option.value,$('engine').value==='python');for(const option of $('decoder').options)option.textContent=decoderLabel(option.value,option.textContent);updateEngineHelp();for(const [id,render] of messages)$(id).textContent=render();});
prepare();
