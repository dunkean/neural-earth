/* Curated model choices for the normal Neural Earth viewer. */
(() => {
  const t=(key,fallback)=>window.NeuralEarthLocale?.text('models.'+key) ?? fallback;
  const messages=new WeakMap();
  function message(element,key,fallback,suffix='') {messages.set(element,{key,fallback,suffix});element.textContent=t(key,fallback)+suffix;}
  function errorMessage(element,value) {messages.delete(element);element.textContent=value;}
  const engine = document.getElementById('nnEngine');
  const panel = document.createElement('fieldset');
  panel.id = 'distillModels';
  panel.innerHTML = '<legend>Comparaison des NN</legend><p>Carte continue : choisis chaque modèle séparément.</p>' +
    '<label>Moteur d’inférence <select id="inferenceRuntime" aria-label="Moteur d’inférence"></select></label>' +
    '<button id="applyRuntime" type="button">Changer de moteur</button>' +
    '<small id="runtimeStatus" role="status"></small>' +
    '<label>Coarse <select id="distillCoarse" aria-label="Modèle coarse"></select></label>' +
    '<label>Base <select id="distillBase" aria-label="Modèle base"></select></label>' +
    '<label>Decoder <select id="distillDecoder" aria-label="Modèle decoder"></select></label>' +
    '<div class="panelActions"><button id="distillApply" type="button">Appliquer les NN</button>' +
    '<button id="distillReset" type="button">Reset après Orogen</button></div>' +
    '<small>Le reset recalcule coarse, base et decoder. Orogen, seed et caméra sont conservés.</small>' +
    '<p><a href="/tile40/">Tuile 40 × 40 km · base résiduel 8k · CPU / GPU navigateur</a></p>' +
    '<small id="distillStatus" role="status"></small>';
  const menu = document.createElement('details');
  menu.id = 'distillPanel';
  menu.innerHTML = '<summary aria-label="Modèles neuronaux">NN</summary><div class="toolPanel"></div>';
  menu.querySelector('.toolPanel').append(panel);
  document.getElementById('controls').insertBefore(menu, document.getElementById('mapActivity'));
  menu.addEventListener('toggle', () => {
    if (menu.open) for (const other of document.querySelectorAll('#controls > details')) if (other !== menu) other.open = false;
  });
  for (const other of document.querySelectorAll('#controls > details')) {
    if (other !== menu) other.addEventListener('toggle', () => { if (other.open) menu.open = false; });
  }
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && menu.open) { menu.open = false; menu.querySelector('summary').focus(); }
  });
  const stages = ['coarse', 'base', 'decoder'];
  const selects = Object.fromEntries(stages.map(stage => [stage, panel.querySelector('#distill' + stage[0].toUpperCase() + stage.slice(1))]));
  const status = panel.querySelector('#distillStatus');
  const runtimeSelect = panel.querySelector('#inferenceRuntime');
  const runtimeStatus = panel.querySelector('#runtimeStatus');
  const browserRuntime = window.NeuralEarthRuntime?.kind === 'wasm';
  const currentRuntime = browserRuntime ? 'wasm' : 'python';
  let runtimeOptions = [];
  function runtimeOptionLabel(item) {
    const label=item.id==='wasm'?t('wasmLabel','Rust / WASM · WebGPU navigateur'):item.label;
    return label+(item.available?'':' · '+t('unbuilt','à construire'));
  }
  function translate() {
    menu.querySelector('summary').setAttribute('aria-label',t('menu','Modèles neuronaux'));
    panel.querySelector('legend').textContent=t('title','Comparaison des NN');
    panel.querySelector('p').textContent=t('hint','Carte continue : choisis chaque modèle séparément.');
    for(const [input,key,fallback] of [[runtimeSelect,'runtime','Moteur d’inférence'],[selects.coarse,'coarse','Modèle coarse'],[selects.base,'base','Modèle base'],[selects.decoder,'decoder','Modèle decoder']]) {
      const label=input.closest('label');if(label?.firstChild?.nodeType===Node.TEXT_NODE)label.firstChild.textContent=t(key,fallback)+' ';
      input.setAttribute('aria-label',t(key,fallback));
    }
    for(const [selector,key,fallback] of [['#applyRuntime','changeRuntime','Changer de moteur'],['#distillApply','apply','Appliquer les NN'],['#distillReset','reset','Reset après Orogen']])panel.querySelector(selector).textContent=t(key,fallback);
    panel.querySelector('small:not([id])').textContent=t('resetHint','Le reset recalcule coarse, base et decoder. Orogen, seed et caméra sont conservés.');
    panel.querySelector('a').textContent=t('tile','Tuile 40 × 40 km · base résiduel 8k · CPU / GPU navigateur');
    for(const option of runtimeSelect.options) {const item=runtimeOptions.find(entry=>entry.id===option.value);if(item)option.textContent=runtimeOptionLabel(item);}
    for(const element of [status,runtimeStatus]) {const saved=messages.get(element);if(saved)element.textContent=t(saved.key,saved.fallback)+saved.suffix;}
  }
  window.addEventListener('neural-locale-change',translate);
  translate();
  async function readRuntimes() {
    // Engine discovery contacts the host; inference still uses TerrainTransport.
    try {
      if (browserRuntime && window.NeuralEarthRuntime?.basePath !== '/browser/') throw Error('Standalone runtime');
      const response = await fetch('/api/runtime', {cache:'no-store'});
      if (!response.ok) throw Error('Standalone runtime');
      const data = await response.json();
      if (!Array.isArray(data.options)) throw Error('No runtime catalog');
      runtimeOptions = data.options;
    } catch (_) {
      runtimeOptions = [{id:currentRuntime, label:browserRuntime ? 'Rust / WASM · WebGPU navigateur' : 'Python / PyTorch · CUDA', available:true}];
    }
    runtimeSelect.replaceChildren(...runtimeOptions.map(item => {
      const option = new Option(runtimeOptionLabel(item), item.id);
      option.disabled = !item.available;
      return option;
    }));
    runtimeSelect.value = currentRuntime;
    if(browserRuntime)message(runtimeStatus,'browserRuntime','Actif : Rust compilé en WASM, calcul sur le GPU du navigateur. Base NF4 B75, 30 m minimum. Le changement de moteur peut modifier le terrain.');
    else message(runtimeStatus,'pythonRuntime','Actif : Python / PyTorch sur CUDA. Moteur par défaut. Le WebGPU de rendu est indépendant de l’inférence.');
  }
  panel.querySelector('#applyRuntime').onclick = async () => {
    if (runtimeSelect.value === currentRuntime) return;
    if (!world || worldLoading || generationPending || restoringShare) {
      message(runtimeStatus,'wait','Attends la fin de la génération.'); return;
    }
    const target = runtimeOptions.find(item => item.id === runtimeSelect.value && item.available);
    if (!target?.url) return;
    const button = panel.querySelector('#applyRuntime'); button.disabled = true;
    try {
      const snapshot = captureShare();
      if (target.id === 'wasm') {
        const settings = snapshot.generation.settings;
        if (snapshot.generation.profile !== 'orogen' || settings.world_topology !== 'sphere' ||
            settings.height_source !== 'orogen' || settings.climate_source !== 'orogen' ||
            !['orogen','original'].includes(settings.relief_pipeline))
          throw Error(t('runtimeConstraint','Le moteur navigateur prend en charge les planètes Orogen avec érosion Orogen ou sans érosion.'));
        if (!['render','relief','biomes','temperature','precipitation','orogen-height','fantasy-ink','fantasy-color'].includes(snapshot.camera.layer))
          throw Error(t('layerConstraint','Choisis une couche prise en charge par WASM : Render, Relief, Biomes ou Fantasy.'));
        if (snapshot.camera.lod !== 'auto' && Number(snapshot.camera.lod) < 0)
          throw Error(t('lodConstraint','WASM descend à 30 m : choisis une LOD automatique ou positive.'));
        for (const stage of ['relief','propagation','post','climate','erosion','raster']) settings['orogen_gpu_'+stage] = false;
        snapshot.rendering.gpuRender = false; snapshot.rendering.coarseGpu = false;
      }
      snapshot.rendering.nnEngine = 'exact';
      const response = await (window.TerrainTransport?.fetch || fetch)('/api/share', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({generation:snapshot.generation, rendering:snapshot.rendering})});
      const recipes = await response.json();
      if (!response.ok) throw Error(recipes.error || t('transfer','Transfert du monde impossible'));
      const path = TerrainShare.url({...snapshot.camera, ...recipes});
      const url = new URL(path, location.origin); url.pathname = target.url;
      location.assign(url.href);
    } catch (error) { errorMessage(runtimeStatus,error.message); }
    finally { button.disabled = false; }
  };
  readRuntimes();
  for (const note of panel.querySelectorAll('small')) { note.style.display = 'block'; note.style.marginTop = '6px'; }
  function selectionLabels(data) {
    return stages.map(stage => data.options[stage].find(option => option.id === data.selection[stage]).label).join(' / ');
  }
  let state;
  async function read() {
    const response = await (window.TerrainTransport?.fetch||fetch)('/api/models', {cache: 'no-store'});
    if (!response.ok) throw Error(t('unavailable','Modèles indisponibles'));
    return response.json();
  }
  function render(data) {
    state = data;
    for (const stage of stages) {
      selects[stage].replaceChildren(...data.options[stage].map(option => new Option(option.label, option.id)));
      selects[stage].value = data.selection[stage];
    }
    message(status,'active','Actif',' · '+selectionLabels(data));
  }
  async function apply(reset) {
    if (generationPending) { message(status,'waitModels','Attends la fin de la génération en cours.'); return; }
    const selection = Object.fromEntries(stages.map(stage => [stage, selects[stage].value]));
    const seed = world?.seed || document.getElementById('seed').value;
    const profile = selectedProfile, generation = activeGeneration;
    const controls = [...panel.querySelectorAll('button,select')];
    controls.forEach(control => control.disabled = true);
    generationPending = true;
    for (const flight of inflight.values()) flight.controller.abort();
    overviewFlight?.controller.abort();
    clearTimeout(timer);
    message(status,'applying','Application…');
    try {
      const response = await (window.TerrainTransport?.fetch||fetch)('/api/models', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({selection, reset})});
      const change = await response.json();
      if (!response.ok) throw Error(change.error || t('refused','Changement refusé'));
      const deadline = Date.now() + 180000;
      let ready = null;
      while (Date.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, 700));
        try { const data = await read(); if (!data.switching && data.revision === change.revision) { ready = data; break; } } catch (_) {}
      }
      if (!ready) throw Error(t('timeout','Le serveur ne répond pas encore. Recharge la page pour réessayer.'));
      render(ready);
      generationPending = false;
      // The existing viewer clears RAM/WebGPU and reacquires the new neural
      // namespace, retaining its exact camera and generation settings.
      nnEngine = 'exact'; document.getElementById('nnEngine').value = nnEngine;
      const opened = await openWorld(seed, profile, {preserveCamera: true, generation});
      if (!opened) throw Error(t('viewFailed','Modèles appliqués, mais la vue n’a pas pu être recalculée.'));
      message(status,reset?'cacheReset':'applied',reset?'Cache NN réinitialisé':'NN appliqués',' · '+selectionLabels(ready));
    } catch (error) { errorMessage(status,error.message); }
    finally { generationPending = false; controls.forEach(control => control.disabled = false); draw(); schedule(); }
  }
  panel.querySelector('#distillApply').onclick = () => apply(false);
  panel.querySelector('#distillReset').onclick = () => apply(true);
  read().then(render).catch(error => { errorMessage(status,error.message); });
})();
