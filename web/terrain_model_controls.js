/* Curated model choices for the normal Neural Earth viewer. */
(() => {
  const engine = document.getElementById('nnEngine');
  const panel = document.createElement('fieldset');
  panel.id = 'distillModels';
  panel.innerHTML = '<legend>Comparaison des NN</legend><p>Carte continue : choisis chaque modèle séparément.</p>' +
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
  for (const note of panel.querySelectorAll('small')) { note.style.display = 'block'; note.style.marginTop = '6px'; }
  function selectionLabels(data) {
    return stages.map(stage => data.options[stage].find(option => option.id === data.selection[stage]).label).join(' / ');
  }
  let state;
  async function read() {
    const response = await (window.TerrainTransport?.fetch||fetch)('/api/models', {cache: 'no-store'});
    if (!response.ok) throw Error('Modèles indisponibles');
    return response.json();
  }
  function render(data) {
    state = data;
    for (const stage of stages) {
      selects[stage].replaceChildren(...data.options[stage].map(option => new Option(option.label, option.id)));
      selects[stage].value = data.selection[stage];
    }
    status.textContent = 'Actif · ' + selectionLabels(data);
  }
  async function apply(reset) {
    if (generationPending) { status.textContent = 'Attends la fin de la génération en cours.'; return; }
    const selection = Object.fromEntries(stages.map(stage => [stage, selects[stage].value]));
    const seed = world?.seed || document.getElementById('seed').value;
    const profile = selectedProfile, generation = activeGeneration;
    const controls = [...panel.querySelectorAll('button,select')];
    controls.forEach(control => control.disabled = true);
    generationPending = true;
    for (const flight of inflight.values()) flight.controller.abort();
    overviewFlight?.controller.abort();
    clearTimeout(timer);
    status.textContent = 'Application…';
    try {
      const response = await (window.TerrainTransport?.fetch||fetch)('/api/models', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({selection, reset})});
      const change = await response.json();
      if (!response.ok) throw Error(change.error || 'Changement refusé');
      const deadline = Date.now() + 180000;
      let ready = null;
      while (Date.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, 700));
        try { const data = await read(); if (!data.switching && data.revision === change.revision) { ready = data; break; } } catch (_) {}
      }
      if (!ready) throw Error('Le serveur ne répond pas encore. Recharge la page pour réessayer.');
      render(ready);
      generationPending = false;
      // The existing viewer clears RAM/WebGPU and reacquires the new neural
      // namespace, retaining its exact camera and generation settings.
      nnEngine = 'exact'; document.getElementById('nnEngine').value = nnEngine;
      const opened = await openWorld(seed, profile, {preserveCamera: true, generation});
      if (!opened) throw Error('Modèles appliqués, mais la vue n’a pas pu être recalculée.');
      status.textContent = (reset ? 'Cache NN réinitialisé · ' : 'NN appliqués · ') + selectionLabels(ready);
    } catch (error) { status.textContent = error.message; }
    finally { generationPending = false; controls.forEach(control => control.disabled = false); draw(); schedule(); }
  }
  panel.querySelector('#distillApply').onclick = () => apply(false);
  panel.querySelector('#distillReset').onclick = () => apply(true);
  read().then(render).catch(error => { status.textContent = error.message; });
})();
