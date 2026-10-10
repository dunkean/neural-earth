/* Loaded only by the explicit local distillation review server. */
(() => {
  const panel = document.createElement('fieldset');
  panel.id = 'distillModels';
  panel.innerHTML = '<legend>Comparaison des NN</legend><p>Chaque modèle peut être remplacé séparément.</p>' +
    '<label>Coarse <select id="distillCoarse" aria-label="Modèle coarse"></select></label>' +
    '<label>Base <select id="distillBase" aria-label="Modèle base"></select></label>' +
    '<label>Decoder <select id="distillDecoder" aria-label="Modèle decoder"></select></label>' +
    '<div class="panelActions"><button id="distillApply" type="button">Appliquer les NN</button>' +
    '<button id="distillReset" type="button">Reset après Orogen</button></div>' +
    '<small>Le reset recalcule coarse, base et decoder. Orogen, seed et caméra sont conservés.</small>' +
    '<small id="distillStatus" role="status"></small>';
  const menu = document.createElement('details');
  menu.id = 'distillPanel';
  menu.innerHTML = '<summary aria-label="Modèles neuronaux">NN</summary><div class="toolPanel"></div>';
  menu.querySelector('.toolPanel').append(panel);
  document.getElementById('controls').append(menu);
  const stages = ['coarse', 'base', 'decoder'];
  const selects = Object.fromEntries(stages.map(stage => [stage, panel.querySelector('#distill' + stage[0].toUpperCase() + stage.slice(1))]));
  const status = panel.querySelector('#distillStatus');
  let state;
  async function read() {
    const response = await fetch('/api/distill/models', {cache: 'no-store'});
    if (!response.ok) throw Error('Modèles indisponibles');
    return response.json();
  }
  function render(data) {
    state = data;
    for (const stage of stages) {
      selects[stage].replaceChildren(...data.options[stage].map(option => new Option(option.label, option.id)));
      selects[stage].value = data.selection[stage];
    }
    status.textContent = 'Actif · ' + stages.map(stage => data.selection[stage]).join(' / ');
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
      const response = await fetch('/api/distill/models', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({selection, reset})});
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
      status.textContent = (reset ? 'Cache NN réinitialisé · ' : 'NN appliqués · ') + stages.map(stage => ready.selection[stage]).join(' / ');
    } catch (error) { status.textContent = error.message; }
    finally { generationPending = false; controls.forEach(control => control.disabled = false); draw(); schedule(); }
  }
  panel.querySelector('#distillApply').onclick = () => apply(false);
  panel.querySelector('#distillReset').onclick = () => apply(true);
  read().then(render).catch(error => { status.textContent = error.message; });
})();
