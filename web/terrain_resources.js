/* Resource layers: legend, season control, GPU packages and cell inspector.
 * Tiles are physical tiles shaded by the WGSL evaluator (terrain_resources_gpu.js);
 * this module loads one package per world and mode and explains the view.
 * Inspector values are those of the LOD-10 authority cell. */
window.TerrainResources=(()=>{
  const PREFIX='resources-';
  let specs={},panel=null,body=null,inspector=null,seasonRow=null,progress=null,ticket=0,loadTicket=0;
  let renderer=null,redraw=()=>{},world=null,packages=new Map(),current=null,season=.5,lastLoadMs=0;
  const esc=value=>String(value).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const fmt=value=>value==null?'—':Math.abs(value)>=1e4||(Math.abs(value)>0&&Math.abs(value)<1e-2)?Number(value).toExponential(2):Number(value).toLocaleString('fr-FR',{maximumFractionDigits:3});
  const months=['janv.','févr.','mars','avr.','mai','juin','juil.','août','sept.','oct.','nov.','déc.'];
  function mount(){
    if(panel)return;
    panel=document.createElement('div');panel.id='resourceLegend';panel.hidden=true;
    panel.style.cssText='max-width:360px;margin-bottom:8px;line-height:1.5;font-size:12px';
    panel.innerHTML='<div data-resource-body></div><div data-resource-progress hidden style="margin:4px 0"><div style="height:4px;background:#ffffff22;border-radius:2px"><div style="height:4px;width:0;background:#9cc9ff;border-radius:2px;transition:width .4s"></div></div><small></small></div><label data-resource-season hidden style="display:block;margin-top:4px">Saison <input type="range" min="0" max="1" step="0.01" value="0.5" style="width:150px;vertical-align:middle"> <span></span></label><p data-resource-inspector style="margin:6px 0 0;white-space:pre-line">Cliquer sur la carte pour inspecter la cellule LOD 10 (30,72 km).</p><a href="/docs/RESOURCE_LAYERS.md" target="_blank" rel="noopener" style="color:#9cc9ff">Méthode des ressources</a>';
    body=panel.querySelector('[data-resource-body]');progress=panel.querySelector('[data-resource-progress]');inspector=panel.querySelector('[data-resource-inspector]');seasonRow=panel.querySelector('[data-resource-season]');
    const input=seasonRow.querySelector('input');
    input.oninput=()=>{season=Number(input.value);syncSeason();renderer?.setResourceSeason?.(season);redraw();};
    (document.getElementById('hud')||document.body).prepend(panel);
  }
  function syncSeason(){const m=season*12-.5,i=((Math.round(m)%12)+12)%12;seasonRow.querySelector('span').textContent=months[i]+' (calendrier nord)';seasonRow.querySelector('input').value=String(season);}
  function legend(spec){
    const status=`<small>Statut : ${esc(spec.status)}${spec.proxy?' · <b>proxy</b> d’un champ de rendu (Render v2)':''} · détail calculé sur GPU</small>`;
    if(spec.classes&&spec.kind!=='ramp'){
      const items=spec.classes.map(([name,color])=>`<span style="display:inline-block;margin-right:10px"><i style="display:inline-block;width:9px;height:9px;margin-right:4px;background:#${color};border:1px solid #0006"></i>${esc(name)}</span>`).join('');
      const note=spec.kind==='points'?'<br><small>Couleur : famille · taille : tonnage (log) · teinte sombre : faible part accessible au Moyen Âge. Au zoom : filons, amas, lentilles, placers, halo et chapeau de fer.</small>':spec.kind==='mix'?(spec.label.startsWith('Contexte')?'<br><small>Contexte dominant : les fractions sont des appartenances floues (distance aux frontières de plaques), contacts déformés.</small>':'<br><small>Géologie réalisée : les mêmes unités à tous les zooms (provinces issues des contextes tectoniques, ceintures le long de la structure) ; les parts de chaque cellule de 30 km en sont mesurées.</small>'):'';
      return `<strong>${esc(spec.label)}</strong><br>${items}${note}<br>${status}`;
    }
    const stops=spec.colors.map((c,i)=>`#${c} ${Math.round(100*i/(spec.colors.length-1))}%`).join(',');
    return `<strong>${esc(spec.label)}</strong><div style="height:10px;margin:4px 0;background:linear-gradient(90deg,${stops})"></div>`+
      `<div style="display:flex;justify-content:space-between"><span>${fmt(spec.range[0])}</span><span>${esc(spec.unit)}${spec.log?' (échelle log)':''}</span><span>${fmt(spec.range[1])}</span></div>${note(spec)}${status}`;
  }
  // Loading state: bar, stage and estimate; the relief map stays visible underneath.
  function showProgress(text,fraction,error){
    if(!progress)return;
    progress.hidden=text==null;if(text==null)return;
    progress.querySelector('small').textContent=text;progress.querySelector('small').style.color=error?'#ff9c9c':'';
    progress.firstElementChild.firstElementChild.style.width=Math.round(100*Math.max(.03,Math.min(1,fraction||0)))+'%';
  }
  const wait=ms=>new Promise(ok=>setTimeout(ok,ms));
  // What a mineral layer shows: a potential (zones) or the stock of the deposits.
  function note(spec){
    if(spec.label.includes('(gisements)'))return '<small>Stock des gisements : densité lissée à l’échelle des districts (écart-type ~45 km) aux petites échelles, amas au zoom moyen, empreintes (filons, amas, placers) au zoom proche ; noyaux normalisés par l’aire sphérique (tolérance numérique mesurée).</small><br>';
    if(spec.label.startsWith('Potentiel'))return '<small>Potentiel en zones : dotation typique par km² (densité × prospectivité × tonnage médian × teneur, modèles de Cox et Singer), indépendant du tirage des gisements.</small><br>';
    return '';
  }
  async function load(mode){
    if(!renderer?.setResources||!world)return;
    const key=world.seed+'|'+mode,mine=++loadTicket,started=performance.now();
    current=null;   // until this package is on the GPU
    // No package bound: the renderer keeps the relief map (not a blank layer).
    renderer.resetResources?.();renderer.markResourceTiles?.();redraw();
    try{
      let pkg=packages.get(key);
      while(!pkg){
        showProgress('Chargement de la couche…',.05);
        const r=await fetch(`/api/resources/${encodeURIComponent(world.seed)}/gpu?mode=${encodeURIComponent(mode)}&world_profile=${encodeURIComponent(world.generation_profile||'orogen')}`);
        if(mine!==loadTicket)return;
        if(r.status===202){            // world resources still compiling in the background
          const s=await r.json();
          if(s.state==='error')throw Error(s.error||'échec du calcul des ressources');
          showProgress((s.message||'Calcul des ressources du monde…')+(s.waiting_for?' — cette couche attend une étape suivante':''),s.progress);
          await wait(1000);if(mine!==loadTicket)return;continue;
        }
        if(!r.ok)throw Error((await r.json()).error);
        const bytes=await r.arrayBuffer(),length=new DataView(bytes).getUint32(0,true);
        const header=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,4,length)));
        pkg={header,data:window.TerrainResourcesGPU.parse(header,bytes.slice(4+length))};
        packages.set(key,pkg);if(packages.size>6)packages.delete(packages.keys().next().value);
      }
      if(mine!==loadTicket)return;
      renderer.setResources(pkg.header,pkg.data,{season});current=mode;lastLoadMs=performance.now()-started;showProgress(null);redraw();
    }catch(error){if(mine===loadTicket){showProgress('Couche indisponible : '+error.message,1,true);}}
  }
  // Start the world's background compile as soon as the world opens.
  async function prepare(){
    if(!world)return;
    try{await fetch(`/api/resources/${encodeURIComponent(world.seed)}/status?world_profile=${encodeURIComponent(world.generation_profile||'orogen')}`);}catch(_){}
  }
  function sync(mode){
    mount();
    const spec=specs[mode],on=!!(mode&&mode.startsWith(PREFIX)&&spec);
    panel.hidden=!on;
    if(!on){current=null;return;}
    body.innerHTML=legend(spec);seasonRow.hidden=!spec.seasonal;syncSeason();
    if(current!==mode)load(mode);
  }
  function setWorld(data){
    world=data;specs=data?.resource_layers||{};packages.clear();current=null;
    if(Object.keys(specs).length)prepare();
    for(const button of document.querySelectorAll('#layerPanel button[data-mode^="'+PREFIX+'"]'))button.hidden=!specs[button.dataset.mode];
    for(const option of document.querySelectorAll('#mapMode option[value^="'+PREFIX+'"]')){option.hidden=!specs[option.value];option.disabled=!specs[option.value];}
    sync(document.getElementById('mapMode')?.value);
  }
  function setRenderer(r,onRedraw){renderer=r;redraw=onRedraw||redraw;current=null;sync(document.getElementById('mapMode')?.value);}
  async function inspect({x,y,seed,profile,fetchJSON}){
    mount();if(panel.hidden)return false;
    const mine=++ticket;inspector.textContent='Lecture de la cellule…';
    try{
      const response=await (fetchJSON||fetch)(`/api/resources/${encodeURIComponent(seed)}/inspect?x=${x}&y=${y}&world_profile=${encodeURIComponent(profile)}`);
      const data=await response.json();if(mine!==ticket)return true;
      if(response.status===202){inspector.textContent=data.message||'Calcul des ressources du monde en cours…';return true;}
      if(!response.ok)throw Error(data.error);
      const mode=document.getElementById('mapMode').value,value=data.values?.[mode];
      const lines=[`Cellule LOD ${data.cell.lod} (${data.cell.i}, ${data.cell.j}) · ${fmt(data.cell.area_km2)} km²`,
        `Contexte dominant : ${data.dominant.context} · roche dominante : ${data.dominant.rock}`];
      if(value)lines.push(`${value.label} : ${fmt(value.value)} ${value.unit}${value.proxy?' (proxy)':''}${specs[mode]?.seasonal?' (moyenne annuelle)':''}`);
      for(const d of data.deposits.slice(0,6)){
        const metals=Object.entries(d.contained).map(([m,v])=>`${m} ${fmt(v)} ${d.contained_units[m]}`).join(', ');
        lines.push(`• ${d.label} à ${fmt(d.distance_m/1000)} km : minerai ${fmt(d.ore_t)} t, ${metals} ; toit ${fmt(d.depth_top_m)} m, accessible ${Math.round(100*d.medieval_accessible_share)} %${d.outcrop?', affleurant':''}${d.gossan?', chapeau de fer':''}`);
      }
      if(!data.deposits.length)lines.push('Aucun gisement à moins de 60 km.');
      inspector.textContent=lines.join('\n');
    }catch(error){if(mine===ticket)inspector.textContent='Inspection impossible : '+error.message;}
    return true;
  }
  return {PREFIX,mount,sync,setWorld,setRenderer,inspect,specs:()=>specs,season:()=>season,
    state:()=>({current,loading:loadTicket,lastLoadMs,progress:progress&&!progress.hidden?progress.textContent:null,header:renderer?.res?.header?.mode||null,generation:renderer?.resGeneration||0,lastUploadMs:renderer?.lastResourceUploadMs||0})};
})();
document.getElementById('mapMode')?.addEventListener('change',event=>window.TerrainResources.sync(event.target.value));
