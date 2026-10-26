/* Resource layers: legend and cell inspector for orogen-resources-* modes.
 * Tiles come from the existing source-diagnostic path; this module only
 * explains them. Values shown are those of the LOD-10 authority cell. */
window.TerrainResources=(()=>{
  const PREFIX='orogen-resources-';
  let specs={},panel=null,body=null,inspector=null,ticket=0;
  const esc=value=>String(value).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const fmt=value=>value==null?'—':Math.abs(value)>=1e4||(Math.abs(value)>0&&Math.abs(value)<1e-2)?Number(value).toExponential(2):Number(value).toLocaleString('fr-FR',{maximumFractionDigits:3});
  function mount(){
    if(panel)return;
    panel=document.createElement('div');panel.id='resourceLegend';panel.hidden=true;
    panel.style.cssText='max-width:360px;margin-bottom:8px;line-height:1.5;font-size:12px';
    panel.innerHTML='<div data-resource-body></div><p data-resource-inspector style="margin:6px 0 0;white-space:pre-line">Cliquer sur la carte pour inspecter la cellule LOD 10 (30,72 km).</p><a href="/docs/RESOURCE_LAYERS.md" target="_blank" rel="noopener" style="color:#9cc9ff">Méthode des ressources</a>';
    body=panel.querySelector('[data-resource-body]');inspector=panel.querySelector('[data-resource-inspector]');
    (document.getElementById('hud')||document.body).prepend(panel);
  }
  function legend(spec){
    const status=`<small>Statut : ${esc(spec.status)}${spec.proxy?' · <b>proxy</b> d’un champ de rendu (Render v2)':''}</small>`;
    if(spec.classes&&spec.kind!=='ramp'){
      const items=spec.classes.map(([name,color])=>`<span style="display:inline-block;margin-right:10px"><i style="display:inline-block;width:9px;height:9px;margin-right:4px;background:#${color};border:1px solid #0006"></i>${esc(name)}</span>`).join('');
      const note=spec.kind==='points'?'<br><small>Couleur : famille · taille : tonnage de minerai (log) · teinte sombre : faible part accessible au Moyen Âge.</small>':spec.kind==='mix'?'<br><small>Couleurs mélangées selon les fractions.</small>':'';
      return `<strong>${esc(spec.label)}</strong><br>${items}${note}<br>${status}`;
    }
    const stops=spec.colors.map((c,i)=>`#${c} ${Math.round(100*i/(spec.colors.length-1))}%`).join(',');
    return `<strong>${esc(spec.label)}</strong><div style="height:10px;margin:4px 0;background:linear-gradient(90deg,${stops})"></div>`+
      `<div style="display:flex;justify-content:space-between"><span>${fmt(spec.range[0])}</span><span>${esc(spec.unit)}${spec.log?' (échelle log)':''}</span><span>${fmt(spec.range[1])}</span></div>${status}`;
  }
  function sync(mode){
    mount();
    const spec=specs[mode];
    panel.hidden=!(mode&&mode.startsWith(PREFIX)&&spec);
    if(!panel.hidden){body.innerHTML=legend(spec);}
  }
  function setWorld(data){
    specs=data?.resource_layers||{};
    // Hide menu entries that this world does not serve (e.g. fantasy disabled).
    for(const button of document.querySelectorAll('#layerPanel button[data-mode^="'+PREFIX+'"]'))button.hidden=!specs[button.dataset.mode];
    for(const option of document.querySelectorAll('#mapMode option[value^="'+PREFIX+'"]'))option.hidden=!specs[option.value];
    sync(document.getElementById('mapMode')?.value);
  }
  async function inspect({x,y,seed,profile,fetchJSON}){
    mount();if(panel.hidden)return false;
    const current=++ticket;inspector.textContent='Lecture de la cellule…';
    try{
      const response=await (fetchJSON||fetch)(`/api/resources/${encodeURIComponent(seed)}/inspect?x=${x}&y=${y}&world_profile=${encodeURIComponent(profile)}`);
      const data=await response.json();if(current!==ticket)return true;
      if(!response.ok)throw Error(data.error);
      const mode=document.getElementById('mapMode').value,value=data.values?.[mode];
      const lines=[`Cellule LOD ${data.cell.lod} (${data.cell.i}, ${data.cell.j}) · ${fmt(data.cell.area_km2)} km²`,
        `Contexte dominant : ${data.dominant.context} · roche dominante : ${data.dominant.rock}`];
      if(value)lines.push(`${value.label} : ${fmt(value.value)} ${value.unit}${value.proxy?' (proxy)':''}`);
      for(const d of data.deposits.slice(0,6)){
        const metals=Object.entries(d.contained).map(([m,v])=>`${m} ${fmt(v)} ${d.contained_units[m]}`).join(', ');
        lines.push(`• ${d.label} à ${fmt(d.distance_m/1000)} km : minerai ${fmt(d.ore_t)} t, ${metals} ; toit ${fmt(d.depth_top_m)} m, accessible ${Math.round(100*d.medieval_accessible_share)} %${d.outcrop?', affleurant':''}${d.gossan?', chapeau de fer':''}`);
      }
      if(!data.deposits.length)lines.push('Aucun gisement à moins de 60 km.');
      inspector.textContent=lines.join('\n');
    }catch(error){if(current===ticket)inspector.textContent='Inspection impossible : '+error.message;}
    return true;
  }
  return {PREFIX,mount,sync,setWorld,inspect,specs:()=>specs};
})();
document.getElementById('mapMode')?.addEventListener('change',event=>window.TerrainResources.sync(event.target.value));
