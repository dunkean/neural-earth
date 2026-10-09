window.TerrainToolbar=(()=>{
  let sync=()=>{};
  function install(){
    const $=id=>document.getElementById(id),controls=$('controls');
    const icons={fit:['◎','World view'],initialView:['▣','Initial region'],native:['⌖','30 m / pixel'],measureDistance:['↔','Measure distance'],random:['⚄','New seed'],apply:['▶','Generate']};
    for(const [id,[icon,label]] of Object.entries(icons)){const button=$(id);button.textContent=icon;button.title=label;button.setAttribute('aria-label',label);button.classList.add('iconButton');}
    for(const id of ['plus','minus'])$(id).classList.add('iconButton');
    const gridLabel=$('grid').closest('label');gridLabel.className='iconToggle';gridLabel.title='Visible tiles';gridLabel.lastChild.textContent='▦';$('grid').setAttribute('aria-label','Visible tiles');
    const render=$('renderPanel').querySelector('.toolPanel');
    const depths=document.createElement('fieldset');depths.innerHTML='<legend>Loading & refinement</legend>';
    const refinement=$('refinementDepth').closest('label');refinement.firstChild.textContent='Refinement · LOD depth ';refinement.title='0: disabled. Each visible LOD finishes before the next begins.';
    const prefetch=$('prefetch').closest('label');prefetch.firstChild.textContent='Prefetch neighboring tiles ';prefetch.title='Preloads nearby tiles at the current LOD, independently of refinement.';
    depths.append(prefetch,refinement);render.querySelector('.panelHeading').after(depths);
    $('renderPanel').querySelector('summary').textContent='◉';$('renderPanel').querySelector('summary').title='Rendering';$('renderPanel').querySelector('summary').setAttribute('aria-label','Rendering');render.querySelector('.panelHeading strong').textContent='Rendering';
    $('runtimePanel').querySelector('summary').textContent='⋯';$('runtimePanel').querySelector('summary').title='Tools';$('runtimePanel').querySelector('summary').setAttribute('aria-label','Tools');
    render.querySelector('p').textContent='Maximum retained difference for finer cached LODs. Available coarse parents remain visible during refinement.';
    const select=$('mapMode'),label=select.closest('label'),panel=document.createElement('details');panel.id='layerPanel';
    panel.innerHTML='<summary title="Layer visible" aria-label="Layer visible"><span class="selectedLayer">Relief</span><span aria-hidden="true">▾</span></summary><div class="toolPanel"><div class="panelHeading"><strong>Layer visible</strong><button type="button" class="panelClose" aria-label="Close layer menu">×</button></div><div class="layerGroups"></div></div>';
    label.hidden=true;controls.append(label,panel);
    const groups=panel.querySelector('.layerGroups'),buttons=[],labels=new Map([
      ['relief','Relief'],['render','Render'],['soil','Soils'],['pedology','Pedology'],['orogen-biomes','Biomes'],['temperature','Annual temperature'],['precipitation','Annual precipitation'],['orogen-koppen','Köppen']
    ]);
    const pedologyLegend=document.createElement('div');pedologyLegend.id='pedologyLegend';pedologyLegend.hidden=true;
    pedologyLegend.style.cssText='max-width:340px;margin-bottom:8px;line-height:1.6';
    pedologyLegend.innerHTML='<strong>Pedology</strong> · regional mixtures<br>'+[
      ['#b09163','Sandy'],['#94876e','Calcareous'],['#7a5940','Clay-rich'],['#8f4d2e','Ferrallitic'],
      ['#4a402e','Organic'],['#6e6e5e','Podzolic'],['#80786b','Mineral']
    ].map(([color,name])=>`<span style="display:inline-block;margin-right:10px"><i style="display:inline-block;width:9px;height:9px;margin-right:4px;background:${color}"></i>${name}</span>`).join('');
    $('hud').prepend(pedologyLegend);
    const sections=[
      ['Main maps',['relief','render','soil','pedology','orogen-biomes','temperature','precipitation','orogen-koppen']],
      ['Map styles',Object.keys(window.TerrainStyles||{})],
      ['Seasonal climate',['orogen-temperature-summer','orogen-temperature-winter','orogen-precip-summer','orogen-precip-winter']],
      ['Atmosphere',['orogen-pressure-summer','orogen-pressure-winter','orogen-wind-summer','orogen-wind-winter','orogen-rain-shadow','orogen-continentality']],
      ['Oceans',['orogen-currents-summer','orogen-currents-winter']],
      ['Tectonic plates',['orogen-plates','orogen-superplates','orogen-crust','orogen-boundaries','orogen-convergence']],
      ['Relief formation',['orogen-height','orogen-uplift','orogen-orogeny','orogen-back-arc','orogen-fold-ridges','orogen-hotspots','orogen-erosion']]
    ];
    const options=new Map();
    // SNR remains an internal diagnostic, outside the visible layer menu.
    for(const option of select.options){if(option.value==='biomes'||option.value.startsWith('snr-')){option.hidden=true;continue;}options.set(option.value,option);}
    const assigned=new Set(sections.flatMap(([,modes])=>modes)),remaining=[...options.keys()].filter(mode=>!assigned.has(mode));
    if(remaining.length)sections.push(['Other layers',remaining]);
    for(const [name,modes]of sections){
      const available=modes.filter(mode=>options.has(mode));if(!available.length)continue;
      const section=document.createElement('fieldset'),heading=document.createElement('legend'),choices=document.createElement('div');
      heading.textContent=name;choices.className='layerChoices';section.append(heading,choices);groups.append(section);
      for(const mode of available){const option=options.get(mode),button=document.createElement('button');button.type='button';button.textContent=labels.get(mode)||option.textContent;button.title=option.textContent;button.dataset.mode=mode;button.onclick=()=>{select.value=mode;select.dispatchEvent(new Event('change',{bubbles:true}));panel.open=false;sync();};choices.append(button);buttons.push([button,option]);}
    }
    const generators=document.createElement('div');generators.className='generatorIcons';generators.setAttribute('role','group');generators.setAttribute('aria-label','Initial relief');
    const generatorButtons=[];
    for(const [value,icon,name] of [['natural','▧','noise'],['orogen','⛰','Tectonic'],['custom','⚙','Custom']]){const button=document.createElement('button');button.type='button';button.textContent=icon;button.title=name;button.setAttribute('aria-label',name);button.onclick=()=>{$('worldGenerator').value=value;$('worldGenerator').dispatchEvent(new Event('change',{bubbles:true}));sync();};generators.append(button);generatorButtons.push([button,value]);}
    $('worldGenerator').closest('label').after(generators);
    sync=()=>{pedologyLegend.hidden=select.value!=='pedology';for(const [button,option] of buttons){button.disabled=option.disabled;button.setAttribute('aria-pressed',String(select.value===option.value));}for(const [button,value] of generatorButtons)button.setAttribute('aria-pressed',String($('worldGenerator').value===value));const selected=labels.get(select.value)||select.selectedOptions[0]?.textContent||'Relief';panel.querySelector('.selectedLayer').textContent=selected;panel.querySelector('summary').title='Layer visible · '+selected;panel.querySelector('summary').setAttribute('aria-label','Layer visible · '+selected);};
    select.addEventListener('change',sync);new MutationObserver(sync).observe(select,{subtree:true,attributes:true});sync();
  }
  return {install,sync:()=>sync()};
})();
