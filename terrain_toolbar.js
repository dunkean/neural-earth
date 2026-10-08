window.TerrainToolbar=(()=>{
  let sync=()=>{};
  function install(){
    const $=id=>document.getElementById(id),controls=$('controls');
    const icons={fit:['◎','Vue du monde'],initialView:['▣','Région initiale'],native:['⌖','30 m / pixel'],measureDistance:['↔','Mesurer une distance'],random:['⚄','Nouvelle seed'],apply:['▶','Générer']};
    for(const [id,[icon,label]] of Object.entries(icons)){const button=$(id);button.textContent=icon;button.title=label;button.setAttribute('aria-label',label);button.classList.add('iconButton');}
    for(const id of ['plus','minus'])$(id).classList.add('iconButton');
    const gridLabel=$('grid').closest('label');gridLabel.className='iconToggle';gridLabel.title='Tiles visibles';gridLabel.lastChild.textContent='▦';$('grid').setAttribute('aria-label','Tiles visibles');
    const render=$('renderPanel').querySelector('.toolPanel');
    const depths=document.createElement('fieldset');depths.innerHTML='<legend>Chargement & raffinement</legend>';
    const refinement=$('refinementDepth').closest('label');refinement.firstChild.textContent='Raffinement · profondeur LOD ';refinement.title='0 : désactivé. Chaque LOD visible est terminée avant la suivante.';
    const prefetch=$('prefetch').closest('label');prefetch.firstChild.textContent='Précharger les tiles voisines ';prefetch.title='Anticipe les déplacements au LOD courant, indépendamment du raffinement.';
    depths.append(prefetch,refinement);render.querySelector('.panelHeading').after(depths);
    $('renderPanel').querySelector('summary').textContent='◉';$('renderPanel').querySelector('summary').title='Rendu';$('renderPanel').querySelector('summary').setAttribute('aria-label','Rendu');render.querySelector('.panelHeading strong').textContent='Rendu';
    $('runtimePanel').querySelector('summary').textContent='⋯';$('runtimePanel').querySelector('summary').title='Outils';$('runtimePanel').querySelector('summary').setAttribute('aria-label','Outils');
    render.querySelector('p').textContent='Écart maximal de l’historique plus fin. Les parents plus grossiers disponibles servent toujours de secours pendant le raffinement.';
    const select=$('mapMode'),label=select.closest('label'),panel=document.createElement('details');panel.id='layerPanel';
    panel.innerHTML='<summary title="Layer visible" aria-label="Layer visible"><span class="selectedLayer">Relief</span><span aria-hidden="true">▾</span></summary><div class="toolPanel"><div class="panelHeading"><strong>Layer visible</strong><button type="button" class="panelClose" aria-label="Close layer menu">×</button></div><div class="layerGroups"></div></div>';
    label.hidden=true;controls.append(label,panel);
    const groups=panel.querySelector('.layerGroups'),buttons=[],labels=new Map([
      ['relief','Relief'],['orogen-biomes','Biomes'],['temperature','Température annuelle'],['precipitation','Précipitations annuelles'],['orogen-koppen','Köppen']
    ]);
    const sections=[
      ['Cartes principales',['relief','orogen-biomes','temperature','precipitation','orogen-koppen']],
      ['Climat saisonnier',['orogen-temperature-summer','orogen-temperature-winter','orogen-precip-summer','orogen-precip-winter']],
      ['Atmosphère',['orogen-pressure-summer','orogen-pressure-winter','orogen-wind-summer','orogen-wind-winter','orogen-rain-shadow','orogen-continentality']],
      ['Océans',['orogen-currents-summer','orogen-currents-winter']],
      ['Plaques tectoniques',['orogen-plates','orogen-superplates','orogen-crust','orogen-boundaries','orogen-convergence']],
      ['Formation du relief',['orogen-height','orogen-uplift','orogen-orogeny','orogen-back-arc','orogen-fold-ridges','orogen-hotspots','orogen-erosion']]
    ];
    const options=new Map();
    // SNR remains an internal diagnostic, outside the visible layer menu.
    for(const option of select.options){if(option.value==='biomes'||option.value.startsWith('snr-')){option.hidden=true;continue;}options.set(option.value,option);}
    const assigned=new Set(sections.flatMap(([,modes])=>modes)),remaining=[...options.keys()].filter(mode=>!assigned.has(mode));
    if(remaining.length)sections.push(['Autres couches',remaining]);
    for(const [name,modes]of sections){
      const available=modes.filter(mode=>options.has(mode));if(!available.length)continue;
      const section=document.createElement('fieldset'),heading=document.createElement('legend'),choices=document.createElement('div');
      heading.textContent=name;choices.className='layerChoices';section.append(heading,choices);groups.append(section);
      for(const mode of available){const option=options.get(mode),button=document.createElement('button');button.type='button';button.textContent=labels.get(mode)||option.textContent;button.title=option.textContent;button.dataset.mode=mode;button.onclick=()=>{select.value=mode;select.dispatchEvent(new Event('change',{bubbles:true}));panel.open=false;sync();};choices.append(button);buttons.push([button,option]);}
    }
    const generators=document.createElement('div');generators.className='generatorIcons';generators.setAttribute('role','group');generators.setAttribute('aria-label','Relief initial');
    const generatorButtons=[];
    for(const [value,icon,name] of [['natural','▧','noise'],['orogen','⛰','Tectonique'],['custom','⚙','Custom']]){const button=document.createElement('button');button.type='button';button.textContent=icon;button.title=name;button.setAttribute('aria-label',name);button.onclick=()=>{$('worldGenerator').value=value;$('worldGenerator').dispatchEvent(new Event('change',{bubbles:true}));sync();};generators.append(button);generatorButtons.push([button,value]);}
    $('worldGenerator').closest('label').after(generators);
    sync=()=>{for(const [button,option] of buttons){button.disabled=option.disabled;button.setAttribute('aria-pressed',String(select.value===option.value));}for(const [button,value] of generatorButtons)button.setAttribute('aria-pressed',String($('worldGenerator').value===value));const selected=labels.get(select.value)||select.selectedOptions[0]?.textContent||'Relief';panel.querySelector('.selectedLayer').textContent=selected;panel.querySelector('summary').title='Layer visible · '+selected;panel.querySelector('summary').setAttribute('aria-label','Layer visible · '+selected);};
    select.addEventListener('change',sync);new MutationObserver(sync).observe(select,{subtree:true,attributes:true});sync();
  }
  return {install,sync:()=>sync()};
})();
