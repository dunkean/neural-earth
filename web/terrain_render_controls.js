/* Display-only material settings. Exact RGB values survive/reset independently
 * of the color input's eight-bit preview and the informational biome palette. */
window.TerrainRender=(()=>{
  const defaults={season:.5,variation:1,forest:1,moisture:0,rock_slope:40,snow:1,vegetation_tint:[1,1,1],rock_tint:[1,1,1],snow_color:[.94,.96,.97]};
  const limits={season:[0,1,.01],variation:[0,2,.05],forest:[0,2,.05],moisture:[-1,1,.05],rock_slope:[15,75,1],snow:[0,2,.05]};
  const clone=x=>JSON.parse(JSON.stringify(x));
  function normalize(value){
    const s=clone(defaults);
    for(const [k,v] of Object.entries(value||{})){
      if(k in limits){const [a,b]=limits[k];if(Number.isFinite(v))s[k]=Math.max(a,Math.min(b,v));}
      else if(k in defaults&&Array.isArray(v)&&v.length===3&&v.every(x=>Number.isFinite(x)&&x>=0&&x<=1))s[k]=[...v];
    }return s;
  }
  let settings=clone(defaults),sync=()=>{},changed=()=>{};
  try{settings=normalize(JSON.parse(new URLSearchParams(location.search).get('render_settings')||'null'));}catch(_){}
  function reset(){settings=clone(defaults);sync();changed();}
  function mount(container,onChange){
    changed=onChange;
    const field=document.createElement('fieldset');field.id='renderMaterialSettings';
    field.innerHTML='<legend>Render · natural materials</legend><div class="materialControls"></div><div class="panelActions"><button type="button" id="renderMaterialReset">Reset Render</button></div><small>Materials derived from continuous climate, slope and terrain position (ridges, valleys). Seasons run from northern winter to summer and back; the south follows its own climate. Changes apply immediately.</small>';
    container.prepend(field);const controls=field.querySelector('.materialControls'),inputs={};
    const labels={season:'Annual cycle (January → December)',variation:'Material variation',forest:'Forest cover',moisture:'Moisture (arid ↔ lush)',rock_slope:'Rock slope (°)',snow:'Snow cover',vegetation_tint:'Vegetation tint',rock_tint:'Rock tint',snow_color:'Snow color'};
    for(const [key,baseline] of Object.entries(defaults)){
      const label=document.createElement('label');label.textContent=labels[key];
      if(Array.isArray(baseline)){
        const input=document.createElement('input');input.type='color';input.id='render_'+key;input.setAttribute('aria-label',labels[key]);label.append(input);inputs[key]=[input];
        input.oninput=()=>{settings[key]=[1,3,5].map(i=>parseInt(input.value.slice(i,i+2),16)/255);sync();changed();};
      }else{
        const pair=document.createElement('span');pair.className='numericControl';inputs[key]=[];
        for(const type of ['range','number']){const input=document.createElement('input');input.type=type;input.id='render_'+key+(type==='range'?'_range':'');const [min,max,step]=limits[key];Object.assign(input,{min,max,step});input.setAttribute('aria-label',labels[key]);pair.append(input);inputs[key].push(input);input.oninput=()=>{if(!Number.isFinite(input.valueAsNumber))return;settings[key]=Math.max(min,Math.min(max,input.valueAsNumber));sync();changed();};}label.append(pair);
      }controls.append(label);
    }
    sync=()=>{for(const [key,pair] of Object.entries(inputs))for(const input of pair){const value=settings[key];if(Array.isArray(value)){input.dataset.rgb=JSON.stringify(value);input.value='#'+value.map(x=>Math.round(x*255).toString(16).padStart(2,'0')).join('');}else input.value=value;}};
    field.querySelector('#renderMaterialReset').onclick=reset;sync();
  }
  return {normalize,mount,reset,set(value){settings=normalize(value);sync();},get:()=>clone(settings),defaults:()=>clone(defaults),serialize:()=>JSON.stringify(settings)};
})();

/* Request-level terrain detail. Before/Hybrid/Phasor/MSE retain separate tile sets. */
window.TerrainAmplification=(()=>{
  const key='neural-earth-amplification-v2',legacyKey='neural-earth-amplification-v1',params=new URLSearchParams(location.search);
  const labels={hybrid:'Hybride (recommandé)',phasor:'Phasor (Grenier 2024)',mse:'MSE (expérimental, Schott 2024)'};
  const t=(name,fallback)=>window.NeuralEarthLocale?.text('amplification.'+name)||fallback;
  const modeLabel=mode=>t(mode,labels[mode]||mode);
  const normalize=value=>({mode:['off',...Object.keys(labels)].includes(value?.mode)?value.mode:value?.enabled?'hybrid':'off',intensity:Math.round(Math.min(2,Math.max(0,Number.isFinite(value?.intensity)?value.intensity:1))*100)/100,before:!!value?.before});
  let saved={};try{saved=JSON.parse(localStorage.getItem(key)||localStorage.getItem(legacyKey))||{};}catch(_){}
  let state=normalize(saved),available=[],unavailable=[],changed=()=>{},sync=()=>{},badge=null,displayedModes=[];
  if(params.has('amplify')){const n=Number(params.get('amplify')),mode=params.get('amplify_mode')||'hybrid',active=mode in labels&&(n>0||n===0&&params.has('amplify_mode'));state=normalize({mode:active?mode:'off',intensity:active?n:1,before:params.get('amplify_before')==='1'});}
  function save(){try{localStorage.setItem(key,JSON.stringify(state));}catch(_){}}
  save();
  const effectiveMode=()=>!state.before&&available.includes(state.mode)?state.mode:'off';
  const effective=()=>effectiveMode()==='off'?0:state.intensity;
  function set(value,notify=false){state=normalize(value);save();sync();if(notify)changed();}
  function compare(){if(available.includes(state.mode))set({...state,before:!state.before},true);}
  function cycle(){const sequence=['off',...available],current=effectiveMode(),next=sequence[(sequence.indexOf(current)+1)%sequence.length];if(next==='off')set({...state,before:true},true);else set({...state,mode:next,before:false},true);}
  function writeParams(p){if(state.mode in labels&&window.NeuralEarthRuntime?.kind!=='wasm'){p.set('amplify',String(state.intensity));p.set('amplify_mode',state.mode);if(state.before)p.set('amplify_before','1');else p.delete('amplify_before');}else{for(const k of ['amplify','amplify_mode','amplify_before'])p.delete(k);}}
  function mount(container,onChange){
    changed=onChange;const field=document.createElement('fieldset');field.id='amplificationSettings';field.hidden=true;
    field.innerHTML='<legend>Amplification du terrain (expérimental)</legend><label>Mode <select id="amplificationMode" aria-label="Amplification du terrain"></select></label><small id="amplificationUnavailable" role="status" hidden></small><label>Intensité <input id="amplificationIntensity" type="range" min="0" max="2" step="0.05" value="1"><output></output></label><div class="panelActions"><button id="amplificationCompare" type="button">Avant / après (B)</button><button id="amplificationCycle" type="button">Avant → Hybride → Phasor → MSE (Maj+B)</button></div><small>Détail procédural plausible, pas un terrain mesuré.</small>';
    container.prepend(field);const selector=field.querySelector('select'),slider=field.querySelector('input'),button=field.querySelector('#amplificationCompare'),cycleButton=field.querySelector('#amplificationCycle');
    badge=document.createElement('small');badge.id='amplificationBadge';badge.hidden=true;container.closest('#controls')?.append(badge);
    sync=()=>{field.hidden=!available.length&&!unavailable.length;field.querySelector('legend').textContent=t('title','Amplification du terrain (expérimental)');selector.setAttribute('aria-label',t('title','Amplification du terrain'));selector.closest('label').firstChild.textContent=t('mode','Mode')+' ';slider.closest('label').firstChild.textContent=t('intensity','Intensité')+' ';field.querySelector('small:last-child').textContent=t('hint','Détail procédural plausible, pas un terrain mesuré.');selector.replaceChildren(...[['off',t('off','Classique (raffinement actuel)')],...available.map(mode=>[mode,modeLabel(mode)])].map(([value,text])=>new Option(text,value)));selector.value=available.includes(state.mode)?state.mode:'off';slider.value=state.intensity;slider.disabled=!available.includes(state.mode);field.querySelector('output').textContent=state.intensity.toFixed(2);button.disabled=slider.disabled;button.textContent=t(effectiveMode()==='off'?'before':'after',effectiveMode()==='off'?'Avant':'Après')+' · '+t('compare','comparer (B)');button.setAttribute('aria-pressed',String(effectiveMode()!=='off'));cycleButton.hidden=available.length<2;cycleButton.textContent=t('cycle','Avant → Hybride → Phasor → MSE (Maj+B)');const notice=field.querySelector('#amplificationUnavailable');notice.hidden=!unavailable.length;notice.textContent=(state.mode!=='off'&&!available.includes(state.mode)?t('disabled','Amplification désactivée.')+' ':'')+unavailable.map(item=>modeLabel(item.mode)+' '+t('unavailable','indisponible')+' : '+(item.reason||t('noModule','module non disponible sur ce serveur'))).join(' · ');if(badge&&displayedModes.length)badge.textContent=displayedModes.map(mode=>modeLabel(mode)).join(' / ')+' · '+t('experimental','expérimental');};
    selector.onchange=()=>set({...state,mode:selector.value,before:false},true);slider.oninput=()=>set({...state,intensity:slider.valueAsNumber},true);button.onclick=compare;cycleButton.onclick=cycle;
    document.addEventListener('keydown',e=>{if(available.length&&e.key.toLowerCase()==='b'&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&!e.target.isContentEditable&&!/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)){if(e.shiftKey&&available.length>1){e.preventDefault();cycle();}else if(available.includes(state.mode)){e.preventDefault();compare();}}});window.addEventListener('neural-locale-change',()=>sync());sync();
  }
  return {normalize,get:()=>({...state}),set,effective,effectiveMode,compare,cycle,writeParams,mount,label:mode=>mode in labels?modeLabel(mode):t('before','Avant'),
    setSupport(value){const wasm=window.NeuralEarthRuntime?.kind==='wasm';available=wasm?[]:Object.keys(labels).filter(mode=>value?.modes?value.modes[mode]?.available:mode==='mse'&&(value===true||value?.available));unavailable=wasm||!value?.modes?[]:Object.keys(labels).filter(mode=>!value.modes[mode]?.available).map(mode=>({mode,reason:value.modes[mode]?.reason}));if(!available.length&&badge)badge.hidden=true;sync();},
    displayed(modes){if(!badge)return;displayedModes=Array.isArray(modes)?modes:typeof modes==='string'?[modes]:[];badge.hidden=!displayedModes.length;badge.textContent=displayedModes.map(mode=>modeLabel(mode)).join(' / ')+' · '+t('experimental','expérimental');}};
})();
