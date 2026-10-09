/* Lighting preferences are display-only and can be overridden for each LOD. */
window.TerrainLighting=(()=>{
  const defaults={strength:1,ambient:.35,contrast:1,exaggeration:6,azimuth:315,altitude:45};
  const limits={strength:[0,1,.01],ambient:[0,1,.01],contrast:[.1,3,.05],exaggeration:[0,20,.1],azimuth:[0,360,1],altitude:[5,90,1]};
  let settings={global:{...defaults},lods:{}};
  function normalize(value){const result={...defaults};for(const [key,[min,max]]of Object.entries(limits)){const v=value?.[key];if(Number.isFinite(v))result[key]=Math.max(min,Math.min(max,v));}return result;}
  try{const saved=JSON.parse(new URLSearchParams(location.search).get('lighting')||'null');if(saved){settings.global=normalize(saved.global);for(const [lod,value]of Object.entries(saved.lods||{}))if(Number.isInteger(Number(lod))&&lod>=-3&&lod<=11)settings.lods[lod]=normalize(value);}}catch(_){}
  function get(lod){return {...(settings.lods[lod]||settings.global)};}
  function vectors(lod){const s=get(lod),az=s.azimuth*Math.PI/180,alt=s.altitude*Math.PI/180;return [s.strength,s.ambient,s.contrast,s.exaggeration/6,Math.sin(az)*Math.cos(alt),-Math.cos(az)*Math.cos(alt),Math.sin(alt),0];}
  function mount(container,onChange){
    const field=document.createElement('fieldset');field.id='lightingSettings';
    field.innerHTML='<legend>Lighting</legend><label>Settings <select id="lightingScope"><option value="global">Global · all LODs</option>'+Array.from({length:15},(_,i)=>`<option value="${i-3}">LOD ${i-3}</option>`).join('')+'</select></label><label id="lightingInheritLabel">Inherit global settings <input id="lightingInherit" type="checkbox"></label><div class="lightingControls"></div><div class="panelActions"><button type="button" id="lightingFlat">No shadows</button><button type="button" id="lightingSoft">Soft</button><button type="button" id="lightingReset">Default</button></div><small>Changes apply immediately. Each LOD can use its own lighting, including 3, 2, 1 and 0. The globe adds spherical lighting to the global layer.</small>';
    container.append(field);
    const scope=field.querySelector('#lightingScope'),inherit=field.querySelector('#lightingInherit'),controls=field.querySelector('.lightingControls'),inputs={};
    const labels={strength:'Shadow strength',ambient:'Ambient light',contrast:'Shadow contrast',exaggeration:'Lighting relief',azimuth:'Sun azimuth (°)',altitude:'Sun altitude (°)'};
    for(const [key,[min,max,step]]of Object.entries(limits)){const label=document.createElement('label');label.textContent=labels[key];const pair=document.createElement('span');pair.className='numericControl';const range=document.createElement('input'),number=document.createElement('input');range.type='range';number.type='number';for(const input of [range,number]){input.min=min;input.max=max;input.step=step;input.setAttribute('aria-label',labels[key]);}pair.append(range,number);label.append(pair);controls.append(label);inputs[key]=[range,number];for(const input of [range,number])input.oninput=()=>{if(input.value===''||!Number.isFinite(input.valueAsNumber))return;const s=get(scope.value);s[key]=Math.max(min,Math.min(max,input.valueAsNumber));if(scope.value==='global')settings.global=s;else settings.lods[scope.value]=s;sync();onChange();};}
    function sync(){const global=scope.value==='global',s=get(scope.value);field.querySelector('#lightingInheritLabel').hidden=global;inherit.checked=!settings.lods[scope.value];for(const [key,pair]of Object.entries(inputs))for(const input of pair){input.value=s[key];input.disabled=!global&&inherit.checked;}}
    scope.onchange=sync;inherit.onchange=()=>{if(inherit.checked)delete settings.lods[scope.value];else settings.lods[scope.value]=get(scope.value);sync();onChange();};
    for(const [id,value]of [['lightingFlat',{...defaults,strength:0}],['lightingSoft',{...defaults,strength:.45,ambient:.65,contrast:.7}],['lightingReset',defaults]])field.querySelector('#'+id).onclick=()=>{if(scope.value==='global')settings.global={...value};else settings.lods[scope.value]={...value};sync();onChange();};
    sync();
  }
  return {get,vectors,mount,serialize:()=>JSON.stringify(settings)};
})();
