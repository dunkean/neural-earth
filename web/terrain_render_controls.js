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
