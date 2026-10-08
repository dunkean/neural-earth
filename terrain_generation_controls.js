// Draft controls only. Network work happens through the explicit callbacks.
window.TerrainGenerationControls=(()=>{
  const clone=value=>value==null?null:JSON.parse(JSON.stringify(value));
  const channels=['Elevation','Temperature','Temperature variation','Precipitation','Precipitation variation'];
  let schema=null;
  function defaults(profile){
    if(schema?.defaults_by_profile?.[profile])return clone(schema.defaults_by_profile[profile]);
    const natural=profile==='natural';
    return {height_source:natural?'natural':'native',climate_source:natural?'natural':'native',
      continental_style:profile.startsWith('terrestrial-')?profile.slice(12):'earthlike',
      continental_strength:.8,macro_scale_km:600,frequency_mult:[1,1,1,1,1],
      octaves:[4,2,4,4,4],cond_snr:natural?[.5,.5,.5,.5,.5]:[.05,.5,.5,.5,.5],drop_water_pct:.5,
      snr_altitude_gain:[1,1,1,1,1],snr_altitude_range_m:[500,3000],snr_driver_channel:'temperature',
      snr_driver_gain:[1,1,1,1,1],snr_driver_range:[5,-10],snr_bins:16};
  }
  function mount({onApply,onReset,currentSnapshot}){
    const panel=document.createElement('details');panel.id='generationPanel';
    panel.innerHTML=`<summary>Generation settings <span id="generationSummary">Default profile</span></summary>
      <div class="generationBody"><div class="generationSources">
      <label>Elevation <select id="heightSource"><option value="natural">Natural</option><option value="native">Earth atlas</option><option value="natural-continental">Natural continental · experimental</option></select></label>
      <label>Climate <select id="climateSource"><option value="natural">Natural</option><option value="native">Terrestrial</option></select></label>
      <label>Continents <select id="continentalStyle"><option value="gondwana">Gondwana</option><option value="continents">Continents</option><option value="earthlike">Terrestrial</option><option value="archipelago">Archipelago</option></select></label>
      <label>Continental strength <input id="continentalStrength" aria-label="Continental strength" type="text" inputmode="decimal" required></label>
      <label>Smoothing radius (σ) <input id="macroScale" aria-label="Smoothing radius" type="text" inputmode="decimal" required> km</label>
      <label title="0: original population; 1: land only; 0.5: historical reference.">Land / sea bias <input id="dropWater" aria-label="Land / sea bias" type="text" inputmode="decimal" required></label></div>
      <small id="generationSourceHelp"></small>
      <table class="generationChannels"><thead><tr><th>Channel</th>${channels.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>
      ${[['Frequency','frequency'],['Octaves','octaves'],['Allowed noise (SNR)','snr'],['Mountain noise multiplier','snrAltitude'],['Climate noise multiplier','snrDriver']].map(([label,key])=>`<tr><th>${label}</th>${channels.map((channel,i)=>`<td><input id="${key}${i}" aria-label="${label} · ${channel}" type="text" inputmode="${key==='octaves'?'numeric':'decimal'}" required></td>`).join('')}</tr>`).join('')}</tbody></table>
      <div class="generationSources"><label>Altitude ramp starts <input id="snrAltitudeLow" aria-label="SNR altitude start" type="text" inputmode="decimal"> m</label>
      <label>Altitude ramp ends <input id="snrAltitudeHigh" aria-label="SNR altitude end" type="text" inputmode="decimal"> m</label>
      <label>Other driver <select id="snrDriverChannel"><option value="temperature">Temperature (°C)</option><option value="temperature_variation">Temperature variation (BIO4, °C × 100)</option><option value="precipitation">Precipitation (mm/year)</option><option value="precipitation_variation">Precipitation variation (%)</option></select></label>
      <label>Driver ramp starts <input id="snrDriverLow" aria-label="SNR driver start" type="text" inputmode="decimal"></label>
      <label>Driver ramp ends <input id="snrDriverHigh" aria-label="SNR driver end" type="text" inputmode="decimal"></label>
      <label>Ramp levels <input id="snrBins" aria-label="SNR ramp levels" type="text" inputmode="numeric"></label></div>
      <small>Regional noise modulation is experimental. Multipliers of 1 disable it. Altitude uses the mean land height; the other driver uses the existing climate inputs. A decreasing temperature ramp permits more noise in cold regions. Larger SNR permits more learned correction; it follows the inputs less closely.</small>
      <small>Lower SNR follows the inputs more closely and gives the network less freedom. Settings take effect when applied. Enter decimals with a dot; scientific notation is supported.</small>
      <div class="generationActions"><label>Preset <select id="generationPreset"><option value="custom">Custom</option><option value="current">Current profile defaults</option><option value="natural">Natural · reference</option><option value="continental">Natural continental · strength 0.8</option><option value="mountains">Mountain detail · experimental</option><option value="glacial">Cold-region detail · experimental</option></select></label>
      <button id="applyGeneration">Apply settings</button><button id="resetGeneration">Reset profile defaults</button>
      <button id="saveGenerationA">Save A</button><button id="switchGenerationAB" disabled>Show A</button><span id="generationDraftStatus" role="status"></span></div></div>`;
    document.getElementById('status').before(panel);
    const $=id=>document.getElementById(id);let base='terrestrial-earthlike',draftBase=base,savedA=null,savedB=null,showingA=false;
    function activeControls(){
      const height=$('heightSource').value,climate=$('climateSource').value,native=height==='native',continental=height==='natural-continental';
      for(const id of ['frequency0','octaves0','dropWater'])$(id).disabled=native;
      for(const id of ['continentalStrength','macroScale'])$(id).disabled=!continental;
      $('continentalStyle').disabled=height==='natural'&&climate==='natural';
      $('generationSourceHelp').textContent=native?'The atlas fixes continents and elevation. Elevation frequency, octaves, and land / sea bias are available for natural sources.':continental?'Continental strength and smoothing radius shape this source. Climate frequencies and octaves remain adjustable.':'Continental strength and smoothing apply to Natural continental. The continent style also controls terrestrial climate.';
    }
    function write(settings){
      settings={...defaults(base),...settings};
      $('heightSource').value=settings.height_source;$('climateSource').value=settings.climate_source;$('continentalStyle').value=settings.continental_style;
      $('continentalStrength').value=settings.continental_strength;$('macroScale').value=settings.macro_scale_km;$('dropWater').value=settings.drop_water_pct;
      for(let i=0;i<5;i++){ $('frequency'+i).value=settings.frequency_mult[i];$('octaves'+i).value=settings.octaves[i];$('snr'+i).value=settings.cond_snr[i];$('snrAltitude'+i).value=settings.snr_altitude_gain[i];$('snrDriver'+i).value=settings.snr_driver_gain[i]; }
      $('snrAltitudeLow').value=settings.snr_altitude_range_m[0];$('snrAltitudeHigh').value=settings.snr_altitude_range_m[1];
      $('snrDriverChannel').value=settings.snr_driver_channel;$('snrDriverLow').value=settings.snr_driver_range[0];$('snrDriverHigh').value=settings.snr_driver_range[1];$('snrBins').value=settings.snr_bins;activeControls();
    }
    function number(id,{minimum,maximum,positive=false,integer=false}={}){
      const input=$(id),text=input.value.trim(),value=Number(text),label=input.getAttribute('aria-label')||id;
      if(!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(text)||!Number.isFinite(value))throw Error(`${label}: enter a finite number using a decimal dot (e.g. 0.125 or 1e-3).`);
      if(integer&&!Number.isInteger(value))throw Error(`${label}: enter an integer.`);
      if(positive&&value<=0)throw Error(`${label}: enter a value greater than 0.`);
      if(minimum!==undefined&&value<minimum)throw Error(`${label}: enter a value of at least ${minimum}.`);
      if(maximum!==undefined&&value>maximum)throw Error(`${label}: enter a value of at most ${maximum}.`);
      return value;
    }
    function read(){
      return {height_source:$('heightSource').value,climate_source:$('climateSource').value,continental_style:$('continentalStyle').value,
        continental_strength:number('continentalStrength'),macro_scale_km:number('macroScale',{minimum:0}),drop_water_pct:number('dropWater',{minimum:0,maximum:1}),
        frequency_mult:Array.from({length:5},(_,i)=>number('frequency'+i)),octaves:Array.from({length:5},(_,i)=>number('octaves'+i,{minimum:1,integer:true})),cond_snr:Array.from({length:5},(_,i)=>number('snr'+i,{positive:true})),
        snr_altitude_gain:Array.from({length:5},(_,i)=>number('snrAltitude'+i,{minimum:.03125,maximum:32})),
        snr_altitude_range_m:[number('snrAltitudeLow'),number('snrAltitudeHigh')],snr_driver_channel:$('snrDriverChannel').value,
        snr_driver_gain:Array.from({length:5},(_,i)=>number('snrDriver'+i,{minimum:.03125,maximum:32})),
        snr_driver_range:[number('snrDriverLow'),number('snrDriverHigh')],snr_bins:number('snrBins',{minimum:2,maximum:32,integer:true})};
    }
    panel.addEventListener('input',event=>{if(event.target.id==='generationPreset')return;activeControls();$('generationDraftStatus').textContent='Unapplied changes';$('generationPreset').value='custom'});
    $('generationPreset').onchange=()=>{
      const preset=$('generationPreset').value;draftBase=preset==='natural'||preset==='continental'?'natural':base;
      const settings=defaults(draftBase);if(preset==='continental'){settings.height_source='natural-continental';settings.continental_strength=.8;settings.cond_snr=[.5,.5,.5,.5,.5];}
      if(preset==='mountains')settings.snr_altitude_gain=[4,1,1,1,1];
      if(preset==='glacial')settings.snr_driver_gain=[3,1,1,1,1];
      if(preset!=='custom')write(settings);$('generationDraftStatus').textContent='Unapplied preset';
    };
    async function submit(settings,profile){
      $('generationDraftStatus').textContent='Applying…';const success=await onApply(settings,profile);
      if(success)$('generationDraftStatus').textContent='Settings applied';
      else if($('generationDraftStatus').textContent==='Applying…')$('generationDraftStatus').textContent='Settings were not applied';
      return !!success;
    }
    async function action(callback){
      for(const id of ['applyGeneration','resetGeneration','saveGenerationA','switchGenerationAB'])$(id).disabled=true;
      try{return await callback()}catch(error){$('generationDraftStatus').textContent=error.message;return false}
      finally{for(const id of ['applyGeneration','resetGeneration','saveGenerationA'])$(id).disabled=false;$('switchGenerationAB').disabled=!savedA;}
    }
    $('applyGeneration').onclick=()=>action(()=>submit(read(),draftBase));
    $('resetGeneration').onclick=()=>action(async()=>{const success=await onReset();if(success)$('generationDraftStatus').textContent='Profile defaults restored';return success});
    $('saveGenerationA').onclick=()=>{savedA=clone(currentSnapshot());savedB=null;showingA=false;$('switchGenerationAB').disabled=false;$('switchGenerationAB').textContent='Show A';$('generationDraftStatus').textContent='Settings A saved'};
    $('switchGenerationAB').onclick=()=>action(async()=>{if(!savedA)return false;const candidateB=clone(currentSnapshot()),target=showingA?savedB:savedA;
      if(!await submit(target.settings,target.profile))return false;
      if(!showingA)savedB=candidateB;showingA=!showingA;$('switchGenerationAB').textContent=showingA?'Show B':'Show A';return true;});
    return {sync(profile,settings,metadata,custom=false){if(metadata)schema=metadata;base=draftBase=profile;write(settings||defaults(profile));$('generationPreset').value=custom?'custom':'current';$('generationSummary').textContent=custom?'Custom settings':'Default profile';$('generationDraftStatus').textContent='';},read,defaults,showError(message){panel.open=true;$('generationDraftStatus').textContent=message}};
  }
  return {mount,defaults};
})();
