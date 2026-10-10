// Draft controls only. Network work happens through the explicit callbacks.
window.TerrainGenerationControls=(()=>{
  const clone=value=>value==null?null:JSON.parse(JSON.stringify(value));
  const channels=['Elevation','Temperature','Temperature variation','Precipitation','Precipitation variation'];
  const cityFields=[['city_erosion_strength','Dose',0,2,.01,1],['city_erosion_iterations','Iterations',1,64,1,12],['city_erosion_talus','Talus slope',.01,4,.01,.6],['city_erosion_motif_km','Drainage scale (km)',10,5000,10,300]];
  const gpuFields=[['orogen_gpu_relief','Relief · plates, collisions, noise'],['orogen_gpu_propagation','Propagation · stress and distances'],['orogen_gpu_post','Orogen postprocessing · local passes'],['orogen_gpu_climate','Climate · winds, currents, rain, temperature'],['orogen_gpu_raster','Projection · GPU spatial search']];
  const tectonicFields=[["orogen_plate_count", "Plate count", 4, 120, true], ["orogen_continent_count", "Continent count", 1, 10, true], ["orogen_land_coverage", "Land coverage", 0.05, 0.9, false], ["orogen_continent_variety", "Continent variety", 0.0, 1.0, false], ["orogen_motion_strength", "Motion strength", 0.0, 4.0, false], ["orogen_convergence_threshold", "Convergence threshold", 0.05, 4.0, false], ["orogen_detail", "Mesh vertices", 20000, 1000000, true], ["orogen_spread", "Orogen propagation", 1.0, 12.0, false], ["orogen_roughness", "Roughness", 0.0, 1.0, false], ["orogen_warp", "Terrain warp", 0.0, 1.0, false], ["orogen_smoothing", "Smoothing", 0.0, 1.0, false], ["orogen_hydraulic", "Hydraulic erosion", 0.0, 1.0, false], ["orogen_thermal", "Thermal erosion", 0.0, 1.0, false], ["orogen_glacial", "Glacial erosion", 0.0, 1.0, false], ["orogen_sharpening", "Ridge sharpening", 0.0, 1.0, false], ["orogen_temperature_offset", "Temperature offset (C)", -15.0, 15.0, false], ["orogen_precipitation_offset", "Precipitation offset", -1.0, 1.0, false]];
  let schema=null;
  const preferencesKey='neural-earth-preferences-v1';
  function localPreferences(){
    try{return JSON.parse(localStorage.getItem(preferencesKey))||{}}catch(_){return {}}
  }
  function savePreferences(change){
    try{localStorage.setItem(preferencesKey,JSON.stringify({...localPreferences(),...change}))}catch(_){}
  }
  function saveGeneration(profile,settings){
    // Stage references belong to a particular generated world, rather than local defaults.
    const local=Object.fromEntries(Object.entries(settings).filter(([key])=>!key.endsWith('_stage')));
    savePreferences({generation:{profile,settings:local}});
  }
  function withGpu(settings,enabled){
    return {...settings,...Object.fromEntries([...gpuFields.map(([key])=>key),'orogen_gpu_erosion'].map(key=>[key,enabled]))};
  }
  async function initializePreferences(){
    let preferences=localPreferences();
    if(typeof preferences.gpuAcceleration==='boolean')return {preferences,firstLaunch:false};
    const dialog=document.createElement('dialog');dialog.id='gpuWelcome';
    dialog.innerHTML='<form method="dialog"><h2>Enable GPU acceleration?</h2><p>Enable all GPU options for Orogen relief, propagation, erosion, climate and projection, plus map rendering?</p><p>Orogen acceleration uses NVIDIA CUDA. Map rendering uses WebGPU. You can adjust each option later in Settings and Rendering. Your choices are saved on this browser.</p><div class="panelActions"><button value="no" autofocus>Use CPU options</button><button value="yes">Enable GPU acceleration</button></div></form>';
    document.body.append(dialog);
    const enabled=await new Promise(resolve=>{dialog.addEventListener('close',()=>resolve(dialog.returnValue==='yes'),{once:true});dialog.showModal()});
    dialog.remove();
    preferences={...preferences,gpuAcceleration:enabled,rendering:{gpuRender:enabled,coarseGpu:enabled}};
    savePreferences(preferences);
    return {preferences,firstLaunch:true};
  }
  function classify(settings){
    return settings.height_source==='orogen'?'orogen':settings.height_source==='natural'?'natural':'custom';
  }
  function defaults(profile){
    const propertyDefaults=Object.fromEntries(Object.entries(schema?.properties||{}).filter(([,spec])=>spec.default!==undefined).map(([key,spec])=>[key,clone(spec.default)]));
    if(schema?.defaults_by_profile?.[profile])return withGpu({...propertyDefaults,...clone(schema.defaults_by_profile[profile]),climate_source:'orogen'},localPreferences().gpuAcceleration===true);
    const natural=profile==='natural';
    return {...propertyDefaults,world_diameter_km:40000/Math.PI,world_topology:"sphere",...withGpu({},localPreferences().gpuAcceleration===true),...Object.fromEntries(cityFields.map(([key,label,min,max,step,value])=>[key,value])),height_source:natural?'natural':profile==='orogen'?'orogen':'native',climate_source:'orogen',relief_pipeline:profile==='orogen'?'orogen':'original',
      continental_style:profile.startsWith('terrestrial-')?profile.slice(12):profile==='orogen'?'continents':'earthlike',
      continental_strength:.8,macro_scale_km:600,frequency_mult:[1,1,1,1,1],
      octaves:[4,2,4,4,4],cond_snr:natural?[.5,.5,.5,.5,.5]:[.05,.5,.5,.5,.5],drop_water_pct:.5,
      snr_adaptive_enabled:true,snr_detail_mode:'global',snr_altitude_gain:[1,1,1,1,1],snr_altitude_range_m:[500,3000],snr_driver_channel:'temperature',
      snr_driver_gain:[1,1,1,1,1],snr_driver_range:[5,-10],snr_bins:16,snr_latitude_gain:1,snr_latitude_range:[0,75],snr_lod:Array(15).fill(0),orogen_plate_count:80,orogen_continent_count:profile==='orogen'?3:4,orogen_land_coverage:.3,orogen_continent_variety:.85,orogen_motion_strength:1,orogen_convergence_threshold:.75,orogen_roughness:.4,orogen_detail:100000,orogen_spread:5.0,orogen_warp:0.75,orogen_smoothing:0.1,orogen_hydraulic:0.5,orogen_thermal:0.1,orogen_glacial:0.5,orogen_sharpening:0.5,orogen_temperature_offset:0.0,orogen_precipitation_offset:0.0};
  }
  function mount({onApply,onReset,currentSnapshot,onGeneratorChange}){
    const panel=document.createElement('details');panel.id='generationPanel';
    panel.innerHTML=`<summary>Generation settings <span id="generationSummary">Default profile</span></summary>
      <div class="generationBody"><div class="generationSources">
      <label>Elevation <select id="heightSource"><option value="natural">Natural</option><option value="native">Earth atlas</option><option value="orogen">Orogen</option><option value="natural-continental">Natural continental · experimental</option></select></label>
      <label>Relief processing <select id="reliefPipeline"><option value="original">None</option><option value="orogen-gpu">Orogen · GPU</option><option value="orogen">Orogen · CPU</option><option value="city-gpu">City · GPU</option></select></label>
      <label>Climate <select id="climateSource"><option value="natural">Natural</option><option value="native">Terrestrial</option><option value="orogen">Orogen</option></select></label>
      <label>Continents <select id="continentalStyle"><option value="gondwana">Gondwana</option><option value="continents">3 continents</option><option value="earthlike">Terrestrial</option><option value="archipelago">Archipelago</option></select></label>
      <label>Continental strength <input id="continentalStrength" aria-label="Continental strength" type="text" inputmode="decimal" required></label>
      <label>Smoothing radius (km) <input id="macroScale" aria-label="Smoothing radius" type="text" inputmode="decimal" required></label>
      <label title="0: original population; 1: land only; 0.5: historical reference.">Land / sea bias <input id="dropWater" aria-label="Land / sea bias" type="text" inputmode="decimal" required></label></div>
      <small id="generationSourceHelp"></small>
      <fieldset id="orogenParameters" hidden><legend>Orogen · plates</legend><div class="generationSources">${tectonicFields.map(([key,label])=>`<label>${label} <input id="${key}" aria-label="${label}" type="text" inputmode="decimal" required></label>`).join('')}</div><small>Original Orogen relief, superplates and seasonal climate. Plate settings apply to the Orogen initial source; relief processing and climate work with every source. Mesh vertices control detail and initial generation time. Save A / Show A compares configurations.</small></fieldset>
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
      <button id="applyGeneration">Apply</button><button id="resetGeneration">Reset defaults</button>
      <button id="saveGenerationA">Save A</button><button id="switchGenerationAB" disabled>Show A</button><span id="generationDraftStatus" role="status"></span></div></div>`;
    document.getElementById('runtimePanel').before(panel);
    panel.querySelector('summary').firstChild.textContent='Settings ';
    const body=panel.querySelector('.generationBody');
    const heading=document.createElement('div');heading.className='panelHeading';
    heading.innerHTML='<strong>Generation settings</strong><button type="button" class="panelClose" aria-label="Close panel">×</button>';
    body.prepend(heading);
    const orogen=body.querySelector('#orogenParameters'),tectonics=orogen.querySelector('.generationSources');
    orogen.querySelector('legend').textContent='Orogen · source parameters';
    for(const [title,fields,open] of [['Orogen plates',tectonicFields.slice(0,8),false],['Initial relief · shape',tectonicFields.slice(8,10),false],['Erosion Orogen',tectonicFields.slice(10,15),false],['Climate offsets',tectonicFields.slice(15),false]]){
      const section=document.createElement('details');section.className='advancedSettings';section.open=open;
      const summary=document.createElement('summary');summary.textContent=title;section.append(summary);
      const controls=document.createElement('div');controls.className='generationSources';
      for(const [id] of fields)controls.append(document.getElementById(id).closest('label'));
      section.append(controls);tectonics.before(section);
    }
    tectonics.remove();
    orogen.querySelector('small').textContent='Apply regenerates the world; Save A compares variants.';
    // Keep the original input IDs and draft/apply contract while grouping the form.
    function group(title,nodes){const field=document.createElement('fieldset'),legend=document.createElement('legend');legend.textContent=title;field.append(legend);nodes[0].before(field);field.append(...nodes);return field;}
    const sourceOverrides=group('Source overrides',[body.querySelector('.generationSources')]);
    const erosionSection=document.createElement('fieldset');erosionSection.id='erosionSettings';
    erosionSection.innerHTML='<legend>Erosion</legend><div class="generationSources"></div><small>Erosion runs before climate and neural refinement. Choose None, Orogen CPU or GPU, or City GPU.</small>';
    orogen.before(erosionSection);
    erosionSection.querySelector('.generationSources').append(document.getElementById('reliefPipeline').closest('label'));
    const cityControls=document.createElement('details');cityControls.id='cityErosionParameters';cityControls.className='advancedSettings';cityControls.open=true;
    cityControls.innerHTML='<summary>City · GPU settings</summary><div class="generationSources">'+cityFields.map(([key,label])=>`<label>${label} <input id="${key}" aria-label="City erosion · ${label}" type="text" inputmode="decimal"></label>`).join('')+'</div><small>Hydraulic incision, thermal relaxation and diffusion. Strength 0 preserves relief; the engine preserves sea level and oceans.</small>';
    erosionSection.append(cityControls);
    const gpuControls=document.createElement('details');gpuControls.id='orogenGpuParameters';gpuControls.className='advancedSettings';gpuControls.open=true;
    gpuControls.innerHTML='<summary>Orogen · optional GPU acceleration</summary><div class="generationSources">'+gpuFields.map(([key,label])=>`<label><input id="${key}" type="checkbox" aria-label="GPU · ${label}"> ${label}</label>`).join('')+'</div><small>GPU options follow your saved local settings. NVIDIA CUDA; unported stages run on CPU. GPU propagation and erosion change terrain shape. City is a separate erosion engine. Apply, then Save A / Show A to compare.</small>';
    erosionSection.after(gpuControls);
    const sourceSection=document.createElement('fieldset');sourceSection.id='sourceSettings';
    sourceSection.innerHTML='<legend>1 · World source</legend><div class="generationSources"><label>Generator <select id="generatorType"><option value="natural">noise</option><option value="custom">Custom</option><option value="orogen">Orogen</option><option value="mixed" disabled>Mixed sources · advanced</option></select></label><label id="customSourceLabel">Custom relief <select id="customSource"><option value="native">Continental atlas</option><option value="natural-continental">Procedural continents</option></select></label></div>';
    sourceOverrides.before(sourceSection);
    const geometry=document.createElement('fieldset');geometry.id='worldGeometry';
    geometry.innerHTML='<legend>World creation</legend><div class="generationSources"><label>Geometry <select id="worldTopology"><option value="sphere">Spherical planet</option><option value="plane">Plane · unprojected map</option></select></label><label>Diameter (km) <input id="worldDiameter" type="number" min="10" max="100000" step="any" aria-label="World diameter in km"></label></div><small>Planet: equirectangular map with width pi times diameter. Plane: a square with side equal to diameter, Cartesian distances and finite edges. Globe view is disabled. The selected generator supplies elevations. Applying these settings creates a new world and its caches.</small>';
    sourceSection.prepend(geometry);
    sourceSection.querySelector('.generationSources').append(...['continentalStyle','continentalStrength','macroScale','dropWater'].map(id=>document.getElementById(id).closest('label')));
    sourceSection.append(body.querySelector('#generationSourceHelp'));
    const mixing=document.createElement('details');mixing.className='advancedSettings';mixing.innerHTML='<summary>Advanced · combine different sources</summary><small>Override elevation, erosion and climate independently. This produces a mixed configuration.</small>';
    sourceOverrides.before(mixing);mixing.append(sourceOverrides);
    for(const [id,label] of [['heightSource','Elevation source '],['reliefPipeline','Erosion pipeline '],['climateSource','Climate source ']])document.getElementById(id).closest('label').firstChild.textContent=label;
    for(const select of sourceOverrides.querySelectorAll('select'))for(const option of select.options){if(option.value==='natural')option.textContent='noise · procedural noise';if(option.value==='native')option.textContent=select.id==='climateSource'?'Custom · terrestrial model':'Custom · continental atlas';if(option.value==='natural-continental')option.textContent='Custom · procedural continents';if(option.value==='original')option.textContent='None · preserve source relief';}
    document.getElementById('continentalStyle').closest('label').firstChild.textContent='Continent layout ';
    const table=body.querySelector('.generationChannels'),channelField=document.createElement('fieldset');
    channelField.innerHTML='<legend>2 · NN refinement · all generators</legend><small>The NN adds detail to the source world. SNR is allowed noise: lower values follow the source more closely.</small><label class="channelSelect">Channel <select id="generationChannel"></select></label>';
    const procedural=document.createElement('fieldset');procedural.id='proceduralParameters';procedural.innerHTML='<legend>Source fields · procedural noise</legend><label class="channelSelect">Channel <select id="sourceChannel"></select></label><small>Frequency and octaves build the NN input fields; they do not control NN refinement.</small>';
    sourceSection.append(procedural);
    const sourceChannel=procedural.querySelector('select');
    const channelSelect=channelField.querySelector('select');
    channels.forEach((channel,i)=>{
      channelSelect.add(new Option(channel,String(i)));
      sourceChannel.add(new Option(channel,String(i)));
      const card=document.createElement('div');card.className='generationSources';card.dataset.channel=String(i);card.hidden=i!==0;
      const sourceCard=document.createElement('div');sourceCard.className='generationSources';sourceCard.dataset.sourceChannel=String(i);sourceCard.hidden=i!==0;
      for(const [index,row] of [...table.tBodies[0].rows].entries()){const label=document.createElement('label');label.append(document.createTextNode(row.cells[0].textContent),row.cells[i+1].querySelector('input'));(index<2?sourceCard:card).append(label);}
      procedural.append(sourceCard);
      channelField.append(card);
    });
    table.replaceWith(channelField);
    channelSelect.addEventListener('change',()=>{for(const card of channelField.querySelectorAll('[data-channel]'))card.hidden=card.dataset.channel!==channelSelect.value;});
    function selectSourceChannel(){for(const card of procedural.querySelectorAll('[data-source-channel]'))card.hidden=card.dataset.sourceChannel!==sourceChannel.value;}
    sourceChannel.addEventListener('change',selectSourceChannel);
    const advanced=document.createElement('details');advanced.className='advancedSettings';advanced.innerHTML='<summary>Regional noise modulation</summary>';
    const ramps=channelField.nextElementSibling,help=ramps.nextElementSibling;advanced.append(ramps,help);channelField.append(advanced);
    // Three independent panels share one generation draft and apply contract.
    function settingsPanel(id,title,icon){
      const details=document.createElement('details');details.id=id;
      details.innerHTML=`<summary title="${title}" aria-label="${title}">${icon}</summary><div class="generationBody"><div class="panelHeading"><strong>${title}</strong><button type="button" class="panelClose" aria-label="Close ${title}">×</button></div></div>`;
      panel.after(details);return details;
    }
    const climatePanel=settingsPanel('climatePanel','Climate','☀');
    const noisePanel=settingsPanel('noisePanel','Relief noise · SNR','≋');
    const climateBody=climatePanel.querySelector('.generationBody'),noiseBody=noisePanel.querySelector('.generationBody');
    panel.querySelector('summary').innerHTML='⚒';panel.querySelector('summary').title='Generation';panel.querySelector('summary').setAttribute('aria-label','Generation');
    // Keep the summary status for consumers, outside the icon title.
    const summaryStatus=document.createElement('span');summaryStatus.id='generationSummary';summaryStatus.hidden=true;panel.querySelector('summary').append(summaryStatus);
    heading.querySelector('strong').textContent='Generation · initial relief';
    mixing.hidden=true;sourceOverrides.hidden=true;document.getElementById('generatorType').querySelector('[value=mixed]').remove();
    const climateSource=document.getElementById('climateSource');climateSource.innerHTML='<option value="orogen">Orogen</option>';
    const erosionToggle=document.getElementById('reliefPipeline').closest('label');sourceSection.append(erosionToggle);
    erosionToggle.firstChild.textContent='Erosion Orogen ';
    document.getElementById('reliefPipeline').options[0].textContent='None';
    const layoutLabel=document.getElementById('continentalStyle').closest('label');layoutLabel.firstChild.textContent='Continents · preset ';
    const layoutHelp=document.createElement('small');layoutHelp.textContent='Presets initialize plates (Orogen), noise (Noise), or the atlas (Custom). Custom retains the choice of atlas or procedural continents.';layoutLabel.after(layoutHelp);
    orogen.querySelector('legend').textContent='Relief · options';sourceSection.append(orogen);
    const offsets=[document.getElementById('orogen_temperature_offset').closest('details')];climateBody.append(...offsets);
    const intro=document.createElement('small');intro.textContent='Orogen supplies the shared climate for Noise, Orogen and Custom: seasons, winds, rain, classification and biomes.';climateBody.querySelector('.panelHeading').after(intro);
    for(const id of ['snrAltitudeLow','snrAltitudeHigh']){const label=document.getElementById(id).closest('label');label.firstChild.textContent+=' (m) ';label.lastChild.remove();}
    noiseBody.append(channelField);for(const note of [...body.querySelectorAll(':scope > small')])noiseBody.append(note);channelField.querySelector('legend').textContent='Relief noise · SNR';
    channelField.querySelector('small').textContent='SNR here means noise / signal amplitude. Lower values constrain relief more closely to its inputs. Climate remains shared.';
    channelField.querySelector('.channelSelect').hidden=true;
    for(const card of channelField.querySelectorAll('[data-channel]'))card.hidden=card.dataset.channel!=='0';
    advanced.open=true;advanced.querySelector('summary').textContent='Adaptive SNR · relief / temperature / latitude';
    const latitudeControls=document.createElement('div');latitudeControls.className='generationSources';
    latitudeControls.innerHTML='<label>Latitude multiplier <input id="snrLatitudeGain" aria-label="SNR latitude multiplier" value="1"></label><label>Latitude ramp starts (°) <input id="snrLatitudeLow" aria-label="SNR latitude start" value="0"></label><label>Latitude ramp ends (°) <input id="snrLatitudeHigh" aria-label="SNR latitude end" value="75"></label>';
    advanced.append(latitudeControls);
    const lodSettings=document.createElement('details');lodSettings.className='advancedSettings';
    lodSettings.innerHTML='<summary>Per-LOD SNR · relief detail</summary><small>0 inherits global SNR. Per-LOD values scale reconstructed relief noise (the local residual after the NN). They do not change climate channels or shared network computation. Experimental settings.</small><div class="generationSources">'+Array.from({length:15},(_,i)=>`<label>LOD ${i-3} · ${30*2**(i-3)} m <input id="snrLod${i}" aria-label="Relief SNR LOD ${i-3}" value="0"></label>`).join('')+'</div>';
    noiseBody.append(lodSettings);
    for(const details of [climatePanel,noisePanel]){const actions=document.createElement('div');actions.className='generationActions';actions.innerHTML='<button type="button" data-shared-action="applyGeneration">Apply</button><button type="button" data-shared-action="resetGeneration">Reset defaults</button><span class="sharedDraftStatus" role="status"></span>';details.querySelector('.generationBody').append(actions);for(const button of actions.querySelectorAll('button'))button.onclick=()=>document.getElementById(button.dataset.sharedAction).click();}
    new MutationObserver(()=>{for(const details of [climatePanel,noisePanel]){details.querySelector('.sharedDraftStatus').textContent=document.getElementById('generationDraftStatus').textContent;for(const button of details.querySelectorAll('[data-shared-action]'))button.disabled=document.getElementById(button.dataset.sharedAction).disabled;}}).observe(document.getElementById('generationDraftStatus').parentElement,{subtree:true,childList:true,attributes:true});
    const noisePreset=document.getElementById('generationPreset');noisePreset.closest('label').firstChild.textContent='Preset SNR ';for(const option of [...noisePreset.options])if(['natural','continental'].includes(option.value))option.remove();noiseBody.querySelector('.generationActions').prepend(noisePreset.closest('label'));
    const numericControls=[];
    const extraFields=[];
    const groupNames={wind:'Atmosphere & winds',ocean:'Ocean currents',temperature:'Temperature',precipitation:'Rain & moisture','heuristic-precip':'Zonal rain model',koppen:'Köppen classification','biome-altitude':'Alpine & snow lines','biome-rendering':'Biome appearance','biome-palette':'Biome colors'};
    function installClimateControls(){
      if(extraFields.length||!schema?.properties)return;
      const properties=Object.entries(schema.properties).filter(([id,spec])=>id.startsWith('orogen_')&&spec.group);
      if(!properties.length)return;
      const section=document.createElement('details');section.className='advancedSettings';
      section.open=true;section.innerHTML='<summary>Climate parameters</summary><label class="climateSearch">Find <input id="orogenClimateSearch" type="search" placeholder="Rain, snow, pressure…" aria-label="Find Orogen settings"></label><label class="channelSelect">Section <select id="orogenClimateSection"></select></label><div class="climateFields"></div><small>Source climate and Original Orogen biomes. Changes take effect with Apply.</small>';
      climateBody.querySelector('.generationActions').before(section);
      const select=section.querySelector('select'),container=section.querySelector('.climateFields'),groups=new Map();
      for(const [id,spec] of properties){
        if(!groups.has(spec.group)){const card=document.createElement('div');card.className='generationSources';card.dataset.group=spec.group;card.hidden=groups.size>0;groups.set(spec.group,card);container.append(card);select.add(new Option(groupNames[spec.group]||spec.group,spec.group));}
        const label=document.createElement('label');label.append(document.createTextNode(spec.label));
        const input=document.createElement('input');input.id=id;input.setAttribute('aria-label',spec.label);
        if(spec.color){
          input.type='color';input.value='#000000';input.dataset.rgb=JSON.stringify(spec.default);
          input.addEventListener('input',()=>{input.dataset.rgb=JSON.stringify([1,3,5].map(i=>parseInt(input.value.slice(i,i+2),16)/255));});
          label.append(input);
        }else{input.type='number';label.append(input);groups.get(spec.group).append(label);slider(id,spec.minimum,spec.maximum,spec.step||.01);}
        groups.get(spec.group).append(label);extraFields.push([id,spec]);
      }
      function filter(){const query=$('orogenClimateSearch').value.trim().toLowerCase();select.disabled=!!query;for(const [key,card] of groups){let matches=0;for(const label of card.querySelectorAll('label')){label.hidden=!!query&&!label.textContent.toLowerCase().includes(query);if(!label.hidden)matches++;}card.hidden=query?matches===0:key!==select.value;}}
      select.addEventListener('change',filter);$('orogenClimateSearch').addEventListener('input',filter);
    }
    function slider(id,min,max,step){
      const input=document.getElementById(id),pair=document.createElement('span');pair.className='numericControl';
      const range=document.createElement('input');range.type='range';range.min=min;range.max=max;range.step=step;range.id=id+'Slider';range.setAttribute('aria-label',input.getAttribute('aria-label')+' slider');
      input.type='text';input.inputMode=step===1||step===1000?'numeric':'decimal';input.before(pair);pair.append(range,input);
      range.addEventListener('input',()=>{input.value=range.value;input.dispatchEvent(new Event('input',{bubbles:true}));});
      numericControls.push({input,range});
    }
    for(const [id,label,min,max,integer] of tectonicFields)slider(id,min,max,id==='orogen_detail'?1000:integer?1:.01);
    for(const [id,label,min,max,step] of cityFields)slider(id,min,max,step);
    slider('snrLatitudeGain',.03125,8,.03125);
    for(let i=0;i<15;i++)slider('snrLod'+i,0,2,.01);
    slider('continentalStrength',0,2,.01);slider('macroScale',0,2000,10);slider('dropWater',0,1,.01);slider('snrBins',2,32,1);
    for(let i=0;i<5;i++){slider('frequency'+i,0,4,.01);slider('octaves'+i,1,12,1);slider('snr'+i,.01,2,.01);slider('snrAltitude'+i,.03125,8,.03125);slider('snrDriver'+i,.03125,8,.03125);}
    for(const id of ['snrAltitudeLow','snrAltitudeHigh','snrDriverLow','snrDriverHigh','snrLatitudeLow','snrLatitudeHigh']){const input=document.getElementById(id);input.type='number';input.step='any';}
    function syncSliders(){for(const {input,range} of numericControls){range.value=input.value;range.disabled=input.disabled;}}
    const $=id=>document.getElementById(id);let base='orogen',draftBase=base,savedA=null,savedB=null,showingA=false,stageReferences={};
    function activeControls(){
      const height=$('heightSource').value,native=height==='native'||height==='orogen',continental=height==='natural-continental';
      $('climateSource').value='orogen';$('orogenParameters').hidden=false;
      $('cityErosionParameters').hidden=$('reliefPipeline').value!=='city-gpu';
      const orogenErosion=['orogen','orogen-gpu'].includes($('reliefPipeline').value);
      $('orogenGpuParameters').hidden=height!=='orogen';
      for(const [id] of cityFields)$(id).disabled=$('reliefPipeline').value!=='city-gpu';
      const type=classify({height_source:height});$('generatorType').value=type;onGeneratorChange?.(type);
      $('customSourceLabel').hidden=type!=='custom';$('customSource').value=height;
      $('continentalStyle').disabled=false;layoutLabel.hidden=false;
      for(const id of ['continentalStrength','macroScale']){$(id).disabled=!continental;$(id).closest('label').hidden=!continental;}
      $('dropWater').disabled=native;$('dropWater').closest('label').hidden=native;
      for(let i=0;i<5;i++)for(const key of ['frequency','octaves']){const input=$(key+i);input.disabled=i===0?native:true;input.closest('label').hidden=input.disabled;}
      procedural.hidden=native;sourceChannel.value='0';procedural.querySelector('.channelSelect').hidden=true;selectSourceChannel();
      for(const section of orogen.querySelectorAll(':scope > details')){section.hidden=height!=='orogen';for(const input of section.querySelectorAll('input'))input.disabled=height!=='orogen';}
      for(const id of ['orogen_warp','orogen_smoothing','orogen_hydraulic','orogen_thermal','orogen_glacial','orogen_sharpening','orogen_gpu_post']){$(id).disabled=!orogenErosion;$(id).closest('label').hidden=!orogenErosion;}
      for(const [id] of extraFields)$(id).disabled=false;
      $('generationSourceHelp').textContent=type==='orogen'?'Orogen plates → initial relief.':type==='custom'?(continental?'Procedural continents → initial relief.':'Continental atlas → initial relief.'):'Procedural noise → initial neural inputs.';
      $('generationSourceHelp').textContent+=' Then optional erosion and shared Orogen climate.';

      const adaptive=$('snrAdaptive').checked;advanced.hidden=!adaptive;
      for(const id of ['snrAltitudeLow','snrAltitudeHigh','snrDriverChannel','snrDriverLow','snrDriverHigh','snrBins','snrLatitudeGain','snrLatitudeLow','snrLatitudeHigh',...channels.flatMap((_,i)=>['snrAltitude'+i,'snrDriver'+i])]){$(id).disabled=!adaptive;if(id.startsWith('snrAltitude')&&/^snrAltitude\d$/.test(id)||/^snrDriver\d$/.test(id))$(id).closest('label').hidden=!adaptive;}
      lodSettings.hidden=$('snrDetailMode').value!=='per-lod';
      for(let i=0;i<15;i++)$('snrLod'+i).disabled=$('snrDetailMode').value!=='per-lod';
      syncSliders();
    }
    function setGenerator(type){
      if(type===classify({height_source:$('heightSource').value}))return;
      const common=read();draftBase=type==='natural'?'natural':type==='orogen'?'orogen':'terrestrial-'+$('continentalStyle').value;
      const settings={...defaults(draftBase),...common,height_source:type==='natural'?'natural':type==='orogen'?'orogen':$('customSource').value==='natural-continental'?'natural-continental':'native',climate_source:'orogen'};
      write(settings);saveGeneration(draftBase,settings);$('generationPreset').value='custom';$('generationDraftStatus').textContent='Unapplied generator change';
    }
    $('generatorType').onchange=()=>{setGenerator($('generatorType').value)};
    $('customSource').onchange=()=>{$('heightSource').value=$('customSource').value;activeControls();$('generationDraftStatus').textContent='Unapplied source change';};
    function write(settings){
      settings={...defaults(base),...settings};
      $("worldTopology").value=settings.world_topology;$("worldDiameter").value=settings.world_diameter_km;
      for(const [id,spec] of extraFields){const input=$(id),value=settings[id]??spec.default;if(spec.color){input.dataset.rgb=JSON.stringify(value);input.value='#'+value.map(v=>Math.round(v*255).toString(16).padStart(2,'0')).join('');}else input.value=value;}
      for(const [id] of tectonicFields)$(id).value=settings[id];
      for(const [id,label,min,max,step,value] of cityFields)$(id).value=settings[id]??value;
      for(const [id] of gpuFields)$(id).checked=settings[id]??false;
      $('snrAdaptive').checked=settings.snr_adaptive_enabled??true;
      $('snrDetailMode').value=settings.snr_detail_mode||(settings.snr_lod?.some(v=>v>0)?'per-lod':'global');
      $('reliefPipeline').value=settings.relief_pipeline==='orogen'&&settings.orogen_gpu_erosion?'orogen-gpu':settings.relief_pipeline;$('heightSource').value=settings.height_source;$('climateSource').value='orogen';$('continentalStyle').value=settings.continental_style;
      $('continentalStrength').value=settings.continental_strength;$('macroScale').value=settings.macro_scale_km;$('dropWater').value=settings.drop_water_pct;
      for(let i=0;i<5;i++){ $('frequency'+i).value=settings.frequency_mult[i];$('octaves'+i).value=settings.octaves[i];$('snr'+i).value=settings.cond_snr[i];$('snrAltitude'+i).value=settings.snr_altitude_gain[i];$('snrDriver'+i).value=settings.snr_driver_gain[i]; }
      $('snrAltitudeLow').value=settings.snr_altitude_range_m[0];$('snrAltitudeHigh').value=settings.snr_altitude_range_m[1];
      $('snrDriverChannel').value=settings.snr_driver_channel;$('snrDriverLow').value=settings.snr_driver_range[0];$('snrDriverHigh').value=settings.snr_driver_range[1];$('snrBins').value=settings.snr_bins;$('snrLatitudeGain').value=settings.snr_latitude_gain??1;$('snrLatitudeLow').value=settings.snr_latitude_range?.[0]??0;$('snrLatitudeHigh').value=settings.snr_latitude_range?.[1]??75;for(let i=0;i<15;i++)$('snrLod'+i).value=settings.snr_lod?.[i]??0;activeControls();
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
      return {world_topology:$("worldTopology").value,world_diameter_km:number("worldDiameter",{minimum:10,maximum:100000}),...stageReferences,...Object.fromEntries(gpuFields.map(([id])=>[id,$(id).checked])),orogen_gpu_erosion:$('reliefPipeline').value==='orogen-gpu',...Object.fromEntries(cityFields.map(([id,label,minimum,maximum,step])=>[id,number(id,{minimum,maximum,integer:step===1})])),...Object.fromEntries(extraFields.map(([id,spec])=>[id,spec.color?JSON.parse($(id).dataset.rgb):number(id,spec)])),...Object.fromEntries(tectonicFields.map(([id,label,minimum,maximum,integer])=>[id,number(id,{minimum,maximum,integer})])),relief_pipeline:$('reliefPipeline').value==='orogen-gpu'?'orogen':$('reliefPipeline').value,height_source:$('heightSource').value,climate_source:$('climateSource').value,continental_style:$('continentalStyle').value,
        snr_adaptive_enabled:$('snrAdaptive').checked,snr_detail_mode:$('snrDetailMode').value,continental_strength:number('continentalStrength'),macro_scale_km:number('macroScale',{minimum:0}),drop_water_pct:number('dropWater',{minimum:0,maximum:1}),
        frequency_mult:Array.from({length:5},(_,i)=>number('frequency'+i)),octaves:Array.from({length:5},(_,i)=>number('octaves'+i,{minimum:1,integer:true})),cond_snr:Array.from({length:5},(_,i)=>number('snr'+i,{positive:true})),
        snr_altitude_gain:Array.from({length:5},(_,i)=>number('snrAltitude'+i,{minimum:.03125,maximum:32})),
        snr_altitude_range_m:[number('snrAltitudeLow'),number('snrAltitudeHigh')],snr_driver_channel:$('snrDriverChannel').value,
        snr_driver_gain:Array.from({length:5},(_,i)=>number('snrDriver'+i,{minimum:.03125,maximum:32})),
        snr_driver_range:[number('snrDriverLow'),number('snrDriverHigh')],snr_bins:number('snrBins',{minimum:2,maximum:32,integer:true}),snr_latitude_gain:number('snrLatitudeGain',{minimum:.03125,maximum:32}),snr_latitude_range:[number('snrLatitudeLow',{minimum:0,maximum:90}),number('snrLatitudeHigh',{minimum:0,maximum:90})],snr_lod:Array.from({length:15},(_,i)=>number('snrLod'+i,{minimum:0}))};
    }
    for(const draftPanel of [panel,climatePanel,noisePanel])draftPanel.addEventListener('input',event=>{if(['generatorType','customSource','sourceChannel','generationPreset','generationChannel','orogenClimateSection','orogenClimateSearch'].includes(event.target.id)||event.target.type==='range')return;if(event.target.id==='continentalStyle'){const values={gondwana:[80,1,.3],continents:[80,3,.3],earthlike:[80,4,.3],archipelago:[110,8,.18]}[$('continentalStyle').value];['orogen_plate_count','orogen_continent_count','orogen_land_coverage'].forEach((id,i)=>$(id).value=values[i]);if($('heightSource').value==='natural'){const preset={gondwana:[.5,3,.55],continents:[.8,4,.5],earthlike:[1,4,.5],archipelago:[1.6,5,.35]}[$('continentalStyle').value];['frequency0','octaves0','dropWater'].forEach((id,i)=>$(id).value=preset[i]);}}activeControls();$('generationDraftStatus').textContent='Unapplied changes';$('generationPreset').value='custom'});
    $('generationPreset').onchange=()=>{
      const preset=$('generationPreset').value;if(preset==='custom')return;
      const settings=read(),baseline=defaults(draftBase);
      for(const key of ['cond_snr','snr_adaptive_enabled','snr_detail_mode','snr_altitude_gain','snr_altitude_range_m','snr_driver_channel','snr_driver_gain','snr_driver_range','snr_bins','snr_latitude_gain','snr_latitude_range','snr_lod'])settings[key]=clone(baseline[key]);
      if(preset==='mountains')settings.snr_altitude_gain=[4,1,1,1,1];
      if(preset==='glacial')settings.snr_driver_gain=[3,1,1,1,1];
      write(settings);$('generationDraftStatus').textContent='Unapplied SNR preset';
    };
    let applySequence=0;
    async function submit(settings,profile,stage='all',options={}){
      const sequence=++applySequence;$('generationDraftStatus').textContent='Computing…';const success=await onApply(settings,profile,stage,options);
      if(sequence!==applySequence)return false;
      if(success){
        if(stage!=='all'&&stage!=='restore'){
          const applied=currentSnapshot().settings;
          const changed=Object.fromEntries((schema?.stage_groups?.[stage]||[]).map(key=>[key,applied[key]]));
          write({...settings,...changed,...stageReferences});
        }
        $('generationDraftStatus').textContent=stage==='all'?'Relief, erosion and climate generated.':`${{relief:'Relief generated',erosion:'Erosion generated',climate:'Climate generated',settings:'SNR applied',restore:'Variant loaded'}[stage]}. Other stages are retained.`;
      }
      else if($('generationDraftStatus').textContent==='Computing…')$('generationDraftStatus').textContent='Settings were not applied.';
      return !!success;
    }
    async function action(callback){
      for(const id of ['saveGenerationA','switchGenerationAB'])$(id).disabled=true;
      try{return await callback()}catch(error){$('generationDraftStatus').textContent=error.message;return false}
      finally{for(const id of ['applyGeneration','resetGeneration','saveGenerationA','generateRelief','generateErosion','generateClimate','applySnr'])$(id).disabled=false;$('switchGenerationAB').disabled=!savedA;}
    }
    $('applyGeneration').onclick=()=>action(()=>submit(read(),draftBase));
    $('resetGeneration').onclick=()=>action(async()=>{
      const sequence=++applySequence;
      const baseline=defaults(draftBase);
      // Restore every draft control immediately, even if generation fails or
      // a previous input is invalid. Color values and their RGB data reset together.
      stageReferences={};write(baseline);$('generationPreset').value='current';
      $('generationDraftStatus').textContent='Resetting all settings…';
      const success=await onReset(draftBase);
      if(sequence!==applySequence)return false;
      $('generationDraftStatus').textContent=success?'All generation settings have been reset.':'Settings reset; generation was not applied.';
      return success;
    });
    function comparisonSnapshot(){const snapshot=clone(currentSnapshot());snapshot.settings={...(snapshot.settings||defaults(snapshot.profile)),...stageReferences};return snapshot;}
    $('saveGenerationA').onclick=()=>{savedA=comparisonSnapshot();savedB=null;showingA=false;$('switchGenerationAB').disabled=false;$('switchGenerationAB').textContent='Show A';$('generationDraftStatus').textContent='Settings A saved'};
    $('switchGenerationAB').onclick=()=>action(async()=>{if(!savedA)return false;const candidateB=comparisonSnapshot(),target=showingA?savedB:savedA;
      if(!await submit(target.settings,target.profile,'restore',{seed:target.seed}))return false;
      if(!showingA)savedB=candidateB;showingA=!showingA;$('switchGenerationAB').textContent=showingA?'Show B':'Show A';return true;});
    installTabs();
    for(const draftPanel of [panel,climatePanel,noisePanel]){
      for(const event of ['input','change'])draftPanel.addEventListener(event,()=>{
        try{saveGeneration(draftBase,read())}catch(_){} // Keep the last valid draft while editing a number.
      });
    }
    return {sync(profile,settings,metadata,custom=false,stages={}){if(metadata)schema=metadata;installClimateControls();base=draftBase=profile;
      stageReferences={...Object.fromEntries(['relief','erosion','climate'].map(name=>['orogen_'+name+'_stage',stages[name]?.id||settings?.['orogen_'+name+'_stage']||''])),orogen_height_stage:stages.height_stage||settings?.orogen_height_stage||'erosion'};
      write(settings||defaults(profile));for(const name of ['relief','erosion','climate']){const state=stages[name];$('stageStatus'+name).textContent=!state?'Not generated':state.stale?'Update needed: input changed.':state.settings_pending?'Update needed: settings changed.':'Generated';}
      $('generationPreset').value=custom?'custom':'current';$('generationSummary').textContent=custom?'Custom settings':'Default profile';$('generationDraftStatus').textContent='';},read,defaults,setGenerator,getGenerator:()=>$('generatorType').value,draftSnapshot:()=>({profile:draftBase,settings:read()}),showError(message){panel.open=true;$('generationDraftStatus').textContent=message}};

    function installTabs(){
      function tabs(container,entries){
        const nav=document.createElement('div');nav.className='settingsTabs';nav.setAttribute('role','tablist');container.append(nav);
        const panes=[];
        for(const [key,title] of entries){
          const button=document.createElement('button');button.id='tab'+key;button.type='button';button.textContent=title;button.setAttribute('role','tab');
          const pane=document.createElement('div');pane.id='pane'+key;pane.className='settingsPane';pane.setAttribute('role','tabpanel');pane.setAttribute('aria-labelledby',button.id);
          button.setAttribute('aria-controls',pane.id);nav.append(button);container.append(pane);panes.push([key,button,pane]);
          button.onclick=()=>{for(const [name,b,p]of panes){const selected=name===key;b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1;p.hidden=!selected;}};
          button.onkeydown=event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const index=panes.findIndex(([name])=>name===key),next=event.key==='Home'?0:event.key==='End'?panes.length-1:(index+(event.key==='ArrowRight'?1:-1)+panes.length)%panes.length;panes[next][1].click();panes[next][1].focus();};
        }
        panes[0][1].click();return Object.fromEntries(panes.map(([key,button,pane])=>[key,pane]));
      }
      heading.querySelector('strong').textContent='Settings';panel.querySelector('summary').title='Settings';panel.querySelector('summary').setAttribute('aria-label','Settings');
      const actions=body.querySelector(':scope > .generationActions');
      const stagePanes=tabs(body,[['Relief','Relief'],['Erosion','Erosion'],['Climate','Climate']]);
      stagePanes.Relief.append(sourceSection,mixing,gpuControls);
      erosionSection.querySelector('.generationSources').append(erosionToggle);
      erosionToggle.firstChild.textContent='Erosion engine ';
      stagePanes.Erosion.append(erosionSection);
      const erosionOptions=$('orogen_hydraulic').closest('details');erosionSection.append(erosionOptions);
      erosionOptions.querySelector('.generationSources').prepend($('orogen_warp').closest('label'));
      for(const id of ['orogen_gpu_post'])erosionSection.append($(id).closest('label'));
      climateBody.querySelector('.panelHeading').remove();climatePanel.querySelector(':scope > summary').hidden=true;climatePanel.open=true;
      climateBody.querySelector('.generationActions').hidden=true;climateBody.prepend($('orogen_gpu_climate').closest('label'));
      stagePanes.Climate.append(climatePanel);
      gpuControls.querySelector('small').textContent='NVIDIA CUDA; GPU options follow your saved local settings. Parallel propagation may change relief.';
      for(const [name,label,scope] of [['Relief','relief','relief'],['Erosion','erosion','erosion'],['Climate','climate','climate']]){
        const footer=document.createElement('div');footer.className='stageActions';const status=document.createElement('small');status.id='stageStatus'+scope;status.setAttribute('role','status');
        const button=document.createElement('button');button.type='button';button.id='generate'+name;button.textContent='Generate '+label;button.onclick=()=>action(()=>submit(read(),draftBase,scope));footer.append(status,button);stagePanes[name].append(footer);
      }
      const help=document.createElement('small');help.textContent='Each tab generates its own stage. Full pipeline: relief, erosion, then climate. Soils use the current relief and climate.';actions.before(help);body.append(actions);
      $('applyGeneration').textContent='Generate full pipeline';
      const rendering=$('renderPanel').querySelector('.toolPanel'),renderHeading=rendering.querySelector('.panelHeading');
      const renderNodes=[...rendering.children].filter(node=>node!==renderHeading);const renderPanes=tabs(rendering,[['Rendering','Rendering'],['Snr','SNR']]);renderPanes.Rendering.append(...renderNodes);
      noiseBody.querySelector('.panelHeading').remove();noisePanel.querySelector(':scope > summary').hidden=true;noisePanel.open=true;renderPanes.Snr.append(noisePanel);
      const adaptiveToggle=document.createElement('label');adaptiveToggle.innerHTML='<input id="snrAdaptive" type="checkbox" checked> Enable adaptive SNR';channelField.before(adaptiveToggle);
      const detailMode=document.createElement('label');detailMode.innerHTML='Detail settings <select id="snrDetailMode"><option value="global">Global</option><option value="per-lod">Per LOD</option></select>';lodSettings.before(detailMode);
      lodSettings.querySelector('summary').textContent='Per-LOD detail amplitude';
      lodSettings.querySelector('small').textContent='Controls displayed relief at each LOD. 0 inherits global settings. Intermediate stages may show a coarser LOD while loading.';
      channelField.querySelector('small').textContent='Global SNR controls neural freedom: lower values follow the source relief more closely. Its neural effect appears after refinement, rather than on the source overview. Per-LOD mode controls displayed detail separately.';
      const detail=$('orogen_detail').closest('label');detail.firstChild.textContent='Mesh vertices ';sourceSection.append(detail);
      const meshHelp=document.createElement('small');meshHelp.textContent='The mesh sets source relief and climate detail. Changing it requires generating relief; display LOD is set in Rendering.';sourceSection.append(meshHelp);
      for(const [pane,text] of [[stagePanes.Relief,'Generating relief retains previous erosion and climate results. The new raw relief becomes visible.'],[stagePanes.Erosion,'Generating erosion uses the latest raw relief without accumulating previous passes. Climate is retained.'],[stagePanes.Climate,'Generating climate uses the currently visibrelief and preserves its elevations exactly.']]){const note=document.createElement('small');note.textContent=text;pane.prepend(note);note.after(pane.querySelector('.stageActions'));}
      const snrActions=noiseBody.querySelector('.generationActions');
      const apply=snrActions.querySelector('[data-shared-action=applyGeneration]');apply.id='applySnr';apply.textContent='Apply SNR';apply.onclick=()=>action(()=>submit(read(),draftBase,'settings'));
      const reset=snrActions.querySelector('[data-shared-action=resetGeneration]');reset.textContent='Reset SNR';reset.onclick=()=>{noisePreset.value='current';noisePreset.dispatchEvent(new Event('change',{bubbles:true}));};
    }
  }
  return {mount,defaults,classify,initializePreferences,savePreferences,saveGeneration,withGpu};
})();
