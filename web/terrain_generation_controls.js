// Draft controls only. Network work happens through the explicit callbacks.
window.TerrainGenerationControls=(()=>{
  const clone=value=>value==null?null:JSON.parse(JSON.stringify(value));
  const channels=['Elevation','Temperature','Temperature variation','Precipitation','Precipitation variation'];
  const cityFields=[['city_erosion_strength','Dose',0,2,.01,1],['city_erosion_iterations','Iterations',1,64,1,12],['city_erosion_talus','Talus slope',.01,4,.01,.6],['city_erosion_motif_km','Drainage scale (km)',10,5000,10,300]];
  const gpuFields=[['orogen_gpu_relief','Relief · plates, collisions, noise'],['orogen_gpu_propagation','Propagation · stress and distances'],['orogen_gpu_post','Orogen postprocessing · local passes'],['orogen_gpu_climate','Climate · winds, currents, rain, temperature'],['orogen_gpu_raster','Projection · GPU spatial search']];
  const tectonicFields=[["orogen_plate_count", "Plate count", 4, 120, true], ["orogen_continent_count", "Continent count", 1, 10, true], ["orogen_land_coverage", "Land coverage", 0.05, 0.9, false], ["orogen_continent_variety", "Continent variety", 0.0, 1.0, false], ["orogen_motion_strength", "Motion strength", 0.0, 4.0, false], ["orogen_convergence_threshold", "Convergence threshold", 0.05, 4.0, false], ["orogen_detail", "Mesh vertices", 20000, 1000000, true], ["orogen_spread", "Orogen propagation", 1.0, 12.0, false], ["orogen_roughness", "Roughness", 0.0, 1.0, false], ["orogen_warp", "Terrain warp", 0.0, 1.0, false], ["orogen_smoothing", "Smoothing", 0.0, 1.0, false], ["orogen_hydraulic", "Hydraulic erosion", 0.0, 1.0, false], ["orogen_thermal", "Thermal erosion", 0.0, 1.0, false], ["orogen_glacial", "Glacial erosion", 0.0, 1.0, false], ["orogen_sharpening", "Ridge sharpening", 0.0, 1.0, false], ["orogen_temperature_offset", "Temperature offset (C)", -15.0, 15.0, false], ["orogen_precipitation_offset", "Precipitation offset", -1.0, 1.0, false]];
  let schema=null;
  const preferencesKey=window.NeuralEarthRuntime?.preferencesKey||'neural-earth-preferences-v1';
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
  function rememberWorld(profile,seed,settings){
    // The opened world, reopened at the next launch instead of a new random one.
    const local=Object.fromEntries(Object.entries(settings).filter(([key])=>!key.endsWith('_stage')));
    savePreferences({lastWorld:{profile,seed:String(seed),settings:local}});
  }
  function withGpu(settings,enabled){
    return {...settings,relief_pipeline:'orogen',...Object.fromEntries([...gpuFields.map(([key])=>key),'orogen_gpu_erosion'].map(key=>[key,enabled]))};
  }
  async function initializePreferences(defaultGpu){
    let preferences=localPreferences();
    if(typeof preferences.gpuAcceleration==='boolean'){
      const resetErosion=preferences.erosionDefaultsVersion!==1;
      if(resetErosion){
        preferences={...preferences,erosionDefaultsVersion:1};
        if(preferences.generation)preferences.generation={...preferences.generation,settings:{...preferences.generation.settings,relief_pipeline:'orogen',orogen_gpu_erosion:preferences.gpuAcceleration}};
        savePreferences(preferences);
      }
      return {preferences,firstLaunch:false,resetErosion};
    }
    // The launcher's --gpu/--cpu answers the first-launch question.
    if(typeof defaultGpu==='boolean'){
      preferences={...preferences,gpuAcceleration:defaultGpu,erosionDefaultsVersion:1,rendering:{gpuRender:defaultGpu,coarseGpu:defaultGpu}};
      savePreferences(preferences);
      return {preferences,firstLaunch:true};
    }
    const dialog=document.createElement('dialog');dialog.id='gpuWelcome';
    dialog.innerHTML='<form method="dialog"><h2>Enable GPU acceleration?</h2><p>Enable all GPU options for Orogen relief, propagation, erosion, climate and projection, plus map rendering?</p><p>Orogen acceleration uses NVIDIA CUDA. Map rendering uses WebGPU. You can adjust each option later in Settings and Rendering. Your choices are saved on this browser.</p><div class="panelActions"><button value="no" autofocus>Use CPU options</button><button value="yes">Enable GPU acceleration</button></div></form>';
    document.body.append(dialog);
    const translate=()=>{
      const locale=window.NeuralEarthLocale;if(!locale)return;
      const t=key=>locale.text('welcome.'+key),paragraphs=dialog.querySelectorAll('p');
      dialog.querySelector('h2').textContent=t('title');paragraphs[0].textContent=t('prompt');paragraphs[1].textContent=t('details');
      dialog.querySelector('[value="no"]').textContent=t('cpu');dialog.querySelector('[value="yes"]').textContent=t('gpu');
    };
    translate();window.addEventListener('neural-locale-change',translate);
    const enabled=await new Promise(resolve=>{dialog.addEventListener('close',()=>resolve(dialog.returnValue==='yes'),{once:true});dialog.showModal()});
    window.removeEventListener('neural-locale-change',translate);
    dialog.remove();
    preferences={...preferences,gpuAcceleration:enabled,erosionDefaultsVersion:1,rendering:{gpuRender:enabled,coarseGpu:enabled}};
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
    return {...propertyDefaults,world_diameter_km:40000/Math.PI,world_topology:"sphere",...withGpu({},localPreferences().gpuAcceleration===true),...Object.fromEntries(cityFields.map(([key,label,min,max,step,value])=>[key,value])),height_source:natural?'natural':profile==='orogen'?'orogen':'native',climate_source:'orogen',relief_pipeline:'orogen',
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
      const summary=document.createElement('summary');summary.textContent=title;summary.dataset.generationSection=title;section.append(summary);
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
    channelField.id='neuralRefinement';
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
    const advanced=document.createElement('details');advanced.id='adaptiveSnr';advanced.className='advancedSettings';advanced.innerHTML='<summary>Regional noise modulation</summary>';
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
    const layoutHelp=document.createElement('small');layoutHelp.id='continentPresetHelp';layoutHelp.textContent='Presets initialize plates (Orogen), noise (Noise), or the atlas (Custom). Custom retains the choice of atlas or procedural continents.';layoutLabel.after(layoutHelp);
    orogen.querySelector('legend').textContent='Relief · options';sourceSection.append(orogen);
    const offsets=[document.getElementById('orogen_temperature_offset').closest('details')];climateBody.append(...offsets);
    const intro=document.createElement('small');intro.id='climateIntro';intro.textContent='Orogen supplies the shared climate for Noise, Orogen and Custom: seasons, winds, rain, classification and biomes.';climateBody.querySelector('.panelHeading').after(intro);
    for(const id of ['snrAltitudeLow','snrAltitudeHigh']){const label=document.getElementById(id).closest('label');label.firstChild.textContent+=' (m) ';label.lastChild.remove();}
    noiseBody.append(channelField);for(const note of [...body.querySelectorAll(':scope > small')])noiseBody.append(note);channelField.querySelector('legend').textContent='Relief noise · SNR';
    channelField.querySelector('small').textContent='SNR here means noise / signal amplitude. Lower values constrain relief more closely to its inputs. Climate remains shared.';
    channelField.querySelector('.channelSelect').hidden=true;
    for(const card of channelField.querySelectorAll('[data-channel]'))card.hidden=card.dataset.channel!=='0';
    advanced.open=true;advanced.querySelector('summary').textContent='Adaptive SNR · relief / temperature / latitude';
    const latitudeControls=document.createElement('div');latitudeControls.className='generationSources';
    latitudeControls.innerHTML='<label>Latitude multiplier <input id="snrLatitudeGain" aria-label="SNR latitude multiplier" value="1"></label><label>Latitude ramp starts (°) <input id="snrLatitudeLow" aria-label="SNR latitude start" value="0"></label><label>Latitude ramp ends (°) <input id="snrLatitudeHigh" aria-label="SNR latitude end" value="75"></label>';
    advanced.append(latitudeControls);
    const lodSettings=document.createElement('details');lodSettings.id='lodSnrSettings';lodSettings.className='advancedSettings';
    lodSettings.innerHTML='<summary>Per-LOD SNR · relief detail</summary><small>0 inherits global SNR. Per-LOD values scale reconstructed relief noise (the local residual after the NN). They do not change climate channels or shared network computation. Experimental settings.</small><div class="generationSources">'+Array.from({length:15},(_,i)=>`<label>LOD ${i-3} · ${30*2**(i-3)} m <input id="snrLod${i}" aria-label="Relief SNR LOD ${i-3}" value="0"></label>`).join('')+'</div>';
    noiseBody.append(lodSettings);
    for(const details of [climatePanel,noisePanel]){const actions=document.createElement('div');actions.className='generationActions';actions.innerHTML='<button type="button" data-shared-action="applyGeneration">Apply</button><button type="button" data-shared-action="resetGeneration">Reset defaults</button><span class="sharedDraftStatus" role="status"></span>';details.querySelector('.generationBody').append(actions);for(const button of actions.querySelectorAll('button'))button.onclick=()=>document.getElementById(button.dataset.sharedAction).click();}
    new MutationObserver(()=>{for(const details of [climatePanel,noisePanel]){details.querySelector('.sharedDraftStatus').textContent=document.getElementById('generationDraftStatus').textContent;for(const button of details.querySelectorAll('[data-shared-action]'))button.disabled=document.getElementById(button.dataset.sharedAction).disabled;}}).observe(document.getElementById('generationDraftStatus').parentElement,{subtree:true,childList:true,attributes:true});
    const noisePreset=document.getElementById('generationPreset');noisePreset.closest('label').firstChild.textContent='Preset SNR ';for(const option of [...noisePreset.options])if(['natural','continental'].includes(option.value))option.remove();noiseBody.querySelector('.generationActions').prepend(noisePreset.closest('label'));
    const numericControls=[];
    const extraFields=[];
    const groupNames={wind:'Atmosphere & winds',ocean:'Ocean currents',temperature:'Temperature',precipitation:'Rain & moisture','heuristic-precip':'Zonal rain model',koppen:'Köppen classification','biome-altitude':'Alpine & snow lines','biome-rendering':'Biome appearance','biome-palette':'Biome colors'};
    const climateUi={
      parameters:['Climate parameters','Paramètres climatiques','气候参数','Parámetros climáticos','Parâmetros climáticos'],
      find:['Find','Rechercher','查找','Buscar','Buscar'],
      placeholder:['Rain, snow, pressure…','Pluie, neige, pression…','降雨、降雪、气压…','Lluvia, nieve, presión…','Chuva, neve, pressão…'],
      searchLabel:['Find Orogen settings','Rechercher les paramètres Orogen','查找 Orogen 设置','Buscar ajustes de Orogen','Buscar configurações do Orogen'],
      section:['Section','Section','类别','Sección','Seção'],
      help:['Source climate and Original Orogen biomes. Changes take effect with Apply.','Climat source et biomes Orogen d’origine. Les changements prennent effet après application.','源气候和原始 Orogen 生物群系。更改在应用后生效。','Clima de origen y biomas originales de Orogen. Los cambios se aplican al pulsar Aplicar.','Clima de origem e biomas originais do Orogen. As alterações entram em vigor ao aplicar.'],
      wind:['Atmosphere & winds','Atmosphère et vents','大气与风','Atmósfera y vientos','Atmosfera e ventos'],
      ocean:['Ocean currents','Courants océaniques','洋流','Corrientes oceánicas','Correntes oceânicas'],
      temperature:['Temperature','Température','温度','Temperatura','Temperatura'],
      precipitation:['Rain & moisture','Pluie et humidité','降雨与湿度','Lluvia y humedad','Chuva e umidade'],
      'heuristic-precip':['Zonal rain model','Modèle zonal de pluie','纬向降雨模型','Modelo zonal de lluvia','Modelo zonal de chuva'],
      koppen:['Köppen classification','Classification de Köppen','柯本气候分类','Clasificación de Köppen','Classificação de Köppen'],
      'biome-altitude':['Alpine & snow lines','Étages alpins et limites de neige','高山与雪线','Pisos alpinos y líneas de nieve','Pisos alpinos e linhas de neve'],
      'biome-rendering':['Biome appearance','Apparence des biomes','生物群系外观','Aspecto de los biomas','Aparência dos biomas'],
      'biome-palette':['Biome colors','Couleurs des biomes','生物群系颜色','Colores de los biomas','Cores dos biomas'],
    };
    // Labels are keyed by the schema ID, never by a translated label or value.
    const climateFieldText={
      orogen_ocean_coast_range:['Current coastal range','Portée côtière des courants','洋流沿岸范围','Alcance costero de corrientes','Alcance costeiro das correntes'],
      orogen_ocean_warmth_range:['Coastal warmth range multiplier','Multiplicateur de portée de chaleur côtière','沿岸暖流范围倍数','Multiplicador del alcance del calor costero','Multiplicador do alcance do calor costeiro'],
      orogen_ocean_west_strength:['Western boundary current strength','Intensité du courant de bord ouest','西岸边界流强度','Intensidad de corriente de borde occidental','Intensidade da corrente de contorno oeste'],
      orogen_ocean_east_strength:['Eastern boundary current strength','Intensité du courant de bord est','东岸边界流强度','Intensidad de corriente de borde oriental','Intensidade da corrente de contorno leste'],
      orogen_ocean_west_deflection:['Western zonal attenuation','Atténuation zonale ouest','西部纬向衰减','Atenuación zonal occidental','Atenuação zonal oeste'],
      orogen_ocean_east_deflection:['Eastern zonal attenuation','Atténuation zonale est','东部纬向衰减','Atenuación zonal oriental','Atenuação zonal leste'],
      orogen_ocean_circumpolar_strength:['Circumpolar current strength','Intensité du courant circumpolaire','绕极洋流强度','Intensidad de la corriente circumpolar','Intensidade da corrente circumpolar'],
      orogen_ocean_smoothing_km:['Current smoothing (km)','Lissage des courants (km)','洋流平滑（km）','Suavizado de corrientes (km)','Suavização das correntes (km)'],
      orogen_ocean_warmth_smoothing_km:['Ocean warmth smoothing (km)','Lissage de la chaleur océanique (km)','海洋热量平滑（km）','Suavizado del calor oceánico (km)','Suavização do calor oceânico (km)'],
      orogen_temperature_smoothing:['Temperature smoothing passes','Passes de lissage de température','温度平滑次数','Pasadas de suavizado de temperatura','Passagens de suavização da temperatura'],
      orogen_temperature_ocean_diffusion_km:['Coastal heat diffusion (km)','Diffusion de chaleur côtière (km)','沿岸热量扩散（km）','Difusión del calor costero (km)','Difusão do calor costeiro (km)'],
      orogen_temperature_tropical_width:['Tropical plateau width (°)','Largeur du plateau tropical (°)','热带平台宽度（°）','Anchura de la meseta tropical (°)','Largura do platô tropical (°)'],
      orogen_temperature_equator:['Equatorial temperature (°C)','Température équatoriale (°C)','赤道温度（°C）','Temperatura ecuatorial (°C)','Temperatura equatorial (°C)'],
      orogen_temperature_polar_drop:['Pole temperature drop (°C)','Baisse de température aux pôles (°C)','极地温度下降（°C）','Descenso de temperatura polar (°C)','Queda de temperatura polar (°C)'],
      orogen_temperature_latitude_power:['Latitude falloff exponent','Exposant de décroissance selon la latitude','纬度衰减指数','Exponente de descenso por latitud','Expoente de queda por latitude'],
      orogen_temperature_reference_itcz:['Thermal reference latitude (°)','Latitude thermique de référence (°)','热参考纬度（°）','Latitud térmica de referencia (°)','Latitude térmica de referência (°)'],
      orogen_temperature_wet_lapse:['Wet lapse rate (°C/km)','Gradient thermique humide (°C/km)','湿绝热温度递减率（°C/km）','Gradiente térmico húmedo (°C/km)','Gradiente térmico úmido (°C/km)'],
      orogen_temperature_dry_lapse_extra:['Dry lapse increment (°C/km)','Supplément de gradient sec (°C/km)','干绝热递减率增量（°C/km）','Incremento del gradiente seco (°C/km)','Incremento do gradiente seco (°C/km)'],
      orogen_temperature_ocean_effect:['Ocean heat influence (°C)','Influence de la chaleur océanique (°C)','海洋热量影响（°C）','Influencia del calor oceánico (°C)','Influência do calor oceânico (°C)'],
      orogen_temperature_coastal_effect:['Coastal heat influence (°C)','Influence de la chaleur côtière (°C)','沿岸热量影响（°C）','Influencia del calor costero (°C)','Influência do calor costeiro (°C)'],
      orogen_temperature_cloud_moderation:['Cloud moderation','Modération par les nuages','云层缓和作用','Moderación por nubes','Moderação por nuvens'],
      orogen_temperature_clear_amplification:['Clear sky amplification','Amplification par ciel clair','晴空放大作用','Amplificación por cielo despejado','Amplificação por céu limpo'],
      orogen_temperature_seasonal_boost:['Seasonal temperature boost (°C)','Amplification saisonnière de température (°C)','季节性温度增强（°C）','Aumento estacional de temperatura (°C)','Aumento sazonal de temperatura (°C)'],
      orogen_temperature_maritime_factor:['Maritime seasonal factor','Facteur saisonnier maritime','海洋性季节因子','Factor estacional marítimo','Fator sazonal marítimo'],
      orogen_temperature_continental_extra:['Continental seasonal increment','Supplément saisonnier continental','大陆性季节增量','Incremento estacional continental','Incremento sazonal continental'],
      orogen_precip_advection_km:['Moisture advection reach (km)','Portée de l’advection d’humidité (km)','水汽平流范围（km）','Alcance de advección de humedad (km)','Alcance da advecção de umidade (km)'],
      orogen_precip_elevation_smoothing_km:['Orographic smoothing (km)','Lissage orographique (km)','地形平滑（km）','Suavizado orográfico (km)','Suavização orográfica (km)'],
      orogen_precip_convergence_smoothing_km:['Frontal smoothing (km)','Lissage des fronts (km)','锋面平滑（km）','Suavizado frontal (km)','Suavização frontal (km)'],
      orogen_precip_ocean_moisture:['Ocean baseline moisture','Humidité océanique de référence','海洋基准湿度','Humedad oceánica de referencia','Umidade oceânica de referência'],
      orogen_precip_ocean_warmth:['Warm water moisture contribution','Apport d’humidité des eaux chaudes','暖水水汽贡献','Aporte de humedad de aguas cálidas','Contribuição de umidade das águas quentes'],
      orogen_precip_offshore_moisture:['Offshore coastal moisture','Humidité côtière au large','近海沿岸湿度','Humedad costera mar adentro','Umidade costeira ao largo'],
      orogen_precip_retention:['Flat terrain moisture retention','Rétention d’humidité sur terrain plat','平坦地形水汽保留','Retención de humedad en terreno llano','Retenção de umidade em terreno plano'],
      orogen_precip_elevation_depletion:['Altitude moisture depletion','Épuisement de l’humidité par altitude','海拔导致的水汽减少','Agotamiento de humedad por altitud','Redução da umidade pela altitude'],
      orogen_precip_itcz_width:['ITCZ rain width (°)','Largeur des pluies de la ZCIT (°)','热带辐合带降雨宽度（°）','Anchura de lluvia de la ZCIT (°)','Largura da chuva da ZCIT (°)'],
      orogen_precip_itcz_core_boost:['ITCZ core rain boost','Renforcement des pluies au cœur de la ZCIT','热带辐合带核心降雨增强','Aumento de lluvia en el núcleo de la ZCIT','Aumento de chuva no núcleo da ZCIT'],
      orogen_precip_itcz_add:['ITCZ rain addition','Apport de pluie de la ZCIT','热带辐合带降雨增量','Aporte de lluvia de la ZCIT','Acréscimo de chuva da ZCIT'],
      orogen_precip_frontal_scale:['Convergence sensitivity','Sensibilité à la convergence','辐合敏感度','Sensibilidad a la convergencia','Sensibilidade à convergência'],
      orogen_precip_frontal_boost:['Frontal rain boost','Renforcement des pluies frontales','锋面降雨增强','Aumento de lluvia frontal','Aumento de chuva frontal'],
      orogen_precip_frontal_add:['Frontal moisture contribution','Apport d’humidité des fronts','锋面水汽贡献','Aporte de humedad frontal','Contribuição de umidade frontal'],
      orogen_precip_uplift_scale:['Windward gradient sensitivity','Sensibilité au gradient au vent','迎风坡梯度敏感度','Sensibilidad al gradiente de barlovento','Sensibilidade ao gradiente de barlavento'],
      orogen_precip_uplift_add:['Windward local rain boost','Renforcement local des pluies au vent','迎风坡局地降雨增强','Aumento local de lluvia de barlovento','Aumento local de chuva de barlavento'],
      orogen_precip_shadow_scale:['Leeward gradient sensitivity','Sensibilité au gradient sous le vent','背风坡梯度敏感度','Sensibilidad al gradiente de sotavento','Sensibilidade ao gradiente de sotavento'],
      orogen_precip_shadow_local:['Local rain shadow strength','Intensité locale de l’ombre pluviométrique','局地雨影强度','Intensidad local de sombra pluviométrica','Intensidade local da sombra de chuva'],
      orogen_precip_subtrop_summer:['Summer subtropical suppression','Atténuation subtropicale estivale','夏季副热带抑制','Supresión subtropical de verano','Supressão subtropical de verão'],
      orogen_precip_subtrop_winter:['Winter subtropical suppression','Atténuation subtropicale hivernale','冬季副热带抑制','Supresión subtropical de invierno','Supressão subtropical de inverno'],
      orogen_precip_monsoon_relief:['Coastal monsoon relief','Atténuation côtière par la mousson','沿岸季风缓解','Alivio monzónico costero','Alívio costeiro pela monção'],
      orogen_precip_high_pressure:['High pressure rain reduction','Réduction des pluies par haute pression','高压降雨抑制','Reducción de lluvia por alta presión','Redução de chuva por alta pressão'],
      orogen_precip_low_pressure:['Low pressure rain relief','Atténuation par basse pression','低压降雨缓解','Alivio de lluvia por baja presión','Alívio de chuva por baixa pressão'],
      orogen_precip_polar_base:['Polar rain baseline','Pluie polaire de référence','极地基准降雨','Lluvia polar de referencia','Chuva polar de referência'],
      orogen_precip_polar_coastal:['Polar coastal rain','Pluie côtière polaire','极地沿岸降雨','Lluvia costera polar','Chuva costeira polar'],
      orogen_precip_polar_boost:['Polar rain multiplier','Multiplicateur de pluie polaire','极地降雨倍数','Multiplicador de lluvia polar','Multiplicador de chuva polar'],
      orogen_precip_continental_dryness:['Continental drying','Assèchement continental','大陆干燥作用','Secado continental','Ressecamento continental'],
      orogen_precip_lee_add:['Lee cyclogenesis rain','Pluie des dépressions sous le vent','背风气旋降雨','Lluvia de ciclogénesis de sotavento','Chuva de ciclogênese de sotavento'],
      orogen_precip_ocean_base:['Ocean rain baseline','Pluie océanique de référence','海洋基准降雨','Lluvia oceánica de referencia','Chuva oceânica de referência'],
      orogen_precip_cutoff_start_km:['Dry interior starts (km)','Début de l’intérieur sec (km)','干燥内陆起点（km）','Inicio del interior seco (km)','Início do interior seco (km)'],
      orogen_precip_cutoff_end_km:['Dry interior ends (km)','Fin de l’intérieur sec (km)','干燥内陆终点（km）','Fin del interior seco (km)','Fim do interior seco (km)'],
      orogen_precip_land_drying:['High land coverage drying','Assèchement des fortes couvertures terrestres','高陆地覆盖率干燥作用','Secado por alta cobertura terrestre','Ressecamento por alta cobertura terrestre'],
      orogen_precip_shadow_min_height:['Rain shadow minimum height (km)','Hauteur minimale de l’ombre pluviométrique (km)','雨影最低高度（km）','Altura mínima de sombra pluviométrica (km)','Altura mínima da sombra de chuva (km)'],
      orogen_precip_shadow_reach_km:['Downwind rain shadow (km)','Portée de l’ombre sous le vent (km)','下风雨影范围（km）','Alcance de sombra a sotavento (km)','Alcance da sombra a sotavento (km)'],
      orogen_precip_shadow_retention:['Downwind shadow retention','Persistance de l’ombre sous le vent','下风雨影保持率','Retención de sombra a sotavento','Retenção da sombra a sotavento'],
      orogen_precip_windward_reach_km:['Upwind rain reach (km)','Portée des pluies au vent (km)','迎风降雨范围（km）','Alcance de lluvia a barlovento (km)','Alcance da chuva a barlavento (km)'],
      orogen_precip_windward_retention:['Upwind rain retention','Persistance des pluies au vent','迎风降雨保持率','Retención de lluvia a barlovento','Retenção da chuva a barlavento'],
      orogen_precip_shadow_propagated:['Propagated shadow reduction','Réduction de l’ombre propagée','传播雨影的削减','Reducción de sombra propagada','Redução da sombra propagada'],
      orogen_precip_windward_propagated:['Propagated windward rain','Pluie propagée au vent','传播的迎风降雨','Lluvia propagada a barlovento','Chuva propagada a barlavento'],
      orogen_precip_smoothing_km:['Rain smoothing (km)','Lissage des pluies (km)','降雨平滑（km）','Suavizado de lluvia (km)','Suavização da chuva (km)'],
      orogen_precip_heuristic_blend:['Heuristic rain proportion','Proportion de pluie heuristique','启发式降雨比例','Proporción de lluvia heurística','Proporção de chuva heurística'],
      orogen_precip_heuristic_wind_blend:['Heuristic wind proportion','Proportion de vent heuristique','启发式风比例','Proporción de viento heurístico','Proporção de vento heurístico'],
      orogen_precip_interior_cap:['Interior rain cap reduction','Réduction du plafond de pluie intérieure','内陆降雨上限降低','Reducción del límite de lluvia interior','Redução do limite de chuva interior'],
      orogen_heuristic_itcz_shift:['Heuristic ITCZ displacement','Déplacement heuristique de la ZCIT','启发式热带辐合带位移','Desplazamiento heurístico de la ZCIT','Deslocamento heurístico da ZCIT'],
      orogen_heuristic_summer:['Heuristic summer rain factor','Facteur heuristique de pluie estivale','启发式夏季降雨系数','Factor heurístico de lluvia estival','Fator heurístico de chuva de verão'],
      orogen_heuristic_winter:['Heuristic winter rain factor','Facteur heuristique de pluie hivernale','启发式冬季降雨系数','Factor heurístico de lluvia invernal','Fator heurístico de chuva de inverno'],
      orogen_heuristic_med_drying:['Mediterranean interior drying','Assèchement de l’intérieur méditerranéen','地中海内陆干燥作用','Secado del interior mediterráneo','Ressecamento do interior mediterrâneo'],
      orogen_heuristic_med_coast:['Mediterranean west coast drying','Assèchement de la côte ouest méditerranéenne','地中海西岸干燥作用','Secado de la costa occidental mediterránea','Ressecamento da costa oeste mediterrânea'],
      orogen_heuristic_continental:['Heuristic continental drying','Assèchement continental heuristique','启发式大陆干燥作用','Secado continental heurístico','Ressecamento continental heurístico'],
      orogen_heuristic_uplift:['Heuristic windward enhancement','Renforcement heuristique au vent','启发式迎风增强','Refuerzo heurístico a barlovento','Aumento heurístico a barlavento'],
      orogen_heuristic_shadow:['Heuristic shadow reduction','Réduction heuristique de l’ombre','启发式雨影削减','Reducción heurística de sombra','Redução heurística da sombra'],
      orogen_heuristic_shadow_height:['Full shadow height (km)','Hauteur de l’ombre complète (km)','完全雨影高度（km）','Altura de sombra completa (km)','Altura da sombra completa (km)'],
      orogen_koppen_ice:['Ice cap warmest month (°C)','Mois le plus chaud de la calotte (°C)','冰盖最暖月（°C）','Mes más cálido de la capa de hielo (°C)','Mês mais quente da calota de gelo (°C)'],
      orogen_koppen_tundra:['Tundra warmest month (°C)','Mois le plus chaud de la toundra (°C)','苔原最暖月（°C）','Mes más cálido de la tundra (°C)','Mês mais quente da tundra (°C)'],
      orogen_koppen_tropical:['Tropical coldest month (°C)','Mois tropical le plus froid (°C)','热带最冷月（°C）','Mes tropical más frío (°C)','Mês tropical mais frio (°C)'],
      orogen_koppen_temperate:['Temperate coldest month (°C)','Mois tempéré le plus froid (°C)','温带最冷月（°C）','Mes templado más frío (°C)','Mês temperado mais frio (°C)'],
      orogen_koppen_hot_summer:['Hot summer threshold (°C)','Seuil d’été chaud (°C)','炎热夏季阈值（°C）','Umbral de verano cálido (°C)','Limiar de verão quente (°C)'],
      orogen_koppen_shoulder:['Warm shoulder month (°C)','Mois intermédiaire chaud (°C)','暖过渡季月份（°C）','Mes cálido de transición (°C)','Mês quente de transição (°C)'],
      orogen_koppen_extreme_cold:['Extreme cold threshold (°C)','Seuil de froid extrême (°C)','极寒阈值（°C）','Umbral de frío extremo (°C)','Limiar de frio extremo (°C)'],
      orogen_koppen_hot_arid:['Hot arid annual mean (°C)','Moyenne annuelle aride chaude (°C)','炎热干旱年均温（°C）','Media anual árida cálida (°C)','Média anual árida quente (°C)'],
      orogen_koppen_aridity_temperature:['Aridity temperature coefficient','Coefficient thermique d’aridité','干旱温度系数','Coeficiente térmico de aridez','Coeficiente térmico de aridez'],
      orogen_koppen_aridity_summer:['Summer aridity addition (mm)','Supplément d’aridité estivale (mm)','夏季干旱增量（mm）','Incremento de aridez estival (mm)','Acréscimo de aridez de verão (mm)'],
      orogen_koppen_aridity_mixed:['Mixed aridity addition (mm)','Supplément d’aridité mixte (mm)','混合干旱增量（mm）','Incremento de aridez mixta (mm)','Acréscimo de aridez mista (mm)'],
      orogen_koppen_summer_fraction:['Summer rain fraction','Fraction de pluie estivale','夏季降雨比例','Fracción de lluvia estival','Fração de chuva de verão'],
      orogen_koppen_winter_fraction:['Winter rain fraction','Fraction de pluie hivernale','冬季降雨比例','Fracción de lluvia invernal','Fração de chuva de inverno'],
      orogen_koppen_desert_fraction:['Desert / steppe aridity ratio','Rapport d’aridité désert / steppe','沙漠 / 草原干旱比','Relación de aridez desierto / estepa','Relação de aridez deserto / estepe'],
      orogen_koppen_dry_summer_mm:['Dry summer monthly rain (mm)','Pluie mensuelle d’été sec (mm)','干燥夏季月降雨量（mm）','Lluvia mensual de verano seco (mm)','Chuva mensal de verão seco (mm)'],
      orogen_koppen_dry_summer_ratio:['Dry summer season ratio','Rapport de saison estivale sèche','干燥夏季比例','Relación de estación seca estival','Relação de estação seca de verão'],
      orogen_koppen_dry_winter_ratio:['Dry winter season ratio','Rapport de saison hivernale sèche','干燥冬季比例','Relación de estación seca invernal','Relação de estação seca de inverno'],
      orogen_koppen_rainforest_mm:['Rainforest driest month (mm)','Mois le plus sec de la forêt pluviale (mm)','雨林最干月（mm）','Mes más seco de selva lluviosa (mm)','Mês mais seco da floresta pluvial (mm)'],
      orogen_koppen_monsoon_factor:['Monsoon annual rain factor','Facteur annuel de pluie de mousson','季风年降雨系数','Factor anual de lluvia monzónica','Fator anual de chuva de monção'],
      orogen_koppen_monsoon_mm:['Monsoon reference rain (mm)','Pluie de référence de mousson (mm)','季风参考降雨量（mm）','Lluvia monzónica de referencia (mm)','Chuva de referência da monção (mm)'],
      orogen_biome_tropical_alpine:['Tropical alpine line (km)','Limite alpine tropicale (km)','热带高山线（km）','Límite alpino tropical (km)','Limite alpino tropical (km)'],
      orogen_biome_tropical_snow:['Tropical snow line (km)','Limite des neiges tropicale (km)','热带雪线（km）','Línea de nieve tropical (km)','Linha de neve tropical (km)'],
      orogen_biome_arid_alpine:['Arid alpine line (km)','Limite alpine aride (km)','干旱区高山线（km）','Límite alpino árido (km)','Limite alpino árido (km)'],
      orogen_biome_arid_snow:['Arid snow line (km)','Limite des neiges aride (km)','干旱区雪线（km）','Línea de nieve árida (km)','Linha de neve árida (km)'],
      orogen_biome_temperate_alpine:['Temperate alpine line (km)','Limite alpine tempérée (km)','温带高山线（km）','Límite alpino templado (km)','Limite alpino temperado (km)'],
      orogen_biome_temperate_snow:['Temperate snow line (km)','Limite des neiges tempérée (km)','温带雪线（km）','Línea de nieve templada (km)','Linha de neve temperada (km)'],
      orogen_biome_continental_alpine:['Continental alpine line (km)','Limite alpine continentale (km)','大陆高山线（km）','Límite alpino continental (km)','Limite alpino continental (km)'],
      orogen_biome_continental_snow:['Continental snow line (km)','Limite des neiges continentale (km)','大陆雪线（km）','Línea de nieve continental (km)','Linha de neve continental (km)'],
      orogen_biome_subarctic_alpine:['Subarctic alpine line (km)','Limite alpine subarctique (km)','亚寒带高山线（km）','Límite alpino subártico (km)','Limite alpino subártico (km)'],
      orogen_biome_subarctic_snow:['Subarctic snow line (km)','Limite des neiges subarctique (km)','亚寒带雪线（km）','Línea de nieve subártica (km)','Linha de neve subártica (km)'],
      orogen_biome_tundra_alpine:['Tundra alpine line (km)','Limite alpine de la toundra (km)','苔原高山线（km）','Límite alpino de tundra (km)','Limite alpino da tundra (km)'],
      orogen_biome_tundra_snow:['Tundra snow line (km)','Limite des neiges de la toundra (km)','苔原雪线（km）','Línea de nieve de tundra (km)','Linha de neve da tundra (km)'],
      orogen_biome_ice_alpine:['Ice alpine line (km)','Limite alpine glaciaire (km)','冰区高山线（km）','Límite alpino glaciar (km)','Limite alpino glacial (km)'],
      orogen_biome_ice_snow:['Ice snow line (km)','Limite des neiges glaciaire (km)','冰区雪线（km）','Línea de nieve glaciar (km)','Linha de neve glacial (km)'],
      orogen_biome_low_darkening:['Lowland shade strength','Intensité de l’ombre en plaine','低地阴影强度','Intensidad de sombra en tierras bajas','Intensidade da sombra nas terras baixas'],
      orogen_biome_mid_darkening:['Mountain shade strength','Intensité de l’ombre en montagne','山地阴影强度','Intensidad de sombra en montaña','Intensidade da sombra nas montanhas'],
      orogen_biome_snow_transition:['Snow transition height (km)','Altitude de transition de neige (km)','雪过渡高度（km）','Altura de transición de nieve (km)','Altitude de transição da neve (km)'],
      orogen_biome_rock_color:['Rock color','Couleur de la roche','岩石颜色','Color de roca','Cor da rocha'],
      orogen_biome_snow_color:['Snow color','Couleur de la neige','雪色','Color de nieve','Cor da neve'],
      orogen_itcz_sample_radius:['ITCZ sampling radius (°)','Rayon d’échantillonnage de la ZCIT (°)','热带辐合带采样半径（°）','Radio de muestreo de la ZCIT (°)','Raio de amostragem da ZCIT (°)'],
      orogen_pressure_baseline:['Baseline pressure (hPa)','Pression de référence (hPa)','基准气压（hPa）','Presión de referencia (hPa)','Pressão de referência (hPa)'],
      orogen_pressure_subtropical_lat:['Subtropical high latitude (°)','Latitude de l’anticyclone subtropical (°)','副热带高压纬度（°）','Latitud del anticiclón subtropical (°)','Latitude do anticiclone subtropical (°)'],
      orogen_pressure_subtropical_width:['Pressure belt width (°)','Largeur de la ceinture de pression (°)','气压带宽度（°）','Anchura del cinturón de presión (°)','Largura da faixa de pressão (°)'],
      orogen_pressure_subpolar_lat:['Subpolar low latitude (°)','Latitude de la dépression subpolaire (°)','副极地低压纬度（°）','Latitud de la baja subpolar (°)','Latitude da baixa subpolar (°)'],
      orogen_pressure_polar_lat:['Polar high latitude (°)','Latitude de l’anticyclone polaire (°)','极地高压纬度（°）','Latitud del anticiclón polar (°)','Latitude do anticiclone polar (°)'],
      orogen_pressure_polar_width:['Polar pressure width (°)','Largeur de la pression polaire (°)','极地气压宽度（°）','Anchura de presión polar (°)','Largura da pressão polar (°)'],
      orogen_geostrophic_lat:['Deflection full at latitude (°)','Latitude de déviation maximale (°)','完全偏转纬度（°）','Latitud de desviación máxima (°)','Latitude de desvio máximo (°)'],
      orogen_wind_friction_speed:['Surface wind speed factor','Facteur de vitesse du vent de surface','地表风速系数','Factor de velocidad del viento superficial','Fator de velocidade do vento superficial'],
      orogen_pressure_noise_frequency:['Pressure noise frequency','Fréquence du bruit de pression','气压噪声频率','Frecuencia del ruido de presión','Frequência do ruído de pressão'],
      orogen_pressure_continental_start:['Thermal continentality starts','Début de continentalité thermique','热大陆性起点','Inicio de continentalidad térmica','Início da continentalidade térmica'],
      orogen_pressure_continental_full:['Thermal continentality full','Continentalité thermique maximale','热大陆性最大值','Continentalidad térmica máxima','Continentalidade térmica máxima'],
      orogen_ocean_itcz_width:['Countercurrent width (°)','Largeur du contre-courant (°)','逆流宽度（°）','Anchura de contracorriente (°)','Largura da contracorrente (°)'],
      orogen_ocean_trade_lat:['Trade current ends (°)','Fin du courant des alizés (°)','信风洋流终点（°）','Fin de corriente de alisios (°)','Fim da corrente dos alísios (°)'],
      orogen_ocean_westerly_start:['Westerly current starts (°)','Début du courant d’ouest (°)','西风洋流起点（°）','Inicio de corriente occidental (°)','Início da corrente de oeste (°)'],
      orogen_ocean_westerly_end:['Westerly current ends (°)','Fin du courant d’ouest (°)','西风洋流终点（°）','Fin de corriente occidental (°)','Fim da corrente de oeste (°)'],
      orogen_ocean_polar_lat:['Polar current starts (°)','Début du courant polaire (°)','极地洋流起点（°）','Inicio de corriente polar (°)','Início da corrente polar (°)'],
      orogen_ocean_warm_trade_end:['Warm trade belt ends (°)','Fin de la bande chaude des alizés (°)','暖信风带终点（°）','Fin de la franja cálida de alisios (°)','Fim da faixa quente dos alísios (°)'],
      orogen_ocean_warm_westerly_end:['Warm westerly belt ends (°)','Fin de la bande chaude des vents d’ouest (°)','暖西风带终点（°）','Fin de la franja cálida occidental (°)','Fim da faixa quente dos ventos de oeste (°)'],
      orogen_ocean_circumpolar_min:['Circumpolar belt starts (°)','Début de la bande circumpolaire (°)','绕极带起点（°）','Inicio de la franja circumpolar (°)','Início da faixa circumpolar (°)'],
      orogen_ocean_circumpolar_max:['Circumpolar belt ends (°)','Fin de la bande circumpolaire (°)','绕极带终点（°）','Fin de la franja circumpolar (°)','Fim da faixa circumpolar (°)'],
      orogen_ocean_circumpolar_center:['Circumpolar center (°)','Centre circumpolaire (°)','绕极带中心（°）','Centro circumpolar (°)','Centro circumpolar (°)'],
      orogen_ocean_polar_speed:['Polar current speed','Vitesse du courant polaire','极地洋流速度','Velocidad de la corriente polar','Velocidade da corrente polar'],
      orogen_ocean_polar_transition:['Polar current transition strength','Intensité de transition du courant polaire','极地洋流过渡强度','Intensidad de transición de corriente polar','Intensidade da transição da corrente polar'],
      orogen_ocean_circumpolar_width:['Circumpolar half width (°)','Demi-largeur circumpolaire (°)','绕极带半宽度（°）','Semianchura circumpolar (°)','Meia largura circumpolar (°)'],
      orogen_ocean_circumpolar_deflection:['Circumpolar meridional attenuation','Atténuation méridienne circumpolaire','绕极经向衰减','Atenuación meridional circumpolar','Atenuação meridional circumpolar'],
      orogen_temperature_blend_start:['Thermal latitude blending starts (°)','Début du mélange thermique en latitude (°)','纬度温度混合起点（°）','Inicio de mezcla térmica latitudinal (°)','Início da mistura térmica por latitude (°)'],
      orogen_temperature_cloud_threshold:['Cloud moderation starts at rain','Pluie au début de modération nuageuse','云层缓和作用开始的降雨阈值','Lluvia al inicio de moderación nubosa','Chuva no início da moderação por nuvens'],
      orogen_temperature_clear_threshold:['Clear sky amplification below rain','Amplification par ciel clair sous cette pluie','晴空放大作用的降雨上限','Amplificación por cielo despejado bajo esta lluvia','Amplificação por céu limpo abaixo desta chuva'],
      orogen_temperature_seasonal_start:['Seasonal boost starts (°)','Début de l’amplification saisonnière (°)','季节性增强起点（°）','Inicio del aumento estacional (°)','Início do aumento sazonal (°)'],
      orogen_temperature_seasonal_peak:['Seasonal boost peak (°)','Maximum de l’amplification saisonnière (°)','季节性增强峰值（°）','Máximo del aumento estacional (°)','Pico do aumento sazonal (°)'],
      orogen_temperature_seasonal_fade:['Seasonal boost fades from (°)','Déclin de l’amplification saisonnière à partir de (°)','季节性增强衰减起点（°）','Descenso del aumento estacional desde (°)','Queda do aumento sazonal a partir de (°)'],
      orogen_temperature_ocean_speed:['Ocean heat speed sensitivity','Sensibilité de chaleur océanique à la vitesse','海洋热量速度敏感度','Sensibilidad del calor oceánico a la velocidad','Sensibilidade do calor oceânico à velocidade'],
      orogen_temperature_coast_limit:['Coastal continentality cutoff','Limite de continentalité côtière','沿岸大陆性截止值','Límite de continentalidad costera','Limite da continentalidade costeira'],
      orogen_precip_subtrop_summer_lat:['Summer subtropical center (°)','Centre subtropical estival (°)','夏季副热带中心（°）','Centro subtropical estival (°)','Centro subtropical de verão (°)'],
      orogen_precip_subtrop_winter_lat:['Winter subtropical center (°)','Centre subtropical hivernal (°)','冬季副热带中心（°）','Centro subtropical invernal (°)','Centro subtropical de inverno (°)'],
      orogen_precip_subtrop_summer_width:['Summer subtropical width (°)','Largeur subtropicale estivale (°)','夏季副热带宽度（°）','Anchura subtropical estival (°)','Largura subtropical de verão (°)'],
      orogen_precip_subtrop_winter_width:['Winter subtropical width (°)','Largeur subtropicale hivernale (°)','冬季副热带宽度（°）','Anchura subtropical invernal (°)','Largura subtropical de inverno (°)'],
      orogen_precip_itcz_core_width:['ITCZ core width (°)','Largeur du cœur de la ZCIT (°)','热带辐合带核心宽度（°）','Anchura del núcleo de la ZCIT (°)','Largura do núcleo da ZCIT (°)'],
      orogen_precip_itcz_outer_boost:['ITCZ outer rain boost','Renforcement des pluies en périphérie de la ZCIT','热带辐合带外围降雨增强','Aumento de lluvia exterior de la ZCIT','Aumento de chuva externa da ZCIT'],
      orogen_precip_monsoon_reach:['Monsoon coastal reach fraction','Fraction de portée côtière de la mousson','季风沿岸影响范围比例','Fracción de alcance costero de monzón','Fração de alcance costeiro da monção'],
      orogen_precip_monsoon_wind:['Monsoon wind threshold','Seuil de vent de mousson','季风风速阈值','Umbral de viento monzónico','Limiar de vento da monção'],
      orogen_precip_high_pressure_threshold:['High pressure rain threshold (hPa)','Seuil de pluie en haute pression (hPa)','高压降雨阈值（hPa）','Umbral de lluvia de alta presión (hPa)','Limiar de chuva de alta pressão (hPa)'],
      orogen_precip_low_pressure_threshold:['Low pressure rain threshold (hPa)','Seuil de pluie en basse pression (hPa)','低压降雨阈值（hPa）','Umbral de lluvia de baja presión (hPa)','Limiar de chuva de baixa pressão (hPa)'],
      orogen_precip_polar_start:['Polar rain starts (°)','Début des pluies polaires (°)','极地降雨起点（°）','Inicio de lluvia polar (°)','Início da chuva polar (°)'],
      orogen_precip_polar_full:['Polar rain full at (°)','Pluie polaire maximale à (°)','极地降雨最大值纬度（°）','Lluvia polar máxima en (°)','Chuva polar máxima em (°)'],
      orogen_precip_lee_height:['Lee cyclone minimum height (km)','Hauteur minimale du cyclone sous le vent (km)','背风气旋最低高度（km）','Altura mínima del ciclón de sotavento (km)','Altura mínima do ciclone de sotavento (km)'],
      orogen_precip_lee_height_full:['Lee cyclone full height (km)','Hauteur maximale du cyclone sous le vent (km)','背风气旋完全发展高度（km）','Altura plena del ciclón de sotavento (km)','Altura plena do ciclone de sotavento (km)'],
      orogen_precip_lee_gradient:['Lee cyclone slope threshold','Seuil de pente du cyclone sous le vent','背风气旋坡度阈值','Umbral de pendiente del ciclón de sotavento','Limiar de inclinação do ciclone de sotavento'],
      orogen_precip_shadow_height_start:['Shadow height ramp starts (km)','Début de la rampe de hauteur d’ombre (km)','雨影高度渐变起点（km）','Inicio de rampa de altura de sombra (km)','Início da rampa de altura da sombra (km)'],
      orogen_precip_shadow_height_range:['Shadow height ramp range (km)','Étendue de la rampe de hauteur d’ombre (km)','雨影高度渐变范围（km）','Alcance de rampa de altura de sombra (km)','Alcance da rampa de altura da sombra (km)'],
      orogen_precip_shadow_smoothing_km:['Rain shadow smoothing (km)','Lissage de l’ombre pluviométrique (km)','雨影平滑（km）','Suavizado de sombra pluviométrica (km)','Suavização da sombra de chuva (km)'],
      orogen_precip_shadow_propagated_scale:['Propagated shadow sensitivity','Sensibilité de l’ombre propagée','传播雨影敏感度','Sensibilidad de sombra propagada','Sensibilidade da sombra propagada'],
      orogen_precip_elevation_depletion_max:['Maximum altitude moisture loss','Perte maximale d’humidité par altitude','海拔导致的最大水汽损失','Pérdida máxima de humedad por altitud','Perda máxima de umidade pela altitude'],
      orogen_precip_ocean_warmth_min:['Minimum moisture warmth','Chaleur minimale pour l’humidité','水汽最低温度','Calor mínimo para humedad','Calor mínimo para umidade'],
      orogen_precip_land_drying_start:['Land drying starts at coverage','Début de l’assèchement selon la couverture terrestre','陆地干燥作用开始的覆盖率','Inicio de secado según cobertura terrestre','Início do ressecamento pela cobertura terrestre'],
      orogen_precip_land_drying_range:['Land drying coverage range','Plage de couverture de l’assèchement terrestre','陆地干燥作用覆盖范围','Rango de cobertura del secado terrestre','Faixa de cobertura do ressecamento terrestre'],
      orogen_precip_interior_cap_start:['Interior rain cap starts','Début du plafond de pluie intérieure','内陆降雨上限起点','Inicio del límite de lluvia interior','Início do limite de chuva interior'],
      orogen_precip_lee_reach_km:['Lee cyclone coastal reach (km)','Portée côtière du cyclone sous le vent (km)','背风气旋沿岸范围（km）','Alcance costero del ciclón de sotavento (km)','Alcance costeiro do ciclone de sotavento (km)'],
      orogen_heuristic_core_lat:['Heuristic core latitude','Latitude du cœur heuristique','启发式核心纬度','Latitud del núcleo heurístico','Latitude do núcleo heurístico'],
      orogen_heuristic_outer_lat:['Heuristic outer latitude','Latitude périphérique heuristique','启发式外围纬度','Latitud exterior heurística','Latitude externa heurística'],
      orogen_heuristic_desert_end_lat:['Heuristic desert end latitude','Latitude de fin du désert heuristique','启发式沙漠终点纬度','Latitud final del desierto heurístico','Latitude final do deserto heurístico'],
      orogen_heuristic_desert_core_lat:['Heuristic desert core latitude','Latitude du cœur désertique heuristique','启发式沙漠核心纬度','Latitud del núcleo desértico heurístico','Latitude do núcleo desértico heurístico'],
      orogen_heuristic_wet_mid_lat:['Heuristic wet mid latitude','Latitude humide moyenne heuristique','启发式湿润中纬度','Latitud media húmeda heurística','Latitude média úmida heurística'],
      orogen_heuristic_polar_lat:['Heuristic polar latitude','Latitude polaire heuristique','启发式极地纬度','Latitud polar heurística','Latitude polar heurística'],
      orogen_heuristic_core_rain:['Heuristic core rain','Pluie du cœur heuristique','启发式核心降雨','Lluvia del núcleo heurístico','Chuva do núcleo heurístico'],
      orogen_heuristic_outer_drop:['Heuristic outer drop','Baisse périphérique heuristique','启发式外围降幅','Descenso exterior heurístico','Queda externa heurística'],
      orogen_heuristic_desert_rain:['Heuristic desert rain','Pluie désertique heuristique','启发式沙漠降雨','Lluvia desértica heurística','Chuva desértica heurística'],
      orogen_heuristic_desert_drop:['Heuristic desert drop','Baisse désertique heuristique','启发式沙漠降幅','Descenso desértico heurístico','Queda desértica heurística'],
      orogen_heuristic_mid_base:['Heuristic mid-latitude baseline','Base heuristique de moyenne latitude','启发式中纬度基准','Base heurística de latitud media','Base heurística de latitude média'],
      orogen_heuristic_mid_gain:['Heuristic mid-latitude gain','Gain heuristique de moyenne latitude','启发式中纬度增益','Ganancia heurística de latitud media','Ganho heurístico de latitude média'],
      orogen_heuristic_subpolar_base:['Heuristic subpolar baseline','Base subpolaire heuristique','启发式副极地基准','Base subpolar heurística','Base subpolar heurística'],
      orogen_heuristic_subpolar_drop:['Heuristic subpolar drop','Baisse subpolaire heuristique','启发式副极地降幅','Descenso subpolar heurístico','Queda subpolar heurística'],
      orogen_heuristic_polar_base:['Heuristic polar baseline','Base polaire heuristique','启发式极地基准','Base polar heurística','Base polar heurística'],
      orogen_heuristic_polar_drop:['Heuristic polar drop','Baisse polaire heuristique','启发式极地降幅','Descenso polar heurístico','Queda polar heurística'],
      orogen_heuristic_coastal_smoothing_km:['Mediterranean coast influence (km)','Influence de la côte méditerranéenne (km)','地中海沿岸影响范围（km）','Influencia de la costa mediterránea (km)','Influência da costa mediterrânea (km)'],
      orogen_heuristic_med_start:['Heuristic Mediterranean start','Début méditerranéen heuristique','启发式地中海起点','Inicio mediterráneo heurístico','Início mediterrâneo heurístico'],
      orogen_heuristic_med_full:['Heuristic Mediterranean maximum','Maximum méditerranéen heuristique','启发式地中海最大值','Máximo mediterráneo heurístico','Máximo mediterrâneo heurístico'],
      orogen_heuristic_med_fade:['Heuristic Mediterranean fade','Déclin méditerranéen heuristique','启发式地中海衰减','Declive mediterráneo heurístico','Declínio mediterrâneo heurístico'],
      orogen_heuristic_med_end:['Heuristic Mediterranean end','Fin méditerranéenne heuristique','启发式地中海终点','Fin mediterráneo heurístico','Fim mediterrâneo heurístico'],
      orogen_heuristic_wind_itcz:['Heuristic ITCZ wind','Vent heuristique de la ZCIT','启发式热带辐合带风','Viento heurístico de la ZCIT','Vento heurístico da ZCIT'],
      orogen_heuristic_wind_trade_east:['Heuristic trade wind east','Composante est heuristique des alizés','启发式信风东向分量','Componente este heurística de alisios','Componente leste heurística dos alísios'],
      orogen_heuristic_wind_trade_north:['Heuristic trade wind north','Composante nord heuristique des alizés','启发式信风北向分量','Componente norte heurística de alisios','Componente norte heurística dos alísios'],
      orogen_heuristic_wind_westerly_east:['Heuristic westerly wind east','Composante est heuristique du vent d’ouest','启发式西风东向分量','Componente este heurística del viento occidental','Componente leste heurística do vento de oeste'],
      orogen_heuristic_wind_westerly_north:['Heuristic westerly wind north','Composante nord heuristique du vent d’ouest','启发式西风北向分量','Componente norte heurística del viento occidental','Componente norte heurística do vento de oeste'],
      orogen_heuristic_wind_polar_east:['Heuristic polar wind east','Composante est heuristique du vent polaire','启发式极地风东向分量','Componente este heurística del viento polar','Componente leste heurística do vento polar'],
      orogen_heuristic_wind_polar_north:['Heuristic polar wind north','Composante nord heuristique du vent polaire','启发式极地风北向分量','Componente norte heurística del viento polar','Componente norte heurística do vento polar'],
      orogen_koppen_shoulder_fraction:['Shoulder season interpolation','Interpolation de l’intersaison','过渡季插值','Interpolación de temporada intermedia','Interpolação da estação intermediária'],
      orogen_biome_low_height:['Lowland shading height (km)','Hauteur d’ombrage des plaines (km)','低地阴影高度（km）','Altura de sombreado de tierras bajas (km)','Altura de sombreamento das terras baixas (km)'],
      orogen_biome_rock_transition:['Fallback rock transition (km)','Transition de roche de repli (km)','备用岩石过渡高度（km）','Transición de roca alternativa (km)','Transição de rocha alternativa (km)'],
      orogen_itcz_ocean_lat:['ITCZ ocean latitude (°)','Latitude océanique de la ZCIT (°)','海洋热带辐合带纬度（°）','Latitud oceánica de la ZCIT (°)','Latitude oceânica da ZCIT (°)'],
      orogen_itcz_land_pull:['ITCZ land displacement (°)','Déplacement continental de la ZCIT (°)','陆地热带辐合带位移（°）','Desplazamiento continental de la ZCIT (°)','Deslocamento continental da ZCIT (°)'],
      orogen_itcz_elevation_pull:['ITCZ elevation displacement (°/km)','Déplacement de la ZCIT par altitude (°/km)','热带辐合带海拔位移（°/km）','Desplazamiento de la ZCIT por altitud (°/km)','Deslocamento da ZCIT por altitude (°/km)'],
      orogen_itcz_land_saturation:['ITCZ land saturation','Saturation continentale de la ZCIT','陆地热带辐合带饱和度','Saturación continental de la ZCIT','Saturação continental da ZCIT'],
      orogen_itcz_min_lat:['ITCZ minimum latitude (°)','Latitude minimale de la ZCIT (°)','热带辐合带最低纬度（°）','Latitud mínima de la ZCIT (°)','Latitude mínima da ZCIT (°)'],
      orogen_itcz_max_lat:['ITCZ maximum latitude (°)','Latitude maximale de la ZCIT (°)','热带辐合带最高纬度（°）','Latitud máxima de la ZCIT (°)','Latitude máxima da ZCIT (°)'],
      orogen_pressure_itcz_strength:['ITCZ low pressure (hPa)','Basse pression de la ZCIT (hPa)','热带辐合带低压（hPa）','Baja presión de la ZCIT (hPa)','Baixa pressão da ZCIT (hPa)'],
      orogen_pressure_itcz_width:['ITCZ pressure width (°)','Largeur de pression de la ZCIT (°)','热带辐合带气压宽度（°）','Anchura de presión de la ZCIT (°)','Largura de pressão da ZCIT (°)'],
      orogen_seasonal_shift:['Seasonal belt displacement (°)','Déplacement saisonnier des ceintures (°)','季节性气压带位移（°）','Desplazamiento estacional de cinturones (°)','Deslocamento sazonal das faixas (°)'],
      orogen_pressure_subtropical_strength:['Subtropical high pressure (hPa)','Haute pression subtropicale (hPa)','副热带高压（hPa）','Alta presión subtropical (hPa)','Alta pressão subtropical (hPa)'],
      orogen_pressure_land_weakening:['High pressure land weakening','Atténuation continentale des hautes pressions','陆地高压减弱','Debilitamiento continental de altas presiones','Enfraquecimento continental das altas pressões'],
      orogen_pressure_subpolar_strength:['Subpolar low pressure (hPa)','Basse pression subpolaire (hPa)','副极地低压（hPa）','Baja presión subpolar (hPa)','Baixa pressão subpolar (hPa)'],
      orogen_pressure_polar_strength:['Polar high pressure (hPa)','Haute pression polaire (hPa)','极地高压（hPa）','Alta presión polar (hPa)','Alta pressão polar (hPa)'],
      orogen_pressure_summer_land:['Summer continental low (hPa)','Dépression continentale estivale (hPa)','夏季大陆低压（hPa）','Baja continental de verano (hPa)','Baixa continental de verão (hPa)'],
      orogen_pressure_winter_land:['Winter continental high (hPa)','Anticyclone continental hivernal (hPa)','冬季大陆高压（hPa）','Alta continental de invierno (hPa)','Alta continental de inverno (hPa)'],
      orogen_pressure_elevation:['Pressure altitude effect (hPa/km)','Effet de l’altitude sur la pression (hPa/km)','海拔对气压的影响（hPa/km）','Efecto de altitud sobre la presión (hPa/km)','Efeito da altitude na pressão (hPa/km)'],
      orogen_pressure_noise:['Pressure noise (hPa)','Bruit de pression (hPa)','气压噪声（hPa）','Ruido de presión (hPa)','Ruído de pressão (hPa)'],
      orogen_geostrophic_angle:['Geostrophic deflection (°)','Déviation géostrophique (°)','地转偏转角（°）','Desviación geostrófica (°)','Desvio geostrófico (°)'],
      orogen_friction_angle:['Surface friction angle (°)','Angle de frottement de surface (°)','地表摩擦角（°）','Ángulo de fricción superficial (°)','Ângulo de atrito superficial (°)'],
      orogen_continentality_km:['Continentality range (km)','Portée de la continentalité (km)','大陆性范围（km）','Alcance de continentalidad (km)','Alcance da continentalidade (km)'],
      orogen_continentality_smoothing_km:['Continentality smoothing (km)','Lissage de la continentalité (km)','大陆性平滑（km）','Suavizado de continentalidad (km)','Suavização da continentalidade (km)'],
      orogen_pressure_smoothing_km:['Pressure smoothing (km)','Lissage de la pression (km)','气压平滑（km）','Suavizado de presión (km)','Suavização da pressão (km)'],
    };
    function installClimateControls(){
      if(extraFields.length||!schema?.properties)return;
      const properties=Object.entries(schema.properties).filter(([id,spec])=>id.startsWith('orogen_')&&spec.group);
      if(!properties.length)return;
      const section=document.createElement('details');section.id='orogenClimateParameters';section.className='advancedSettings';
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
      function translateClimate(){
        const index=Math.max(0,['en','fr','zh','es','pt'].indexOf(window.NeuralEarthLocale?.locale()||'en'));
        const t=name=>climateUi[name][index];
        section.querySelector(':scope > summary').textContent=t('parameters');
        const search=$('orogenClimateSearch');search.closest('label').firstChild.textContent=t('find')+' ';
        search.placeholder=t('placeholder');search.setAttribute('aria-label',t('searchLabel'));
        select.closest('label').firstChild.textContent=t('section')+' ';
        section.querySelector(':scope > small').textContent=t('help');
        for(const option of select.options)if(climateUi[option.value])option.textContent=t(option.value);
        for(const [id,spec] of extraFields){
          let labels=climateFieldText[id];
          if(!labels&&spec.group==='biome-palette'&&spec.color){
            const code=spec.label.match(/^([A-Za-z]+) color$/)?.[1];
            if(code)labels=[`${code} color`,`Couleur ${code}`,`${code} 颜色`,`Color ${code}`,`Cor ${code}`];
          }
          if(!labels)continue;
          const input=document.getElementById(id),label=input.closest('label');
          label.firstChild.textContent=labels[index];input.setAttribute('aria-label',labels[index]);
          const range=document.getElementById(id+'Slider');if(range)range.setAttribute('aria-label',labels[index]);
        }
        filter();
      }
      select.addEventListener('change',filter);$('orogenClimateSearch').addEventListener('input',filter);
      translateClimate();window.addEventListener('neural-locale-change',translateClimate);
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
    const generationMessages={
      sourceOrogen:['Orogen plates → initial relief. Then optional erosion and shared Orogen climate.','Plaques Orogen → relief initial. Puis érosion facultative et climat Orogen commun.','Orogen 板块 → 初始地形。之后可选侵蚀，并使用共享 Orogen 气候。','Placas Orogen → relieve inicial. Después, erosión opcional y clima Orogen común.','Placas Orogen → relevo inicial. Depois, erosão opcional e clima Orogen compartilhado.'],
      sourceContinents:['Procedural continents → initial relief. Then optional erosion and shared Orogen climate.','Continents procéduraux → relief initial. Puis érosion facultative et climat Orogen commun.','程序生成大陆 → 初始地形。之后可选侵蚀，并使用共享 Orogen 气候。','Continentes procedurales → relieve inicial. Después, erosión opcional y clima Orogen común.','Continentes procedurais → relevo inicial. Depois, erosão opcional e clima Orogen compartilhado.'],
      sourceAtlas:['Continental atlas → initial relief. Then optional erosion and shared Orogen climate.','Atlas continental → relief initial. Puis érosion facultative et climat Orogen commun.','大陆图集 → 初始地形。之后可选侵蚀，并使用共享 Orogen 气候。','Atlas continental → relieve inicial. Después, erosión opcional y clima Orogen común.','Atlas continental → relevo inicial. Depois, erosão opcional e clima Orogen compartilhado.'],
      sourceNoise:['Procedural noise → initial neural inputs. Then optional erosion and shared Orogen climate.','Bruit procédural → entrées neuronales initiales. Puis érosion facultative et climat Orogen commun.','程序噪声 → 初始神经网络输入。之后可选侵蚀，并使用共享 Orogen 气候。','Ruido procedural → entradas neuronales iniciales. Después, erosión opcional y clima Orogen común.','Ruído procedural → entradas neurais iniciais. Depois, erosão opcional e clima Orogen compartilhado.'],
      finite:['enter a finite number using a decimal dot (e.g. 0.125 or 1e-3).','saisissez un nombre fini avec un point décimal (ex. 0.125 ou 1e-3).','请输入有限数值，使用小数点（例如 0.125 或 1e-3）。','introduce un número finito con punto decimal (p. ej., 0.125 o 1e-3).','insira um número finito com ponto decimal (ex.: 0.125 ou 1e-3).'],
      integer:['enter an integer.','saisissez un nombre entier.','请输入整数。','introduce un número entero.','insira um número inteiro.'],
      positive:['enter a value greater than 0.','saisissez une valeur supérieure à 0.','请输入大于 0 的值。','introduce un valor mayor que 0.','insira um valor maior que 0.'],
      minimum:['enter a value of at least {value}.','saisissez une valeur supérieure ou égale à {value}.','请输入大于或等于 {value} 的值。','introduce un valor de al menos {value}.','insira um valor de pelo menos {value}.'],
      maximum:['enter a value of at most {value}.','saisissez une valeur inférieure ou égale à {value}.','请输入小于或等于 {value} 的值。','introduce un valor de como máximo {value}.','insira um valor de no máximo {value}.'],
      tabRelief:['Relief','Relief','地形','Relieve','Relevo'],tabErosion:['Erosion','Érosion','侵蚀','Erosión','Erosão'],tabClimate:['Climate','Climat','气候','Clima','Clima'],tabRendering:['Rendering','Rendu','渲染','Renderizado','Renderização'],
      generateRelief:['Generate relief','Générer le relief','生成地形','Generar relieve','Gerar relevo'],generateErosion:['Generate erosion','Générer l’érosion','生成侵蚀','Generar erosión','Gerar erosão'],generateClimate:['Generate climate','Générer le climat','生成气候','Generar clima','Gerar clima'],
      fullPipeline:['Generate full pipeline','Générer toute la chaîne','生成完整流程','Generar proceso completo','Gerar processo completo'],applySnr:['Apply SNR','Appliquer le SNR','应用 SNR','Aplicar SNR','Aplicar SNR'],resetSnr:['Reset SNR','Réinitialiser le SNR','重置 SNR','Restablecer SNR','Redefinir SNR'],
      stagesHelp:['Each tab generates its own stage. Full pipeline: relief, erosion, then climate. Soils use the current relief and climate.','Chaque onglet génère son étape. Chaîne complète : relief, érosion, puis climat. Les sols utilisent le relief et le climat actuels.','每个选项卡生成对应阶段。完整流程为地形、侵蚀、气候；土壤使用当前地形和气候。','Cada pestaña genera su etapa. Proceso completo: relieve, erosión y clima. Los suelos usan el relieve y clima actuales.','Cada aba gera sua etapa. Processo completo: relevo, erosão e clima. Os solos usam o relevo e clima atuais.'],
      meshHelp:['The mesh sets source relief and climate detail. Changing it requires generating relief; display LOD is set in Rendering.','Le maillage détermine le détail du relief et du climat source. Le modifier nécessite de générer le relief ; le LOD d’affichage se règle dans Rendu.','网格决定源地形和气候的细节。更改后需生成地形；显示 LOD 在渲染中设置。','La malla determina el detalle del relieve y clima de origen. Cambiarla requiere generar el relieve; el LOD de pantalla se ajusta en Renderizado.','A malha determina o detalhe do relevo e clima de origem. Alterá-la exige gerar o relevo; o LOD de exibição é configurado em Renderização.'],
      reliefHelp:['Generating relief retains previous erosion and climate results. The new raw relief becomes visible.','Générer le relief conserve les résultats précédents d’érosion et de climat. Le nouveau relief brut devient visible.','生成地形会保留先前的侵蚀和气候结果，并显示新的原始地形。','Generar el relieve conserva los resultados anteriores de erosión y clima. El nuevo relieve bruto se hace visible.','Gerar o relevo mantém os resultados anteriores de erosão e clima. O novo relevo bruto fica visível.'],
      erosionHelp:['Generating erosion uses the latest raw relief without accumulating previous passes. Climate is retained.','Générer l’érosion utilise le dernier relief brut sans accumuler les passes précédentes. Le climat est conservé.','生成侵蚀使用最新原始地形，不累积以前的处理；气候保持不变。','Generar erosión usa el último relieve bruto sin acumular pasadas anteriores. Se conserva el clima.','Gerar erosão usa o relevo bruto mais recente sem acumular passagens anteriores. O clima é mantido.'],
      climateHelp:['Generating climate uses the currently visible relief and preserves its elevations exactly.','Générer le climat utilise le relief actuellement visible et conserve exactement ses altitudes.','生成气候使用当前可见地形，并精确保留其高程。','Generar clima usa el relieve visible y conserva exactamente sus alturas.','Gerar clima usa o relevo visível e preserva exatamente suas altitudes.'],
      plates:['Orogen plates','Plaques Orogen','Orogen 板块','Placas Orogen','Placas Orogen'],shape:['Initial relief · shape','Relief initial · forme','初始地形 · 形状','Relieve inicial · forma','Relevo inicial · forma'],erosionOptions:['Orogen erosion','Érosion Orogen','Orogen 侵蚀','Erosión Orogen','Erosão Orogen'],offsets:['Climate offsets','Décalages climatiques','气候偏移','Desplazamientos climáticos','Desvios climáticos'],
      detailMode:['Detail settings','Réglages du détail','细节设置','Ajustes de detalle','Configurações de detalhe'],global:['Global','Global','全局','Global','Global'],perLod:['Per LOD','Par LOD','按 LOD','Por LOD','Por LOD'],adaptive:['Enable adaptive SNR','Activer le SNR adaptatif','启用自适应 SNR','Activar SNR adaptativo','Ativar SNR adaptativo'],
      generatorDraft:['Unapplied generator change','Changement de générateur non appliqué','生成器更改尚未应用','Cambio de generador sin aplicar','Alteração do gerador não aplicada'],
      sourceDraft:['Unapplied source change','Changement de source non appliqué','源更改尚未应用','Cambio de fuente sin aplicar','Alteração da fonte não aplicada'],
      draft:['Unapplied changes','Modifications non appliquées','更改尚未应用','Cambios sin aplicar','Alterações não aplicadas'],
      snrDraft:['Unapplied SNR preset','Préréglage SNR non appliqué','SNR 预设尚未应用','Preajuste SNR sin aplicar','Predefinição SNR não aplicada'],
      computing:['Computing…','Calcul en cours…','正在计算…','Calculando…','Calculando…'],
      allGenerated:['Relief, erosion and climate generated.','Relief, érosion et climat générés.','地形、侵蚀和气候已生成。','Relieve, erosión y clima generados.','Relevo, erosão e clima gerados.'],
      reliefGenerated:['Relief generated. Other stages are retained.','Relief généré. Les autres étapes sont conservées.','地形已生成，其他阶段保持不变。','Relieve generado. Se conservan las demás etapas.','Relevo gerado. As demais etapas são mantidas.'],
      erosionGenerated:['Erosion generated. Other stages are retained.','Érosion générée. Les autres étapes sont conservées.','侵蚀已生成，其他阶段保持不变。','Erosión generada. Se conservan las demás etapas.','Erosão gerada. As demais etapas são mantidas.'],
      climateGenerated:['Climate generated. Other stages are retained.','Climat généré. Les autres étapes sont conservées.','气候已生成，其他阶段保持不变。','Clima generado. Se conservan las demás etapas.','Clima gerado. As demais etapas são mantidas.'],
      settingsGenerated:['SNR applied. Other stages are retained.','SNR appliqué. Les autres étapes sont conservées.','SNR 已应用，其他阶段保持不变。','SNR aplicado. Se conservan las demás etapas.','SNR aplicado. As demais etapas são mantidas.'],
      restoreGenerated:['Variant loaded. Other stages are retained.','Variante chargée. Les autres étapes sont conservées.','变体已加载，其他阶段保持不变。','Variante cargada. Se conservan las demás etapas.','Variante carregada. As demais etapas são mantidas.'],
      notApplied:['Settings were not applied.','Les réglages n’ont pas été appliqués.','设置未应用。','No se aplicaron los ajustes.','As configurações não foram aplicadas.'],
      resetting:['Resetting all settings…','Réinitialisation de tous les réglages…','正在重置所有设置…','Restableciendo todos los ajustes…','Redefinindo todas as configurações…'],
      resetDone:['All generation settings have been reset.','Tous les réglages de génération ont été réinitialisés.','所有生成设置已重置。','Se restablecieron todos los ajustes de generación.','Todas as configurações de geração foram redefinidas.'],
      resetFailed:['Settings reset; generation was not applied.','Réglages réinitialisés ; génération non appliquée.','设置已重置；生成未应用。','Ajustes restablecidos; no se aplicó la generación.','Configurações redefinidas; geração não aplicada.'],
      savedA:['Settings A saved','Réglages A enregistrés','设置 A 已保存','Ajustes A guardados','Configurações A salvas'],
      noStage:['Not generated','Non généré','未生成','No generado','Não gerado'],
      staleInput:['Update needed: input changed.','Mise à jour nécessaire : entrée modifiée.','需要更新：输入已更改。','Actualización necesaria: entrada modificada.','Atualização necessária: entrada alterada.'],
      staleSettings:['Update needed: settings changed.','Mise à jour nécessaire : réglages modifiés.','需要更新：设置已更改。','Actualización necesaria: ajustes modificados.','Atualização necessária: configurações alteradas.'],
      generated:['Generated','Généré','已生成','Generado','Gerado'],
      custom:['Custom settings','Réglages personnalisés','自定义设置','Ajustes personalizados','Configurações personalizadas'],
      default:['Default profile','Profil par défaut','默认配置','Perfil predeterminado','Perfil padrão'],
      showA:['Show A','Afficher A','显示 A','Mostrar A','Mostrar A'],showB:['Show B','Afficher B','显示 B','Mostrar B','Mostrar B'],
    };
    const generationFieldText={
      snrAltitudeLow:['Altitude ramp starts (m)','Début de la rampe d’altitude (m)','海拔渐变起点（m）','Inicio de rampa de altitud (m)','Início da rampa de altitude (m)'],snrAltitudeHigh:['Altitude ramp ends (m)','Fin de la rampe d’altitude (m)','海拔渐变终点（m）','Fin de rampa de altitud (m)','Fim da rampa de altitude (m)'],
      snrDriverChannel:['Other driver','Autre variable','其他驱动变量','Otra variable','Outra variável'],snrDriverLow:['Driver ramp starts','Début de la rampe de variable','驱动变量渐变起点','Inicio de rampa de variable','Início da rampa da variável'],snrDriverHigh:['Driver ramp ends','Fin de la rampe de variable','驱动变量渐变终点','Fin de rampa de variable','Fim da rampa da variável'],snrBins:['Ramp levels','Niveaux de rampe','渐变级数','Niveles de rampa','Níveis da rampa'],
      snrLatitudeGain:['Latitude multiplier','Multiplicateur de latitude','纬度倍数','Multiplicador de latitud','Multiplicador de latitude'],snrLatitudeLow:['Latitude ramp starts (°)','Début de la rampe de latitude (°)','纬度渐变起点（°）','Inicio de rampa de latitud (°)','Início da rampa de latitude (°)'],snrLatitudeHigh:['Latitude ramp ends (°)','Fin de la rampe de latitude (°)','纬度渐变终点（°）','Fin de rampa de latitud (°)','Fim da rampa de latitude (°)'],generationPreset:['SNR preset','Préréglage SNR','SNR 预设','Preajuste SNR','Predefinição SNR'],
      orogen_plate_count:['Plate count','Nombre de plaques','板块数量','Número de placas','Número de placas'],orogen_continent_count:['Continent count','Nombre de continents','大陆数量','Número de continentes','Número de continentes'],
      orogen_land_coverage:['Land coverage','Couverture terrestre','陆地覆盖率','Cobertura terrestre','Cobertura terrestre'],orogen_continent_variety:['Continent variety','Variété des continents','大陆多样性','Variedad de continentes','Variedade dos continentes'],
      orogen_motion_strength:['Motion strength','Intensité du mouvement','运动强度','Intensidad del movimiento','Intensidade do movimento'],orogen_convergence_threshold:['Convergence threshold','Seuil de convergence','汇聚阈值','Umbral de convergencia','Limiar de convergência'],
      orogen_detail:['Mesh vertices','Sommets du maillage','网格顶点','Vértices de malla','Vértices da malha'],orogen_spread:['Orogen propagation','Propagation Orogen','Orogen 传播','Propagación Orogen','Propagação Orogen'],
      orogen_roughness:['Roughness','Rugosité','粗糙度','Rugosidad','Rugosidade'],orogen_warp:['Terrain warp','Déformation du terrain','地形扭曲','Deformación del terreno','Distorção do terreno'],orogen_smoothing:['Smoothing','Lissage','平滑','Suavizado','Suavização'],
      orogen_hydraulic:['Hydraulic erosion','Érosion hydraulique','水力侵蚀','Erosión hidráulica','Erosão hidráulica'],orogen_thermal:['Thermal erosion','Érosion thermique','热侵蚀','Erosión térmica','Erosão térmica'],orogen_glacial:['Glacial erosion','Érosion glaciaire','冰川侵蚀','Erosión glaciar','Erosão glacial'],orogen_sharpening:['Ridge sharpening','Accentuation des crêtes','山脊锐化','Acentuación de crestas','Realce das cristas'],
      orogen_temperature_offset:['Temperature offset (°C)','Décalage de température (°C)','温度偏移（°C）','Desplazamiento de temperatura (°C)','Desvio de temperatura (°C)'],orogen_precipitation_offset:['Precipitation offset','Décalage des précipitations','降水偏移','Desplazamiento de precipitación','Desvio de precipitação'],
      city_erosion_strength:['Dose','Intensité','强度','Intensidad','Intensidade'],city_erosion_iterations:['Iterations','Itérations','迭代次数','Iteraciones','Iterações'],city_erosion_talus:['Talus slope','Pente de talus','堆积坡坡度','Pendiente de talud','Inclinação do talude'],city_erosion_motif_km:['Drainage scale (km)','Échelle de drainage (km)','排水尺度（km）','Escala de drenaje (km)','Escala de drenagem (km)'],
      orogen_gpu_relief:['Relief · plates, collisions, noise','Relief · plaques, collisions, bruit','地形 · 板块、碰撞、噪声','Relieve · placas, colisiones, ruido','Relevo · placas, colisões, ruído'],orogen_gpu_propagation:['Propagation · stress and distances','Propagation · contraintes et distances','传播 · 应力和距离','Propagación · tensión y distancias','Propagação · tensão e distâncias'],orogen_gpu_post:['Orogen postprocessing · local passes','Post-traitement Orogen · passes locales','Orogen 后处理 · 局部处理','Posprocesado Orogen · pasadas locales','Pós-processamento Orogen · passagens locais'],orogen_gpu_climate:['Climate · winds, currents, rain, temperature','Climat · vents, courants, pluie, température','气候 · 风、洋流、降雨、温度','Clima · vientos, corrientes, lluvia, temperatura','Clima · ventos, correntes, chuva, temperatura'],orogen_gpu_raster:['Projection · GPU spatial search','Projection · recherche spatiale GPU','投影 · GPU 空间搜索','Proyección · búsqueda espacial GPU','Projeção · busca espacial GPU'],
    };
    const translateGeneration=key=>generationMessages[key][Math.max(0,['en','fr','zh','es','pt'].indexOf(window.NeuralEarthLocale?.locale()||'en'))];
    let draftMessage=null,stageMessages={},summaryMessage='default';
    function setDraft(key){draftMessage=key;$('generationDraftStatus').textContent=key?translateGeneration(key):'';}
    function rawDraft(message){draftMessage=null;$('generationDraftStatus').textContent=message;}
    function refreshSourceHelp(){
      const height=$('heightSource').value,type=classify({height_source:height});
      const key=type==='orogen'?'sourceOrogen':type==='custom'?(height==='natural-continental'?'sourceContinents':'sourceAtlas'):'sourceNoise';
      $('generationSourceHelp').textContent=translateGeneration(key);
    }
    function refreshGenerationMessages(){
      if(draftMessage)$('generationDraftStatus').textContent=translateGeneration(draftMessage);
      for(const [name,key] of Object.entries(stageMessages)){const target=$('stageStatus'+name);if(target)target.textContent=translateGeneration(key);}
      $('generationSummary').textContent=translateGeneration(summaryMessage);
      $('switchGenerationAB').textContent=translateGeneration(showingA?'showB':'showA');
      refreshSourceHelp();
      for(const [id,key] of Object.entries({tabRelief:'tabRelief',tabErosion:'tabErosion',tabClimate:'tabClimate',tabRendering:'tabRendering',generateRelief:'generateRelief',generateErosion:'generateErosion',generateClimate:'generateClimate',applyGeneration:'fullPipeline',applySnr:'applySnr',generationStagesHelp:'stagesHelp',generationMeshHelp:'meshHelp'})){
        const element=$(id);if(element)element.textContent=translateGeneration(key);
      }
      const resetSnr=noiseBody.querySelector('[data-shared-action="resetGeneration"]');if(resetSnr)resetSnr.textContent=translateGeneration('resetSnr');
      for(const [pane,key] of Object.entries({paneRelief:'reliefHelp',paneErosion:'erosionHelp',paneClimate:'climateHelp'})){
        const element=document.querySelector('[data-stage-help="'+pane+'"]');if(element)element.textContent=translateGeneration(key);
      }
      for(const [title,key] of Object.entries({'Orogen plates':'plates','Initial relief · shape':'shape','Erosion Orogen':'erosionOptions','Climate offsets':'offsets'})){
        const element=document.querySelector('[data-generation-section="'+title+'"]');if(element)element.textContent=translateGeneration(key);
      }
      const index=Math.max(0,['en','fr','zh','es','pt'].indexOf(window.NeuralEarthLocale?.locale()||'en'));
      for(const [id,values] of Object.entries(generationFieldText)){
        const input=$(id);if(!input)continue;const label=input.closest('label'),textNode=[...label.childNodes].find(node=>node.nodeType===Node.TEXT_NODE);
        if(textNode)textNode.textContent=values[index]+' ';input.setAttribute('aria-label',values[index]);
        const range=$(id+'Slider');if(range)range.setAttribute('aria-label',values[index]);
      }
      const channelNames=[['Elevation','Altitude','海拔','Elevación','Elevação'],['Temperature','Température','温度','Temperatura','Temperatura'],['Temperature variation','Variation de température','温度变化','Variación de temperatura','Variação de temperatura'],['Precipitation','Précipitations','降水','Precipitación','Precipitação'],['Precipitation variation','Variation des précipitations','降水变化','Variación de precipitación','Variação de precipitação']];
      const fieldNames={frequency:['Frequency','Fréquence','频率','Frecuencia','Frequência'],octaves:['Octaves','Octaves','倍频程','Octavas','Oitavas'],snr:['Allowed noise (SNR)','Bruit autorisé (SNR)','允许的噪声（SNR）','Ruido permitido (SNR)','Ruído permitido (SNR)'],snrAltitude:['Mountain noise multiplier','Multiplicateur de bruit en montagne','山地噪声倍数','Multiplicador de ruido de montaña','Multiplicador de ruído de montanha'],snrDriver:['Climate noise multiplier','Multiplicateur de bruit climatique','气候噪声倍数','Multiplicador de ruido climático','Multiplicador de ruído climático']};
      for(const [prefix,names] of Object.entries(fieldNames))for(let i=0;i<5;i++){
        const input=$(prefix+i),label=input?.closest('label');if(!label)continue;label.firstChild.textContent=names[index]+' ';input.setAttribute('aria-label',names[index]+' · '+channelNames[i][index]);
        const range=$(prefix+i+'Slider');if(range)range.setAttribute('aria-label',names[index]+' · '+channelNames[i][index]);
      }
      for(const selectId of ['sourceChannel','generationChannel']){const select=$(selectId);if(select)for(const option of select.options)option.textContent=channelNames[Number(option.value)]?.[index]||option.textContent;}
      const optionText={custom:['Custom','Personnalisé','自定义','Personalizado','Personalizado'],current:['Current profile defaults','Valeurs par défaut du profil actuel','当前配置默认值','Valores predeterminados del perfil actual','Padrões do perfil atual'],mountains:['Mountain detail · experimental','Détail montagne · expérimental','山地细节 · 实验性','Detalle de montaña · experimental','Detalhe de montanha · experimental'],glacial:['Cold-region detail · experimental','Détail des régions froides · expérimental','寒冷地区细节 · 实验性','Detalle de regiones frías · experimental','Detalhe de regiões frias · experimental']};
      for(const option of $('generationPreset').options)if(optionText[option.value])option.textContent=optionText[option.value][index];
      const adaptive=$('snrAdaptive');if(adaptive){const node=[...adaptive.closest('label').childNodes].find(item=>item.nodeType===Node.TEXT_NODE);if(node)node.textContent=' '+translateGeneration('adaptive');}
      const detail=$('snrDetailMode');if(detail){detail.closest('label').firstChild.textContent=translateGeneration('detailMode')+' ';detail.querySelector('[value="global"]').textContent=translateGeneration('global');detail.querySelector('[value="per-lod"]').textContent=translateGeneration('perLod');}
    }
    window.addEventListener('neural-locale-change',refreshGenerationMessages);
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
      refreshSourceHelp();

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
      write(settings);saveGeneration(draftBase,settings);$('generationPreset').value='custom';setDraft('generatorDraft');
    }
    $('generatorType').onchange=()=>{setGenerator($('generatorType').value)};
    $('customSource').onchange=()=>{$('heightSource').value=$('customSource').value;activeControls();setDraft('sourceDraft');};
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
      if(!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(text)||!Number.isFinite(value))throw Error(`${label}: ${translateGeneration('finite')}`);
      if(integer&&!Number.isInteger(value))throw Error(`${label}: ${translateGeneration('integer')}`);
      if(positive&&value<=0)throw Error(`${label}: ${translateGeneration('positive')}`);
      if(minimum!==undefined&&value<minimum)throw Error(`${label}: ${translateGeneration('minimum').replace('{value}',minimum)}`);
      if(maximum!==undefined&&value>maximum)throw Error(`${label}: ${translateGeneration('maximum').replace('{value}',maximum)}`);
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
    for(const draftPanel of [panel,climatePanel,noisePanel])draftPanel.addEventListener('input',event=>{if(['generatorType','customSource','sourceChannel','generationPreset','generationChannel','orogenClimateSection','orogenClimateSearch'].includes(event.target.id)||event.target.type==='range')return;if(event.target.id==='continentalStyle'){const values={gondwana:[80,1,.3],continents:[80,3,.3],earthlike:[80,4,.3],archipelago:[110,8,.18]}[$('continentalStyle').value];['orogen_plate_count','orogen_continent_count','orogen_land_coverage'].forEach((id,i)=>$(id).value=values[i]);if($('heightSource').value==='natural'){const preset={gondwana:[.5,3,.55],continents:[.8,4,.5],earthlike:[1,4,.5],archipelago:[1.6,5,.35]}[$('continentalStyle').value];['frequency0','octaves0','dropWater'].forEach((id,i)=>$(id).value=preset[i]);}}activeControls();setDraft('draft');$('generationPreset').value='custom'});
    $('generationPreset').onchange=()=>{
      const preset=$('generationPreset').value;if(preset==='custom')return;
      const settings=read(),baseline=defaults(draftBase);
      for(const key of ['cond_snr','snr_adaptive_enabled','snr_detail_mode','snr_altitude_gain','snr_altitude_range_m','snr_driver_channel','snr_driver_gain','snr_driver_range','snr_bins','snr_latitude_gain','snr_latitude_range','snr_lod'])settings[key]=clone(baseline[key]);
      if(preset==='mountains')settings.snr_altitude_gain=[4,1,1,1,1];
      if(preset==='glacial')settings.snr_driver_gain=[3,1,1,1,1];
      write(settings);setDraft('snrDraft');
    };
    let applySequence=0;
    async function submit(settings,profile,stage='all',options={}){
      const sequence=++applySequence;setDraft('computing');const success=await onApply(settings,profile,stage,options);
      if(sequence!==applySequence)return false;
      if(success){
        if(stage!=='all'&&stage!=='restore'){
          const applied=currentSnapshot().settings;
          const changed=Object.fromEntries((schema?.stage_groups?.[stage]||[]).map(key=>[key,applied[key]]));
          write({...settings,...changed,...stageReferences});
        }
        setDraft(stage==='all'?'allGenerated':stage+'Generated');
      }
      else if(draftMessage==='computing')setDraft('notApplied');
      return !!success;
    }
    async function action(callback){
      for(const id of ['saveGenerationA','switchGenerationAB'])$(id).disabled=true;
      try{return await callback()}catch(error){rawDraft(error.message);return false}
      finally{for(const id of ['applyGeneration','resetGeneration','saveGenerationA','generateRelief','generateErosion','generateClimate','applySnr'])$(id).disabled=false;$('switchGenerationAB').disabled=!savedA;}
    }
    $('applyGeneration').onclick=()=>action(()=>submit(read(),draftBase));
    $('resetGeneration').onclick=()=>action(async()=>{
      const sequence=++applySequence;
      const baseline=defaults(draftBase);
      // Restore every draft control immediately, even if generation fails or
      // a previous input is invalid. Color values and their RGB data reset together.
      stageReferences={};write(baseline);$('generationPreset').value='current';
      setDraft('resetting');
      const success=await onReset(draftBase);
      if(sequence!==applySequence)return false;
      setDraft(success?'resetDone':'resetFailed');
      return success;
    });
    function comparisonSnapshot(){const snapshot=clone(currentSnapshot());snapshot.settings={...(snapshot.settings||defaults(snapshot.profile)),...stageReferences};return snapshot;}
    $('saveGenerationA').onclick=()=>{savedA=comparisonSnapshot();savedB=null;showingA=false;$('switchGenerationAB').disabled=false;$('switchGenerationAB').textContent=translateGeneration('showA');setDraft('savedA')};
    $('switchGenerationAB').onclick=()=>action(async()=>{if(!savedA)return false;const candidateB=comparisonSnapshot(),target=showingA?savedB:savedA;
      if(!await submit(target.settings,target.profile,'restore',{seed:target.seed}))return false;
      if(!showingA)savedB=candidateB;showingA=!showingA;$('switchGenerationAB').textContent=translateGeneration(showingA?'showB':'showA');return true;});
    installTabs();
    refreshGenerationMessages();
    for(const draftPanel of [panel,climatePanel,noisePanel]){
      for(const event of ['input','change'])draftPanel.addEventListener(event,()=>{
        try{saveGeneration(draftBase,read())}catch(_){} // Keep the last valid draft while editing a number.
      });
    }
    return {sync(profile,settings,metadata,custom=false,stages={}){if(metadata)schema=metadata;installClimateControls();base=draftBase=profile;
      stageReferences={...Object.fromEntries(['relief','erosion','climate'].map(name=>['orogen_'+name+'_stage',stages[name]?.id||settings?.['orogen_'+name+'_stage']||''])),orogen_height_stage:stages.height_stage||settings?.orogen_height_stage||'erosion'};
      write(settings||defaults(profile));for(const name of ['relief','erosion','climate']){const state=stages[name];stageMessages[name]=!state?'noStage':state.stale?'staleInput':state.settings_pending?'staleSettings':'generated';$('stageStatus'+name).textContent=translateGeneration(stageMessages[name]);}
      $('generationPreset').value=custom?'custom':'current';summaryMessage=custom?'custom':'default';$('generationSummary').textContent=translateGeneration(summaryMessage);setDraft(null);},read,defaults,setGenerator,getGenerator:()=>$('generatorType').value,draftSnapshot:()=>({profile:draftBase,settings:read()}),showError(message){panel.open=true;rawDraft(message)}};

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
      const help=document.createElement('small');help.id='generationStagesHelp';help.textContent='Each tab generates its own stage. Full pipeline: relief, erosion, then climate. Soils use the current relief and climate.';actions.before(help);body.append(actions);
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
      const meshHelp=document.createElement('small');meshHelp.id='generationMeshHelp';meshHelp.textContent='The mesh sets source relief and climate detail. Changing it requires generating relief; display LOD is set in Rendering.';sourceSection.append(meshHelp);
      for(const [pane,text] of [[stagePanes.Relief,'Generating relief retains previous erosion and climate results. The new raw relief becomes visible.'],[stagePanes.Erosion,'Generating erosion uses the latest raw relief without accumulating previous passes. Climate is retained.'],[stagePanes.Climate,'Generating climate uses the currently visibrelief and preserves its elevations exactly.']]){const note=document.createElement('small');note.dataset.stageHelp=pane.id;note.textContent=text;pane.prepend(note);note.after(pane.querySelector('.stageActions'));}
      const snrActions=noiseBody.querySelector('.generationActions');
      const apply=snrActions.querySelector('[data-shared-action=applyGeneration]');apply.id='applySnr';apply.textContent='Apply SNR';apply.onclick=()=>action(()=>submit(read(),draftBase,'settings'));
      const reset=snrActions.querySelector('[data-shared-action=resetGeneration]');reset.textContent='Reset SNR';reset.onclick=()=>{noisePreset.value='current';noisePreset.dispatchEvent(new Event('change',{bubbles:true}));};
    }
  }
  return {mount,defaults,classify,initializePreferences,savePreferences,saveGeneration,rememberWorld,withGpu};
})();
