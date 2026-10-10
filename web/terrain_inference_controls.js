window.TerrainInferenceControls=(()=>{
  const choices=[1,2,4,8,16];
  const gpuModes=[['auto','Auto (recommended)'],['single','One GPU'],['throughput','All GPUs · max throughput'],['consistent','Two GPUs · consistent']];
  const gpuHelp={
    auto:'Uses every eligible GPU (BF16, at least 10 GB) for maximum throughput; otherwise the best GPU alone.',
    single:'The best GPU only, as on a single-GPU machine.',
    throughput:'Coarse windows stay on the coarse GPU. Latent/decoder tiles are shared between GPUs by blocks of 2×2 tiles. Tiles from different GPU models may differ by a few metres at block borders.',
    consistent:'Coarse on one GPU, latent/decoder on the other: each window always comes from the same GPU. Uses a different coarse GPU, hence a new world cache.'};
  function gib(bytes){return Math.round(bytes/1024**3)+' GB'}
  function mountGpus(panel,{fetchGpu=(path,options)=>fetch(path,options),reload=()=>location.reload()}={}){
    const section=document.createElement('div');
    section.id='gpuSettings';
    section.innerHTML='<label>GPUs <select id="gpuMode" aria-label="GPU mode"></select></label><ul id="gpuDevices" aria-label="GPU devices"></ul><small id="gpuHelp"></small><small id="gpuStatus" role="status"></small><button id="gpuRestart" type="button" hidden>Restart server to apply</button>';
    panel.append(section);
    const select=section.querySelector('#gpuMode'),list=section.querySelector('#gpuDevices'),help=section.querySelector('#gpuHelp'),status=section.querySelector('#gpuStatus'),restart=section.querySelector('#gpuRestart');
    for(const [value,label] of gpuModes)select.add(new Option(label,value));
    let state=null,busy=false,version=0;
    function render(data){
      state=data;
      const locked=new Set(data.environment||[]);
      select.value=data.settings.mode;
      select.disabled=busy||locked.has('mode')||data.devices.length<2;
      help.textContent=gpuHelp[data.settings.mode]||'';
      list.replaceChildren(...data.devices.map(device=>{
        const item=document.createElement('li');
        const box=document.createElement('input');
        box.type='checkbox';box.dataset.uuid=device.uuid;box.checked=device.enabled;
        // The primary GPU always runs; ineligible devices never join a multi-GPU plan.
        box.disabled=busy||device.primary||!device.eligible||locked.has('devices');
        box.title=device.primary?'Primary GPU, always used':device.eligible?'Use this GPU for detail tiles':'Needs BF16 (compute capability 8.0+) and 10 GB';
        box.setAttribute('aria-label','Use '+device.name);
        box.addEventListener('change',()=>{
          const enabled=[...list.querySelectorAll('input[data-uuid]')].filter(b=>b.checked).map(b=>b.dataset.uuid);
          apply({devices:enabled.length===data.devices.length?'auto':enabled});
        });
        const roles=device.roles.length?device.roles.join(' + '):'idle';
        const work=device.roles.length?` · ${device.tiles} tile${device.tiles===1?'':'s'}`+(device.current_job?' · busy':''):'';
        const loading=device.preload&&!['ready','released'].includes(device.preload.state)&&device.preload.state!=='idle'?` · ${device.preload.state}`:'';
        const label=document.createElement('label');
        label.append(box,` ${device.name} · ${gib(device.memory_bytes)} — ${roles}${work}${loading}`);
        item.append(label);return item;
      }));
      const plan=data.plan;
      restart.hidden=!data.restart_required;
      restart.disabled=busy||!data.restart_available;
      if(data.restart_required)status.textContent=data.restart_available?'Saved · restart the server to change the coarse GPU':'Saved · restart the server manually to apply';
      else status.textContent=(plan.mode==='single'?'Active · one GPU':`Active · ${plan.devices.length} GPUs (${plan.mode})`)+(plan.reason&&plan.requested_mode!==plan.mode?' · '+plan.reason:'')+(locked.size?' · set by server environment':'');
    }
    async function request(options){
      const token=++version;
      try{
        const response=await fetchGpu('/api/gpu',options);
        const data=await response.json();
        if(token!==version)return null;
        if(!response.ok)throw new Error(data.error||'GPU settings unavailable');
        render(data);return data;
      }catch(error){if(token===version){if(state)render(state);status.textContent=error.message}return null}
    }
    async function apply(change){
      busy=true;if(state)render(state);status.textContent='Applying GPU settings…';
      try{await request({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(change)})}
      finally{busy=false;if(state)render(state)}
    }
    select.addEventListener('change',()=>apply({mode:select.value}));
    restart.addEventListener('click',async()=>{
      const previous=state&&state.pid;busy=true;render(state);status.textContent='Restarting the server…';
      try{
        const response=await fetchGpu('/api/server/restart',{method:'POST'});
        if(!response.ok)throw new Error((await response.json()).error||'Restart refused');
        const deadline=Date.now()+180000;
        while(Date.now()<deadline){
          await new Promise(resolve=>setTimeout(resolve,1000));
          try{const r=await fetchGpu('/api/gpu');if(r.ok){const data=await r.json();if(data.pid!==previous){reload();return}}}catch(_){}
        }
        throw new Error('The server did not come back; see server-error.log');
      }catch(error){busy=false;if(state)render(state);status.textContent=error.message}
    });
    request();
    return {sync:()=>request(),observe:data=>{if(data&&!busy&&data.settings)render(data)},get:()=>state};
  }
  function mount({fetchSettings,onChange=()=>{},fetchGpu,reload}){
    const panel=document.createElement('fieldset');
    panel.id='inferenceSettings';
    panel.innerHTML='<legend>Neural computation</legend><label>Streams coarse <select id="coarseStreams" aria-label="Streams coarse"></select></label><small>4 is the recommended balance. Higher counts may speed up background preparation, using more memory and longer work groups. Shared across tabs using this engine.</small><small id="coarseStreamsStatus" role="status"></small>';
    const container=document.getElementById('paneRendering')||document.querySelector('#renderPanel .toolPanel');
    container.prepend(panel);
    const select=panel.querySelector('select'),status=panel.querySelector('[role=status]');
    for(const count of choices)select.add(new Option(String(count),String(count)));
    let current=4,version=0;
    select.value=String(current);
    async function sync(desired){
      const token=++version;
      select.disabled=true;status.textContent='Applying settings…';
      try{
        const options=desired===undefined?undefined:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({coarse:desired})};
        const response=await fetchSettings('/api/inference/streams',options);
        const data=await response.json();
        if(token!==version)return;
        if(!response.ok||!choices.includes(data.coarse))throw new Error(data.error||'Settings unavailable');
        current=data.coarse;select.value=String(current);status.textContent='Active · '+current+' stream'+(current>1?'s':'');onChange(current);
      }catch(error){if(token===version){select.value=String(current);status.textContent=error.message;}}
      finally{if(token===version)select.disabled=false;}
    }
    select.addEventListener('change',()=>sync(Number(select.value)));
    const query=Number(new URLSearchParams(location.search).get('coarse_streams'));
    sync(choices.includes(query)?query:undefined);
    // GPU settings belong to the main server, even when the reference engine is displayed.
    const gpus=mountGpus(panel,{fetchGpu,reload});
    return {sync:count=>sync(count),get:()=>current,observe:(value,runtime)=>{
      if(select.disabled||!choices.includes(value))return;
      if(value!==current){current=value;select.value=String(value);onChange(value);}
      status.textContent=runtime&&runtime.requested===value&&runtime.effective!==value
        ?'Settings '+value+' · using '+runtime.effective+' stream (available mode)'
        :'Active · '+value+' stream'+(value>1?'s':'');
    },observeGpus:gpus.observe,gpus};
  }
  return {mount,mountGpus};
})();
