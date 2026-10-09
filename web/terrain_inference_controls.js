window.TerrainInferenceControls=(()=>{
  const choices=[1,2,4,8,16];
  function mount({fetchSettings,onChange=()=>{}}){
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
    return {sync:()=>sync(),get:()=>current,observe:(value,runtime)=>{
      if(select.disabled||!choices.includes(value))return;
      if(value!==current){current=value;select.value=String(value);onChange(value);}
      status.textContent=runtime&&runtime.requested===value&&runtime.effective!==value
        ?'Settings '+value+' · using '+runtime.effective+' stream (available mode)'
        :'Active · '+value+' stream'+(value>1?'s':'');
    }};
  }
  return {mount};
})();
