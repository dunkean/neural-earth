window.TerrainInferenceControls=(()=>{
  const choices=[1,2,4,8,16];
  function mount({fetchSettings,onChange=()=>{}}){
    const panel=document.createElement('fieldset');
    panel.id='inferenceSettings';
    panel.innerHTML='<legend>Calcul neuronal</legend><label>Streams coarse <select id="coarseStreams" aria-label="Streams coarse"></select></label><small>4 : compromis conseillé. Un nombre plus élevé peut accélérer la préparation de fond, avec davantage de mémoire et des groupes plus longs. Réglage partagé entre les onglets utilisant ce moteur.</small><small id="coarseStreamsStatus" role="status"></small>';
    const container=document.getElementById('paneRendering')||document.querySelector('#renderPanel .toolPanel');
    container.prepend(panel);
    const select=panel.querySelector('select'),status=panel.querySelector('[role=status]');
    for(const count of choices)select.add(new Option(String(count),String(count)));
    let current=4,version=0;
    select.value=String(current);
    async function sync(desired){
      const token=++version;
      select.disabled=true;status.textContent='Application du réglage…';
      try{
        const options=desired===undefined?undefined:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({coarse:desired})};
        const response=await fetchSettings('/api/inference/streams',options);
        const data=await response.json();
        if(token!==version)return;
        if(!response.ok||!choices.includes(data.coarse))throw new Error(data.error||'Réglage indisponible');
        current=data.coarse;select.value=String(current);status.textContent='Actif · '+current+' stream'+(current>1?'s':'');onChange(current);
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
        ?'Réglage '+value+' · calcul à '+runtime.effective+' stream (mode disponible)'
        :'Actif · '+value+' stream'+(value>1?'s':'');
    }};
  }
  return {mount};
})();
