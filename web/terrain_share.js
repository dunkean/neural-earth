/* NE1 codes keep saved generation/rendering recipes short and the camera editable. */
window.TerrainShare=(()=>{
  const uid=/^[gr][A-Za-z0-9_-]{18}$/;
  const number=value=>{if(value===''||!Number.isFinite(Number(value)))throw Error('Invalid camera number');return Number(value)};
  function parse(text){
    text=text.trim();
    if(/^https?:\/\//.test(text)){const url=new URL(text),p=url.searchParams;p.set('recipe',new URLSearchParams(url.hash.slice(1)).get('recipe')||p.get('recipe')||'');return fromParams(p);}
    const [head,...fields]=text.split(/\s+/),parts=head.split(':');
    if(parts.length!==3||parts[0]!=='NE1')throw Error('Paste an NE1 share code or a share link');
    const params=new URLSearchParams({share:parts[1]+'.'+parts[2]});
    for(const field of fields){const at=field.indexOf('=');if(at<1)throw Error('Invalid view field');const key=field.slice(0,at);if(params.has(key))throw Error('Duplicate view field');params.set(key,field.slice(at+1));}
    return fromParams(params);
  }
  function fromParams(params){
    const parts=(params.get('share')||'').split('.');
    if(parts.length!==2||!uid.test(parts[0])||parts[0][0]!=='g'||!uid.test(parts[1])||parts[1][0]!=='r')throw Error('Invalid share UID');
    const layer=params.get('layer'),view=params.get('view'),zoom=number(params.get('zoom')??''),position=(params.get('position')||'').split(',').map(number),size=(params.get('size')||'').split(',').map(number),lod=params.get('lod');
    if(!/^[a-z0-9-]+$/.test(layer||'')||!['map','globe'].includes(view)||zoom<=0||position.length!==2||size.length!==2||size.some(n=>!Number.isInteger(n)||n<1||n>16384)||!(lod==='auto'||Number.isInteger(number(lod??''))&&Number(lod)>=-3&&Number(lod)<=11))throw Error('Invalid shared view');
    const orbit=view==='globe'?(params.get('orbit')||'').split(',').map(number):null;
    if(orbit&&(orbit.length!==3||orbit[2]<1e-9||orbit[2]>11||Math.abs(orbit[1])>Math.PI/2))throw Error('Invalid globe camera');
    const recipe=params.get('recipe')||null;if(recipe&&(!/^[A-Za-z0-9_-]+$/.test(recipe)||recipe.length>174764))throw Error('Invalid portable recipe');
    const pins=params.get('pins')||'',pairs=pins?pins.split(';').map(pair=>pair.split(',').map(number)):[];
    if(pairs.length>256||pairs.some(pair=>pair.length!==2))throw Error('Invalid pin coordinates');
    const latentGeometry=Number(params.get('latent_geometry')||0);if(![0,1,2].includes(latentGeometry))throw Error('Invalid latent geometry');
    return {generation:parts[0],rendering:parts[1],layer,view,zoom,position,size,lod,orbit,recipe,pins,latentGeometry};
  }
  function params(state){
    const p=new URLSearchParams({share:state.generation+'.'+state.rendering,layer:state.layer,view:state.view,zoom:String(state.zoom),position:state.position.join(','),size:state.size.join(','),lod:String(state.lod)});
    if(state.orbit)p.set('orbit',state.orbit.join(','));if(state.pins)p.set('pins',state.pins);if(state.latentGeometry)p.set('latent_geometry',state.latentGeometry);return p;
  }
  function format(state){return 'NE1:'+state.generation+':'+state.rendering+' '+[...params(state)].filter(([key])=>key!=='share').map(([key,value])=>key+'='+value).join(' ')+(state.recipe?' recipe='+state.recipe:'')}
  function url(state){return '/?'+params(state).toString().replace(/%2C/gi,',').replace(/%3B/gi,';')+(state.recipe?'#recipe='+state.recipe:'')}
  async function request(path,options){const response=await fetch(path,options),data=await response.json();if(!response.ok)throw Error(data.error||'Sharing unavailable');return data;}
  function mount({capture,canShare}){
    const panel=document.createElement('details');panel.id='sharePanel';
    panel.innerHTML='<summary aria-label="Share view" title="Share or open an exact view">Share</summary><div class="toolPanel"><div class="panelHeading"><strong>Share this view</strong><button type="button" class="panelClose" aria-label="Close share panel">×</button></div><p>One portable code restores generation, rendering options and the camera on another installation. Keep the recipe field when sharing between servers. Zoom is metres per pixel; position is world metres.</p><label>Generation UID <input id="generationUid" readonly style="width:205px"></label><textarea id="shareCode" aria-label="Share code" rows="5" style="width:100%;resize:vertical;background:#10191e;color:inherit"></textarea><div class="panelActions"><button id="copyShare" type="button">Copy code</button><button id="copyShareLink" type="button">Copy link</button></div><label for="openShareCode">Open a code or link</label><textarea id="openShareCode" aria-label="Code or link to open" rows="3" style="width:100%;resize:vertical;background:#10191e;color:inherit"></textarea><div class="panelActions"><button id="openShare" type="button">Open view</button><button id="unlockShareSize" type="button" hidden>Use full window</button></div><small id="shareStatus" role="status"></small></div>';
    document.getElementById('controls').append(panel);
    const compatibility=document.createElement('p');compatibility.id='shareCompatibility';compatibility.setAttribute('role','status');panel.querySelector('.toolPanel').append(compatibility);
    const $=id=>document.getElementById(id),cache=new Map();let timer=null,sequence=0,lastScheduled=null;
    async function save(){
      if(!canShare())throw Error('Wait for generation to finish before sharing');
      const snapshot=capture(),key=JSON.stringify([snapshot.generation,snapshot.rendering]);
      let recipes=cache.get(key);
      if(!recipes){recipes=await request('/api/share',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({generation:snapshot.generation,rendering:snapshot.rendering})});cache.set(key,recipes);if(cache.size>16)cache.delete(cache.keys().next().value);}
      return {...snapshot.camera,...recipes};
    }
    async function publish(copy){
      const token=++sequence;
      try{
        const state=await save();if(!copy&&token!==sequence)return;
        const link=new URL(url(state),location.origin).href,code=format(state);
        $('generationUid').value=state.generation;$('shareCode').value=code;
        history.replaceState({terrainLocal:true},'',link);
        if(copy){try{await navigator.clipboard.writeText(copy==='link'?link:code);$('shareStatus').textContent='Copied '+(copy==='link'?'link':'code');}catch(_){$('shareCode').focus();$('shareCode').select();$('shareStatus').textContent='Select and copy the code above';}}
        else $('shareStatus').textContent='View saved';
      }catch(error){$('shareStatus').textContent=error.message;}
    }
    $('copyShare').onclick=()=>publish('code');$('copyShareLink').onclick=()=>publish('link');
    $('openShare').onclick=()=>{try{const state=parse($('openShareCode').value);location.assign(url(state));}catch(error){$('shareStatus').textContent=error.message;}};
    panel.addEventListener('toggle',()=>{if(panel.open)publish();});
    return {updateURL(){if(!canShare())return;const key=JSON.stringify(capture());if(key===lastScheduled)return;lastScheduled=key;clearTimeout(timer);++sequence;timer=setTimeout(()=>publish(),200);},save,format,params};
  }
  return {parse,fromParams,format,params,url,mount,load:state=>state.recipe?request('/api/share/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({generation:state.generation,rendering:state.rendering,recipe:state.recipe})}):request('/api/share/'+state.generation+'/'+state.rendering)};
})();
