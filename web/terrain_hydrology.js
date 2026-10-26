/* Geographic hydrology overlay and inspector. Geometry belongs to the server's
 * physical network; zoom only changes projection and visibility, never seeds. */
(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.TerrainHydrology = api;
})(typeof window === 'undefined' ? globalThis : window, function() {
  'use strict';
  // Optional server work, all off by default: terrainDetail is the slow regional
  // terrain fit; the two cosmetics reshape the drawn course on the server.
  // Generated meanders were removed: old URLs and recipes asking for them are ignored.
  // One live viewer controller owns display geometry; reset/dispose never leave
  // a previous world's network available to the GPU renderer.
  let displayController=null;
  const riverDisplay=()=>displayController?.read()||null;
  const regionalRiverDisplayEnabled=()=>!!displayController?.enabled();
  const acknowledgeRiverDisplay=(key,rects,surfaceKey=key)=>displayController?.ack(key,rects,surfaceKey);
  const COSMETICS = Object.freeze(['valleyFit','smoothBanks']);
  const COSMETIC_LABELS = {valleyFit:'Fit course to valley floor (slower)', smoothBanks:'Limit bend radius to 1.5 widths (slower)'};
  // Display-only settings: [min, max, default]. They filter and style the
  // stored network and never reach the server or a cache key.
  // Lakes under 1 km² are mostly filled Base pits of a few 240 m cells.
  const RANGES = Object.freeze({density:[0,2,1], minArea:[0,1e7,0], widthScale:[.5,2,1], minLakeArea:[0,1e7,1], minLakeDepth:[0,1e4,0]});
  const FLAGS = Object.freeze({intermittent:false, lakes:true, smoothShores:false});
  const DEFAULTS = Object.freeze({enabled:true, month:0, diagnostic:'water', terrainDetail:false,
    ...Object.fromEntries(COSMETICS.map(name=>[name,false])),
    ...Object.fromEntries(Object.entries(RANGES).map(([name,r])=>[name,r[2]])), ...FLAGS});
  // URL names of the display settings; only non-default values are written.
  const URL_NAMES = {density:'hydroDensity', minArea:'hydroMinArea', widthScale:'hydroWidth', minLakeArea:'hydroLakeArea', minLakeDepth:'hydroLakeDepth',
    intermittent:'hydroIntermittent', lakes:'hydroLakes', smoothShores:'hydroShores'};
  const SHORE = '#1a4d5c', THIN = '#2d7894', LAKE = '#215a6d', DRY = '#9b7a50';
  // Inside ready physical patches, rivers at least this wide are left to the raster.
  const PHYSICAL_WIDTH = 2.5;
  // Cartographic hierarchy. Automatic threshold catchment
  // A = REFERENCE_AREA * (m/px / 120)^AREA_EXPONENT / density^1.5 (density
  // exponent as in Magna Urbis), calibrated on cached seed-42 and user views.
  // A reach shows when its mean flow reaches that of A at REFERENCE_RUNOFF and
  // it drains at least AREA_SHARE * A: dry climates need larger catchments,
  // inherited trunks stay visible, tributaries appear as the view zooms in.
  const REFERENCE_AREA = 250e6, AREA_EXPONENT = .85, REFERENCE_RUNOFF = .3/31557600, AREA_SHARE = .25;
  // Stroke: MIN_STROKE px at the threshold flow, +STROKE_PER_DECADE per decade
  // of flow above it, capped; a resolved channel keeps its physical width.
  const MIN_STROKE = .7, STROKE_PER_DECADE = .8, MAX_STROKE = 4.5;
  // Cartographic class from mean annual flow (m³/s), for the inspector.
  const CLASSES = [[100,'Main river'],[5,'River'],[.2,'Stream'],[0,'Rill']];
  const MONTHS = Array.from({length:12},(_,i)=>new Date(2000,i,1).toLocaleString('en',{month:'short'}));
  // Cached local routes follow terrain arrival, at most about twice a second.
  const REVISION_INTERVAL_MS = 500;
  function number(value,[lo,hi,fallback]) {
    const n=value===null||value===undefined||value===''?NaN:Number(value);
    return Number.isFinite(n) ? Math.max(lo,Math.min(hi,n)) : fallback;
  }
  function normalize(value={}) {
    return {enabled:value.enabled !== false,
      month:Number.isInteger(Number(value.month)) ? Math.max(0,Math.min(12,Number(value.month))) : 0,
      diagnostic:['water','basins','discharge','lowflow','navigation'].includes(value.diagnostic) ? value.diagnostic : 'water',
      terrainDetail:value.terrainDetail === true,
      ...Object.fromEntries(COSMETICS.map(name=>[name,value[name] === true])),
      ...Object.fromEntries(Object.entries(RANGES).map(([name,range])=>[name,number(value[name],range)])),
      ...Object.fromEntries(Object.entries(FLAGS).map(([name,on])=>[name,on ? value[name] !== false : value[name] === true]))};
  }
  function cosmeticList(value) {return COSMETICS.filter(name=>value[name]);}
  function fromParams(params) {
    const cosmetics=(params.get('hydroCosmetics')||'').split(','),flag=name=>{const v=params.get(URL_NAMES[name]);return v===null?undefined:v==='1';};
    return normalize({enabled:params.get('hydro') !== '0',month:Number(params.get('hydroMonth')||0),diagnostic:params.get('hydroView')||'water',
      terrainDetail:params.get('hydroDetail') === '1',...Object.fromEntries(COSMETICS.map(name=>[name,cosmetics.includes(name)])),
      ...Object.fromEntries(Object.keys(RANGES).map(name=>[name,params.get(URL_NAMES[name])])),
      ...Object.fromEntries(Object.keys(FLAGS).map(name=>[name,flag(name)]))});
  }
  function writeParams(params, value) {
    const s=normalize(value),cosmetics=cosmeticList(s);
    for(const name of ['hydro','hydroMonth','hydroView','hydroDetail','hydroCosmetics',...Object.values(URL_NAMES)]) params.delete(name);
    if (!s.enabled) params.set('hydro','0');
    if (s.month) params.set('hydroMonth',String(s.month));
    if (s.diagnostic !== 'water') params.set('hydroView',s.diagnostic);
    if (s.terrainDetail) params.set('hydroDetail','1');
    if (cosmetics.length) params.set('hydroCosmetics',cosmetics.join(','));
    for(const name of Object.keys(RANGES)) if(s[name]!==DEFAULTS[name]) params.set(URL_NAMES[name],String(s[name]));
    for(const name of Object.keys(FLAGS)) if(s[name]!==DEFAULTS[name]) params.set(URL_NAMES[name],s[name]?'1':'0');
  }
  // Server options shared by /api/hydrology and the physical tile requests.
  function requestParams(value) {
    const s=normalize(value),cosmetics=cosmeticList(s),out={};
    if (s.terrainDetail) out.hydro_detail='1';
    if (cosmetics.length) out.hydro_cosmetics=cosmetics.join(',');
    return out;
  }
  // Empty for the default fast network, so default display caches keep their keys.
  function optionsIdentity(value) {
    const s=normalize(value);
    return s.enabled ? Object.entries(requestParams(s)).map(([k,v])=>`/${k}-${v}`).join('') : '';
  }
  // Physical tile identity. Only detailed carving changes terrain (with its
  // month and cosmetics); the fast network is drawn as vectors over unchanged
  // tiles, so visibility, month and cosmetics never reach a fast tile key.
  function tileIdentity(value) {
    const s=normalize(value);
    return s.enabled&&s.terrainDetail ? `/hydro-${s.month}${optionsIdentity(s)}` : '';
  }
  // RGBA tile fields (level, coverage, class, authority): any authoritative
  // river coverage. A ready tile without it leaves every river to the vectors.
  function hasRiver(fields) {
    if(!fields) return false;
    for(let i=0;i<fields.length;i+=4) if(fields[i+2]>2.5&&fields[i+1]>0&&fields[i+3]>.5) return true;
    return false;
  }
  // V2 lake-only tiles also own their water appearance (including ice).
  // Keep hasRiver for existing consumers with its narrower meaning.
  function hasInlandWater(fields) {
    if(!fields) return false;
    for(let i=0;i<fields.length;i+=4) if(fields[i+2]>=1.5&&fields[i+1]>0&&fields[i+3]>.5) return true;
    return false;
  }
  function v2DisplayIdentity(value,month) {
    const s=normalize(value);if(!s.enabled)return '';
    return `/v2water-v2-month-${month}/lakes-${Number(s.lakes)}/minArea-${s.minLakeArea*1e6}`;
  }
  function v2TileParams(value,month) {
    const s=normalize(value);if(!s.enabled)return null;
    return {hydro:'1',hydro_month:String(month),v2_lakes:s.lakes?'1':'0',
      v2_lake_min_area_m2:String(s.minLakeArea*1e6),...requestParams(s)};
  }
  function contains(a,b) {return a&&b&&a[0]<=b[0]&&a[1]<=b[1]&&a[2]>=b[2]&&a[3]>=b[3];}
  function segmentDistance(p,a,b) {
    const dx=b[0]-a[0],dy=b[1]-a[1],d=dx*dx+dy*dy;
    const t=d?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/d)):0;
    return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy);
  }
  function monthlyFlow(reach,month) {
    const values=reach.monthly_flow_m3s;
    return month&&values?.length===12 ? Math.max(0,Number(values[month-1])||0) : Math.max(0,Number(reach.flow_m3s)||0);
  }
  // Flow regime. With a monthly series, the dry months are exactly those whose
  // flow is below DRY_Q (absolute: a large river at 1% of its mean still
  // flows). 0 dry months: perennial; 1-5: seasonal; 6 or more: ephemeral.
  // Twelve equal months are a flat fallback, not evidence: 'unresolved'.
  // Without a series, only a dry annual mean is classified; a backend
  // `dry_months` count is reported as a count, never as invented months.
  const DRY_Q = .01;
  function regime(reach) {
    const annual=Math.max(0,Number(reach.flow_m3s)||0),values=reach.monthly_flow_m3s;
    const count=Number.isInteger(reach.low_flow_months)?reach.low_flow_months:Number.isInteger(reach.dry_months)?reach.dry_months:null;
    if(values?.length===12) {
      const dry=values.map((q,i)=>Number(q)<DRY_Q?i+1:0).filter(Boolean);
      const flat=values.every(q=>Math.abs(Number(q)-Number(values[0]))<=1e-6*Math.max(1e-9,annual));
      return {regime:dry.length?dry.length<6?'seasonal':'ephemeral':flat?'unresolved':'perennial',dry,count};
    }
    if(annual<DRY_Q||reach.regime==='ephemeral'||reach.regime==='dry') return {regime:'ephemeral',dry:MONTHS.map((_,i)=>i+1),count};
    return {regime:'unresolved',dry:[],count};
  }
  // Navigation in `month`, derived as the server does from the reach's vessel
  // classes: the largest navigable that month, or all year for the annual
  // mean. The server's own value is used for the month it was computed for.
  function navigation(reach,month,served) {
    const nav=reach.navigation||{};
    if(month===served||!Array.isArray(nav.vessels)) return {navigable:!!(reach.navigable||nav.navigable),label:nav.label};
    let capability=month?-1:nav.year_round_class??-1;
    if(month) (reach.navigation_months||[]).forEach((flags,v)=>{if(flags?.[month-1])capability=v;});
    return {navigable:capability>=0,label:capability>=0?nav.vessels[capability]:'no continuous route'};
  }
  // A reply serves every month when its monthly navigation flags are complete;
  // flows are always selected here from monthly_flow_m3s.
  function servesMonth(network,month) {
    return Number(network.month??0)===month||(network.reaches||[]).every(r=>
      (!r.navigation||Array.isArray(r.navigation.vessels))&&(!r.navigation_months||r.navigation_months.every(f=>f?.length===12)));
  }
  function riverClass(annual) {return CLASSES.find(([q])=>annual>=q)[1];}
  // Display threshold catchment (m²) for a view at `mpp` metres per pixel.
  function thresholdArea(value,mpp) {
    const s=normalize(value);
    return s.minArea>0 ? s.minArea*1e6 : REFERENCE_AREA*(Math.max(.01,mpp)/120)**AREA_EXPONENT/Math.max(.15,s.density)**1.5;
  }
  // Per-network index, built once per reply: records sorted by mean flow (the
  // largest drawn last, on top), grouped by reach id (fragments and connectors
  // share their parent's id), with the downstream group of each group.
  function prepare(network) {
    const entries=[],groups=new Map();
    for(const reach of network.reaches||[]) {
      const xy=reach.xy||reach.points||[];if(xy.length<2) continue;
      const id=reach.id??`#${entries.length}`;let group=groups.get(id);
      if(group===undefined){group=groups.size;groups.set(id,group);}
      const area=Number(reach.area_m2);
      entries.push({reach,xy,group,annual:Math.max(0,Number(reach.flow_m3s)||0),area:Number.isFinite(area)?area:Infinity,...regime(reach)});
    }
    const down=new Int32Array(groups.size).fill(-1),flow=new Float64Array(groups.size),area=new Float64Array(groups.size),representatives=new Array(groups.size);
    for(const e of entries) {
      representatives[e.group]??=e.reach;
      const target=e.reach.display_downstream_id??e.reach.downstream_id;
      const d=groups.get(target)??groups.get(e.reach.downstream_id);
      if(d!==undefined&&d!==e.group) down[e.group]=d;
      flow[e.group]=Math.max(flow[e.group],e.annual);area[e.group]=Math.max(area[e.group],e.area);
    }
    entries.sort((a,b)=>a.annual-b.annual);
    return {network,entries,down,flow,area,representatives};
  }
  // Groups shown at a threshold: physical drainage, then every downstream
  // group, so a displayed river always continues to its outlet in the reply.
  function visibleGroups(index,area,qMin,settings,mpp) {
    const shown=new Uint8Array(index.down.length);
    for(let g=0;g<shown.length;g++) {
      if(index.flow[g]<qMin||index.area[g]<area*AREA_SHARE) continue;
      const r=index.representatives[g];
      // Continental maps need fewer source headwaters. Only the source order
      // is comparable here; local Base orders must never be compared with it.
      // Apply this to seeds, then retain their entire downstream continuation.
      if(!settings.minArea&&mpp>1000&&r.strahler_scale==='source'&&r.strahler<3&&
          (Number(r.width_m)||7.2*Math.sqrt(index.flow[g]))/mpp<.12/Math.max(.15,settings.density)**1.5) continue;
      for(let i=g;i>=0&&!shown[i];i=index.down[i]) shown[i]=1;
    }
    return shown;
  }
  function color(reach, diagnostic, month, served) {
    if(diagnostic==='basins') return `hsl(${((Number(reach.basin_id)||0)*137.508)%360} 58% 55%)`;
    if(diagnostic==='navigation') return navigation(reach,month,served).navigable ? '#75cba8' : '#ba936a';
    if(diagnostic==='discharge'||diagnostic==='lowflow') {
      const q=diagnostic==='lowflow' ? Math.min(...(reach.monthly_flow_m3s||[reach.flow_m3s||0])) : monthlyFlow(reach,month);
      return `hsl(${205-Math.min(1,Math.log1p(Math.max(0,q))/Math.log(10001))*55} 65% 55%)`;
    }
    return SHORE;
  }
  function formatFlow(q) {return `${q<.01?q.toPrecision(2):q<10?q.toFixed(2):q<100?q.toFixed(1):Math.round(q).toLocaleString('en')} m³/s`;}
  function km2(m2) {return `${(Number(m2||0)/1e6).toLocaleString('en',{maximumFractionDigits:m2<1e7?2:0})} km²`;}
  function monthList(months) {return months.length>4?`${months.length} months`:months.map(m=>MONTHS[m-1]).join(', ');}
  function regimeText(e) {
    if(e.regime==='unresolved') return `regime unresolved (no monthly variation${e.count?`; backend reports ${e.count} low-flow months`:''})`;
    return e.regime==='perennial'?'perennial':`${e.regime}, below ${DRY_Q} m³/s in ${monthList(e.dry)}`;
  }
  // Screen continuity between consecutive vertices. On the map, a step drawn
  // much longer than its wrapped world length crosses the longitude seam; long
  // coarse segments at fine zoom stay continuous. The globe hides its far side
  // (null), so only a screen-size limit applies there.
  function stepper(state,mpp) {
    const period=state.wrapLongitude&&state.worldBounds?state.worldBounds[2]-state.worldBounds[0]:0,limit=Math.max(state.width,state.height)*1.5;
    if(state.view==='globe') return (a,b,qa,qb)=>Math.hypot(qb[0]-qa[0],qb[1]-qa[1])<=limit;
    return (a,b,qa,qb)=>{
      let dx=Math.abs(b[0]-a[0]);if(period){dx%=period;dx=Math.min(dx,period-dx);}
      return Math.hypot(qb[0]-qa[0],qb[1]-qa[1])<=1.5*Math.hypot(dx,b[1]-a[1])/mpp+4;
    };
  }
  // Projects a ring like the basin outline; null when a vertex is hidden or wraps.
  function projectRing(ring,state,continuous) {
    if(!Array.isArray(ring)||ring.length<3) return null;
    const out=[];
    for(let i=0;i<ring.length;i++){const q=state.project(ring[i][0],ring[i][1]);if(!q||i&&!continuous(ring[i-1],ring[i],out[i-1],q))return null;out.push(q);}
    return out;
  }
  function traceRing(ctx,p,smooth) {
    if(!smooth){ctx.moveTo(p[0][0],p[0][1]);for(let i=1;i<p.length;i++)ctx.lineTo(p[i][0],p[i][1]);ctx.closePath();return;}
    // Quadratic curves through edge midpoints: a cheap rounded shore, no resampling.
    const mid=(a,b)=>[(a[0]+b[0])/2,(a[1]+b[1])/2],start=mid(p[p.length-1],p[0]);
    ctx.moveTo(start[0],start[1]);
    for(let i=0;i<p.length;i++){const m=mid(p[i],p[(i+1)%p.length]);ctx.quadraticCurveTo(p[i][0],p[i][1],m[0],m[1]);}
    ctx.closePath();
  }
  function insideRings(x,y,rings) {
    let inside=false;
    for(const p of rings)for(let i=0,j=p.length-1;i<p.length;j=i++)
      if((p[i][1]>y)!==(p[j][1]>y)&&x<(p[j][0]-p[i][0])*(y-p[i][1])/(p[j][1]-p[i][1])+p[i][0]) inside=!inside;
    return inside;
  }
  function create({getState, requestFrame=()=>{}, onChange=()=>{}, fetchJSON=null, initial=null}={}) {
    if(typeof getState!=='function') throw Error('Hydrology needs a geographic view state');
    let settings=normalize(initial||fromParams(new URLSearchParams(typeof location==='undefined'?'':location.search)));
    // `identity` is the world of the stored network; `networkOptions` and
    // `networkMonth` are the server options and month it was requested with.
    let network=null, index=null, identity=null, networkOptions=null, networkMonth=null, queryBounds=null, queryLod=Infinity, queryRevision=null, refreshAt=0, controller=null, timer=null, fetching=false, nextAttempt=0;
    let networkRevision=0,displayAck=null,displayCache=null;
    let status='idle', failure=null, panel=null, statusLabel=null, inspector=null, selected=null, selection=null, catchment=null, inspection=0, shownCount=0;
    const projected=[], lakeShapes=[];
    const fetcher=fetchJSON||((url,options)=>fetch(url,options));
    function cancel() {
      controller?.abort(); controller=null; fetching=false; nextAttempt=0;
      if(timer!==null) clearTimeout(timer); timer=null;
    }
    function reset() {
      cancel(); network=null; index=null; identity=null; networkOptions=null; networkMonth=null; queryBounds=null; queryLod=Infinity; queryRevision=null; refreshAt=0; selected=null; selection=null;
      status='idle';displayAck=null;displayCache=null;networkRevision++; catchment=null;inspection++;projected.length=0;lakeShapes.length=0;shownCount=0;
      if(inspector)inspector.textContent='Click a river or lake to inspect it.';
    }
    function showSelection() {if(inspector&&selection)inspector.textContent=describe(selection.entry)+selection.suffix;}
    // Server options refetch in the background while the stored network of
    // the same world stays drawn; the month and display settings only redraw.
    function set(value) {
      const before=settings; settings=normalize({...settings,...value});
      if(optionsIdentity(before)!==optionsIdentity(settings)||!settings.enabled) {cancel(); failure=null;}
      if(before.month!==settings.month) showSelection();
      syncControls(); onChange({...settings},before); requestFrame();
    }
    function syncStatus() {
      if(statusLabel) statusLabel.textContent=!settings.enabled?'Hidden':failure?`Hydrology: ${failure}`:
        status==='building'?'Preparing river network…':network?`${shownCount} of ${network.reaches?.length||0} river reaches shown${network.lakes?.length?` · ${network.lakes.length} lakes`:''}${network.regional?.errors?.length?' · some fine detail unavailable':''}`:'River network loads with the map';
    }
    function syncControls() {
      if(!panel) return;
      for(const input of panel.querySelectorAll('[data-hydro]')) {
        const name=input.dataset.hydro;
        if(input.type==='checkbox') input.checked=settings[name]; else input.value=String(settings[name]);
      }
      for(const out of panel.querySelectorAll('[data-hydro-value]')) out.textContent=`${settings[out.dataset.hydroValue].toFixed(2)}×`;
      panel.querySelector('[data-hydro="density"]').disabled=settings.minArea>0;
      syncStatus();
    }
    function mount(parent) {
      if(!parent||panel) return;
      const box=(name,label,title='')=>`<label${title?` title="${title}"`:''}>${label} <input data-hydro="${name}" type="checkbox"></label>`;
      const range=(name,label,step,title)=>`<label title="${title}">${label} <input data-hydro="${name}" type="range" min="${RANGES[name][0]}" max="${RANGES[name][1]}" step="${step}"> <output data-hydro-value="${name}"></output></label>`;
      const field=(name,label,step,title)=>`<label title="${title}">${label} <input data-hydro="${name}" type="number" min="0" step="${step}" style="width:6em"></label>`;
      panel=document.createElement('details'); panel.id='hydrologySettings';
      panel.innerHTML='<summary>Water</summary><div class="toolPanel"><div class="panelHeading"><strong>Rivers</strong><button type="button" class="panelClose" aria-label="Close water settings">×</button></div>'+
        box('enabled','Show water')+
        '<label>Flow month <select data-hydro="month"><option value="0">Annual mean</option>'+Array.from({length:12},(_,i)=>`<option value="${i+1}">${new Date(2000,i,1).toLocaleString('en',{month:'long'})}</option>`).join('')+'</select></label>'+
        '<label>Display <select data-hydro="diagnostic"><option value="water">Water</option><option value="basins">River basins</option><option value="discharge">Discharge</option><option value="lowflow">Climatological low flow</option><option value="navigation">Navigability estimate</option></select></label>'+
        range('density','Stream density',.05,'Scales the automatic catchment threshold as 1/density^1.5; 0 keeps main rivers only. Display only.')+
        field('minArea','Min catchment (km², 0 auto)',1,'Fixed display threshold instead of the zoom-dependent one; the mean flow must also match this catchment at 300 mm/yr of runoff.')+
        range('widthScale','Line width',.1,'Scales the cartographic stroke; resolved channels keep their physical width.')+
        box('intermittent','Seasonal and dry streams','Dashed blue: seasonal; dashed brown: dry in the selected month or most of the year.')+
        '<small>Lakes</small>'+box('lakes','Show lakes')+
        field('minLakeArea','Min lake area (km²)',.1,'Display filter; routing and water budgets are unchanged.')+
        field('minLakeDepth','Min lake depth (m)',.5,'Display filter on the lake maximum depth.')+
        box('smoothShores','Smooth shorelines','Rounds drawn shore corners; display only.')+
        '<small>Server work (slower)</small>'+
        box('terrainDetail','Detailed channel carving (slow)','Generates extra regional neural terrain and carves physical river channels into it; much slower')+
        COSMETICS.map(name=>box(name,COSMETIC_LABELS[name])).join('')+
        '<small>Local rivers refine as terrain loads.</small><small data-hydro-status style="display:block"></small><p data-hydro-inspector style="line-height:1.6;margin-bottom:4px">Click a river or lake to inspect it.</p><a href="/docs/methods/hydrology-rendering.md" target="_blank" rel="noopener">Hydrology methods</a></div>';
      parent.append(panel); statusLabel=panel.querySelector('[data-hydro-status]'); inspector=panel.querySelector('[data-hydro-inspector]');
      panel.querySelector('.panelClose').onclick=()=>{panel.open=false;};
      for(const input of panel.querySelectorAll('[data-hydro]')) {
        const name=input.dataset.hydro,read=()=>input.type==='checkbox'?input.checked:input.tagName==='SELECT'&&name==='diagnostic'?input.value:Number(input.value);
        input[input.type==='range'?'oninput':'onchange']=()=>set({[name]:read()});
      }
      syncControls();
    }
    function key(state) {return `${state.seed}/${state.profile}/${state.worldIdentity||''}`;}
    async function ensure(state) {
      if(!settings.enabled||!state||state.seed==null||!state.bounds||!state.worldBounds||state.engine==='wasm') return;
      const id=key(state),options=optionsIdentity(settings),month=settings.month;
      // Another world never inherits a network, current or stale.
      if(identity!==id) {reset(); identity=id;}
      if(fetching||Date.now()<nextAttempt) return;
      // The default network adds local routes from Base already loaded on the
      // server, so newly arrived terrain may refine a covered view. Coalesce
      // arrivals; one deferred frame catches the last one.
      const revision=state.terrainRevision??0;
      const current=network&&networkOptions===options&&(networkMonth===month||servesMonth(network,month));
      if(current&&contains(queryBounds,state.bounds)&&queryLod<=(state.lod??4)) {
        if(settings.terrainDetail||queryRevision===revision) return;
        const wait=refreshAt-Date.now();
        if(wait>0) {if(timer===null)timer=setTimeout(()=>{timer=null;requestFrame();},wait+10);return;}
      }
      // Pad only cheap overview queries; fine levels fetch exactly the view.
      const pad=(state.lod??4)<=3?0:.3,wb=state.worldBounds,b=state.bounds,px=(b[2]-b[0])*pad,py=(b[3]-b[1])*pad;
      const extent=[state.wrapLongitude?b[0]-px:Math.max(wb[0],b[0]-px),Math.max(wb[1],b[1]-py),state.wrapLongitude?b[2]+px:Math.min(wb[2],b[2]+px),Math.min(wb[3],b[3]+py)];
      if(extent[2]-extent[0]>=wb[2]-wb[0]){extent[0]=wb[0];extent[2]=wb[2];}
      if(extent[2]<=extent[0]||extent[3]<=extent[1]) return;
      controller=new AbortController(); const active=controller; fetching=true;
      const params=new URLSearchParams({world_profile:state.profile,bounds:extent.join(','),lod:String(state.lod??4),month:String(month)});
      if(state.worldIdentity) params.set('world_identity',state.worldIdentity);
      for(const [k,v] of Object.entries(requestParams(settings))) params.set(k,v);
      try {
        const response=await fetcher(`/api/hydrology/${state.seed}?${params}`,{signal:active.signal,cache:'no-cache'});
        if(!response.ok&&response.status!==202) throw Error(`unavailable (${response.status})`);
        const data=await response.json();
        if(identity!==id||controller!==active) return;
        if(data.status==='error') throw Error(data.error||'generation failed');
        status=data.status||'ready'; failure=null;
        // A building reply without reaches keeps the network already drawn.
        if(response.status===202||status==='building'||status==='pending') {
          if(data.reaches?.length){network=data;networkOptions=options;networkMonth=month;queryBounds=extent;queryLod=Infinity;queryRevision=revision;}nextAttempt=Date.now()+1200;
          timer=setTimeout(()=>{timer=null;requestFrame();},1250);
        } else {network=data;networkOptions=options;networkMonth=month;queryBounds=extent;queryLod=state.lod??4;queryRevision=revision;nextAttempt=0;refreshAt=Date.now()+REVISION_INTERVAL_MS;}
        networkRevision++;displayAck=null;displayCache=null;syncControls(); requestFrame();
      } catch(error) {
        if(error.name!=='AbortError'&&identity===id) {failure=error.message;status='error';nextAttempt=Date.now()+10000;syncControls();timer=setTimeout(()=>{timer=null;requestFrame();},10100);}
      } finally {if(controller===active){controller=null;fetching=false;}}
    }
    function stroke(ctx,segments) {
      for(const points of segments){ctx.beginPath();ctx.moveTo(...points[0]);for(let i=1;i<points.length;i++)ctx.lineTo(...points[i]);ctx.stroke();}
    }
    function projectRivers(state) {
      projected.length=0;
      if(!settings.enabled||!network||identity!==key(state)||typeof state.project!=='function') return;
      if(index?.network!==network) index=prepare(network);
      const s=settings,water=s.diagnostic==='water',mpp=Math.max(.01,Number(state.mpp)||30),continuous=stepper(state,mpp);
      const area=thresholdArea(s,mpp),qMin=area*REFERENCE_RUNOFF,shown=visibleGroups(index,area,qMin,s,mpp);
      for(const e of index.entries) {
        if(!shown[e.group]) continue;
        const reach=e.reach,dry=s.month?e.dry.includes(s.month):e.regime==='ephemeral',seasonal=!s.month&&e.regime==='seasonal';
        if((dry||seasonal)&&!s.intermittent) continue;
        const segments=[],levels=[];let run=[],runLevels=[];
        for(let i=0;i<e.xy.length;i++) {
          const p=e.xy[i],q=state.project(p[0],p[1]);
          if(!q||run.length&&!continuous(e.xy[i-1],p,run[run.length-1],q)) {if(run.length>1){segments.push(run);levels.push(runLevels);}run=[];runLevels=[];}
          if(q){run.push(q);runLevels.push(Number.isFinite(Number(reach.water_m?.[i]))?Number(reach.water_m[i]):0);}
        }
        if(run.length>1){segments.push(run);levels.push(runLevels);}
        if(!segments.length) continue;
        const annual=Math.max(1e-9,e.annual),q=monthlyFlow(reach,s.month);
        const fraction=Math.min(1.8,Math.pow(q/annual,.26));
        const physicalWidth=Math.max(.5,Number(reach.width_m)||Math.max(1,3*Math.sqrt(annual)))*fraction;
        const mid=e.xy[Math.floor(e.xy.length/2)],center=state.project(mid[0],mid[1]);
        const across=state.project(mid[0],mid[1]+physicalWidth);
        const pixelWidth=state.view==='globe'&&center&&across?Math.hypot(across[0]-center[0],across[1]-center[1]):physicalWidth/mpp;
        // Width and opacity grow monotonically with flow above the view threshold.
        const decades=Math.log10(Math.max(1,q/qMin));
        const carto=Math.min(MAX_STROKE,MIN_STROKE+STROKE_PER_DECADE*decades)*s.widthScale;
        const width=dry&&water?carto:Math.max(water?carto:Math.max(1.4,carto),pixelWidth);
        const thin=pixelWidth<width,alpha=dry?.55:water?Math.min(1,.6+.13*decades):q>0?1:.5;
        const dash=water?(dry?[3,3]:seasonal?[Math.max(4,3*width),Math.max(3,2*width)]:[]):q>0?[]:[3,4];
        const colour=water?(dry?DRY:thin?THIN:SHORE):color(reach,s.diagnostic,s.month,networkMonth);
        projected.push({reach,entry:e,segments,levels,dry,q,pixelWidth,width,alpha,dash,colour});
      }
    }
    function displayKey(state) {
      return JSON.stringify([key(state),networkRevision,settings.month,optionsIdentity(settings),settings.enabled,
        settings.diagnostic,settings.density,settings.minArea,settings.intermittent,state.bounds,state.mpp,state.width,state.height,state.lod]);
    }
    function readDisplay() {
      const state=getState();
      if(!state||!settings.enabled||!state.renderV2||state.view!=='map'||settings.terrainDetail||settings.diagnostic!=='water'||(state.lod??4)>3)return null;
      const id=displayKey(state),ready=!!(network&&identity===key(state)&&status==='ready'&&
        networkOptions===optionsIdentity(settings)&&(networkMonth===settings.month||servesMonth(network,settings.month))&&
        contains(queryBounds,state.bounds)&&queryLod<=(state.lod??4)&&queryRevision===(state.terrainRevision??0));
      if(displayCache?.key===id&&displayCache.ready===ready)return displayCache;
      const values=[];
      if(ready){
        projectRivers(state);
        for(const item of projected){
          if(item.dry||item.q<=0||!(item.pixelWidth>0))continue;
          for(let j=0;j<item.segments.length;j++){
            const points=item.segments[j],levels=item.levels[j];
            for(let i=1;i<points.length;i++){
              const a=points[i-1],b=points[i],half=item.pixelWidth*.5;
              if(Math.max(a[0],b[0])+half+1<0||Math.min(a[0],b[0])-half-1>state.width||
                 Math.max(a[1],b[1])+half+1<0||Math.min(a[1],b[1])-half-1>state.height)continue;
              values.push(a[0],a[1],b[0],b[1],half,levels[i-1],levels[i],0);
            }
          }
        }
      }
      return displayCache={key:id,ready,month:settings.month,networkRevision,hydroId:network?.hydro_id||null,
        width:state.width,height:state.height,segments:new Float32Array(values),authority:'same-selected-regional-vector-as-Render3D',
        terrainDetail:false};
    }
    const displayOwner={read:readDisplay,enabled:()=>{const s=getState();return !!(settings.enabled&&s?.renderV2&&s.view==='map'&&!settings.terrainDetail&&settings.diagnostic==='water');},ack:(key,rects,surfaceKey)=>{displayAck=surfaceKey?{key,rects,surfaceKey}:null;}};
    displayController=displayOwner;
    function draw(ctx,state) {
      ensure(state);lakeShapes.length=0;projectRivers(state);
      if(!settings.enabled||!network||identity!==key(state)||typeof state.project!=='function')return;
      const s=settings,water=s.diagnostic==='water',mpp=Math.max(.01,Number(state.mpp)||30),continuous=stepper(state,mpp);
      const fastDisplay=!!(state.renderV2&&state.view==='map'&&!s.terrainDetail&&water&&(state.lod??4)<=3&&displayAck?.surfaceKey===displayKey(state));
      const displayActive=fastDisplay&&displayAck?.key===displayKey(state);
      if(shownCount!==projected.length){shownCount=projected.length;syncStatus();}
      // One contract for every layer and the globe. Ready physical patches
      // already show resolved water, so vectors there only draw narrow
      // channels; elsewhere (previews, native coarse, fantasy) the vector is
      // the river. Same geometry on both sides of a patch edge.
      const paint=(physical)=>{
        if(physical&&state.renderV2&&!fastDisplay)return;
        for(const item of projected) {
          if(physical&&fastDisplay&&!item.dry)continue;
          if(physical&&!fastDisplay&&item.pixelWidth>=PHYSICAL_WIDTH) continue;
          ctx.strokeStyle=item.colour;ctx.lineWidth=item.width;ctx.globalAlpha=item.alpha;ctx.setLineDash(item.dash);
          stroke(ctx,item.segments);
        }
      };
      const rects=water&&state.view==='map'?(fastDisplay?(displayActive?displayAck.rects:[]):state.physicalRects||[]):[];
      ctx.save(); ctx.lineCap='round';ctx.lineJoin='round';
      if(water&&state.view!=='map') paint(!!state.physicalWater);
      else if(!rects.length) paint(false);
      else {
        ctx.save();ctx.beginPath();ctx.rect(0,0,state.width,state.height);for(const r of rects)ctx.rect(r.x,r.y,r.width,r.height);ctx.clip('evenodd');paint(false);ctx.restore();
        ctx.save();ctx.beginPath();for(const r of rects)ctx.rect(r.x,r.y,r.width,r.height);ctx.clip();paint(true);ctx.restore();
      }
      // Lakes are filled after rivers: the opaque surface hides inflow ends
      // and connector courses through the lake. Display filters only.
      if(state.renderV2&&water&&rects.length){ctx.beginPath();ctx.rect(0,0,state.width,state.height);for(const r of rects)ctx.rect(r.x,r.y,r.width,r.height);ctx.clip('evenodd');}
      if(s.lakes&&!(state.renderV2&&state.physicalWater)) for(const lake of network.lakes||[]) {
        if(Number(lake.area_m2)<s.minLakeArea*1e6||Number(lake.max_depth_m)<s.minLakeDepth) continue;
        const rings=[];for(const ring of lake.rings||[]){const p=projectRing(ring,state,continuous);if(p)rings.push(p);}
        if(!rings.length) continue;
        let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
        for(const p of rings[0]){x0=Math.min(x0,p[0]);x1=Math.max(x1,p[0]);y0=Math.min(y0,p[1]);y1=Math.max(y1,p[1]);}
        if(x1<0||y1<0||x0>state.width||y0>state.height||x1-x0<.8&&y1-y0<.8) continue;
        ctx.beginPath();for(const p of rings)traceRing(ctx,p,s.smoothShores);
        // A 19.5 km source shoreline at Base zoom is an approximate fallback:
        // outline only: an unknown shore must not flood the detailed terrain.
        const rough=lake.resolution==='source'&&(state.lod??4)<=3;
        ctx.setLineDash(rough?[4,3]:[]);
        if(!rough){ctx.globalAlpha=1;ctx.fillStyle=LAKE;ctx.fill('evenodd');}
        ctx.globalAlpha=rough?.6:.8;ctx.strokeStyle=rough?THIN:SHORE;ctx.lineWidth=1;ctx.stroke();
        lakeShapes.push({lake,rings});
      }
      if(settings.diagnostic==='basins'&&catchment?.rings?.length){
        ctx.beginPath();let visible=false;
        for(const ring of catchment.rings){const p=projectRing(ring,state,continuous);if(!p)continue;traceRing(ctx,p,false);visible=true;}
        if(visible){ctx.globalAlpha=.18;ctx.fillStyle='#73d5e6';ctx.fill('evenodd');ctx.globalAlpha=.9;ctx.strokeStyle='#9ce2ed';ctx.lineWidth=1.5;ctx.setLineDash([]);ctx.stroke();}
      }
      ctx.restore();
    }
    async function loadCatchment(reach){
      const state=getState(),ticket=++inspection;
      const parent=reach.parent_id??(typeof reach.id==='number'?reach.id:null);
      const p=new URLSearchParams({world_profile:state.profile});
      if(parent!==null)p.set('reach_id',String(parent));else p.set('basin_id',String(reach.basin_id));
      if(state.worldIdentity)p.set('world_identity',state.worldIdentity);
      try{
        const response=await fetcher(`/api/hydrology/${state.seed}?${p}`,{cache:'no-cache'});
        if(!response.ok)return;
        const data=await response.json();if(ticket!==inspection||data.status!=='ready')return;
        catchment=data;
        const label=reach.kind==='tributary'&&reach.parent_id!=null?'parent river watershed':'catchment';
        if(selection?.entry.reach===reach){selection.suffix=` · ${label} ${km2(data.area_m2)} · drains to ${data.outlet_kind}`;showSelection();}
        requestFrame();
      }catch(_){/* The reach inspector remains usable during a world switch. */}
    }
    // Text for a drawn reach in the selected month: class, regime, flows, drainage.
    function describe(e) {
      const r=e.reach,q=monthlyFlow(r,settings.month),nav=navigation(r,settings.month,networkMonth),low=Math.min(...(r.monthly_flow_m3s?.length===12?r.monthly_flow_m3s.map(Number):[e.annual]));
      const lowMonth=r.monthly_flow_m3s?.length===12?` (${MONTHS[r.monthly_flow_m3s.map(Number).indexOf(low)]})`:'';
      return `${riverClass(e.annual)}${r.morphology==='connector'?' through a lake':''} · ${regimeText(e)} · mean ${formatFlow(e.annual)}`+
        `${settings.month?` · ${MONTHS[settings.month-1]} ${formatFlow(q)}`:''} · lowest month ${formatFlow(low)}${lowMonth} · ${km2(r.area_m2)} upstream · basin ${r.basin_id}`+
        `${nav.label?` · navigation estimate: ${nav.label}`:''}`;
    }
    function inspect(x,y) {
      let best=null,distance=8;
      for(const item of projected)for(const run of item.segments)for(let i=1;i<run.length;i++) {
        const d=segmentDistance([x,y],run[i-1],run[i])-item.width*.5;
        if(d<distance){best=item;distance=d;}
      }
      const lake=best?null:lakeShapes.slice().reverse().find(shape=>insideRings(x,y,shape.rings))?.lake;
      if(lake){
        selected=lake.id??null;selection=null;catchment=null;inspection++;
        if(inspector)inspector.textContent=`Lake${lake.resolution==='source'?' (approximate shore)':''} · ${km2(lake.area_m2)}${Number.isFinite(Number(lake.level_m))?` · surface ${Math.round(lake.level_m)} m`:''}`+
          `${Number.isFinite(Number(lake.max_depth_m))?` · max depth ${Number(lake.max_depth_m).toFixed(1)} m`:''}${lake.closed===true?' · closed basin, no outlet':lake.closed===false?' · drains through an outlet':''}`;
        if(panel)panel.open=true;requestFrame();return true;
      }
      if(!best)return false; selected=best.reach.id;selection={entry:best.entry,suffix:''};catchment=null;
      showSelection();
      if(panel)panel.open=true;loadCatchment(best.reach);return true;
    }
    function snapshot() {const state=getState();return {queryRevision,currentTerrainRevision:state?.terrainRevision??0,geometryCurrent:!!(state&&network&&identity===key(state)&&contains(queryBounds,state.bounds)&&queryLod<=(state.lod??4)&&queryRevision===(state.terrainRevision??0)),settings:{...settings},status,error:failure,hydroId:network?.hydro_id||null,reaches:network?.reaches?.length||0,lakes:network?.lakes?.length||0,
      riverDisplay:{networkRevision,active:!!displayAck?.key,gpuSurface:!!displayAck?.surfaceKey,frameKey:displayAck?.key||null,surfaceKey:displayAck?.surfaceKey||null,authority:'regional-vector'},
      shownReaches:shownCount,shownLakes:lakeShapes.length,selected,catchmentArea:catchment?.area_m2||null,fetching};}
    // Tile request parameters, snapshotted into each task with its cache key
    // (tileIdentity). Fast height tiles only request hydrology headers (state,
    // Base revision): the server returns their elevation unchanged. Fast PNGs
    // ask for nothing, so the server bakes no water into the image.
    function tileParams({physical=true}={}) {
      if(!settings.enabled) return null;
      if(settings.terrainDetail) return {hydro:'1',hydro_month:String(settings.month),...requestParams(settings)};
      return physical?{hydro:'1'}:null;
    }
    return {get:()=>({...settings}),set,reset,mount,draw,inspect,snapshot,writeParams:params=>writeParams(params,settings),tileParams,
      optionsIdentity:()=>optionsIdentity(settings),tileIdentity:()=>tileIdentity(settings),dispose:()=>{reset();if(displayController===displayOwner)displayController=null;}};
  }
  return {defaults:()=>({...DEFAULTS}),normalize,fromParams,writeParams,requestParams,optionsIdentity,tileIdentity,hasRiver,hasInlandWater,v2DisplayIdentity,v2TileParams,contains,segmentDistance,monthlyFlow,
    regime,navigation,servesMonth,thresholdArea,riverClass,riverDisplay,regionalRiverDisplayEnabled,acknowledgeRiverDisplay,create};
});
