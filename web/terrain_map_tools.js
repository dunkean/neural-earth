/* Cursor elevation and one-shot distances in the flat map's metre coordinates. */
(() => {
  'use strict';
  function sampleHeight(tile, point) {
    const {heights, heightOptions: options, b} = tile;
    if (!heights || !options || !b) return null;
    const width = options.width, height = options.height || width, halo = options.halo;
    const x = Math.max(0, Math.min(width - 1, halo + (point.x - b[0]) / (b[2] - b[0]) * (width - 2 * halo) - .5));
    const y = Math.max(0, Math.min(height - 1, halo + (point.y - b[1]) / (b[3] - b[1]) * (height - 2 * halo) - .5));
    const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(width - 1, x0 + 1), y1 = Math.min(height - 1, y0 + 1);
    const dx = x - x0, dy = y - y0;
    const value = (heights[y0 * width + x0] * (1 - dx) + heights[y0 * width + x1] * dx) * (1 - dy)
      + (heights[y1 * width + x0] * (1 - dx) + heights[y1 * width + x1] * dx) * dy;
    return Number.isFinite(value) ? (options.encoding==='signed-sqrt'?Math.sign(value)*value*value:value) : null;
  }
  const distanceLabel = metres => metres >= 1000 ? `${(metres / 1000).toFixed(2)} km` : `${metres.toFixed(1)} m`;
  function mount({view, mini, button, altitude, extent, distance, getState, redraw}) {
    let pointer = null, armed = false, segment = null, measuringId = null, frame = null;
    let flight = null, fetchTimer = null, generation = null;
    const samples = new Map(), failures = new WeakMap();
    const inside = (point, b) => point.x >= b[0] && point.x < b[2] && point.y >= b[1] && point.y < b[3];
    const screenPoint = event => { const r = view.getBoundingClientRect(); return {x:(event.clientX-r.left)*view.clientWidth/r.width, y:(event.clientY-r.top)*view.clientHeight/r.height}; };
    const worldPoint = (p, s) => ({x:s.cx+(p.x-s.W/2)*s.mpp, y:s.cy+(p.y-s.H/2)*s.mpp});
    function setArmed(value) {
      armed = value;
      button.setAttribute('aria-pressed', String(value));
      view.classList.toggle('measuring', value);
      button.textContent = value ? '×' : '↔';
      button.title = value ? 'Annuler la mesure' : 'Measure distance';
      button.setAttribute('aria-label', button.title);
    }
    function clearMeasurement() {
      segment = null;
      const id = measuringId;
      measuringId = null;
      setArmed(false);
      if (id !== null && view.hasPointerCapture(id)) view.releasePointerCapture(id);
      redraw();
    }
    function queueUpdate() {
      if (frame === null) frame = requestAnimationFrame(() => { frame = null; update(); });
    }
    async function loadHeight(tile, s) {
      if (s.mode?.startsWith('snr-') || flight || samples.has(tile) || (failures.get(tile) || 0) > Date.now()) return;
      const controller = new AbortController(), request = {controller, tile};
      flight = request;
      try {
        const params = new URLSearchParams({session:s.session, epoch:String(s.publishedEpoch), profile:s.world.cache_profile,
          world_profile:s.generationProfile, mode:s.mode, coarse_revision:String(s.coarseRevision)});
        if (tile.lod===4) params.set('coarse_interpolation',s.coarseInterpolation||'monotone');
        if (tile.source_lod) params.set('source_lod',String(tile.source_lod));
        if (s.world.world_identity) params.set('world_identity', s.world.world_identity);
        const response = await fetch(`${s.backendPrefix||''}/height/${s.world.version}/${tile.path}.bin?${params}`, {signal:controller.signal});
        if (!response.ok) throw Error('Elevation unavailable');
        if(tile.source_lod&&Number(response.headers.get('X-Terrain-Source-Resolution'))!==30*2**tile.source_lod)throw Error('Unexpected elevation source');
        const bytes = await response.arrayBuffer(), width = Number(response.headers.get('X-Terrain-Width') || 304);
        const halo = Number(response.headers.get('X-Terrain-Halo') ?? 24);
        if (!Number.isInteger(width) || width <= 0 || halo < 0 || width <= 2*halo || bytes.byteLength < width*width*4) throw Error('Invalid elevation tile');
        if (getState().epoch !== s.epoch || controller.signal.aborted) return;
        samples.set(tile, {...tile, heights:new Float32Array(bytes, 0, width*width), heightOptions:{width, height:width, halo}});
        while (samples.size > 4) samples.delete(samples.keys().next().value);
      } catch (error) {
        if (error.name !== 'AbortError') failures.set(tile, Date.now()+3000);
      } finally {
        if (flight === request) flight = null;
        queueUpdate();
      }
    }
    function update() {
      if(getState().sceneView==='globe')return;
      const s = getState();
      if (generation !== s.epoch) {
        generation = s.epoch;
        flight?.controller.abort(); flight = null;
        clearTimeout(fetchTimer); fetchTimer = null;
        samples.clear(); clearMeasurement();
      }
      const km = metres => (metres / 1000).toLocaleString(undefined, {minimumFractionDigits:2,maximumFractionDigits:metres < 1000 ? 5 : 2});
      extent.textContent = s.world ? `Visible area: ${km(s.W*s.mpp)} × ${km(s.H*s.mpp)} km · ${s.W} × ${s.H} px (width × height)` : 'Visible area: —';
      distance.textContent = segment ? `Distance on map: ${distanceLabel(Math.hypot(segment.end.x-segment.start.x, segment.end.y-segment.start.y))}`
        : armed ? 'Distance: click and drag · Esc to cancel' : 'Distance: —';
      // A noise-policy view has no elevation payload. Avoid launching terrain
      // inference merely because the pointer crosses this diagnostic layer.
      altitude.hidden = !!s.mode?.startsWith('snr-');
      if (altitude.hidden) {flight?.controller.abort();flight=null;clearTimeout(fetchTimer);fetchTimer=null;return;}
      if (!s.world || !pointer || pointer.x < 0 || pointer.y < 0 || pointer.x >= s.W || pointer.y >= s.H) {
        altitude.textContent = 'Cursor elevation: —'; return;
      }
      const point = worldPoint(pointer, s);
      if (!inside(point, s.worldBounds)) { altitude.textContent = 'Cursor elevation: outside world'; return; }
      const entry = s.visiblePlan.find(entry => inside(point, entry.bounds));
      // The saved initial image protects its rectangle from the draw plan, but
      // the ordinary scheduler still loads its native physical tiles.
      const tile = entry?.tile || (s.initialImage && s.currentLod === 0
        ? [...s.tiles].find(tile => tile.lod === 0 && inside(point, tile.b)) : null);
      const data = tile?.heights ? tile : samples.get(tile);
      const value = data ? sampleHeight(data, point) : null;
      const preview = ['conditioning-preview','orogen-diagnostic'].includes(tile?.stage) ? ' · input preview' : '';
      altitude.textContent = value === null ? 'Cursor elevation: waiting for terrain…' : `Cursor elevation: ${value.toFixed(1)} m${preview}`;
      if (tile && !data && !flight && fetchTimer === null && s.publishedEpoch === s.cameraEpoch && (failures.get(tile) || 0) <= Date.now()) {
        fetchTimer = setTimeout(() => {
          fetchTimer = null;
          const latest = getState();
          if (latest.epoch !== s.epoch || latest.mode !== s.mode || !pointer || latest.publishedEpoch !== latest.cameraEpoch) return;
          const current = latest.visiblePlan.find(entry => inside(worldPoint(pointer, latest), entry.bounds));
          if (current?.tile === tile) loadHeight(tile, latest);
        }, 100);
      }
    }
    function draw(context) {
      if(getState().sceneView==='globe')return;
      update();
      if (!segment) return;
      const s = getState(), screen = p => ({x:(p.x-s.cx)/s.mpp+s.W/2, y:(p.y-s.cy)/s.mpp+s.H/2});
      const a = screen(segment.start), b = screen(segment.end);
      context.save();
      context.lineCap = 'round';
      context.beginPath(); context.moveTo(a.x,a.y); context.lineTo(b.x,b.y);
      context.strokeStyle = '#101e26'; context.lineWidth = 5; context.stroke();
      context.strokeStyle = '#ffe28a'; context.lineWidth = 2; context.stroke();
      for (const p of [a,b]) { context.beginPath(); context.arc(p.x,p.y,4,0,Math.PI*2); context.fillStyle='#ffe28a'; context.fill(); }
      const label = distanceLabel(Math.hypot(segment.end.x-segment.start.x, segment.end.y-segment.start.y));
      context.font = '13px system-ui';
      const width = context.measureText(label).width;
      const x = Math.max(4,Math.min(s.W-width-16,(a.x+b.x)/2-width/2-6));
      const y = Math.max(4,Math.min(s.H-26,(a.y+b.y)/2-30));
      context.fillStyle='#101e26ee'; context.fillRect(x,y,width+12,24);
      context.fillStyle='#ffe28a'; context.fillText(label,x+6,y+17);
      context.restore();
    }
    button.addEventListener('click', () => { const next = !armed; clearMeasurement(); setArmed(next); view.focus(); queueUpdate(); });
    view.addEventListener('pointerdown', event => {
      if (event.altKey || event.target.closest?.('.mapPin') || getState().sceneView==='globe' || !armed || event.button !== 0 || event.target === mini || event.target.closest?.('#hud') || !getState().world) return;
      event.preventDefault(); event.stopImmediatePropagation();
      pointer = screenPoint(event);
      const point = worldPoint(pointer,getState());
      segment = {start:point,end:point}; measuringId = event.pointerId;
      view.setPointerCapture(event.pointerId); view.focus(); redraw();
    }, true);
    view.addEventListener('pointermove', event => {
      if(getState().sceneView==='globe'){pointer=null;return;}
      pointer = event.target === mini ? null : screenPoint(event);
      if (measuringId === event.pointerId) {
        event.preventDefault(); event.stopImmediatePropagation();
        segment.end = worldPoint(screenPoint(event),getState()); redraw();
      }
      queueUpdate();
    }, true);
    view.addEventListener('pointerleave', () => { if (measuringId === null) {pointer=null; queueUpdate();} });
    for (const name of ['pointerup','pointercancel','lostpointercapture']) view.addEventListener(name, event => {
      if (measuringId !== event.pointerId) return;
      event.stopImmediatePropagation();
      if (name === 'pointerup') segment.end = worldPoint(screenPoint(event),getState());
      else segment = null;
      measuringId = null; setArmed(false);
      if (view.hasPointerCapture(event.pointerId)) view.releasePointerCapture(event.pointerId);
      redraw();
    }, true);
    view.addEventListener('keydown', event => {
      if (event.key === 'Escape') {clearMeasurement(); queueUpdate();}
      else if (measuringId !== null) {event.preventDefault(); event.stopImmediatePropagation();}
    }, true);
    view.addEventListener('wheel', event => {if (measuringId !== null) {event.preventDefault(); event.stopImmediatePropagation();}}, {capture:true,passive:false});
    return {draw,cancel(){pointer=null;clearMeasurement();flight?.controller?.abort();}};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {sampleHeight, distanceLabel};
  if (typeof window !== 'undefined') window.TerrainMapTools = {mount};
})();
