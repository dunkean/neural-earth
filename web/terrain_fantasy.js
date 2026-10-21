/* Display-only atlas painting. Shared planner: city_generator/rust/bridge/fantasyPainting.ts. */
window.TerrainFantasy = (() => {
  const painter = window.FantasyPainting;
  const atlases = Object.fromEntries(['fantasy-ink','fantasy-color'].map(style => [style,painter.createBrushLibrary(style,window.FantasyBrushAssets)]));
  const images = new Map(); let redraw = () => {}, cached = null;
  const settings = { density: 1, size: 1, spacing: 1 };
  function set(value) {
    for (const key of Object.keys(settings)) if (Number.isFinite(Number(value?.[key]))) settings[key] = Math.max(.3,Math.min(2,Number(value[key])));
    cached = null;
    for (const key of Object.keys(settings)) { const input = document.getElementById('brush_'+key); if(input) input.value = settings[key]; }
  }
  function mount(draw) {
    redraw = draw;
    const field = document.createElement('fieldset'); field.id = 'fantasySettings'; field.innerHTML = '<legend>Fantasy brushes · oblique view</legend>';
    for (const [key,label] of [['density','Density'],['size','Size'],['spacing','Spacing']]) {
      const row = document.createElement('label'), input = document.createElement('input'); row.textContent = label;
      Object.assign(input,{id:'brush_'+key,type:'range',min:'.3',max:'2',step:'.05',value:String(settings[key])});
      input.oninput = () => { settings[key] = Number(input.value); cached = null; redraw(); }; row.append(input);field.append(row);
    }
    document.querySelector('#renderPanel .toolPanel').append(field);
    const exportButton = document.createElement('button'); exportButton.type='button'; exportButton.id='fantasyExport'; exportButton.textContent='Export view · PNG';
    exportButton.onclick=()=>{
      redraw();
      requestAnimationFrame(()=>{
        const source=document.getElementById('map'), canvas=document.createElement('canvas'); canvas.width=source.width;canvas.height=source.height;
        const context=canvas.getContext('2d');
        for(const id of ['map','gpuMap','annotations']) {const layer=document.getElementById(id);context.filter=layer.style.filter||'none';context.drawImage(layer,0,0);}
        canvas.toBlob(blob=>{if(!blob)return;const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='neural-earth-fantasy.png';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);});
      });
    };field.append(exportButton);
  }
  function heightAt(tile,x,y) {
    const o = tile.heightOptions, values = tile.heights;
    if (!o || !values) return NaN;
    const gx = (x-o.origin[0])/o.metresPerSample-.5, gy = (y-o.origin[1])/o.metresPerSample-.5;
    const ix = Math.max(0,Math.min(o.width-2,Math.floor(gx))), iy = Math.max(0,Math.min((o.height||o.width)-2,Math.floor(gy)));
    const u = Math.max(0,Math.min(1,gx-ix)), v = Math.max(0,Math.min(1,gy-iy));
    const h = (values[iy*o.width+ix]*(1-u)+values[iy*o.width+ix+1]*u)*(1-v)+(values[(iy+1)*o.width+ix]*(1-u)+values[(iy+1)*o.width+ix+1]*u)*v;
    return o.encoding === 'signed-sqrt' ? Math.sign(h)*h*h : h;
  }
  function climateAt(tile,x,y,layer,fallback) {
    const o = tile.heightOptions, b = tile.b, n = o?.climateWidth, h = o?.climateHeight;
    if (!o?.climate || !n || !h) return fallback;
    const ix = Math.max(0,Math.min(n-1,Math.round((x-b[0])/(b[2]-b[0])*(n-1))));
    const iy = Math.max(0,Math.min(h-1,Math.round((y-b[1])/(b[3]-b[1])*(h-1))));
    return o.climate[layer*n*h+iy*n+ix] ?? fallback;
  }
  // CPU fallback uses the same physical arrays, never brightness guessed from a PNG.
  function ground(tile,ink) {
    const o = tile.heightOptions, n = o.width-2*o.halo, canvas = document.createElement('canvas'); canvas.width=canvas.height=n;
    const ctx = canvas.getContext('2d'), image = ctx.createImageData(n,n);
    for(let y=0;y<n;y++) for(let x=0;x<n;x++) {
      const p=(y*n+x)*4, xw=o.origin[0]+(x+o.halo+.5)*o.metresPerSample, yw=o.origin[1]+(y+o.halo+.5)*o.metresPerSample;
      const land=heightAt(tile,xw,yw)>0, color=land?(ink?[248,247,242]:[239,231,208]):(ink?[219,224,224]:[174,197,203]);
      image.data.set([...color,255],p);
    }
    ctx.putImageData(image,0,0); return canvas;
  }
  function draw(ctx,{mode,seed,cx,cy,mpp,width,height,entries,revision=0}) {
    if (!painter.isFantasyStyle(mode)) return;
    const atlas = atlases[mode];
    // Discrete cartographic scale: stable world anchors at each tier, bounded screen density.
    const brushPitch = Math.max(20,2**Math.floor(Math.log2(mpp*21))), pitch = brushPitch * settings.spacing;
    const bx = cx-width*mpp/2, by = cy-height*mpp/2;
    const tiles = [...new Map(entries.filter(e=>e.tile.heights).map(e=>[e.tile.key,{tile:e.tile,bounds:e.tile.b}])).values()].sort((a,b)=>(a.tile.sourceResolution||Infinity)-(b.tile.sourceResolution||Infinity));
    if (!tiles.length) return;
    const step = pitch*16, x0 = Math.floor(bx/step)*step-step, y0 = Math.floor(by/step)*step-step;
    const w = Math.ceil(width*mpp/step+3)*step, h = Math.ceil(height*mpp/step+3)*step;
    const key = JSON.stringify([mode,seed,pitch,x0,y0,w,h,revision,settings,tiles.map(e=>e.tile.key)]);
    if (!cached || cached.key!==key) {
      // Index retained physical tiles once; avoid scanning the tile list per brush sample.
      const buckets = new Map();
      for (const entry of tiles) {
        const b=entry.bounds;
        for(let y=Math.floor(b[1]/step);y<=Math.floor(b[3]/step);y++)for(let x=Math.floor(b[0]/step);x<=Math.floor(b[2]/step);x++) {
          const k=x+','+y; if(!buckets.has(k))buckets.set(k,[]); buckets.get(k).push(entry);
        }
      }
      const sample = (x,y) => {
        const entry=(buckets.get(Math.floor(x/step)+','+Math.floor(y/step))||[]).find(e=>x>=e.bounds[0]&&x<e.bounds[2]&&y>=e.bounds[1]&&y<e.bounds[3]);
        if(!entry)return;
        const t=entry.tile,d=t.heightOptions.metresPerSample;
        const height = heightAt(t,x,y), radius = brushPitch*3;
        let sum=0,lo=height,hi=height;
        for(let i=0;i<8;i++) {
          const a=i*Math.PI/4,px=x+Math.cos(a)*radius,py=y+Math.sin(a)*radius;
          const other=(buckets.get(Math.floor(px/step)+','+Math.floor(py/step))||[]).find(e=>px>=e.bounds[0]&&px<e.bounds[2]&&py>=e.bounds[1]&&py<e.bounds[3]);
          const v=heightAt(other?.tile||t,px,py);sum+=v;lo=Math.min(lo,v);hi=Math.max(hi,v);
        }
        return {height,tpi:(height-sum/8)/radius,relief:hi-lo,slope:Math.hypot(heightAt(t,x+d,y)-heightAt(t,x-d,y),heightAt(t,x,y+d)-heightAt(t,x,y-d))/(2*d),temperature:climateAt(t,x,y,0,18),rain:climateAt(t,x,y,2,1000)};
      };
      const start=performance.now();
      const stamps=painter.planPainting({x:x0,y:y0,width:w,height:h},sample,{seed:String(seed),pitch,brushPitch,...settings});
      cached = {key,mode,stamps,families:[...new Set(stamps.map(stamp=>painter.brushFrame(atlas,stamp).family))],ms:performance.now()-start};
    }
    const ready = new Map();
    for (const family of cached.families) {
      const sheet=atlas.families[family];
      let image=images.get(family);
      if(!image){image=new Image();images.set(family,image);image.onload=redraw;image.onerror=()=>console.warn('Fantasy brush sheet unavailable:',sheet.url);image.src=sheet.url;}
      if(image.complete&&image.naturalWidth)ready.set(family,image);
    }
    painter.paintingCanvas(ctx,cached.stamps,ready,atlas,{x:bx,y:by,scale:1/mpp});
  }
  return {isStyle:painter.isFantasyStyle,mount,set,get:()=>({...settings}),draw,ground,heightAt,stats:()=>({mode:cached?.mode,stamps:cached?.stamps.length||0,planningMs:cached?.ms||0,sheets:images.size,decoded:[...images.values()].filter(image=>image.complete&&image.naturalWidth).length,variants:new Set(cached?.stamps.map(s=>s.brush+':'+s.variant)||[]).size})};
})();
