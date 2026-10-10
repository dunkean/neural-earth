/* Coordinate-only pins: Alt+click adds one, clicking its marker removes it. */
window.TerrainPins=(()=>{
  function parse(text){
    if(!text)return [];
    const pins=text.split(';').map(pair=>pair.split(',').map(value=>value.trim()===''?NaN:Number(value)));
    if(pins.length>256||pins.some(pin=>pin.length!==2||!pin.every(Number.isFinite)))throw Error('Invalid pin coordinates');
    return pins;
  }
  const serialize=pins=>pins.map(pin=>pin.join(',')).join(';');
  function mount({view,getState,onChange}){
    let pins=[];const overlay=document.createElement('div');overlay.id='mapPins';overlay.style.cssText='position:absolute;inset:0;pointer-events:none;z-index:2';view.append(overlay);
    function draw(){
      const s=getState();
      for(let i=0;i<pins.length;i++){
        const marker=overlay.children[i],pin=pins[i];
        const p=!s.world?null:s.sceneView==='globe'?s.globe?.worldToScreen(pin[0],pin[1],s.W,s.H,s.worldBounds):[(pin[0]-s.cx)/s.mpp+s.W/2,(pin[1]-s.cy)/s.mpp+s.H/2];
        marker.hidden=!p||p[0]<0||p[1]<0||p[0]>s.W||p[1]>s.H;
        if(p){marker.style.left=p[0]+'px';marker.style.top=p[1]+'px';}
      }
    }
    function rebuild(){
      overlay.replaceChildren();
      for(let i=0;i<pins.length;i++){
        const marker=document.createElement('button');marker.type='button';marker.className='mapPin';marker.title=serialize([pins[i]]);marker.setAttribute('aria-label','Remove pin at '+marker.title);
        marker.style.cssText='position:absolute;width:26px;height:32px;min-height:0;padding:0;border:0;background:transparent;transform:translate(-50%,-100%);pointer-events:auto;cursor:pointer';
        marker.innerHTML='<svg viewBox="0 0 26 32" width="26" height="32" aria-hidden="true"><path d="M13 31C10 26 1 18 1 13a12 12 0 0 1 24 0c0 5-9 13-12 18Z" fill="#ed5353" stroke="#fff" stroke-width="1.5"/><circle cx="13" cy="13" r="4" fill="#fff"/></svg>';
        marker.addEventListener('pointerdown',event=>event.stopPropagation());
        marker.onclick=event=>{event.stopPropagation();pins.splice(i,1);rebuild();onChange();};overlay.append(marker);
      }
      draw();
    }
    const eligible=event=>event.button===0&&event.altKey&&!event.target.closest('#hud,#mini,.mapPin');
    view.addEventListener('pointerdown',event=>{if(eligible(event))event.stopImmediatePropagation();},true);
    view.addEventListener('click',event=>{
      if(!eligible(event))return;
      const s=getState();if(!s.world||pins.length>=256)return;
      const r=view.getBoundingClientRect(),x=(event.clientX-r.left)*s.W/r.width,y=(event.clientY-r.top)*s.H/r.height;
      const pin=s.sceneView==='globe'?s.globe?.screenToWorld(x,y,s.W,s.H,s.worldBounds):[s.cx+(x-s.W/2)*s.mpp,s.cy+(y-s.H/2)*s.mpp];
      if(!pin)return;
      event.stopPropagation();pins.push(pin);rebuild();onChange();
    });
    return {draw,get:()=>pins.map(pin=>[...pin]),serialize:()=>serialize(pins),set(value){pins=typeof value==='string'?parse(value):parse(serialize(value));rebuild();}};
  }
  return {parse,serialize,mount};
})();
