/* Canonical screen coverage: a camera cell contains its target level or a
 * cropped parent. Historical fine tiles never leak into a zoomed-out view. */
(function(root) {
  'use strict';
  const MAX_LOD=11, MAX_METRES_PER_PIXEL=30*2**MAX_LOD;
  function subtract(bounds,protectedBounds){
    if(!protectedBounds)return[bounds];
    const [x0,y0,x1,y1]=bounds,[px0,py0,px1,py1]=protectedBounds;
    const left=Math.max(x0,px0),top=Math.max(y0,py0),right=Math.min(x1,px1),bottom=Math.min(y1,py1);
    if(left>=right||top>=bottom)return[bounds];
    return[[x0,y0,left,y1],[right,y0,x1,y1],[left,y0,right,top],[left,bottom,right,y1]].filter(b=>b[2]>b[0]&&b[3]>b[1]);
  }
  function plan(tiles, targetLod, viewBounds, {tileSize=256,nativeResolution=30,maxLod=MAX_LOD,worldBounds=null,protectedBounds=null}={}) {
    const entries=[...tiles], lookup=new Map();
    for(const tile of entries) if(tile.lod>=targetLod)lookup.set(`${tile.lod}/${tile.tx}/${tile.ty}`,tile);
    const span=tileSize*nativeResolution*2**targetLod,result=[];
    const view=worldBounds?[Math.max(viewBounds[0],worldBounds[0]),Math.max(viewBounds[1],worldBounds[1]),Math.min(viewBounds[2],worldBounds[2]),Math.min(viewBounds[3],worldBounds[3])]:viewBounds;
    if(view[2]<=view[0]||view[3]<=view[1])return result;
    for(let ty=Math.floor(view[1]/span);ty<Math.ceil(view[3]/span);ty++)
      for(let tx=Math.floor(view[0]/span);tx<Math.ceil(view[2]/span);tx++) {
        for(let lod=targetLod;lod<=maxLod;lod++) {
          const scale=2**(lod-targetLod),px=Math.floor(tx/scale),py=Math.floor(ty/scale);
          const tile=lookup.get(`${lod}/${px}/${py}`);if(!tile)continue;
          const bounds=worldBounds?[Math.max(tx*span,worldBounds[0]),Math.max(ty*span,worldBounds[1]),Math.min((tx+1)*span,worldBounds[2]),Math.min((ty+1)*span,worldBounds[3])]:[tx*span,ty*span,(tx+1)*span,(ty+1)*span];
          for(const piece of subtract(bounds,lod>targetLod?protectedBounds:null))result.push({tile,bounds:piece,
            uv:[(piece[0]/span-px*scale)/scale,(piece[1]/span-py*scale)/scale,(piece[2]-piece[0])/span/scale,(piece[3]-piece[1])/span/scale],fallback:lod!==targetLod});break;
        }
      }
    return result;
  }
  // Start with the cheapest learned height source (coarse), then expose each
  // finer stage before scheduling the next. A cached finer parent can satisfy
  // a stage, so revisiting a native region never regenerates its ancestors.
  function refinementLod(tiles,targetLod,viewBounds,{tileSize=256,nativeResolution=30,worldBounds=null,protectedBounds=null}={}) {
    if(targetLod>=4)return targetLod;
    const lookup=new Set();
    for(const t of tiles)if(t.presented!==false&&t.lod>=targetLod&&t.lod<=4)lookup.add(`${t.lod}/${t.tx}/${t.ty}`);
    const span=tileSize*nativeResolution*2**targetLod;
    const b=worldBounds?[Math.max(viewBounds[0],worldBounds[0]),Math.max(viewBounds[1],worldBounds[1]),Math.min(viewBounds[2],worldBounds[2]),Math.min(viewBounds[3],worldBounds[3])]:viewBounds;
    if(b[2]<=b[0]||b[3]<=b[1])return targetLod;
    const missing=[];
    for(let ty=Math.floor(b[1]/span);ty<Math.ceil(b[3]/span);ty++)
      for(let tx=Math.floor(b[0]/span);tx<Math.ceil(b[2]/span);tx++)
        if(!lookup.has(`${targetLod}/${tx}/${ty}`)){
          const cell=[Math.max(tx*span,b[0]),Math.max(ty*span,b[1]),Math.min((tx+1)*span,b[2]),Math.min((ty+1)*span,b[3])];
          if(subtract(cell,protectedBounds).length)missing.push([tx,ty]);
        }
    for(let lod=4;lod>targetLod;lod--){
      const uncovered=missing.some(([tx,ty])=>{
        for(let parent=targetLod+1;parent<=lod;parent++){
          const scale=2**(parent-targetLod);
          if(lookup.has(`${parent}/${Math.floor(tx/scale)}/${Math.floor(ty/scale)}`))return false;
        }
        return true;
      });
      if(uncovered)return lod;
    }
    return targetLod;
  }
  const api=Object.freeze({plan,refinementLod,MAX_LOD,MAX_METRES_PER_PIXEL});root.TerrainLOD=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window==='undefined'?globalThis:window);
