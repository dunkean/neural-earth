// Both viewers use this boundary. The Python viewer retains ordinary fetch.
window.TerrainTransport={fetch:(...args)=>fetch(...args),release:(path,data={})=>navigator.sendBeacon(path,new Blob([JSON.stringify(data)],{type:'application/json'}))};

// Optional RGBA8 erosion surface follows heights, climate and float32 water.
window.TerrainPhysicalPayload={surface(bytes,headers,floatCount,width,height=width){
  const format=headers.get('X-Terrain-Surface-Format'),offset=floatCount*4;
  if(format&&format!=='rgba8unorm')throw Error('Unknown erosion surface format');
  const count=format?width*height*4:0;
  const gradient=headers.get('X-Terrain-Gradient-Format');
  if(gradient&&gradient!=='rg32float')throw Error('Unknown terrain gradient format');
  if(bytes.byteLength!==offset+count+(gradient?width*height*8:0))throw Error('Truncated physical tile');
  if(format&&headers.get('X-Terrain-Surface-Channels')!=='flow,deposit,scree,rock')throw Error('Unknown erosion surface channels');
  return count?new Uint8Array(bytes,offset,count):null;
},gradient(bytes,headers,floatCount,width,height=width){return headers.get('X-Terrain-Gradient-Format')==='rg32float'?new Float32Array(bytes,floatCount*4+(headers.get('X-Terrain-Surface-Format')?width*height*4:0),width*height*2):null;}};
