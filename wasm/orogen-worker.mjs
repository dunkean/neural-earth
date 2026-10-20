// Same maintained Orogen sources as Python's native/orogen/pipeline.mjs.
// The UMD triangulator is converted to an ES module by the static build.
let result;
const send=self.postMessage.bind(self);
self.postMessage=value=>{
  if(value.type==='error')send(value);
  else if(value.type==='done')result=value;
  else if(value.type==='progress')send(value);
};
await import('./orogen/planet-worker.js');
const generate=self.onmessage;
self.onmessage=event=>{
  const request=event.data;
  globalThis.OROGEN_THRESHOLD=request.threshold;
  globalThis.OROGEN_MOTION=request.motion;
  globalThis.OROGEN_SPREAD=request.spread;
  globalThis.OROGEN_SKIP_POST=request.skipPost;
  globalThis.OROGEN_CLIMATE=request.climateSettings;
  result=null;
  try {generate({data:{...request,cmd:'generate'}}); if(result)send(result);}
  catch(error){send({type:'error',message:error.message,stack:error.stack});}
};
send({type:'ready'});
