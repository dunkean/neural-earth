export async function loadModels(progress) {
  const response=await fetch(new URL('./models/manifest.json',import.meta.url));
  if(!response.ok)throw Error('Model manifest missing: run npm run build:wasm');
  const manifest=await response.json(), models=[];
  const cache=await caches.open('neural-earth-models-v1').catch(()=>null);
  const directory=new URL('./models/',import.meta.url);
  const urls=['coarse','base','decoder'].map(name=>{
    const entry=manifest.models[name],url=new URL(entry.file,directory);
    url.searchParams.set('sha256',entry.sha256);return url;
  });
  // Release the old original-base download and outdated checkpoint revisions.
  if(cache)for(const request of await cache.keys()) {
    if(request.url.startsWith(directory.href)&&!urls.some(url=>url.href===request.url))await cache.delete(request);
  }
  for(const name of ['coarse','base','decoder']) {
    const entry=manifest.models[name],url=urls[['coarse','base','decoder'].indexOf(name)];
    progress(`Loading ${name} weights · ${(entry.bytes/1e6).toFixed(1)} MB`);
    let stored=await cache?.match(url), response=stored||await fetch(url);
    if(!response.ok)throw Error(`Cannot load ${name} weights (${response.status})`);
    const buffer=await response.arrayBuffer();
    const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),v=>v.toString(16).padStart(2,'0')).join('');
    if(buffer.byteLength!==entry.bytes||digest!==entry.sha256){await cache?.delete(url);throw Error(`Invalid ${name} weights. Clear the model cache and reload.`);}
    if(!stored&&cache)await cache.put(url,new Response(buffer)).catch(()=>{});
    models.push(new Uint8Array(buffer));
  }
  return {models,manifest};
}
