const utf8=new TextEncoder();
const canonical=value=>Array.isArray(value)?value.map(canonical):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(k=>[k,canonical(value[k])])):value;
const base64=bytes=>btoa(Array.from(bytes,x=>String.fromCharCode(x)).join('')).replaceAll('+','-').replaceAll('/','_').replace(/=+$/,'');
const uid=async(kind,recipe)=>kind+base64(new Uint8Array(await crypto.subtle.digest('SHA-256',utf8.encode(JSON.stringify(canonical({version:1,kind,recipe})))))).slice(0,18);
export async function share(path,data) {
  if(path==='/api/share') {
    const generation=await uid('g',data.generation),rendering=await uid('r',data.rendering);
    const saved={version:1,generation:data.generation,rendering:data.rendering};
    const raw=utf8.encode(JSON.stringify(canonical(saved)));
    if(raw.length>128*1024)throw Error('Share recipe too large');
    const bytes=new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer());
    const recipe=base64(bytes);
    const storage=await caches.open('neural-earth-recipes-v1');
    await storage.put(new URL(`/api/share/${generation}/${rendering}`,self.location.origin),new Response(JSON.stringify(saved)));
    return {generation,rendering,recipe};
  }
  if(path==='/api/share/import') {
    if(!/^[A-Za-z0-9_-]{1,174764}$/.test(data.recipe))throw Error('Invalid recipe');
    const raw=Uint8Array.from(atob(data.recipe.replaceAll('-','+').replaceAll('_','/')),c=>c.charCodeAt(0));
    const reader=new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate')).getReader();
    const chunks=[];let size=0;for(;;){const {value,done}=await reader.read();if(done)break;size+=value.length;if(size>128*1024){await reader.cancel();throw Error('Recipe too large');}chunks.push(value);}
    const saved=JSON.parse(await new Blob(chunks).text());
    if(saved.version!==1||await uid('g',saved.generation)!==data.generation||await uid('r',saved.rendering)!==data.rendering)throw Error('Recipe does not match UIDs');
    return saved;
  }
  const storage=await caches.open('neural-earth-recipes-v1'),response=await storage.match(new URL(path,self.location.origin));
  if(!response)throw Error('Unknown recipe; include the portable recipe in the shared link');
  return response.json();
}
