// Exercise the real response/promotion functions with controlled network races.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync('index.html','utf8');
const script=html.split('<script>')[1].split('</script>')[0];
const prefix=script.slice(0,script.indexOf("view.addEventListener('wheel'"));
const poll=script.slice(script.indexOf('setInterval(async()=>'),script.indexOf("addEventListener('pagehide'"));
const elements=new Map(),callbacks=[],textures=new Map();let resolveResponse;
const context=vm.createContext({console,URLSearchParams,AbortController,performance,Float32Array,Map,Set,Math,Date,
 location:{search:'?prepare=0'},crypto:require('node:crypto').webcrypto,
 document:{hidden:false,getElementById(id){if(!elements.has(id))elements.set(id,{checked:true,dataset:{},getContext:()=>({})});return elements.get(id)}},
 setTimeout:()=>1,clearTimeout:()=>{},requestAnimationFrame:()=>1,setInterval:f=>callbacks.push(f),
 fetch:()=>new Promise(resolve=>{resolveResponse=resolve}),window:{},devicePixelRatio:1});
vm.runInContext(prefix,context);
vm.runInContext(`world={seed:'42',version:'natural-v1',world_profile:'natural',cache_profile:'test',world_bounds:[-20e6,-10e6,20e6,10e6]};
 renderer={available:true,uploadTile:(k)=>textures.set(k,true),deleteTile:k=>textures.delete(k)};`,Object.assign(context,{textures}));
vm.runInContext(poll,context);
function response(stage){return {ok:true,status:200,headers:{get:name=>({'X-Terrain-Width':'1','X-Terrain-Halo':'0','X-Terrain-Stage':stage,'X-Terrain-Resolution':'15360','X-Terrain-Cache':'hit'}[name]||null)},arrayBuffer:async()=>new Float32Array([100]).buffer}}
(async()=>{
 vm.runInContext(`testTask=taskFor(9,0,0,2000);wanted.add(testTask.key);interests.set(testTask.key,testTask);publishedEpoch=cameraEpoch;`,context);
 const pending=vm.runInContext('fetchTile(testTask)',context);
 vm.runInContext('coarseRevision=100',context); // Final revision arrives before response.
 resolveResponse(response('conditioning-preview'));await pending;
 assert(vm.runInContext('tiles.get(testTask.key).needsValidation',context),'late preview must be stale');
 assert(vm.runInContext('Number.isFinite(tiles.get(testTask.key).expiresAt)',context),'preview must expire independently of progress');
 assert(textures.has(vm.runInContext('testTask.key',context)));
 context.fetch=async()=>({ok:true,json:async()=>({coarse_preparation:{seed:'42',world_profile:'natural',progress:{complete_windows:101}}})});
 await callbacks[0]();
 assert(vm.runInContext('tiles.has(testTask.key)',context),'promotion must keep the displayed tile while validating');
 // Complete the replacement and prove old-tile disposal does not delete new GPU texture.
 context.fetch=async()=>response('coarse-area-mean');
 await vm.runInContext('fetchTile(testTask)',context);
 assert.equal(vm.runInContext('tiles.get(testTask.key).stage',context),'coarse-area-mean');
 assert(textures.has(vm.runInContext('testTask.key',context)),'replacement GPU texture must survive');
 assert.equal(vm.runInContext('tiles.get(testTask.key).needsValidation',context),false);
 vm.runInContext(`const p=taskFor(1,0,0,2000);tiles.set(p.key,{stage:'decoder',expiresAt:Date.now()-1});if(!needsValidation(tiles.get(p.key)))throw Error('parent DEM promotion must expire');`,context);
 vm.runInContext(`
  inflight.clear();interests.clear();wanted.clear();tiles.clear();rendererReady=true;publishedEpoch=cameraEpoch;
  const a=taskFor(10,0,0,3000),b=taskFor(10,1,0,3001),missing=taskFor(10,2,0,2000);
  for(const t of [a,b])tiles.set(t.key,{expiresAt:0});
  for(const t of [a,b,missing]){interests.set(t.key,t);wanted.add(t.key)}
  dispatched=[];fetchTile=t=>{dispatched.push(t.key);inflight.set(t.key,{validation:needsValidation(tiles.get(t.key))})};
  inflight.set('existing-validation',{validation:true});queue=[a,b,missing];pump();
  if(dispatched.length!==1||dispatched[0]!==missing.key)throw Error('validation must not block missing coverage');
  inflight.clear();dispatched=[];pump();
  if(dispatched.length!==1||queue.length!==1)throw Error('only one cached validation may occupy the HTTP lane');
 `,context);
 console.log('Preview race, retained display, parent/GPU replacement and serialized promotion with parallel missing coverage: OK');
})().catch(e=>{console.error(e);process.exitCode=1});
