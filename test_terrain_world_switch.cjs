// Reproduce late camera acknowledgements while opening a customized world.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const script=fs.readFileSync('index.html','utf8').split('<script>')[1].split('</script>')[0];
const prefix=script.slice(0,script.indexOf("view.addEventListener('wheel'"));
const elements=new Map(),requests=[];
const context=vm.createContext({console,URLSearchParams,AbortController,performance,Map,Set,Math,Date,
 location:{search:'?coarse_prepare=0'},history:{replaceState(){}},crypto:require('node:crypto').webcrypto,
 document:{getElementById(id){if(!elements.has(id))elements.set(id,{checked:false,dataset:{},options:[],getContext:()=>({}),setAttribute(){}});return elements.get(id)}},
 setTimeout:()=>1,clearTimeout:()=>{},requestAnimationFrame:()=>1,window:{},TerrainLighting:{get:()=>({}),serialize:()=>''},devicePixelRatio:1,
 fetch:(url,options)=>new Promise(resolve=>requests.push({url,options,resolve}))});
vm.runInContext(prefix,context);
const run=code=>vm.runInContext(code,context);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const response=(data,ok=true)=>({ok,status:ok?200:400,json:async()=>data});
const world=seed=>({seed,world_profile:'natural',generation_profile:'natural',gpu:'test',
 world_bounds:[-20e6,-10e6,20e6,10e6],overview_bounds:[-20e6,-10e6,20e6,10e6]});
(async()=>{
 run(`pump=()=>{};startOverview=()=>{};syncWorldGenerator=()=>{};syncSceneView=()=>{};
 desiredView={session,epoch:1,seed:'42',world_profile:'natural',nnEngine,tiles:[]};cameraEpoch=1;`);
 const oldCamera=run('publishView()');
 const customized=run(`openWorld('43','natural',{generation:{height_source:'natural'}})`);
 assert.equal(run('desiredView'),null);
 requests.shift().resolve(response({accepted:true,epoch:1}));await oldCamera;
 assert.equal(run('error'),'','late camera acknowledgement must not dereference a cleared view');
 requests.shift().resolve(response(world('43')));assert.equal(await customized,true);
 assert.equal(run('world.seed'),'43');assert.equal(run('worldLoading'),false);

 // An obsolete camera error must not poison the next world's successful request.
 run(`desiredView={session,epoch:2,seed:'43',world_profile:'natural',nnEngine,tiles:[]};`);
 const rejectedCamera=run('publishView()');
 const nextWorld=run(`openWorld('44','natural',{generation:{height_source:'natural'}})`);
 requests.shift().resolve(response({error:'Invalid epoch'},false));await rejectedCamera;
 assert.equal(run('error'),'');
 requests.shift().resolve(response(world('44')));assert.equal(await nextWorld,true);

 // Two metadata requests finishing out of order must retain the newest world.
 run(`serverStatus={coarse_preparation:{state:'running'},scheduler:{computing:1}};coarseRevision=99;`);
 const first=run(`openWorld('45','natural',{generation:{height_source:'natural'}})`);
 assert.equal(run('serverStatus'),null);assert.equal(run('coarseRevision'),0);
 const second=run(`openWorld('46','natural',{generation:{height_source:'natural'}})`);
 const older=requests.shift(),newer=requests.shift();
 newer.resolve(response(world('46')));assert.equal(await second,true);
 older.resolve(response(world('45')));assert.equal(await first,false);
 assert.equal(run('world.seed'),'46');assert.equal(run('worldLoading'),false);

 // Missing coverage, retries and HTTP transfers do not mean the server is computing.
 run(`wanted.add('missing');overviewWanted=true;queue=[{prefetch:false}];inflight.set('transfer',{task:{prefetch:true}});
 serverStatus={scheduler:{queued:0,computing:0,encoding:0},coarse_preparation:{state:'idle'}};updateActivity();`);
 assert.equal(elements.get('loadingIndicator').hidden,true);
 for(const state of ['queued','computing','encoding']){
  run(`serverStatus.scheduler.${state}=1;updateActivity()`);
  assert.equal(elements.get('loadingIndicator').hidden,false,`${state} background work must show activity`);
  run(`serverStatus.scheduler.${state}=0`);
 }
 run(`viewMode='snr-elevation';serverStatus.coarse_preparation={state:'running'};updateActivity()`);
 assert.equal(elements.get('loadingIndicator').hidden,false,'background preparation also runs in diagnostic views');
 run(`serverStatus.coarse_preparation.state='idle';updateActivity()`);
 assert.equal(elements.get('loadingIndicator').hidden,true);
 console.log('World switching races and computation activity: OK');
})().catch(error=>{console.error(error);process.exitCode=1});
