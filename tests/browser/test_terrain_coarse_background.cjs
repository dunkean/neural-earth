const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
// World preparation stays off by default and starts only after explicit opt-in.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync('index.html','utf8').split('<script>')[1].split('</script>')[0];
const elements=new Map(),requests=[];
const context=vm.createContext({console,URLSearchParams,AbortController,performance,Map,Set,Math,Date,
 location:{search:'?prepare=0'},history:{replaceState(){}},crypto:require('node:crypto').webcrypto,
 document:{getElementById(id){if(!elements.has(id))elements.set(id,{checked:false,value:'0',dataset:{},options:[],getContext:()=>({}),setAttribute(){}});return elements.get(id)}},
 setTimeout:()=>1,clearTimeout(){},requestAnimationFrame:()=>1,window:{},TerrainLighting:{get:()=>({}),serialize:()=>''},devicePixelRatio:1,
 fetch:(url,options)=>new Promise(resolve=>requests.push({url,options,resolve}))});
vm.runInContext(source.slice(0,source.indexOf("view.addEventListener('wheel'")),context);
vm.runInContext(source.slice(source.indexOf("async function prepareCoarse("),source.indexOf("$('prepareWorld').onclick")),context);
const run=code=>vm.runInContext(code,context),tick=()=>new Promise(resolve=>setImmediate(resolve));
const response=data=>({ok:true,json:async()=>data});
const metadata=seed=>({seed,world_profile:'natural',generation_profile:'natural',gpu:'test',world_bounds:[-20e6,-10e6,20e6,10e6],overview_bounds:[-20e6,-10e6,20e6,10e6]});
(async()=>{
 for(const [search,enabled]of [['',false],['?coarse_prepare=1',false],['?prepare_world=1',true]]){
  const startup=vm.createContext({...context,location:{search}});
  vm.runInContext(source.slice(0,source.indexOf("view.addEventListener('wheel'")),startup);
  assert.equal(vm.runInContext('automaticPreparation',startup),enabled,search||'default');
 }
 assert.equal(run('automaticPreparation'),false,'Global preparation is off by default');
 run('syncWorldGenerator=()=>{};syncSceneView=()=>{};');
 const opened=run("openWorld('42','natural',{generation:{height_source:'natural'}})");
 requests.shift().resolve(response(metadata('42')));assert.equal(await opened,true,run('error'));
 assert.equal(requests.length,0,'Opening a world must not start full-world computation');
 run('automaticPreparation=true;prepareCoarse({seed:world.seed,world_profile:generationProfile(),full_world:true})');
 const preparation=requests.shift();assert.equal(preparation.url,'/api/coarse/prepare');
 assert.equal(JSON.parse(preparation.options.body).full_world,true);
 assert.equal(run('overview'),null,'Start before even receiving the overview');
 assert.equal(elements.get('loadingIndicator').hidden,false,'Pending admission already displays the spinner');
 assert.match(elements.get('loadingLabel').textContent,/Computing coarse/);
 preparation.resolve(response({state:'running',seed:'42',world_profile:'natural',budget:null}));await tick();
 assert.equal(run('automaticPreparationPending'),null);
 assert.equal(elements.get('loadingIndicator').hidden,false);
 run("serverStatus={scheduler:{queued:0,computing:0,encoding:0},coarse_preparation:{state:'idle'}};queue=[];inflight.clear();inflight.set('coarse',{task:{native_coarse:true,learned:true}});updateActivity()");
 assert.equal(elements.get('loadingIndicator').hidden,false,'Coarse cannot wait two seconds for server polling');
 assert.match(elements.get('loadingLabel').textContent,/Computing coarse/);
 run('inflight.clear();updateActivity()');assert.equal(elements.get('loadingIndicator').hidden,true);
 // A zoom does not cancel global preparation or start another full-world job.
 run('automaticPreparation=false;');
 const second=run("openWorld('43','natural',{generation:{height_source:'natural'}})");
 requests.shift().resolve(response(metadata('43')));await second;
 assert.equal(requests.length,0,'Explicit pause survives a world change');
 // A new generation cancels old transfers and supersedes the previous request.
 run(`automaticPreparation=true;const oldController=new AbortController();oldSignal=oldController.signal;
 inflight.set('old-flight',{controller:oldController,task:{native_coarse:true},epoch});desiredView={epoch:cameraEpoch};`);
 const firstGeneration=run("applyGenerationSettings({},'orogen','all')");
 assert.equal(run('oldSignal.aborted'),true);assert.equal(run('inflight.size'),0);
 assert.equal(run('desiredView'),null);assert.equal(run('generationPending'),true);
 assert.match(elements.get('loadingLabel').textContent,/Generating world/);
 let stops=requests.splice(0);assert.deepEqual(stops.map(r=>r.url).sort(),['/api/coarse/prepare','/api/view/release']);
 stops.forEach(r=>r.resolve(response({state:'stopped'})));await tick();
 const oldGeneration=requests.shift();assert.equal(oldGeneration.url,'/api/generation/run');
 run("$('seed').value='44'");const latestGeneration=run("applyGenerationSettings({},'orogen','all')");
 assert.equal(oldGeneration.options.signal.aborted,true);
 stops=requests.splice(0);stops.forEach(r=>r.resolve(response({state:'stopped'})));await tick();
 const latestRequest=requests.shift();assert.equal(latestRequest.url,'/api/generation/run');
 assert.ok(JSON.parse(latestRequest.options.body).generation_epoch>JSON.parse(oldGeneration.options.body).generation_epoch);
 run('refresh();pump();startOverview();prepareCoarse({seed:43,full_world:true})');assert.equal(requests.length,0);
 const generated=data=>({...response(data),headers:{get:()=> 'application/json'}});
 oldGeneration.resolve(generated({world_profile:'orogen',generation_settings:{}}));
 assert.equal(await firstGeneration,false);assert.equal(run('generationPending'),true);
 assert.equal(requests.length,0,'Old completion must not restart the coarse');
 latestRequest.resolve(generated({world_profile:'orogen',generation_settings:{}}));await tick();
 const freshWorld=requests.shift();assert.match(freshWorld.url,/api\/world\?seed=44/);
 freshWorld.resolve(response(metadata('44')));assert.equal(await latestGeneration,true);
 assert.equal(run('generationPending'),false);assert.equal(run('world.seed'),'44');
 const resumed=requests.shift();assert.equal(resumed.url,'/api/coarse/prepare');
 assert.equal(JSON.parse(resumed.options.body).seed,'44');assert.equal(requests.length,0);
 resumed.resolve(response({state:'running',seed:'44'}));await tick();
 run('automaticPreparation=false;');
 // Input water is not proof that the learned coarse contains no island.
 run(`renderer={available:true,uploadCoarse(){}};$('gpuRender').checked=true;
 W=256;H=256;mpp=1920;cx=491520;cy=491520;$('seaMaxLod').value='9';
 world.scheduling_land_mask={bounds:world.world_bounds,width:1,height:1,rows:['0']};
 const sea=nativeCoarseTask(0,0,2000);seaProbe=sea;
 tiles.set(sea.key,{...sea,b:tileBounds(sea),stage:'conditioning-preview',elevationMin:-100,elevationMax:-100,expiresAt:0,coarseRevision:0});
 serverStatus={coarse_preparation:{state:'running'}};coarseRevision=1;`);
 assert.equal(run('seaProbe.learned'),false,'Idle global work, rather than forcing sea NN at a land zoom');
 assert.equal(run('seaTaskAllowed(seaProbe)'),true,'Preview water remains eligible for learned discovery');
 assert.equal(run('needsValidation(tiles.get(seaProbe.key))'),true);
 run("tiles.get(seaProbe.key).stage='coarse'");
 assert.equal(run('seaTaskAllowed(seaProbe)'),false,'Already learned empty water may freeze');
 run('tiles.get(seaProbe.key).elevationMax=25');
 assert.equal(run('seaTaskAllowed(seaProbe)'),true,'A learned island returns to visible refinement');
 console.log('Full-world off by default, explicit opt-in, spinner, cancellation and opt-out: OK');
})().catch(e=>{console.error(e);process.exitCode=1});
