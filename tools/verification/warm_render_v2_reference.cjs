// Prepare or warm the real 60-centre decoder cache through the existing viewer.
// Default --plan starts no browser, inference or service. --run is working only.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),manifestFile=path.join(root,'output/render-v2-reference/manifest.json');
const manifest=JSON.parse(fs.readFileSync(manifestFile,'utf8'));
const wanted=process.env.REVIEW_CASES?.split(','),sites=manifest.sites.filter(s=>!wanted||wanted.includes(s.id));
assert(sites.length);if(wanted)assert.equal(sites.length,wanted.length);
const hash=crypto.createHash('sha256').update(fs.readFileSync(manifestFile)).digest('hex').slice(0,12);
const relative='output/render-v2-reference/warm/'+manifest.worldIdentity.slice(0,8)+'-'+hash,out=path.join(root,relative);
const plan={label:'WORKING decoder30m cache warming; not final render validation',sites:sites.length,
 worldIdentity:manifest.worldIdentity,extentMetres:40000,logicalSize:1024,metresPerPixel:39.0625,
 forcedLod:0,viewsPerSite:1,hydrology:false,output:relative,
 limitations:['Uses the existing live viewer and may generate new neural cache entries.',
 'Source biome/relief labels stay provisional until native measurement and visual review.',
 'Production rendering may still change: working PNGs never count toward the final900 images.']};
console.log(JSON.stringify(plan));
if(process.argv.includes('--run'))(async()=>{
 const url='http://127.0.0.1:8765/api/world?'+new URLSearchParams({seed:manifest.seed,world_profile:manifest.generationProfile});
 const response=await fetch(url);assert(response.ok);const world=await response.json();
 assert.equal(world.world_identity,manifest.worldIdentity,'Runtime version changed: do not warm another world');
 assert.equal(world.cache_profile,manifest.runtimeCacheProfile);
 assert.deepEqual(world.generation_stages,manifest.generationStages);
 fs.mkdirSync(out,{recursive:true});
 const selection={label:plan.label,world_identity:manifest.worldIdentity,generation_profile:manifest.generationProfile,
  cache_profile:manifest.runtimeCacheProfile,patches:sites.map(s=>({...s,available:true})),plan};
 const selectionFile=path.join(out,'selection.json');fs.writeFileSync(selectionFile,JSON.stringify(selection,null,2));
 process.env.REVIEW_SELECTION=selectionFile;process.env.REVIEW_OUT=relative;
 process.env.REVIEW_WARM_ONLY='1';process.env.REVIEW_SIZE='1024';process.env.REVIEW_LOD='0';process.env.REVIEW_MPP='39.0625';
 require('./review_render_v2_real_40km.cjs');
})().catch(error=>{console.error(error);process.exitCode=1;});
