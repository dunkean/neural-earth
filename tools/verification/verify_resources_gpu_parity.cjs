// Compile the actual resource WGSL evaluator and compare it with the NumPy
// mirror on tests/fixtures/resources_gpu_parity.json (build_resources_gpu_fixture.py).
// No server, no neural jobs. Requires a real WebGPU adapter.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..');
require(path.join(root,'web/terrain_resources_gpu.js'));
const RG=globalThis.TerrainResourcesGPU;
const fixture=JSON.parse(fs.readFileSync(path.join(root,'tests/fixtures/resources_gpu_parity.json'),'utf8'));
const code=`${RG.core}
struct Case { g:vec4<i32>, a:vec4<f32>, b:vec4<f32>, c:vec4<f32> }
@group(1) @binding(0) var<storage,read_write> out:array<f32>;
@group(1) @binding(1) var<storage,read> cases:array<Case>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3<u32>){
  let k=cases[id.x];let gi=k.g.x;let gj=k.g.y;let lod=k.g.z;let o=id.x*4u;
  if(k.g.w==0){out[o]=rContinuous(gi,gj,lod,k.a.x);}
  else if(k.g.w==1){out[o]=f32(rRealized(gi,gj,lod));out[o+1u]=f32(rDominant(gi,gj,lod));}
  else if(k.g.w==2){out[o]=rCyclic(gi,gj,lod);}
  else if(k.g.w==3){var d=0.;let rgb=rDeposits(gi,gj,lod,30.*exp2(f32(lod)),k.a.x,vec3<f32>(.5),&d);out[o]=rgb.x;out[o+1u]=rgb.y;out[o+2u]=rgb.z;}
  else if(k.g.w==4){out[o]=rStock(gi,gj,lod,30.*exp2(f32(lod)));}
  else {out[o]=rTerms(rFeatures(k.a.x,k.b.xy,k.b.zw,k.a.y,k.a.z,k.a.w,k.c.x));}
}`;
const kinds={positive:0,bounded:0,seasonal:0,'potential-copper':0,'potential-iron':0,geology:1,cyclic:2,points:3,stock:4};
(async()=>{
  const server=http.createServer((_,res)=>res.end('<!doctype html><title>Resources WGSL parity</title>'));
  await new Promise(ok=>server.listen(0,'127.0.0.1',ok));
  let browser;
  try{
    browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
    const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`);
    const runs=fixture.configs.map(c=>({name:c.name,kind:kinds[c.name],params:Array.from(new Uint8Array(RG.pack(c.header,{season:c.season??.5,active:true}))),
      shape:c.header.field_shape,field:c.field,tau:c.tau||null,tauLayers:c.header.thresholds,deposits:c.deposits||null,buckets:c.buckets||null,
      cases:c.cases.map(k=>[k.gi,k.gj,k.lod,kinds[c.name],k.terms??k.valley??0])}));
    runs.push({name:'features',kind:5,params:Array.from(new Uint8Array(RG.pack({...fixture.configs[0].header,coef:fixture.coef},{}))),shape:[1,1,1],field:[0],tau:null,deposits:null,buckets:null,
      cases:fixture.features.map(f=>[0,0,0,5,f.height,f.small,f.large,f.r,f.g[0],f.g[1],f.gl[0],f.gl[1],f.river])});
    const report=await page.evaluate(async({code,runs})=>{
      const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('A real WebGPU adapter is required');
      const device=await adapter.requestDevice();
      const module=device.createShaderModule({label:'resources parity',code});
      const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>`${m.lineNum}:${m.linePos} ${m.message}`);
      if(errors.length)throw Error(errors.join('\n'));
      const L0=device.createBindGroupLayout({entries:[
        {binding:11,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'2d-array'}},
        {binding:12,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'2d-array'}},
        {binding:13,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
        {binding:14,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
        {binding:15,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}}]});
      const L1=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}}]});
      const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[L0,L1]}),compute:{module,entryPoint:'main'}});
      const tex=(bits,[k,h,w])=>{const t=device.createTexture({size:[w,h,k],format:'r16float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});device.queue.writeTexture({texture:t},new Uint16Array(bits),{bytesPerRow:w*2,rowsPerImage:h},[w,h,k]);return t;};
      const buf=(data,usage,min=16)=>{const b=device.createBuffer({size:Math.max(min,Math.ceil(data.byteLength/16)*16),usage:usage|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(b,0,data);return b;};
      const results=[];
      for(const run of runs){
        const field=tex(run.field,run.shape),tau=run.tau?tex(run.tau,[run.tauLayers,run.shape[1],run.shape[2]]):tex([0],[1,1,1]);
        const params=buf(new Uint8Array(run.params),GPUBufferUsage.UNIFORM);
        const deposits=buf(new Uint32Array(run.deposits||[0,0,0,0]),GPUBufferUsage.STORAGE,48),buckets=buf(new Uint32Array(run.buckets||[0,0]),GPUBufferUsage.STORAGE,16);
        const caseData=new ArrayBuffer(run.cases.length*64),ci=new Int32Array(caseData),cf=new Float32Array(caseData);
        run.cases.forEach((c,i)=>{ci.set(c.slice(0,4),i*16);cf.set(c.slice(4),i*16+4);});
        const cases=buf(new Uint8Array(caseData),GPUBufferUsage.STORAGE),bytes=run.cases.length*16;
        const out=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        const g0=device.createBindGroup({layout:L0,entries:[{binding:11,resource:field.createView({dimension:'2d-array'})},{binding:12,resource:tau.createView({dimension:'2d-array'})},{binding:13,resource:{buffer:params}},{binding:14,resource:{buffer:deposits}},{binding:15,resource:{buffer:buckets}}]});
        const g1=device.createBindGroup({layout:L1,entries:[{binding:0,resource:{buffer:out}},{binding:1,resource:{buffer:cases}}]});
        const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,g0);pass.setBindGroup(1,g1);pass.dispatchWorkgroups(run.cases.length);pass.end();
        enc.copyBufferToBuffer(out,0,read,0,bytes);device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);
        results.push({name:run.name,values:Array.from(new Float32Array(read.getMappedRange()))});read.unmap();
      }
      const info=adapter.info;return {results,adapter:info?{vendor:info.vendor,architecture:info.architecture}:{}};
    },{code,runs});
    const summary=[];let failures=0;
    for(const r of report.results){
      const cases=r.name==='features'?fixture.features:fixture.configs.find(c=>c.name===r.name).cases;
      let maxError=0,mismatches=0;
      cases.forEach((c,i)=>{
        const got=r.values.slice(i*4,i*4+4);
        if(r.name==='geology'){if(got[0]!==c.expected||got[1]!==c.dominant)mismatches++;return;}
        if(r.name==='cyclic'){if(got[0]!==c.expected)mismatches++;return;}
        if(r.name==='points'){c.expected.forEach((x,k)=>{const e=Math.abs(got[k]-x);maxError=Math.max(maxError,e);if(e>.004)mismatches++;});return;}
        const e=Math.abs(got[0]-c.expected)/Math.max(1,Math.abs(c.expected));maxError=Math.max(maxError,e);if(e>2e-3)mismatches++;
      });
      summary.push({name:r.name,cases:cases.length,maxError,mismatches});failures+=mismatches;
    }
    const result={ok:failures===0,adapter:report.adapter,summary};
    const target=path.join(root,'output/resources-gpu/wgsl-parity.json');fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,JSON.stringify(result,null,2));
    console.log(JSON.stringify(result));
    assert.equal(failures,0,'WGSL and NumPy disagree');
  }finally{if(browser)await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
