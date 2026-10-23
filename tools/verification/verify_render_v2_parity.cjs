// Compile the actual environmental equations and compare GPU scalar fields
// against the shared Python reference fixture. No neural jobs or live server.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..');
require(path.join(root,'web/terrain_render_v2.js'));
const fixture=JSON.parse(fs.readFileSync(path.join(root,'tests/fixtures/render_v2_environment.json'),'utf8'));
const renderer=fs.readFileSync(path.join(root,'web/terrain_renderer.js'),'utf8');
const first=renderer.indexOf('    fn surfaceHash('),last=renderer.indexOf('    // (unit-variance pattern',first);
assert(first>=0&&last>first,'Actual renderer noise functions must be found');
const fields=Object.keys(fixture.cases[0].expected),f=v=>Number.isInteger(v)?v+'.':String(v);
const vec=(n,v)=>`vec${n}<f32>(${v.map(f).join(',')})`;
const cases=fixture.cases.map((c,i)=>{
 const args=[vec(3,c.point),f(c.height),vec(2,c.gradient),...['tpi','ts','tw','ps','pw','sand','bank','date'].map(k=>f(c[k])),'0.',c.seed+'u'];
 return `case ${i}u:{let e=v2Evaluate(${args.join(',')});${fields.map((key,j)=>`out[${i*fields.length+j}]=e.${key};`).join('')} }`;
}).join('\n');
const code=`fn sstep(a:f32,b:f32,x:f32)->f32{let t=clamp((x-a)/(b-a),0.,1.);return t*t*(3.-2.*t);}
${renderer.slice(first,last)}
${globalThis.TerrainRenderV2.environment}
@group(0) @binding(0) var<storage,read_write> out:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3<u32>){switch(id.x){${cases} default:{}}}`;
(async()=>{
 const server=http.createServer((_,res)=>res.end('<!doctype html><title>Render v2 parity</title>'));
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 let browser;
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
  const page=await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const report=await page.evaluate(async({code,count})=>{
   const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('A real WebGPU adapter is required');
   const device=await adapter.requestDevice();
   const module=device.createShaderModule({label:'Actual Render v2 environment parity',code});
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>`${m.lineNum}:${m.linePos} ${m.message}`);
   if(errors.length)throw Error(errors.join('\n'));
   const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
   const bytes=count*4;
   const output=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
   const read=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(count);pass.end();
   encoder.copyBufferToBuffer(output,0,read,0,bytes);device.queue.submit([encoder.finish()]);
   await read.mapAsync(GPUMapMode.READ);const values=Array.from(new Float32Array(read.getMappedRange()));read.unmap();
   const info=adapter.info;const adapterInfo=info?{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description}:{};
   output.destroy();read.destroy();device.destroy();return{values,adapter:adapterInfo};
  },{code,count:fixture.cases.length*fields.length});
  const comparisons=[];let maxAbsolute=0;
  fixture.cases.forEach((c,i)=>fields.forEach((key,j)=>{
   const expected=c.expected[key],actual=report.values[i*fields.length+j],difference=Math.abs(expected-actual);
   maxAbsolute=Math.max(maxAbsolute,difference);
   comparisons.push({case:c.name,field:key,expected,actual,difference});
   assert(Number.isFinite(actual),`${c.name}/${key}: nonfinite GPU result`);
   assert(difference<.002,`${c.name}/${key}: CPU ${expected}, GPU ${actual}, difference ${difference}`);
  }));
  const result={ok:true,cases:fixture.cases.length,fields:fields.length,maxAbsolute,tolerance:.002,adapter:report.adapter,comparisons};
  const target=path.join(root,'output/render-v2-review/wgsl-parity.json');fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,JSON.stringify(result,null,2));
  console.log(JSON.stringify({ok:true,cases:result.cases,fields:result.fields,maxAbsolute,adapter:result.adapter,report:target}));
 }finally{if(browser)await browser.close();server.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
