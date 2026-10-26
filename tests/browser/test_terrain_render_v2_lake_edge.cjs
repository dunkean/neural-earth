// Technical GPU regression of the production lake/DEM intersection helper.
// This is not a visual acceptance image or a generated terrain substitute.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {loadPlaywright,browserExecutable}=require('../../tools/platform.cjs');
const source=fs.readFileSync(path.resolve(__dirname,'../../web/terrain_render_v2.js'),'utf8');
const helper=source.match(/fn v2LakeCoverage\([\s\S]*?\n      }/)[0];
const samples=[
 // coverage, lake level, DEM height, gradient x/z, footprint
 [.2,100,99,.1,0,.15625], [.8,100,101,.1,0,.15625],
 [.4,100,100,.1,0,.15625], [.2,100,99,.1,0,150],
 [.8,100,101,.1,0,150], [.2,100,99,.1,0,30],
 [.2,100,99,0,0,.15625], [.8,100,101,0,0,.15625],
 [.5,100,99.999,.1,.2,.15625], [.5,100,100.001,.1,.2,.15625]
];
(async()=>{
 const browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});
 try{
  const page=await browser.newPage();
  await page.route('https://lake-edge.test/',route=>route.fulfill({contentType:'text/html',body:'<!doctype html><title>GPU lake intersection</title>'}));
  await page.goto('https://lake-edge.test/');
  const actual=await page.evaluate(async({helper,samples})=>{
   const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('WebGPU adapter required');
   const device=await adapter.requestDevice();
   const rows=samples.map((s,i)=>`out[${i}]=v2LakeCoverage(${s[0]},${s[1]}.,vec3<f32>(${s[2]},${s[3]},${s[4]}),${s[5]});`).join('\n');
   const module=device.createShaderModule({code:`
    fn sstep(a:f32,b:f32,x:f32)->f32{let t=clamp((x-a)/(b-a),0.,1.);return t*t*(3.-2.*t);}
    ${helper}
    @group(0) @binding(0) var<storage,read_write> out:array<f32>;
    @compute @workgroup_size(1) fn main(){${rows}}
   `});
   const info=await module.getCompilationInfo();if(info.messages.some(m=>m.type==='error'))throw Error(JSON.stringify(info.messages));
   const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
   const size=samples.length*4,buffer=device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
   const read=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer}}]});
   const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(1);pass.end();e.copyBufferToBuffer(buffer,0,read,0,size);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const values=Array.from(new Float32Array(read.getMappedRange()));read.unmap();device.destroy();return values;
  },{helper,samples});
  assert.equal(actual[0],1,'Submerged close lake must be opaque');
  assert.equal(actual[1],0,'Emergent island must remain dry');
  assert.equal(actual[2],.5,'Plane intersection must be antialiased');
  assert(Math.abs(actual[3]-.2)<1e-6&&Math.abs(actual[4]-.8)<1e-6,'Distant filtered area coverage changed');
  assert(actual[5]>.2&&actual[5]<1,'Footprint transition is not continuous');
  assert.equal(actual[6],1);assert.equal(actual[7],0);
  assert(actual[8]>.5&&actual[8]<.6&&actual[9]<.5&&actual[9]>.4,'One-pixel edge derivative lost');
  console.log(JSON.stringify({kind:'GPU technical lake intersection',actual}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
