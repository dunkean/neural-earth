// Actual BRDF GPU quadrature; isolated material probe, never visual terrain acceptance.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..');
require(path.join(root,'web/terrain_render_v2.js'));const r3=require(path.join(root,'web/terrain_render3d.js'));
assert(r3.shaders.forest.includes('1.1*delta,1.6*delta/sqrt(delta*delta+.015)'),'Crown geometry changed; regenerate mean LUT');
assert(r3.shaders.forest.includes('.6+.4*(1.-hit.rel*hit.rel)'),'Crown intrinsic AO changed; regenerate mean LUT');
const lighting=r3.shaders.lighting.replace('clamp(c,vec3<f32>(0.),vec3<f32>(1.))','max(c,vec3<f32>(0.))');
const cases=[];for(const rgb of [[.16,.255,.13],[.5,.55,.4]])for(const needle of [0,1])for(const altitude of [5,27,45,63,85])for(const strength of [0,.5,1])cases.push({rgb,needle,altitude,strength});
const branches=cases.map((c,i)=>`case ${i}u:{rgb=vec3<f32>(${c.rgb.join(',')});needle=${c.needle}.;alt=${c.altitude}.;strength=${c.strength};}`).join('\n');
const code=`${lighting}\n${TerrainRenderV2.canopyMeanShader}
@group(0) @binding(0) var<storage,read_write> output:array<f32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3<u32>){
 var rgb=vec3<f32>(0.);var needle=0.;var alt=0.;var strength=0.;switch(id.x){${branches}default:{return;}}
 let a=alt*.01745329252;let l=vec3<f32>(cos(a),sin(a),0.);let l2=vec3<f32>(.75*cos(a)/.70710678,.63*sin(a)/.70710678,-.2*cos(a)/.70710678);
 let albedo=groundToLinear(rgb);let amount=clamp(strength,0.,1.);var integral=vec3<f32>(0.);var displayMean=vec3<f32>(0.);
 for(var i=0;i<32;i++){let r=sqrt((f32(i)+.5)/32.);for(var j=0;j<64;j++){
  let phi=(f32(j)+.5)*6.28318530718/64.;let g=mix(1.1*r,1.6*r/sqrt(r*r+.015),needle);
  var m:GroundMaterial;m.albedo=albedo;m.roughness=.9;m.normal=normalize(vec3<f32>(-g*cos(phi),1.,-g*sin(phi)));m.occlusion=1.-.4*r*r;
  let ambient=albedo*(.34+.16*m.normal.y)*m.occlusion;
  let value=mix(albedo,ambient,amount)+max(strength,0.)*(2.2*groundBrdf(m,vec3<f32>(0.,1.,0.),l)+.58*groundBrdf(m,vec3<f32>(0.,1.,0.),l2));
  integral+=value/2048.;displayMean+=pow(max(value,vec3<f32>(0.)),vec3<f32>(1./2.2))/2048.;
 }}
 var flat:GroundMaterial;flat.albedo=albedo;flat.normal=vec3<f32>(0.,1.,0.);flat.roughness=.9;flat.occlusion=1.;
 let base=shadeTerrainGround(flat,vec3<f32>(0.,1.,0.),l,l2,strength);
 let actual=v2CanopyMeanRatio(rgb,needle,l,l2,strength);let expected=pow(max(integral,vec3<f32>(0.)),vec3<f32>(1./2.2))/max(base,vec3<f32>(.0001));
 for(var c=0;c<3;c++){output[id.x*9u+u32(c)]=actual[c];output[id.x*9u+3u+u32(c)]=expected[c];output[id.x*9u+6u+u32(c)]=displayMean[c]/max(base[c],.0001);}
}`;
(async()=>{let browser;const server=http.createServer((_,res)=>res.end('<!doctype html><title>Canopy mean BRDF probe</title>'));await new Promise(r=>server.listen(0,'127.0.0.1',r));try{
 browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}`);
 const values=await page.evaluate(async({code,count})=>{const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('Actual WebGPU required');const d=await adapter.requestDevice();const module=d.createShaderModule({code});const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(e=>e.message).join('\n'));const pipeline=await d.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}}),bytes=count*9*4;
 const output=d.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=d.createBuffer({size:bytes,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});const group=d.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:output}}]}),e=d.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(count);p.end();e.copyBufferToBuffer(output,0,read,0,bytes);d.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const values=Array.from(new Float32Array(read.getMappedRange()));read.unmap();d.destroy();return values;},{code,count:cases.length});
 let maxError=0,maxDisplayMeanBias=0;const comparisons=cases.map((c,i)=>{const actual=values.slice(i*9,i*9+3),expected=values.slice(i*9+3,i*9+6),displayMean=values.slice(i*9+6,i*9+9);actual.forEach((a,k)=>{maxError=Math.max(maxError,Math.abs(a-expected[k]));maxDisplayMeanBias=Math.max(maxDisplayMeanBias,Math.abs(a-displayMean[k]));});return {...c,actual,expected,displayMean};});assert(maxError<.004,`LUT versus exact production BRDF quadrature error ${maxError}`);
 const out=path.join(root,'output/render-v2-floor/canopy-mean-probe.json');fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify({kind:'Technical intrinsic analytic crown probe, no neighbour shadows/overlap/clumps; not visual acceptance',cases:cases.length,maxError,maxDisplayMeanBias,comparisons},null,2));console.log(JSON.stringify({cases:cases.length,maxError,maxDisplayMeanBias,report:out}));
}finally{await browser?.close();await new Promise(r=>server.close(r));}})().catch(e=>{console.error(e);process.exitCode=1;});
