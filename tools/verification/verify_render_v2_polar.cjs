// Exercise the production polar material directly: CPU/WGSL parity and an
// ocean-only visual atlas. No terrain generation, model or running server.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const {loadPlaywright,browserExecutable,pythonExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/render-v2-polar');fs.mkdirSync(out,{recursive:true});
require(path.join(root,'web/terrain_render_v2.js'));
const scenes=[
 ['open-sea',8,0,1,90],['marginal-pack',-5,0,1,90],['winter-pack',-18,0,1,90],
 ['glacial-bergs',-1,1,1,80],['bergs-and-pack',-10,1,1,120],
 ['frozen-lake',-12,0,2,18],['freezing-river',-8,0,3,8],['cold-river',-18,0,3,8],
];
const footprints=[.5,5,50,500],anchor=[100000,-370000,12000];
const samples=scenes.flatMap((scene,i)=>footprints.flatMap(fp=>[0,1,2].map(j=>({
 scene:i,name:scene[0],point:anchor.map((p,k)=>Math.fround(p+[j*173.5,-j*219.25,j*79.5][k])),fp,
 temperature:scene[1],glacier:scene[2],kind:scene[3],depth:scene[4],cover:1,
}))));
const reference=JSON.parse(execFileSync(pythonExecutable(),['-c',String.raw`
import json,sys,numpy as np
sys.path.insert(0,'backend')
from terrain_render_v2 import water_color,ice_material
from terrain_render import F,smooth
data=json.load(sys.stdin);result=[]
for c in data:
 t=F(c['temperature']);p=np.array([c['point']],F);one=np.ones(1,F)
 e={k:np.zeros(1,F) for k in ['temperature','potential','moisture','exposure','mineral','sand','clay','organic','rock','grass','woody','height','needle','leaf','snow','ice','tone','glacier']}
 e.update(temperature=one*t,moisture=one*F(.8),sand=one*F(.25),mineral=one*F(.5),snow=one*smooth(2,-5,t),ice=one*smooth(-1.8,-8,t),glacier=one*F(c['glacier']))
 d=one*F(c['depth']);kind=one*F(c['kind']);fp=F(c['fp'])
 rgb=water_color(d,kind,one,e,p,fp,17331)
 ice=ice_material(d,kind,e,p,fp,17331)
 result.append(np.concatenate([np.asarray(rgb).reshape(-1),np.asarray(ice).reshape(-1)]).tolist())
# Select an occupied existing fragment for a useful close view. This changes
# only the review camera, never the distribution, source, material or seed.
q=np.arange(-64,65,dtype=F)*20
x,y=np.meshgrid(q,q);p=np.stack((x.ravel()+100000,y.ravel()-370000,np.full(x.size,12000,F)),-1)
one=np.ones(x.size,F)
e={k:np.zeros(x.size,F) for k in e}
e.update(temperature=one*F(-1),moisture=one*F(.8),sand=one*F(.25),mineral=one*F(.5),snow=one*smooth(2,-5,F(-1)),glacier=one)
ice=ice_material(one*80,one,e,p,F(5),17331)
index=int(np.argmax(ice[:,0]+F(.01)*ice[:,1]-F(.000001)*np.linalg.norm(p[:,:2]-np.array([100000,-370000],F),axis=-1)))
json.dump(dict(values=result,berg_anchor=p[index].tolist(),berg_coverage=float(ice[index,0])),sys.stdout)
`],{cwd:root,input:JSON.stringify(samples),maxBuffer:4*1024**2}).toString());
const renderer=fs.readFileSync(path.join(root,'web/terrain_renderer.js'),'utf8'),first=renderer.indexOf('    fn surfaceHash('),last=renderer.indexOf('    // (unit-variance pattern',first);
assert(first>=0&&last>first);
const code=`fn sstep(a:f32,b:f32,x:f32)->f32{let t=clamp((x-a)/(b-a),0.,1.);return t*t*(3.-2.*t);}
${renderer.slice(first,last)}
${globalThis.TerrainRenderV2.environment}
struct Sample{point:vec4<f32>,env:vec4<f32>,media:vec4<f32>}
@group(0) @binding(0) var<storage,read> input:array<Sample>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4<f32>>;
fn environment(s:Sample)->V2Env{var e:V2Env;e.temperature=s.env.x;e.glacier=s.env.y;e.sand=.25;e.moisture=.8;e.mineral=.5;e.snow=sstep(2.,-5.,e.temperature);e.ice=sstep(-1.8,-8.,e.temperature);return e;}
@compute @workgroup_size(1) fn parity(@builtin(global_invocation_id) id:vec3<u32>){
 if(id.x>=arrayLength(&input)){return;}let s=input[id.x];let e=environment(s);
 let c=v2Water(s.env.w,s.env.z,1.,e,s.point.xyz,s.point.w,17331u);
 let ice=v2Ice(s.env.w,s.env.z,e,s.point.xyz,s.point.w,17331u);
 output[id.x*2u]=vec4<f32>(c,ice.x);output[id.x*2u+1u]=vec4<f32>(ice.yzw,0.);
}
@compute @workgroup_size(8,8,1) fn atlas(@builtin(global_invocation_id) id:vec3<u32>){
 if(id.x>=1024u||id.y>=256u||id.z>=arrayLength(&input)){return;}
 let fp=array<f32,4>(.5,5.,50.,500.)[id.x/256u];let s=input[id.z];
 let p=s.point.xyz+vec3<f32>((f32(id.x%256u)-127.5)*fp,(f32(id.y)-127.5)*fp,0.);
 output[(id.z*256u+id.y)*1024u+id.x]=vec4<f32>(v2Water(s.env.w,s.env.z,1.,environment(s),p,fp,17331u),1.);
}`;
const floats=entries=>Float32Array.from(entries.flatMap(c=>[...c.point,c.fp,c.temperature,c.glacier,c.kind,c.depth,1,0,0,0]));
(async()=>{
 const server=http.createServer((_,res)=>res.end('<!doctype html><title>Polar material review</title>'));await new Promise(ok=>server.listen(0,'127.0.0.1',ok));let browser;
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--enable-unsafe-webgpu']});const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result=await page.evaluate(async({code,points,scenes,anchor})=>{
   const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw Error('Actual WebGPU adapter required');const timing=adapter.features.has('timestamp-query');const device=await adapter.requestDevice({requiredFeatures:timing?['timestamp-query']:[]});
   const module=device.createShaderModule({code}),errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(errors.length)throw Error(errors.map(m=>m.message).join('\n'));
   const pipelines=await Promise.all(['parity','atlas'].map(entryPoint=>device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}})));
   const run=async(pipeline,data,count,dispatch)=>{
    const ib=device.createBuffer({size:data.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(ib,0,new Float32Array(data));
    const ob=device.createBuffer({size:count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:count*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:ib}},{binding:1,resource:{buffer:ob}}]});
    const query=timing?device.createQuerySet({type:'timestamp',count:2}):null;
    const resolved=timing?device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}):null;
    const queryRead=timing?device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}):null;
    const start=performance.now(),enc=device.createCommandEncoder(),pass=enc.beginComputePass(query?{timestampWrites:{querySet:query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(...dispatch);pass.end();enc.copyBufferToBuffer(ob,0,read,0,count*4);
    if(query){enc.resolveQuerySet(query,0,2,resolved,0);enc.copyBufferToBuffer(resolved,0,queryRead,0,16);}device.queue.submit([enc.finish()]);await Promise.all([read.mapAsync(GPUMapMode.READ),...(queryRead?[queryRead.mapAsync(GPUMapMode.READ)]:[])]);
    const values=new Float32Array(read.getMappedRange()).slice(),elapsed=performance.now()-start;let gpuMs=null;if(queryRead){const times=new BigUint64Array(queryRead.getMappedRange());gpuMs=Number(times[1]-times[0])/1e6;queryRead.unmap();queryRead.destroy();resolved.destroy();query.destroy();}read.unmap();ib.destroy();ob.destroy();read.destroy();return {values,elapsed,gpuMs};
   };
   const parity=await run(pipelines[0],points,points.length/12*8,[points.length/12]);
   const sceneData=scenes.flatMap(c=>[...(c[5]||anchor),0,c[1],c[2],c[3],c[4],1,0,0,0]);
   const atlas=await run(pipelines[1],sceneData,1024*256*scenes.length*4,[128,32,scenes.length]);
   const images=[];for(let s=0;s<scenes.length;s++){
    const canvas=document.createElement('canvas');canvas.width=1024;canvas.height=284;const ctx=canvas.getContext('2d');ctx.fillStyle='#142127';ctx.fillRect(0,0,1024,284);ctx.fillStyle='white';ctx.font='14px sans-serif';
    [0.5,5,50,500].forEach((fp,k)=>ctx.fillText(scenes[s][0]+' | '+fp+' m/px',k*256+6,19));const pixels=ctx.createImageData(1024,256);
    for(let i=0;i<1024*256;i++)for(let c=0;c<4;c++)pixels.data[i*4+c]=Math.round(Math.max(0,Math.min(1,atlas.values[(s*1024*256+i)*4+c]))*255);ctx.putImageData(pixels,0,28);images.push(canvas.toDataURL());
   }
   const info=adapter.info,report={values:Array.from(parity.values),images,adapter:{vendor:info.vendor,architecture:info.architecture,device:info.device},atlasDispatchAndReadbackMs:atlas.elapsed,atlasComputeGpuMs:atlas.gpuMs};device.destroy();return report;
  },{code,points:Array.from(floats(samples)),scenes:[...scenes,['glacial-berg-detail',-1,1,1,80,reference.berg_anchor]],anchor});
  let worst=0;const comparisons=[];samples.forEach((sample,i)=>reference.values[i].forEach((expected,j)=>{const actual=result.values[i*8+j],difference=Math.abs(actual-expected);assert(Number.isFinite(actual));worst=Math.max(worst,difference);comparisons.push({scene:sample.name,footprint:sample.fp,field:j,expected,actual,difference});assert(difference<.004,JSON.stringify(comparisons.at(-1)));}));
  result.images.forEach((image,i)=>fs.writeFileSync(path.join(out,(scenes[i]?.[0]||'glacial-berg-detail')+'.png'),Buffer.from(image.split(',')[1],'base64')));
  const report={ok:true,samples:samples.length,fields:7,maxAbsolute:worst,tolerance:.004,adapter:result.adapter,atlasDispatchAndReadbackMs:result.atlasDispatchAndReadbackMs,atlasComputeGpuMs:result.atlasComputeGpuMs,atlasPixels:1024*256*9,detailAnchor:reference.berg_anchor,detailCoverage:reference.berg_coverage,comparisons};fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({...report,comparisons:undefined,output:out}));
 }finally{if(browser)await browser.close();server.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
