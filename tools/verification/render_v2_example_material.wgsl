// Independent research prototype from published blending equations.
// CC0 exemplars: Poly Haven rock_face_03 and sandy_gravel_02.
// Scalar luminance histogram preservation; not full RGB/PCA optimal transport.
@group(0) @binding(10) var exampleMaterial:texture_2d_array<f32>;
@group(0) @binding(11) var exampleSampler:sampler;
var<private> exampleOpaqueWitness:bool;
struct ExampleSurface { ratio:f32, gradient:vec3<f32>, rough:f32 }
const EXAMPLE_SIGMA=array<f32,18>(
  .166664475,.142408288,.128786730,.121190957,.114389426,.106402647,.100238119,.094002765,0.,
  .166664935,.151379468,.134828155,.118595805,.099738984,.080739854,.057213536,.027164964,0.);

fn exampleRotate(v:vec2<f32>,turn:u32)->vec2<f32>{
  if(turn==1u){return vec2<f32>(-v.y,v.x);}
  if(turn==2u){return -v;}
  if(turn==3u){return vec2<f32>(v.y,-v.x);}
  return v;
}
fn examplePatch(uv:vec2<f32>,vertex:vec2<i32>,layer:i32,lod:f32,seed:u32)->vec4<f32>{
  let h=groundHash(vec3<i32>(vertex.x,layer*43+71,vertex.y),seed+831u);
  let offset=vec2<f32>(f32(h&65535u),f32(h>>16u))/65536.;
  let turn=(h>>27u)&3u;
  let s=textureSampleLevel(exampleMaterial,exampleSampler,exampleRotate(uv,turn)+offset,layer,lod);
  let g=exampleRotate(s.gb,(4u-turn)&3u);
  return vec4<f32>(s.r,g,s.a);
}
// Equilateral triangle grid: (sqrt(12)*u, -2*u+4*v). The skewed world
// origin is also split in float64 on the CPU; no large cell is cast to f32.
fn examplePlane(local:vec2<f32>,origin:vec2<i32>,axis:i32,layer:i32,lod:f32,seed:u32)->vec4<f32>{
  let skewFrac=bitcast<vec2<f32>>(r3.cells[30].zw);
  let originalFraction=r3.fracs[30].yz;
  let grid=vec2<f32>(skewFrac[axis]+3.464101615*(local.x-originalFraction[axis]),-2.*local.x+4.*local.y);
  let gridOrigin=vec2<i32>(r3.cells[30][axis],-2*origin.x+4*origin.y);
  let fl=floor(grid);let q=grid-fl;let cell=gridOrigin+vec2<i32>(fl);
  let sourceUV=fract(local);
  var vertices=array<vec2<i32>,3>(cell,cell+vec2<i32>(1,0),cell+vec2<i32>(0,1));
  var w=vec3<f32>(1.-q.x-q.y,q.x,q.y);
  if(q.x+q.y>1.){
    vertices=array<vec2<i32>,3>(cell+vec2<i32>(1,1),cell+vec2<i32>(0,1),cell+vec2<i32>(1,0));
    w=vec3<f32>(q.x+q.y-1.,1.-q.x,1.-q.y);
  }
  let a=examplePatch(sourceUV,vertices[0],layer,lod,seed);
  let b=examplePatch(sourceUV,vertices[1],layer,lod,seed);
  let c=examplePatch(sourceUV,vertices[2],layer,lod,seed);
  let correction=inverseSqrt(dot(w,w));
  let blend=(a*w.x+b*w.y+c*w.z-vec4<f32>(.5,0.,0.,0.))*correction+vec4<f32>(.5,0.,0.,0.);
  let j=i32(floor(lod));let next=min(j+1,8);
  let sigma=mix(EXAMPLE_SIGMA[layer*9+j],EXAMPLE_SIGMA[layer*9+next],fract(lod));
  var ratio=1.;
  if(sigma>.00001){
    let standard=(blend.r-.5)/sigma;
    let u=clamp(standard/6.+.5,.5/256.,255.5/256.);
    let inverse=textureSampleLevel(exampleMaterial,exampleSampler,vec2<f32>(u,(lod+.5)/256.),2,0.);
    ratio=select(inverse.r,inverse.g,layer==1);
  }
  return vec4<f32>(ratio,blend.gba);
}
fn exampleTriplanar(p:vec2<f32>,s:vec3<f32>,fp:f32,layer:i32,seed:u32)->ExampleSurface{
  let lod=clamp(log2(max(fp,.01)*256./2.56),0.,8.);
  // fracs31 belongs to the canonical globe/chart mapping. Keep it untouched;
  // only unused fracs30.yz and cells31.xy carry this experiment's split.
  let local=r3.fracs[30].yz+p/2.56;
  let origin=r3.cells[31].xy;
  let hn=s.x/2.56;let hi=i32(floor(hn));let hf=fract(hn);
  let n=abs(normalize(vec3<f32>(-s.y,1.,-s.z)));
  let power=n*n*n*n;let weights=power/(power.x+power.y+power.z);
  var result:ExampleSurface;result.ratio=0.;result.gradient=vec3<f32>(0.);result.rough=0.;
  if(weights.y>.001){
    let v=examplePlane(local,origin,0,layer,lod,seed);
    result.ratio+=weights.y*v.r;result.gradient+=weights.y*vec3<f32>(v.g,0.,v.b);result.rough+=weights.y*v.a;
  }
  if(weights.x>.001){
    let v=examplePlane(vec2<f32>(local.y,hf),vec2<i32>(origin.y,hi),1,layer,lod,seed+112u);
    result.ratio+=weights.x*v.r;result.gradient+=weights.x*vec3<f32>(0.,v.b,v.g);result.rough+=weights.x*v.a;
  }
  if(weights.z>.001){
    let v=examplePlane(vec2<f32>(local.x,hf),vec2<i32>(origin.x,hi),0,layer,lod,seed+227u);
    result.ratio+=weights.z*v.r;result.gradient+=weights.z*vec3<f32>(v.g,v.b,0.);result.rough+=weights.z*v.a;
  }
  return result;
}
fn exampleAt(p:vec2<f32>,s:vec3<f32>,fp:f32,rock:f32,seed:u32)->ExampleSurface{
  var a:ExampleSurface;var b:ExampleSurface;
  a.ratio=1.;a.gradient=vec3<f32>(0.);a.rough=0.;b=a;
  let mixRock=sstep(.1,.8,rock);
  if(mixRock<.999){a=exampleTriplanar(p,s,fp,0,seed);}
  if(mixRock>.001){b=exampleTriplanar(p,s,fp,1,seed);}
  var result:ExampleSurface;result.ratio=mix(a.ratio,b.ratio,mixRock);
  result.gradient=mix(a.gradient,b.gradient,mixRock);result.rough=mix(a.rough,b.rough,mixRock);return result;
}
