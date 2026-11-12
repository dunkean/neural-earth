// Planetary inputs around Magna Urbis' unchanged Satellite material functions.
fn magnaWaterCover(p:vec2<f32>)->f32{
  let pixel=p/params.viewport.w-.5;
  let w=hydroFiltered(pixel,1.);
  return select(0.,clamp(w.y,0.,1.),w.w>.5&&w.z>.5);
}
// Source microdetail is authored for a finite local map. Four overlapping
// 1024 m domains preserve it as a continuous seeded planetary field. Quintic
// weights have zero first/second derivatives at domain boundaries; normalizing
// residual variance prevents the midpoint becoming smoother than the edges.
// Every call uses at most 1024 m coordinates, including far negative tiles.
fn magnaMicro(p:vec2<f32>,s:f32,fp:f32,rock:f32,grass:f32,seed:u32)->vec3<f32>{
  if(fp>=10.){return vec3<f32>(1.);}
  let metres=r3.cells[0].xy*5200;let fraction=r3.fracs[0].xy*5200.;
  let whole=metres/1024;let q=(vec2<f32>(metres%1024)+fraction+p)/1024.;
  let fl=floor(q);let cell=whole+vec2<i32>(fl);let f=q-fl;
  let u=f*f*f*(f*(f*6.-15.)+10.);
  var tone=vec3<f32>(0.);var normal=vec2<f32>(0.);var cover=vec2<f32>(0.);var variance=0.;
  for(var k=0;k<4;k++){
    let offset=vec2<i32>(k&1,k>>1);let point=(f-vec2<f32>(offset))*1024.;
    let weight=select(1.-u.x,u.x,(k&1)!=0)*select(1.-u.y,u.y,(k>>1)!=0);
    let salt=surfaceHash(vec3<i32>(cell+offset,0),seed);
    mu_surfaceGroundPoint=point;
    let value=mu_surfaceMicroLocal(point,s,fp,rock,grass,salt);
    tone+=(value-1.)*weight;normal+=mu_surfaceMicroSlope*weight;cover+=mu_surfaceGroundSlope*weight;variance+=weight*weight;
  }
  let scale=inverseSqrt(max(variance,.25));
  mu_surfaceMicroSlope=normal*scale;mu_surfaceGroundSlope=cover*scale;
  mu_surfaceMicroTone=clamp(1.+tone*scale,vec3<f32>(.55),vec3<f32>(1.45));return mu_surfaceMicroTone;
}
// Reconstruct an exact integer metre origin plus a small fractional remainder.
// This keeps the source's 3.2--45 m snow grain stable near world coordinates
// of millions of metres, without evaluating noise in tile-local coordinates.
fn magnaPattern(p:vec3<f32>,scale:f32,fp:f32,salt:u32)->f32{
  let metres=r3.cells[0].xy*5200;let fraction=r3.fracs[0].xy*5200.;
  var num=metres;var rem=fraction+p.xy;var denominator=i32(scale);
  if(abs(scale-11.)<.01){
    num=vec2<i32>(4*metres.x-3*metres.y,3*metres.x+4*metres.y);
    rem=vec2<f32>(4.*fraction.x-3.*fraction.y,3.*fraction.x+4.*fraction.y)+p.xy*5.;denominator=55;
  }else if(abs(scale-3.2)<.01){num=metres*5;rem=(fraction+p.xy)*5.;denominator=16;}
  denominator=max(denominator,1);
  let whole=num/denominator;let q=(vec2<f32>(num%denominator)+rem)/f32(denominator);
  let fl=floor(q);let cell=vec3<i32>(whole+vec2<i32>(fl),0);let f=vec3<f32>(q-fl,0.);
  let u=f*f*f*(f*(f*6.-15.)+10.);
  let a=surfaceCorner(cell,f,vec3<i32>(0,0,0),salt);let b=surfaceCorner(cell,f,vec3<i32>(1,0,0),salt);
  let c=surfaceCorner(cell,f,vec3<i32>(0,1,0),salt);let d=surfaceCorner(cell,f,vec3<i32>(1,1,0),salt);
  return mix(mix(a,b,u.x),mix(c,d,u.x),u.y)*5.25*(1.-sstep(scale*.12,scale*.3,fp));
}
fn magnaEnvironment(e:V2Env,p:vec2<f32>,s:vec3<f32>,fp:f32)->mu_V2Env{
  var m:mu_V2Env;
  m.temperature=e.temperature;m.potential=e.potential;m.moisture=e.moisture;m.exposure=e.exposure;
  m.mineral=e.mineral;m.sand=e.sand;m.clay=e.clay;m.organic=e.organic;m.rock=e.rock;m.grass=e.grass;
  m.woody=e.woody;m.height=e.height;m.needle=e.needle;m.leaf=e.leaf;m.snow=e.snow;m.ice=e.ice;m.tone=e.tone;m.glacier=e.glacier;
  m.world=p;m.slope=length(s.yz);m.footprint=fp;m.seed=r3Seed();return m;
}
fn magnaSatelliteFragment(s:vec3<f32>,p:vec2<f32>,fp:f32,baseColor:vec3<f32>,macroLight:f32)->vec4<f32>{
  let local=v2At(p,s,fp);let e=v2Appearance(local,fp);let meanLod=max(log2(max(fp,params.viewport.w)/params.viewport.w),0.);
  let retained=r3Aux(p,0,meanLod);let inherited=r3Aux(p,5,meanLod);let sediment=r3Aux(p,1,meanLod).rgb;
  let beach=clamp(r3Aux(p,4,0.).a/.8,0.,1.);let c=r3Aux(p,3,0.);
  let ts=c.r*100.-50.;let tw=c.b*100.-50.;let annual=(ts+tw)*.5;
  let rain=c.g*5.*max(120.,20.*annual+280.);
  mu_global.uMaterialClimate=vec2<f32>(annual+.0065*max(s.x,0.),rain);
  mu_global.uMaterialSeed=r3Seed();mu_global.uQuality=0;mu_global.uV2Season=params.render3.w;
  mu_global.uGroundCover=vec4<f32>(-params.sunlight.x,params.sunlight.z,-params.sunlight.y,1.);
  mu_global.uForestRect=vec3<f32>(0.,0.,params.viewport.w*params.field.x);
  mu_global.uForestTrees=vec4<f32>(1.,1.,1.,1.);mu_global.uHasOccupation=0;mu_global.uMaterialZone=0;
  mu_renderGuideBaseOpen=baseColor;mu_renderGuideBaseClosed=inherited.rgb;
  mu_renderGuideSediment=sediment;mu_renderGuideWoods=e.woody;
  mu_renderGuideBeach=beach*.8;mu_renderGuideFlats=retained.g*(1.-groundSmooth(240.,960.,fp));
  mu_renderGuideShore=select(1000.,0.,beach>.001);mu_renderGuideOpenness=beach;
  let l=normalize(vec3<f32>(-params.sunlight.x,params.sunlight.z,-params.sunlight.y));
  let angle=atan2(l.z,l.x)+2.35619449;let horizontal=length(l.xz)/.70710678;
  let fill=vec2<f32>(.2*cos(angle)+.75*sin(angle),.2*sin(angle)-.75*cos(angle))*horizontal;
  let l2=vec3<f32>(fill.x,.63*l.y/.70710678,fill.y);
  let slope=vec3<f32>(s.x,s.yz*2.*params.lighting.w);
  mu_surfaceGroundHeight=s.x;mu_surfaceGroundGradient=s.yz;
  mu_surfaceGroundEnv=vec4<f32>(e.moisture,e.grass,e.rock,e.snow);
  mu_surfaceGroundShore=mu_renderGuideShore;mu_surfaceGroundBeach=beach;
  mu_surfaceGroundWet=magnaWaterCover(p);mu_surfaceGroundHasInputs=true;
  mu_surfaceGroundOccupation=vec3<f32>(1000.,-1000.,0.);
  v2CanopyTone=e.tone;
  var material=mu_v2GroundEnv(magnaEnvironment(local,p,s,fp),slope,p,fp,l,l2,params.lighting.x,baseColor);
  let microSlope=mu_surfaceMicroSlope+mu_surfaceGroundSlope*mu_surfaceGroundNormalGain;
  material.normal=normalize(material.normal+vec3<f32>(-microSlope.x,0.,-microSlope.y));
  mu_v2MeanOpen.normal=material.normal;mu_v2MeanLitter.normal=material.normal;
  // Keep Neural Earth's precision-safe crown lattice and hydrology on the
  // source material; city occupation, crops and roads have no planetary input.
  var native:GroundMaterial;native.albedo=material.albedo;native.normal=material.normal;
  native.roughness=material.roughness;native.occlusion=material.occlusion;
  v2FloorModulation=mu_v2FloorModulation;v2GroundSnow=e.snow;v2CanopyTone=e.tone;
  v2MacroNormal=normalize(vec3<f32>(-slope.y,1.,-slope.z));v2ReferenceRoughness=mix(.88,.54,e.rock);
  v2BeachCoverage=beach;v2MeanResponseEnabled=true;v2ForestNearReveal=0.;
  var leafOn=e;leafOn.leaf=1.;leafOn.snow=0.;var bare=e;bare.snow=0.;
  let closed=inherited.rgb*v2Canopy(bare)/max(v2Canopy(leafOn),vec3<f32>(.02));
  applyForest(&native,p,fp,l,RenderGuide(baseColor,mu_renderGuideOpen,closed,e.woody*e.leaf),forestClimate(c),false);
  mu_v2ForestReveal=v2ForestNearReveal;
  material.albedo=native.albedo;material.normal=native.normal;material.roughness=native.roughness;material.occlusion=native.occlusion;
  // Avoid shading the DEM twice: retain its canonical lighting while the
  // source BRDF contributes ground/crown microrelief and material response.
  var reference=mu_v2MeanOpen;reference.normal=normalize(vec3<f32>(-slope.y,1.,-slope.z));
  let tilted=mu_shadeTerrainGround(reference,vec3<f32>(0.,1.,0.),l,l2,params.lighting.x);
  reference.normal=vec3<f32>(0.,1.,0.);
  let flat=mu_shadeTerrainGround(reference,vec3<f32>(0.,1.,0.),l,l2,params.lighting.x);
  let raw=mu_v2ShadeGround(material,vec3<f32>(0.,1.,0.),l,l2,params.lighting.x);
  let normalizer=dot(flat,vec3<f32>(.2126,.7152,.0722))/max(.001,dot(tilted,vec3<f32>(.2126,.7152,.0722)));
  var rgb=mu_v2DisplayShoulder(raw*normalizer*macroLight);
  rgb=v2Final(rgb,s,p,fp,e);return vec4<f32>(rgb,1.);
}
