// The same material functions, evaluated as an unresolved population on
// coarse physical terrain. No local city/occupation inputs are synthesized.
var<private> magnaCoarsePopulation:vec4<f32>;
fn magnaCoarseSatellite(e:V2Env,p:vec2<f32>,height:f32,g:vec2<f32>,population:vec3<f32>,l:vec3<f32>,l2:vec3<f32>)->vec3<f32>{
  let fp=params.viewport.w;let world=(p+.5)*fp;
  magnaCoarsePopulation=vec4<f32>(population,e.tone);
  mu_global.uMaterialClimate=vec2<f32>((clim(p,21)+clim(p,24))*.5,clim(p,22)+clim(p,25));
  mu_global.uMaterialSeed=u32(surfaceMeta(6));mu_global.uQuality=0;mu_global.uV2Season=params.render3.w;
  mu_global.uForestRect=vec3<f32>(0.,0.,fp*params.field.x);mu_global.uForestTrees=vec4<f32>(1.);
  mu_global.uHasOccupation=0;mu_global.uMaterialZone=0;
  mu_renderGuideBaseOpen=r3BaseOpen;mu_renderGuideBaseClosed=r3BaseClosed;mu_renderGuideSediment=r3Sediment;
  mu_renderGuideWoods=population.r*select(0.,1.,height>=0.);
  mu_renderGuideBeach=r3Beach;mu_renderGuideFlats=r3Flats*(1.-groundSmooth(240.,960.,fp));
  mu_renderGuideShore=select(1000.,0.,r3Beach>.001);mu_renderGuideOpenness=clamp(r3Beach/.8,0.,1.);
  let slope=vec3<f32>(height,g*2.*params.lighting.w);
  // Bake a flat-light area response; the fragment applies the canonical
  // DEM hillshade and display shoulder once, as on the fine path.
  let m=mu_v2GroundEnv(magnaEnvironment(e,world,vec3<f32>(height,g),fp),vec3<f32>(height,0.,0.),world,fp,l,l2,params.lighting.x,r3BaseOpen);
  return mu_v2ShadeGround(m,vec3<f32>(0.,1.,0.),l,l2,params.lighting.x);
}
