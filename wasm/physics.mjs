// Units and equations mirrored by backend/terrain_conditioning.py and terrain_orogen.py.
// Cross-runtime fixtures guard these conventions.
export function physicalFields(result) {
  const count=result.numRegions, fields=new Float32Array(count*5);
  for(let i=0;i<count;i++) {
    fields[i]=result.r_elevation[i]*1000;
    for(const [channel,key]of ['bio1','bio4','bio12','bio15'].entries())fields[(channel+1)*count+i]=result['r_'+key][i];
  }
  return fields;
}
export async function orogenSeed(seed) {
  if(BigInt(seed)<=0x7fffffffn)return Number(seed);
  const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(`orogen-tectonic-cuda-v1:${seed}`));
  return new DataView(digest).getUint32(0,true)&0x7fffffff;
}
export function orogenRequest(seed,s) {
  return {seed,N:s.orogen_detail??100000,P:s.orogen_plate_count??80,jitter:.75,
    nMag:s.orogen_roughness??.4,numContinents:s.orogen_continent_count??3,continentSizeVariety:s.orogen_continent_variety??.85,
    landCoverage:s.orogen_land_coverage??.3,smoothing:s.orogen_smoothing??.1,terrainWarp:s.orogen_warp??.75,
    hydraulicErosion:s.orogen_hydraulic??.5,thermalErosion:s.orogen_thermal??.1,glacialErosion:s.orogen_glacial??.5,
    ridgeSharpening:s.orogen_sharpening??.5,temperatureOffset:s.orogen_temperature_offset??0,precipitationOffset:s.orogen_precipitation_offset??0,
    skipPost:s.relief_pipeline==='original',climateSettings:Object.fromEntries(Object.entries(s).filter(([k])=>k.startsWith('orogen_')).map(([k,v])=>[k.slice(7),v]))};
}
