// Units and equations mirrored by backend/terrain_conditioning.py and terrain_orogen.py.
// Cross-runtime fixtures guard these conventions.
export function physicalFields(result) {
  const count=result.numRegions, fields=new Float32Array(count*5), correction=Math.sqrt(12/11);
  for(let i=0;i<count;i++) {
    const e=result.r_elevation[i], t=Math.max(0,Math.min(1,e));
    fields[i]=e<=0?e*10000:6000*t**4*(5-4*t)+1000*t*(1-t)**4;
    const ts=result.r_temperature_summer[i]*90-45,tw=result.r_temperature_winter[i]*90-45;
    const ps=result.r_precip_summer[i]*1000,pw=result.r_precip_winter[i]*1000;
    fields[count+i]=(ts+tw)*0.5;
    fields[2*count+i]=Math.abs(ts-tw)*100*correction/(2*Math.sqrt(2));
    fields[3*count+i]=ps+pw;
    fields[4*count+i]=Math.abs(ps-pw)*100*correction/(ps+pw+12);
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
    threshold:s.orogen_convergence_threshold??.75,motion:s.orogen_motion_strength??1,spread:s.orogen_spread??5,
    skipPost:s.relief_pipeline==='original',climateSettings:Object.fromEntries(Object.entries(s).filter(([k])=>k.startsWith('orogen_')).map(([k,v])=>[k.slice(7),v]))};
}
