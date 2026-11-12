// The same Orogen v3 CPU core and monthly-climate adapter as the server.
import {meshFromArrays} from './orogen/upstream/v2/js/core/sim-mesh.js';
import {finishPlanet} from './orogen/upstream/v2/js/finish/finish.js';
import {FINISH} from './orogen/upstream/v2/js/finish/finish-config.js';
import {terrainConfig} from './orogen/upstream/v2/js/terrain/terrain-config.js';
import {setParams} from './orogen/upstream/v2/js/climate/config.js';
import {monthlyClimate} from './orogen/climate.mjs';
const send=self.postMessage.bind(self);
let result;
self.postMessage=value=>{
  if(value.type==='done')result=value;
  else if(value.type==='error')throw Error(value.stack||value.message);
  else if(value.type==='progress')send(value);
};
await import('./orogen/upstream/js/planet-worker.js');
const generate=self.onmessage;
self.onmessage=async({data:request})=>{
  try{
    globalThis.OROGEN_CLIMATE=request.climateSettings||{};
    setParams(Object.fromEntries(Object.entries(request.climateSettings||{}).filter(([k])=>k.startsWith('v3_')).map(([k,v])=>[k.slice(3).toUpperCase(),v])));
    generate({data:{...request,cmd:'generate'}});
    const s=result.shared,f=s.finish,mesh=meshFromArrays(s.mesh);
    const options={greenhouseWm2:request.temperatureOffset||0,precipScale:Math.max(0,1+.5*(request.precipitationOffset||0)),tiltDeg:request.climateSettings?.tilt_deg??23.44};
    const state={mesh:s.mesh,heightKm:s.heightKm,fields:{}};
    if(!request.skipPost){
      const out=await finishPlanet({frame:f.frame,simMesh:meshFromArrays(f.simMesh),seed:f.seed,cells:mesh.numCells,mesh,surface:()=>f.surface,
        cfg:terrainConfig(f.terrain),F:{...FINISH,...f.finish},climateOptions:options,onStage:label=>send({type:'progress',label})});
      state.heightKm=out.z;state.fields.ocean_mask=Float32Array.from(out.ocean);
    }
    const {biomeColor}=await import('./orogen/vendor/color-map.js');
    const fields=monthlyClimate(state,options,biomeColor),xyz=new Float32Array(s.mesh.xyz.length);
    for(let c=0;c<mesh.numCells;c++){xyz[3*c]=mesh.xyz[3*c];xyz[3*c+1]=mesh.xyz[3*c+2];xyz[3*c+2]=mesh.xyz[3*c+1];}
    const out={type:'done',numRegions:mesh.numCells,r_xyz:xyz,r_elevation:state.heightKm,...Object.fromEntries(Object.entries(fields).map(([k,v])=>['r_'+k,v]))};
    send(out,Object.values(out).filter(ArrayBuffer.isView).map(a=>a.buffer));
  }catch(error){send({type:'error',message:error.message,stack:error.stack});}
};
send({type:'ready'});
