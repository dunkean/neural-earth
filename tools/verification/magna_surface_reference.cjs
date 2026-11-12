// Independent numerical oracle: execute the pinned upstream TypeScript, not
// a JavaScript copy of the Python implementation.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const {stripTypeScriptTypes}=require('node:module');
const root=path.resolve(__dirname,'../..'),context=vm.createContext({});
for(const name of ['terrain3dNoise.ts','terrainGround.ts']){
  const source=stripTypeScriptTypes(fs.readFileSync(path.join(root,'native/magna_surface',name),'utf8')).replace(/^export /gm,'');
  // Each source owns private helpers with the same names.
  vm.runInContext('{'+source+';Object.assign(globalThis,{'+(name.includes('3dNoise')?'createTerrain3dNoiseSampler':'createGroundDisplacementSampler')+'});}',context);
}
const points=[[-137.25,83.875],[5.625,-11.25],[171.75,291.125],[-12000001.25,7340012.75],[20000003.375,-10000007.5]];
context.points=points;
console.log(vm.runInContext(`JSON.stringify((()=>{
 const base=(x,y)=>1000+100*Math.sin(x/200)+60*Math.cos(y/140);
 const look={sourceCell:30,seed:17331,refine:true,detail:{amplitude:.008,hurst:1,lacunarity:1.5,octaves:8,start:1,slope:.5,flat:.004,shore:2}};
 const noise=createTerrain3dNoiseSampler(base,look);
 const flat=(x,y)=>2000+.6*x+.4*y;
 const ground=createGroundDisplacementSampler(flat,{seed:17331,sourceCell:30,climate:[14,900],macroSampler:flat});
 return points.map(([x,y])=>({x,y,noise:noise(x,y),ground:ground(x,y,30/64)-flat(x,y)}));
})())`,context));
