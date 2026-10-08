// Optional CUDA source adapter. CPU modules are never rewritten in place.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {parse} from './acorn.mjs';
const here=path.dirname(fileURLToPath(import.meta.url));
const vendor=path.join(here,'../vendor');
const [destination, flagsText]=process.argv.slice(2);
const flags=JSON.parse(flagsText);
fs.mkdirSync(destination,{recursive:true});
function walk(node, fn) {
 if(!node?.type)return; fn(node);
 for(const [k,v] of Object.entries(node))if(k!=='parent') {
  if(Array.isArray(v))v.forEach(x=>walk(x,fn));else if(v?.type)walk(v,fn);
 }
}
function names(node) {
 const declared=new Set(), used=new Set();
 walk(node,n=>{
  if(n.type==='VariableDeclarator')walk(n.id,id=>{if(id.type==='Identifier')declared.add(id.name);});
 });
 function visit(n,parent,key){
  if(!n?.type)return;
  if(n.type==='Identifier' && !(parent?.type==='MemberExpression'&&key==='property'&&!parent.computed)
   && !(parent?.type==='Property'&&key==='key'&&!parent.computed))used.add(n.name);
  for(const [k,v]of Object.entries(n))if(Array.isArray(v))v.forEach(x=>visit(x,n,k));else if(v?.type)visit(v,n,k);
 }
 visit(node);for(const n of declared)used.delete(n);
 for(const n of ['Math','Infinity','undefined','NaN'])used.delete(n);
 return {declared,used};
}
const knownHelpers=new Set(['smoothstep','elevToHeightKm','evaluateSpline','regionPressure','zonalBase','heuristicWind','plateVelocityAt','getPairIntensity','cellNoise','lookupKoppen']);
const allowedMethods=new Set(['fbm','noise3D','ridgedFbm','has']);
let coverage=[];
for(const entry of fs.readdirSync(vendor)) {
 if(!entry.endsWith('.js')){fs.copyFileSync(path.join(vendor,entry),path.join(destination,entry));continue;}
 let source=fs.readFileSync(path.join(vendor,entry),'utf8').replaceAll('\r\n','\n');
 const group=['wind.js','ocean.js','precipitation.js','temperature.js','heuristic-precip.js','climate-util.js','koppen.js'].includes(entry)?'climate':
  entry==='terrain-post.js'?'post': ['coarse-plates.js','elevation.js'].includes(entry)?'relief':null;
 if(entry==='climate-util.js') {
  source=source.replace('return function (lon) {','const lookup = function (lon) {');
  source=source.replace('    };\n}', '    };\n    lookup.gpuItcz={lats:itczLats,start:lonStart,step};\n    return lookup;\n}');
 }
 if(flags.climate && entry==='koppen.js'){
  const original=parse(source,{ecmaVersion:'latest',sourceType:'module'});let codes=[];
  walk(original,n=>{if(n.type==='VariableDeclarator'&&n.id.name==='KOPPEN_CLASSES')codes=n.init.elements.map(x=>x.properties.find(p=>p.key.name==='code').value.value);});
  source=source.replace(/const code = '([CD])' \+ precipPattern \+ tempLetter;\s*const id = CODE_TO_ID\[code\];/g,(_,band)=>`const id=lookupKoppen('${band}',precipPattern,tempLetter);`);
  source=source.replace(/const fallback = 'Df' \+ tempLetter;\s*r_koppen\[r\] = CODE_TO_ID\[fallback\]/g,"r_koppen[r] = lookupKoppen('D','f',tempLetter)");
  source=source.replace(/CODE_TO_ID\['([^']+)'\]/g,(_,code)=>String(codes.indexOf(code)));
  source+='\nfunction lookupKoppen(band,pattern,letter){\n'+codes.map((c,i)=>c.length===3&&'CD'.includes(c[0])?`if(band==='${c[0]}'&&pattern==='${c[1]}'&&letter==='${c[2]}')return ${i};`:'').join('\n')+'\nreturn undefined;\n}\n';
 }
 if(flags.relief && entry==='coarse-plates.js') {
  source=source.replace('let cur = 0;', '// GPU independent starting point');
  source=source.replace('let bestDot = px * coarse_xyz', 'let cur = Math.max(0, Math.min(NC-1,Math.floor((1-pz)*.5*(NC-1))));\n        let bestDot = px * coarse_xyz');
 }
 if(flags.relief && entry==='elevation.js') {
  // Convert seed-set side effects into per-region masks, then compact on CPU.
  const a=source.indexOf('export function findCollisions'),b=source.indexOf('//  Stress propagation');
  let part=source.slice(a,b);
  part=part.replace('const undulOctaves =', 'const mountainMask=new Uint8Array(numRegions),coastMask=new Uint8Array(numRegions),oceanMask=new Uint8Array(numRegions);\n    const undulOctaves =');
  part=part.replace('if (r_subductFactor[r] < 0.55) mountain_r.add(r);','if (r_subductFactor[r] < 0.55) mountainMask[r]=1;');
  part=part.replaceAll('else coastline_r.add(r);','else coastMask[r]=1;');
  part=part.replace('(collided ? coastline_r : ocean_r).add(r);','if(collided)coastMask[r]=1;else oceanMask[r]=1;');
  part=part.replace('(collided ? mountain_r : coastline_r).add(r);','if(collided)mountainMask[r]=1;else coastMask[r]=1;');
  part=part.replace('    return { mountain_r,', '    for(let i=0;i<numRegions;i++){if(mountainMask[i])mountain_r.add(i);if(coastMask[i])coastline_r.add(i);if(oceanMask[i])ocean_r.add(i);}\n    return { mountain_r,');
  const start=part.indexOf('    const pairIntensityCache'),end=part.indexOf('    const mountainMask');
  part=part.slice(0,start)+`    function getPairIntensity(a,b) {
        const lo=Math.min(a,b),hi=Math.max(a,b);
        let h=((lo*16807)^(hi*48271))>>>0;
        h=(((h>>16)^h)*0x45d9f3b)>>>0;
        return .5+(h%10001)/10000;
    }
`+part.slice(end);
  source=source.slice(0,a)+part+source.slice(b);
 }
 const ast=parse(source,{ecmaVersion:'latest',sourceType:'module'});
 const helpers={};walk(ast,n=>{if(n.type==='FunctionDeclaration'&&knownHelpers.has(n.id.name))helpers[n.id.name]=n;});
 const plans=[], edits=[];
 function batch(parent,inner,used){
  if(!parent||parent.type!=='ForStatement'||parent.init?.declarations?.[0]?.init?.value!==0||parent.test?.operator!=='<'||parent.update?.operator!=='++'||parent.body.type!=='BlockStatement')return null;
  const counter=parent.init.declarations[0].id.name;if(used.has(counter))return null;
  const before=[],after=[],swaps=[],extra=[];let seen=false,swapSource=null;
  for(const s of parent.body.body){
   if(s===inner){seen=true;continue;}
   if(s.type==='ExpressionStatement'&&s.expression.type==='CallExpression'){
    const c=s.expression;if(c.callee.type!=='MemberExpression'||c.callee.property.name!=='set'||c.callee.object.type!=='Identifier'||c.arguments.length!==1||c.arguments[0].type!=='Identifier')return null;
    (seen?after:before).push([c.callee.object.name,c.arguments[0].name]);continue;
   }
   if(s.type==='VariableDeclaration'&&s.declarations.length===1&&s.declarations[0].id.name==='swap'&&s.declarations[0].init.type==='Identifier'){swapSource=s.declarations[0].init.name;continue;}
   if(s.type==='ExpressionStatement'&&s.expression.type==='AssignmentExpression'&&s.expression.operator==='='&&s.expression.left.type==='Identifier'&&s.expression.right.type==='Identifier'){
    const {left,right}=s.expression;if(left.name===swapSource){swaps.push([left.name,right.name]);continue;}if(right.name==='swap'&&swaps.at(-1)?.[1]===left.name)continue;return null;
   }
   if(s.type==='ForStatement'){
    const body=s.body.type==='BlockStatement'&&s.body.body.length===1?s.body.body[0]:s.body;
    const x=body?.expression;
    if(x?.type==='AssignmentExpression'&&x.operator==='='&&x.left.type==='MemberExpression'&&x.right.type==='MemberExpression'&&x.left.object.type==='Identifier'&&x.right.object.type==='Identifier'&&x.left.property.name==='r'&&x.right.property.name==='r'){
     (seen?after:before).push([x.left.object.name,x.right.object.name]);continue;
    }
   }
   return null;
  }
  if(!seen)return null;
  for(const pair of [...before,...after,...swaps])extra.push(...pair);
  const bound=sourceText(parent.test.right);const capture=names(parent.test.right).used;
  return {parent,before,after,swaps,extra:[...new Set([...extra,...capture])],bound};
 }
 let sourceText=n=>source.slice(n.start,n.end);
 if(flags[group]) {
  function candidates(n,parentLoop=null){
   if(!n?.type)return;
   if(n.type==='ForStatement'&&n.init?.type==='VariableDeclaration'&&n.init.declarations.length===1) {
    const id=n.init.declarations[0].id.name;
    const first=n.body.type==='BlockStatement'?n.body.body[0]:null;
    const region=first?.type==='VariableDeclaration'&&first.declarations[0]?.id.name==='r'&&first.declarations[0]?.init;
    const uniqueLand=region?.type==='MemberExpression'&&['landCells','interiorLand'].includes(region.object.name)&&region.property.name===id;
    if((id==='r'||uniqueLand)&&n.init.declarations[0].init?.value===0&&n.test?.operator==='<'&&n.test.left.name===id&&n.update?.operator==='++') {
     const {declared,used}=names(n);let safe=true;const outputs=new Set();
     walk(n.body,x=>{
      if(['ForOfStatement','ForInStatement','NewExpression','FunctionDeclaration','ArrowFunctionExpression'].includes(x.type))safe=false;
      if(x.type==='AssignmentExpression'||x.type==='UpdateExpression') {
       const target=x.left||x.argument;
       if(target.type==='Identifier'&&!declared.has(target.name))safe=false;
       if(target.type==='MemberExpression') {
        if(!target.computed||target.object.type!=='Identifier'||!declared.has(id)||!source.slice(target.property.start,target.property.end).match(/\br\b/))safe=false;
        else outputs.add(target.object.name);
       }
      }
      if(x.type==='CallExpression') {
       if(x.callee.type==='Identifier'&&!knownHelpers.has(x.callee.name)&&!used.has(x.callee.name))safe=false;
       if(x.callee.type==='MemberExpression'&&x.callee.object.name!=='Math'&&!allowedMethods.has(x.callee.property.name))safe=false;
      }
     });
     // Read/write neighbour dependencies are not assumed safe by this adapter.
     walk(n.body,x=>{if(x.type==='MemberExpression'&&outputs.has(x.object.name)&&x.computed&&!source.slice(x.property.start,x.property.end).match(/\br\b/))safe=false;});
     if(safe&&outputs.size) {
      const allUsed=new Set(used),needed={};
      function include(name){if(!helpers[name]||needed[name])return;needed[name]=helpers[name];
       const u=names(helpers[name].body).used;
       for(const p of helpers[name].params)if(p.type==='Identifier')u.delete(p.name);else if(p.type==='AssignmentPattern')u.delete(p.left.name);
       for(const v of u){if(helpers[v])include(v);else allUsed.add(v);}
      }
      for(const v of used)include(v);
      for(const v of Object.keys(needed))allUsed.delete(v);
      // Imported scalar helper is shared with the original JS implementation.
      for(const builtin of ['elevToHeightKm','smoothstep'])if(!needed[builtin])allUsed.delete(builtin);
      const batchPlan=batch(parentLoop,n,allUsed);
      if(batchPlan)batchPlan.extra.forEach(x=>allUsed.add(x));
      const env=[...allUsed].sort();
      const plan={label:`${entry}:${n.loc?.start.line||source.slice(0,n.start).split('\n').length}`,group,index:id,body:n.body,bound:n.test.right,helpers:needed,outputs:[...outputs]};
      const index=plans.push(plan)-1;
      if(batchPlan){
       const {parent,before,after,swaps,bound}=batchPlan;
       plan.batchNames=[...new Set([...before,...after,...swaps].flat())];
       const swapCode=swaps.map(([a,b])=>`if((${bound})%2){const swap=${a};${a}=${b};${b}=swap;}`).join('');
       // Serialize the plan after adding batch names below, when emitting module.
       edits.push({start:parent.start,end:parent.end,text:`if(__gpuLoop(__gpuPlans[${index}],{${env.join(',')}},{iterations:${bound},before:${JSON.stringify(before)},after:${JSON.stringify(after)},swaps:${JSON.stringify(swaps)}})){${swapCode}}else{${source.slice(parent.start,parent.end)}}`});
      }else edits.push({start:n.start,end:n.end,text:`if(!__gpuLoop(__gpuPlans[${index}],{${env.join(',')}})){${source.slice(n.start,n.end)}}`});
      coverage.push({module:entry,label:plan.label,outputs:plan.outputs});
      return;
     }
    }
   }
   for(const v of Object.values(n))if(Array.isArray(v))v.forEach(x=>candidates(x,n.type==='ForStatement'?n:parentLoop));else if(v?.type)candidates(v,n.type==='ForStatement'?n:parentLoop);
  }
  candidates(ast);
 }
 const disjoint=edits.filter(e=>!edits.some(other=>other!==e&&other.start<=e.start&&other.end>=e.end));
 for(const e of disjoint.sort((a,b)=>b.start-a.start))source=source.slice(0,e.start)+e.text+source.slice(e.end);
 if(plans.length)source=`const __gpuLoop=globalThis.OROGEN_GPU_LOOP;\nconst __gpuPlans=${JSON.stringify(plans)};\n`+source;
 if(flags.propagation && entry==='elevation.js') {
  // Explicit alternative graph algorithms; no silent claim of CPU parity.
  source=source.replace('export function propagateStress(mesh, r_stress, r_subductFactor, r_plate, plateIsOcean, decayFactor, subductDecayFactor, numPasses) {',
   `export function propagateStress(mesh,r_stress,r_subductFactor,r_plate,plateIsOcean,decayFactor,subductDecayFactor,numPasses){
      return globalThis.OROGEN_GPU_SPECIAL('stress',{adjOffset:mesh.adjOffset,adjList:mesh.adjList,r_stress,r_subductFactor,r_plate,plateIsOcean,decayFactor,subductDecayFactor,numPasses});
   }
   function referencePropagateStress(mesh,r_stress,r_subductFactor,r_plate,plateIsOcean,decayFactor,subductDecayFactor,numPasses) {`);
  source=source.replace('export function assignDistanceField(mesh, seeds, stops, seed) {',
   `export function assignDistanceField(mesh,seeds,stops,seed){
      const result=new Float32Array(mesh.numRegions);
      globalThis.OROGEN_GPU_SPECIAL('distance',{adjOffset:mesh.adjOffset,adjList:mesh.adjList,seeds,stops,result});return result;
   }
   function referenceAssignDistanceField(mesh,seeds,stops,seed) {`);
 }
 if(flags.erosion && entry==='terrain-post.js') {
  const start=source.indexOf('export function erodeComposite('),body=source.indexOf('{',start);
  source=source.slice(0,body+1)+`
    return globalThis.OROGEN_GPU_SPECIAL('erode',{adjOffset:mesh.adjOffset,adjList:mesh.adjList,r_elevation,r_xyz,r_isOcean,
        hIters,K,m,dt,tIters,talusSlope,kThermal,gIters:gIters||0,glacialStrength:glacialStrength||0,neighborDist});
`+source.slice(body+1);
 }
 fs.writeFileSync(path.join(destination,entry),source);
}
fs.writeFileSync(path.join(destination,'gpu-coverage.json'),JSON.stringify(coverage));
