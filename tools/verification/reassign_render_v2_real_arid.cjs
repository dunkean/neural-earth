// Reassign two already captured ARID sites to their measured and visually
// confirmed reliefs. No rendering/generation. Preserve the full initial pass.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const out=path.resolve(__dirname,'../../output/render-v2-40km/real-matrix'),file=path.join(out,'report.json');
const report=JSON.parse(fs.readFileSync(file,'utf8'));
const selectionPath=path.resolve(__dirname,'../../output/render-v2-40km/real-matrix-24-selection.json');
const initial=path.join(out,'attempts','initial-pass');fs.mkdirSync(initial,{recursive:true});
if(!fs.existsSync(path.join(initial,'report.json'))){
 for(const r of report.results)for(const f of r.frames)fs.copyFileSync(path.join(out,f.file),path.join(initial,f.file));
 fs.writeFileSync(path.join(initial,'report.json'),JSON.stringify(report,null,2));
}
const mapping={'arid-plateau':'arid-mountains','arid-mountains':'arid-plateau'};
if(!report.reassignments?.length){
 const originals=Object.fromEntries(Object.values(mapping).map(id=>[id,structuredClone(report.results.find(r=>r.id===id))]));
 const pixels=Object.fromEntries(Object.values(originals).flatMap(r=>r.frames.map(f=>[f.file,fs.readFileSync(path.join(out,f.file))])));
 report.reassignments=[];
 for(const [target,source] of Object.entries(mapping)){
  const old=originals[source],targetSlot=report.selection.slots.find(s=>s.id===target);
  assert(old.protocolOk&&old.frames.length===12);assert.equal(old.selection.biomeFamily,targetSlot.biomeFamily);
  assert.equal(old.nativeRelief,targetSlot.relief);assert(old.familyDominance>=.8);
  const next=structuredClone(old);next.id=target;
  next.reassignedFrom={id:source,originalRequestedRelief:old.selection.relief,originalAcceptance:old.acceptance,reason:'Measured native relief and direct visual review agree; ARID plateau/escarpment and mountain sites were originally predicted in opposite slots.',initialPass:'attempts/initial-pass/report.json'};
  next.selection={...structuredClone(targetSlot),center_m:old.selection.center_m,bounds_m:old.selection.bounds_m,source_stats:old.selection.source_stats,native_stats:old.nativeStats,status:'native-measured-and-visually-confirmed-reassignment',sourcePredictionRelief:old.selection.relief};
  next.acceptance={...old.acceptance,nativeReliefMatched:true,visualReliefReview:'confirmed-by-root-and-review-agent',status:'native-metrics-and-visual-relief-match'};
  for(const f of next.frames){const previous=f.file;f.file=target+previous.slice(source.length);fs.writeFileSync(path.join(out,f.file),pixels[previous]);}
  report.results[report.results.findIndex(r=>r.id===target)]=next;
  report.selection.slots[report.selection.slots.findIndex(s=>s.id===target)]=next.selection;
  report.reassignments.push({target,source,center_m:next.selection.center_m,reason:next.reassignedFrom.reason});
 }
}
for(const r of report.results){
 r.representativeFamily=r.familyDominance>=.70;
 r.usableMixedBiome=r.representativeFamily&&!r.acceptance.biomeFamilyMatched;
 if(r.usableMixedBiome)r.mixedBiomeNote='Representative mixed mountain transition; strict80% biome acceptance remains false. Fractions are in nativeStats.biome_fractions_land.';
}
fs.writeFileSync(file,JSON.stringify(report,null,2));fs.writeFileSync(selectionPath,JSON.stringify(report.selection,null,2));
console.log(JSON.stringify({sites:report.results.length,frames:report.results.reduce((n,r)=>n+r.frames.length,0),reassignments:report.reassignments,strictMatches:report.results.filter(r=>r.acceptance.biomeFamilyMatched&&r.acceptance.nativeReliefMatched).length,mixedRepresentative:report.results.filter(r=>r.usableMixedBiome).map(r=>r.id),initial}));
