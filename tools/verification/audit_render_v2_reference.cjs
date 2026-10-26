// Source candidates never fill a measured relief/biome coverage cell.
const fs=require('node:fs'),path=require('node:path');
const base=path.resolve(__dirname,'../../output/render-v2-reference');
const read=f=>fs.existsSync(path.join(base,f))?JSON.parse(fs.readFileSync(path.join(base,f),'utf8')):null;
const manifest=read('manifest.json'),storedReport=read('report.json'),reviewData=read('reviews.json');
const report=storedReport?.manifest?.worldIdentity===manifest.worldIdentity?storedReport:null;
const reviews=report?.runId&&reviewData?.runId===report.runId?(reviewData.sites||{}):{};
const results=report?.results||[];
const families={AF:['AF','AM'],CFB:['CFB','CFC'],BSK:['BSK','BSH'],ARID:['BWH','BWK'],DFC:['DFC','DSC'],POLAR:['ET','EF']};
const cells=[];
for(const [family,codes]of Object.entries(families))for(const relief of ['plain','rolling','plateau','mountains']){
 const requested=manifest.sites.filter(s=>s.biomeFamily===family&&s.requestedRelief===relief);
 const attempts=requested.map(s=>{
  const r=results.find(r=>r.id===s.id),fractions=r?.nativeStats?.biome_fractions_land;
  const share=fractions?codes.reduce((n,c)=>n+(fractions[c]||0),0):null;
  const review=reviews[s.id];
  return{id:s.id,center_m:s.center_m,sourceStatus:s.source_stats.status,measured:!!r?.nativeStats,
   protocolPassed:r?.protocolOk===true,measuredRelief:r?.nativeRelief??null,biomeFractions:fractions??null,
   familyDominance:share,strictBiomeMatched:share!==null&&share>=.8,
   representativeBiomeMatched:share!==null&&share>=.7,nativeReliefMatched:!!r?.nativeStats&&r.nativeRelief===relief,
   morphologyVisualAccepted:review?.reliefAccepted===true,
   materialVisualAccepted:review?.status==='accepted',review:review||null,failure:r?.failure??null};
 });
 const strict=attempts.filter(a=>a.protocolPassed&&a.strictBiomeMatched&&a.nativeReliefMatched);
 const representative=attempts.filter(a=>a.protocolPassed&&a.representativeBiomeMatched&&a.nativeReliefMatched);
 const visual=strict.filter(a=>a.morphologyVisualAccepted&&a.materialVisualAccepted);
 const visualRepresentative=representative.filter(a=>a.morphologyVisualAccepted&&a.materialVisualAccepted);
 cells.push({biomeFamily:family,requestedRelief:relief,attempts,
  strictNativeMatched:strict.map(a=>a.id),representativeNativeMatched:representative.map(a=>a.id),visuallyAccepted:visual.map(a=>a.id),
  visuallyAcceptedRepresentative:visualRepresentative.map(a=>a.id),
  status:visual.length?'visually-accepted-strict':strict.length?'native-match-visual-review-pending':
   visualRepresentative.length?'visually-accepted-mixed-transition-strict-gap':representative.length?'mixed-transition-only-strict-gap':attempts.some(a=>a.measured)?'native-mismatch-gap':'pending-native-measurement'});
}
// Separate actual sample coverage from success of the source preselection.
// Every counted sample retains its original id, target and provenance.
const actualCells=[];
for(const [family,codes]of Object.entries(families))for(const relief of ['plain','rolling','plateau','mountains']){
 const attempts=[];
 for(const r of results){
  if(!r.protocolOk||!r.nativeStats||r.nativeRelief!==relief)continue;
  const s=manifest.sites.find(s=>s.id===r.id);if(!s)continue;
  const fractions=r.nativeStats.biome_fractions_land,share=codes.reduce((n,c)=>n+(fractions[c]||0),0);
  if(share<=0)continue;
  const sourceShare=(families[s.biomeFamily]||s.biome_codes).reduce((n,c)=>n+(fractions[c]||0),0);
  const sourceTargetMatched=s.requestedRelief===null?null:r.nativeRelief===s.requestedRelief&&sourceShare>=.8;
  const review=reviews[s.id];
  attempts.push({id:s.id,center_m:s.center_m,actualRelief:relief,actualFamily:family,familyDominance:share,
   biomeFractions:fractions,strictBiomeMatched:share>=.8,representativeBiomeMatched:share>=.7,
   sourceFamily:s.biomeFamily,sourceReliefTarget:s.requestedRelief,sourceTargetMatched,
   failedSourceTarget:sourceTargetMatched===false,sourceProvenance:s.aliases,
   morphologyVisualAccepted:review?.actualReliefAccepted===true||sourceTargetMatched===true&&review?.reliefAccepted===true,
   materialVisualAccepted:review?.status==='accepted',review:review||null});
 }
 const strict=attempts.filter(a=>a.strictBiomeMatched),representative=attempts.filter(a=>a.representativeBiomeMatched);
 const accepted=rows=>rows.filter(a=>a.morphologyVisualAccepted&&a.materialVisualAccepted).map(a=>a.id);
 const visual=accepted(strict),visualRepresentative=accepted(representative);
 actualCells.push({biomeFamily:family,actualRelief:relief,attempts,
  strictNativeMatched:strict.map(a=>a.id),representativeNativeMatched:representative.map(a=>a.id),
  visuallyAccepted:visual,visuallyAcceptedRepresentative:visualRepresentative,
  status:visual.length?'actual-visually-accepted-strict':strict.length?'actual-native-strict-visual-pending':
   visualRepresentative.length?'actual-visually-accepted-mixed':representative.length?'actual-native-mixed-visual-pending':'actual-native-gap'});
}
const focus=['user-pack-ocean','warm-coast-beach-detail'].map(id=>{
 const s=manifest.sites.find(s=>s.id===id),r=results.find(r=>r.id===id);
 return{id,center_m:s.center_m,featureProvenance:s.featureProvenance,nearFocus:r?.nearFocus??null,
  protocolPassed:r?.protocolOk===true,review:reviews[id]??null,
  status:!r?.nearFocus?'pending-native160m-measurement':!r.nearFocus.geometryTargetMatched?'geometry-target-mismatch':
   reviews[id]?.status==='accepted'?'visually-accepted':'geometry-confirmed-material-review-pending'};
});
const audit={label:'Final source-target and actual-native coverage, kept separate',runId:report?.runId??null,
 worldIdentity:manifest.worldIdentity,ignoredHistoricalReport:!!storedReport&&!report,
 expectedSites:60,expectedFrames:900,capturedSites:results.length,protocolPassedSites:results.filter(r=>r.protocolOk).length,
 capturedFrames:results.reduce((n,r)=>n+r.frames.length,0),requiredCells:24,
 strictNativeMatchedCells:cells.filter(c=>c.strictNativeMatched.length).length,
 representativeNativeMatchedCells:cells.filter(c=>c.representativeNativeMatched.length).length,
 visuallyAcceptedCells:cells.filter(c=>c.visuallyAccepted.length).length,
 visuallyAcceptedRepresentativeCells:cells.filter(c=>c.visuallyAcceptedRepresentative.length).length,
 requestedTargetMatches:{strictNativeMatchedCells:cells.filter(c=>c.strictNativeMatched.length).length,
  representativeNativeMatchedCells:cells.filter(c=>c.representativeNativeMatched.length).length},
 actualNativeCoverage:{strictNativeMatchedCells:actualCells.filter(c=>c.strictNativeMatched.length).length,
  representativeNativeMatchedCells:actualCells.filter(c=>c.representativeNativeMatched.length).length,
  visuallyAcceptedCells:actualCells.filter(c=>c.visuallyAccepted.length).length,
  visuallyAcceptedRepresentativeCells:actualCells.filter(c=>c.visuallyAcceptedRepresentative.length).length,
  gaps:actualCells.filter(c=>!c.strictNativeMatched.length).map(c=>({biomeFamily:c.biomeFamily,relief:c.actualRelief,status:c.status})),cells:actualCells},
 gaps:cells.filter(c=>!c.strictNativeMatched.length).map(c=>({biomeFamily:c.biomeFamily,relief:c.requestedRelief,status:c.status})),
 cells,focus,limitations:manifest.limitations};
fs.writeFileSync(path.join(base,'coverage.json'),JSON.stringify(audit,null,2));
const table=cells.map(c=>`|${c.biomeFamily}|${c.requestedRelief}|${c.strictNativeMatched.join(', ')||'—'}|${c.representativeNativeMatched.join(', ')||'—'}|${c.status}|`).join('\n');
const actualTable=actualCells.map(c=>`|${c.biomeFamily}|${c.actualRelief}|${c.strictNativeMatched.join(', ')||'—'}|${c.representativeNativeMatched.join(', ')||'—'}|${c.attempts.filter(a=>a.representativeBiomeMatched&&a.failedSourceTarget).map(a=>`${a.id} (cible ${a.sourceReliefTarget})`).join(', ')||'—'}|${c.status}|`).join('\n');
const text=`# Couverture réelle du corpus\n\n${audit.capturedFrames}/900 images, ${audit.protocolPassedSites}/60 situations au protocole complet.\n\n## Réussite des cibles source demandées\n\n${audit.strictNativeMatchedCells}/24 cibles satisfont les critères natifs stricts ; ${audit.representativeNativeMatchedCells}/24 avec transitions représentatives ≥ 70 %. ${audit.visuallyAcceptedCells}/24 ont aussi une acceptation visuelle explicite. Ce premier indicateur mesure la réussite de la présélection, distincte de la couverture réelle ci-dessous.\n\nLes noms source et le nombre total de situations ne prouvent pas la couverture de 4 reliefs × 6 familles. Les échecs de cible restent visibles. Les seuils stricts ≥ 80 % ne sont pas abaissés pour les transitions naturelles.\n\n|Famille source|Relief demandé|Natif strict≥80%|Natif représentatif≥70%|État|\n|---|---|---|---|---|\n${table}\n\n## Détails ciblés\n\n${focus.map(f=>`- ${f.id} : ${f.status}. Centre ${f.center_m.join(', ')}. Une mer confirmée dans le DEM ne prouve pas à elle seule l’apparence de banquise ; un rivage confirmé ne prouve pas la conservation de la plage.`).join('\n')}\n\nLes fractions complètes, reliefs réellement mesurés et erreurs sont dans coverage.json. Les revues visuelles sont séparées dans reviews.json ; l’absence de revue reste pending.\n`;
const actualText=`\n## Couverture des reliefs réellement mesurés\n\n${audit.actualNativeCoverage.strictNativeMatchedCells}/24 cases strictes ; ${audit.actualNativeCoverage.representativeNativeMatchedCells}/24 avec transitions représentatives. ${audit.actualNativeCoverage.visuallyAcceptedCells}/24 ont aussi une acceptation visuelle explicite. Cette lecture utilise la morphologie et les fractions de biome natives indépendamment de la cible source. Aucun centre ne change de nom ; ses échecs et sa provenance restent visibles.\n\n|Famille native|Relief mesuré|Natif strict≥80%|Natif représentatif≥70%|Cible source échouée|État|\n|---|---|---|---|---|---|\n${actualTable}\n\nLes quatre reliefs × six familles sont une couverture demandée, pas une propriété garantie par 60 centres. Toute case sans échantillon réel reste une lacune. Pour accepter un relief différent de la cible, la revue doit porter actualReliefAccepted:true ainsi que status:accepted, liée au runId courant.\n`;
fs.writeFileSync(path.join(base,'coverage.md'),text+actualText);
console.log(JSON.stringify({sites:audit.capturedSites,frames:audit.capturedFrames,strictCells:audit.strictNativeMatchedCells,
 representativeCells:audit.representativeNativeMatchedCells,visualCells:audit.visuallyAcceptedCells,gaps:audit.gaps.length,
 actualStrictCells:audit.actualNativeCoverage.strictNativeMatchedCells,actualRepresentativeCells:audit.actualNativeCoverage.representativeNativeMatchedCells,output:base}));
