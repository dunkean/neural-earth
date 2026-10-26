// Contact sheets made ONLY from completed real terrain PNG captures.
// Browser rendering here is CPU HTML/PNG assembly, not procedural terrain.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const root=path.resolve(__dirname,'../..');
const mainBase=path.resolve(root,process.env.REVIEW_REFERENCE_MAIN_BASE||'output/render-v2-reference');
const hydroBase=path.resolve(root,process.env.REVIEW_REFERENCE_HYDRO_BASE||'output/render-v2-hydrology-reference/captures-reference');
const base=path.resolve(root,process.env.REVIEW_REFERENCE_BASE||'output/render-v2-reference');
const manifest=JSON.parse(fs.readFileSync(path.join(base,'manifest.json'),'utf8'));
const reportFile=path.join(base,'report.json'),rawReport=fs.existsSync(reportFile)?JSON.parse(fs.readFileSync(reportFile,'utf8')):null;
const report=rawReport?.manifest?.worldIdentity===manifest.worldIdentity?rawReport:null;
const results=(report?.results||[]).filter(r=>{
 const site=manifest.sites.find(s=>s.id===r.id);return site&&JSON.stringify(site.center_m)===JSON.stringify((r.site||r.selection)?.center_m);
}),reviewsFile=path.join(base,'reviews.json');
const reviewData=fs.existsSync(reviewsFile)?JSON.parse(fs.readFileSync(reviewsFile,'utf8')):null;
const reviews=report?.runId&&reviewData?.runId===report.runId?(reviewData.sites||{}):{};
const esc=s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const checkedFrames=new Map();
const validFrameAt=(directory,f)=>{
 if(!f?.file)return false;const file=path.resolve(directory,f.file),key=file+'/'+f.sha256;
 if(!checkedFrames.has(key))checkedFrames.set(key,file.startsWith(directory+path.sep)&&fs.existsSync(file)&&(!f.sha256||hash(fs.readFileSync(file))===f.sha256));
 return checkedFrames.get(key);
};
const validFrame=f=>validFrameAt(base,f);
const frames=results.reduce((n,r)=>n+r.frames.filter(validFrame).length,0);
const expectedSites=manifest.sites.length,expectedFrames=expectedSites*manifest.zooms.length*manifest.styles.length;
assert.equal(manifest.expectedFrames,expectedFrames,'Manifest frame count mismatch');
const hasNative=r=>r?.nativeStats?.resolution_m===30&&r.frames.some(f=>f.zoom==='40km'&&f.requiredSourceResolutionMetres===30&&f.requestedLod===0);
const selectionMatches=report?.frozen?.manifestHash===hash(fs.readFileSync(path.join(base,'manifest.json')));
const complete=selectionMatches&&results.length===expectedSites&&frames===expectedFrames&&manifest.sites.every(s=>{
 const r=results.find(r=>r.id===s.id);return r?.protocolOk&&hasNative(r)&&manifest.zooms.every(z=>manifest.styles.every(tag=>validFrame(r.frames.find(f=>f.zoom===z.id&&f.tag===tag))));
});
if(process.argv.includes('--require-complete'))assert(complete,`Final assembly requires ${expectedSites} sites/${expectedFrames} real frames, matching manifest and native40km measurements`);
const gallery=path.join(base,'gallery');fs.mkdirSync(gallery,{recursive:true});
const hydro=!!manifest.hydrologyRequired;
const title=(complete?'Capture protocol complete':'WORKING · capture in progress')+' · '+(hydro?'real hydrology reference':'real terrain reference');
const columns=manifest.zooms.length,sheetWidth=160+224*columns+4*columns+16;
const css=`*{box-sizing:border-box}body{margin:0;background:#172027;color:#e5ebed;font:13px system-ui}a{color:#c7e3ec}h1,h2,p{margin:12px 16px}small{display:block;color:#c2cbd0}nav{padding:10px 16px;display:flex;flex-wrap:wrap;gap:12px}.grid{display:grid;grid-template-columns:160px repeat(${columns},224px);gap:4px;margin:8px}.cell img,.missing{display:block;width:224px;height:224px;object-fit:contain;background:#26343d}.missing{padding:60px 18px;color:#adc1ca}.label{padding:8px;overflow-wrap:anywhere;align-self:center}.cell small{padding:4px}.warn{color:#ffdaab}.case{margin:14px;padding:10px;background:#22303a}.case img{width:calc(${100/columns}% - 5px);min-width:160px;height:auto}.case h2{margin:4px}.meta{padding:8px 16px;white-space:pre-wrap;overflow-wrap:anywhere}.tag{display:inline-block;border:1px solid #586972;padding:3px 6px;margin:2px}table{margin:12px 16px;border-collapse:collapse}th,td{border:1px solid #586972;padding:6px;text-align:left}`;
const docs=`153.6 km uses actual latent data at 240 m, forced LOD 3. Other extents use the decoder at 30 m, forced LOD 0. The 160 m view enlarges shader detail; it does not validate submeter relief. July is austral winter and January austral summer south of the equator. `+(hydro?'Hydrology is enabled. V2 physical buffers require ready headers; modeled flows and source lake levels do not validate neural shorelines. Render3D retains its historical vector overlay.':'Hydrology is disabled; coast and sea remain real DEM. Source labels and actual native morphology remain separate from visual acceptance.');
const relative=(from,to)=>path.relative(from,to).split(path.sep).join('/');
const navigation=from=>`<nav><a href="${esc(relative(from,path.join(mainBase,'index.html')))}">Common reference</a><a href="${esc(relative(from,path.join(mainBase,'gallery/index.html')))}">Main set</a><a href="${esc(relative(from,path.join(hydroBase,'gallery/index.html')))}">Hydrology set</a></nav>`;
function waterSummary(f){
 const e=f?.hydrologyEvidence;if(!e)return 'Hydrology evidence missing';
 const cuts=[...new Set(e.summaries.map(s=>s.maxCut).filter(v=>v!==undefined))];
 return `month ${e.month} · lake ${e.lakeSamples} / river ${e.riverSamples} samples · max cut ${cuts.length?cuts.join('/')+' m':'no physical raster (vector overlay)'} · ${e.status}${e.fetching?' / fetching':''}`;
}
function waterTable(r){
 if(!hydro)return '';
 const rows=(r?.frames||[]).map(f=>{
  const e=f.hydrologyEvidence,d=e?.regionalDisplay,m=e?.maskReadback;
  const regional=f.tag==='render3d-july'?'Historical vector overlay':d?
   `${d.mode} · ${d.ready?'ready':'pending'} · month ${d.month} · ${d.segments} segments · ${m?`${m.wetPixels} wet screen pixels / ${m.coveragePixels.toFixed(2)} coverage pixels`:'no mask readback'} · ${d.bytes} bytes`:'Regional display evidence missing (older capture)';
  return `<tr><td>${esc(f.zoom)}</td><td>${esc(f.tag)}</td><td>${esc(waterSummary(f))}</td><td>${esc(regional)}</td><td>${esc(e?.overlayPixels??'missing')} overlay pixels</td></tr>`;
 }).join('');
 return `<h2>Actual water sources and display</h2><p>Physical counts are DEM samples. Source river samples do not establish the displayed regional course. The separate GPU mask readback counts screen pixels and records the actual uploaded segment authority; the screenshots also include the historical overlay canvas.</p><table><tr><th>Extent</th><th>View</th><th>Physical month / class / cut</th><th>Regional GPU display</th><th>Overlay</th></tr>${rows}</table>`;
}
function setSummary(directory){
 const read=name=>fs.existsSync(path.join(directory,name))?JSON.parse(fs.readFileSync(path.join(directory,name),'utf8')):null;
 const m=read('manifest.json');if(!m)return null;
 const raw=read('report.json'),r=raw?.manifest?.worldIdentity===m.worldIdentity?raw:null;
 const entries=(r?.results||[]).filter(e=>{
  const s=m.sites.find(s=>s.id===e.id);return s&&JSON.stringify(s.center_m)===JSON.stringify((e.site||e.selection)?.center_m);
 });
 const expected=m.sites.length*m.zooms.length*m.styles.length;
 assert.equal(m.expectedFrames,expected,'Manifest summary frame count mismatch');
 const count=entries.reduce((n,e)=>n+e.frames.filter(f=>validFrameAt(directory,f)).length,0);
 const frozenSelection=r?.frozen?.manifestHash===hash(fs.readFileSync(path.join(directory,'manifest.json')));
 const done=frozenSelection&&count===expected&&entries.length===m.sites.length&&m.sites.every(s=>{
  const e=entries.find(e=>e.id===s.id);return e?.protocolOk&&hasNative(e)&&m.zooms.every(z=>m.styles.every(tag=>validFrameAt(directory,e.frames.find(f=>f.zoom===z.id&&f.tag===tag))));
 });
 return{hydrology:!!m.hydrologyRequired,index:relative(mainBase,path.join(directory,'gallery/index.html')),
  expectedSites:m.sites.length,expectedFrames:expected,frames:count,nativeSites:entries.filter(hasNative).length,
  runId:r?.runId||null,productionHash:r?.frozen?.codeHash||null,worldIdentity:m.worldIdentity,complete:done};
}
function status(site,r){
 const review=reviews[site.id]?.status||'pending';
 const requested=site.requestedRelief||'unspecified',actual=r?.nativeRelief||'not yet measured';
 const family=r?.nativeStats?.biome_fractions_land||site.source_stats?.biome_fractions_land||{};
 const fractions=Object.entries(family).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`${k} ${(v*100).toFixed(1)}%`).join(', ')||'OCEAN';
 const range=r?.nativeStats?.elevation_land_m;
 return `<b>${esc(site.id)}</b><small>${esc(site.biomeFamily)} · requested ${esc(requested)} / measured ${esc(actual)}</small><small>${esc(fractions)}</small><small>${range?`Land ${Math.round(range.min)}–${Math.round(range.max)} m`:'No native land range'}</small><small>Protocol: ${r?.protocolOk?'PASS':'pending/failed'} · visual: ${esc(review)}</small>`;
}
function frameCell(site,z,tag){
 const result=results.find(r=>r.id===site.id),f=result?.frames.find(f=>f.zoom===z.id&&f.tag===tag);
 let replay='';
 if(validFrame(f)&&result.url){
  const url=new URL(result.url);
  for(const [key,value]of Object.entries({layer:tag==='render3d-july'?'render3d':'render-v2',
   zoom:z.extentMetres/manifest.size,position:site.center_m.join(','),lod:z.lod,
   size:[manifest.size,manifest.size].join(','),v2season:tag==='v2-january'?0:.5,
   hydroMonth:tag==='v2-january'?1:7}))url.searchParams.set(key,String(value));
  replay=`<small><a href="${esc(url.href)}" target="_blank" rel="noopener">Ouvrir dans Neural Earth</a></small>`;
 }
 return `<div class="cell">${validFrame(f)?`<a href="../${esc(f.file)}"><img src="../${esc(f.file)}" loading="eager"></a>`:'<div class="missing">Not captured</div>'}<small>${esc(z.id)} · ${z.extentMetres/manifest.size} m/px · LOD ${z.lod}</small>${hydro&&f?`<small>${esc(waterSummary(f))}</small>`:''}${replay}</div>`;
}
(async()=>{
 const groups=[];
 for(const family of [...new Set(manifest.sites.map(s=>s.biomeFamily))]){
  const sites=manifest.sites.filter(s=>s.biomeFamily===family);
  for(let i=0;i<sites.length;i+=4)groups.push({id:family.toLowerCase()+'-'+(i/4+1),family,sites:sites.slice(i,i+4)});
 }
 const server=http.createServer((req,res)=>{
  const url=new URL(req.url,'http://local'),name=url.pathname.slice(1);
  const file=path.resolve(base,name);
  if(!file.startsWith(base+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.statusCode=404;return res.end();}
  res.setHeader('Content-Type',file.endsWith('.png')?'image/png':'text/html; charset=utf-8');res.end(fs.readFileSync(file));
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
 try{
  let page;
  if(process.argv.includes('--sheets')){
   browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--disable-gpu']});
   page=await browser.newPage({viewport:{width:sheetWidth,height:1250},deviceScaleFactor:1});
  }
  const links=[];
  for(const group of groups)for(const tag of manifest.styles){
   const name=group.id+'-'+tag;
   const rows=group.sites.map(site=>`<div class="grid"><div class="label">${status(site,results.find(r=>r.id===site.id))}<a href="${site.id}.html">Full site comparison</a></div>${manifest.zooms.map(z=>frameCell(site,z,tag)).join('')}</div>`).join('');
   const html=`<!doctype html><meta charset="utf-8"><title>${esc(name)}</title><style>${css}body{width:${sheetWidth}px}</style>${navigation(gallery)}<h2>${esc(title)} · ${esc(group.family)} · ${esc(tag)}</h2><p>${esc(docs)}</p><p>${frames}/${expectedFrames} frames · exact version ${esc(report?.runId||'not frozen')} · visual acceptance is separate from protocol.</p>${rows}`;
   fs.writeFileSync(path.join(gallery,name+'.html'),html);
   if(process.argv.includes('--sheets')){
    await page.goto(`http://127.0.0.1:${server.address().port}/gallery/${name}.html`);
    await page.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth));
    await page.locator('body').screenshot({path:path.join(gallery,name+'.png')});
   }
   links.push(`<a href="${name}.html">${esc(group.family)} · ${esc(tag)} · ${group.id.split('-').at(-1)}</a>`);
  }
  for(const site of manifest.sites){
   const r=results.find(r=>r.id===site.id);
   const rows=manifest.styles.map(tag=>`<div class="grid"><div class="label"><b>${esc(tag)}</b></div>${manifest.zooms.map(z=>frameCell(site,z,tag)).join('')}</div>`).join('');
   fs.writeFileSync(path.join(gallery,site.id+'.html'),`<!doctype html><meta charset="utf-8"><title>${esc(site.id)}</title><style>${css}</style>${navigation(gallery)}<nav><a href="index.html">All sites in this set</a></nav><h1>${status(site,r)}</h1><p>${esc(docs)}</p><p>Centre ${esc(site.center_m.join(', '))} · source provenance in <a href="../manifest.json">manifest</a>. Open any image for ${manifest.size}px detail.</p>${rows}${waterTable(r)}<pre class="meta">${esc(JSON.stringify({captureVersion:report?.runId,productionHash:report?.frozen?.codeHash,provenance:site.aliases,feature:site.featureProvenance,acceptance:r?.acceptance,hydrology:hydro?{record_type:site.record_type,record_id:site.record_id,level_m:site.level_m,monthly_flow_m3s:site.monthly_flow_m3s,apiEvidence:site.apiEvidence}:undefined,review:reviews[site.id]},null,2))}</pre>`);
  }
  const rows=manifest.sites.map(site=>`<section class="case"><h2><a href="${site.id}.html">${status(site,results.find(r=>r.id===site.id))}</a></h2>${manifest.zooms.map(z=>{
   const f=results.find(r=>r.id===site.id)?.frames.find(f=>f.zoom===z.id&&f.tag==='v2-july');
   return validFrame(f)?`<a href="${site.id}.html"><img loading="lazy" src="../${esc(f.file)}" alt="${esc(site.id+' '+z.id)}"></a>`:'';
  }).join('')}</section>`).join('');
  const index=`<!doctype html><meta charset="utf-8"><title>Real Render v2 reference</title><style>${css}</style>${navigation(gallery)}<h1>${esc(title)}</h1><p>${frames}/${expectedFrames} captured frames · ${results.filter(r=>r.protocolOk).length}/${expectedSites} protocol passes · ${results.filter(hasNative).length}/${expectedSites} native40km measurements · ${Object.values(reviews).filter(r=>r.status==='accepted').length}/${expectedSites} visually accepted.</p><p>${esc(docs)}</p><p>Version ${esc(report?.runId||'pending')} · production ${esc(report?.frozen?.codeHash||'pending')} · ${selectionMatches?'selection matches frozen receipt':'selection pending freeze or updated since captures'}.</p><p>Requested relief classes that fail native measurements remain visible as failures; actual native coverage is documented separately.</p><nav>${links.join('')}</nav>${rows}`;
  fs.writeFileSync(path.join(gallery,'index.html'),index);
  // Local-file navigation is valid even before the other set is assembled.
  for(const directory of [mainBase,hydroBase]){
   const target=path.join(directory,'gallery/index.html');
   if(!fs.existsSync(target)){
    fs.mkdirSync(path.dirname(target),{recursive:true});
    fs.writeFileSync(target,`<!doctype html><meta charset="utf-8"><title>Gallery pending</title><style>${css}</style>${navigation(path.dirname(target))}<h1>Gallery pending assembly</h1><p>No final images are claimed by this placeholder.</p>`);
   }
  }
  const sets=[setSummary(mainBase),setSummary(hydroBase)].filter(Boolean);
  const sameProduction=sets.length===2&&sets.every(s=>s.productionHash)&&sets[0].productionHash===sets[1].productionHash&&sets[0].worldIdentity===sets[1].worldIdentity;
  const combinedComplete=!!sameProduction&&sets.every(s=>s.complete);
  const summary={label:'Separate versioned real sets; visual acceptance is not inferred from protocol',
   complete:combinedComplete,sameProduction:!!sameProduction,expectedSites:sets.reduce((n,s)=>n+s.expectedSites,0),
   expectedFrames:sets.reduce((n,s)=>n+s.expectedFrames,0),frames:sets.reduce((n,s)=>n+s.frames,0),sets};
  fs.mkdirSync(mainBase,{recursive:true});
  fs.writeFileSync(path.join(mainBase,'gallery-summary.json'),JSON.stringify(summary,null,2));
  const coverage=fs.existsSync(path.join(mainBase,'coverage.md'))?'<a href="coverage.md">main coverage audit</a>':'coverage audit pending';
  fs.writeFileSync(path.join(mainBase,'index.html'),`<!doctype html><meta charset="utf-8"><title>Real Render v2 reference</title><style>${css}</style><h1>Real Render v2 reference · ${summary.expectedSites} situations · ${manifest.zooms.length} zooms · ${summary.expectedFrames} expected images</h1><p>${summary.frames}/${summary.expectedFrames} images present · ${combinedComplete?'both capture protocols complete':'WORKING: campaign incomplete'} · ${sameProduction?'same production hash and world':'version coherence pending'}. Visual acceptance remains separate.</p><p>The varied portfolio does not prove all4×6 relief/biome cells: see ${coverage} for the actual native coverage and its explicit gaps.</p>${sets.map(s=>`<section class="case"><h2><a href="${esc(s.index)}">${s.hydrology?'Hydrology complement':'Main set'} · ${s.expectedSites} situations</a></h2><p>${s.frames}/${s.expectedFrames} images · ${s.nativeSites}/${s.expectedSites} native40km measurements · ${s.complete?'protocol complete':'pending/partial'}</p><p>Version ${esc(s.runId||'pending')}<br>Production ${esc(s.productionHash||'pending')}<br>World ${esc(s.worldIdentity)}</p></section>`).join('')}`);
  if(process.argv.includes('--require-complete')&&sets.length===2&&sets.every(s=>s.complete))assert(sameProduction,'Complete sets must share the same production hash and world identity');
  console.log(JSON.stringify({complete,combinedComplete,sameProduction:!!sameProduction,sites:expectedSites,frames,expectedFrames,
   sheets:process.argv.includes('--sheets')?links.length:0,index:path.join(gallery,'index.html'),entry:path.join(mainBase,'index.html')}));
 }finally{if(browser)await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
