// Assemble only live generated-world captures. No synthetic DEM or material fixture.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const out=path.resolve(__dirname,'../../output/render-v2-40km/real-matrix'),file=path.join(out,'report.json');
const esc=s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
(async()=>{
 const report=JSON.parse(fs.readFileSync(file,'utf8')),tags=['v2-july','render3d-july','v2-january'],zooms=['40km','10km','2.56km','640m'];
 fs.mkdirSync(path.join(out,'sheets'),{recursive:true});
 const server=http.createServer((req,res)=>{const f=path.resolve(out,'.'+new URL(req.url,'http://x').pathname);if(!f.startsWith(out+path.sep)||!fs.existsSync(f)){res.statusCode=404;return res.end();}res.setHeader('Content-Type',f.endsWith('.png')?'image/png':'text/html');res.end(fs.readFileSync(f));});
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;let count=0;
 const caption=r=>`${r.id} · H ${Math.round(r.nativeStats?.elevation_min_m??0)}–${Math.round(r.nativeStats?.elevation_max_m??0)} m · ${r.acceptance?.status||'awaiting DEM metrics'}`;
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--disable-gpu']});const page=await browser.newPage({viewport:{width:1160,height:1300}});
  for(const family of Object.keys(report.selection.families))for(const tag of tags){
   const rows=report.selection.reliefs.map(relief=>{
    const r=report.results.find(r=>r.selection.biomeFamily===family&&r.selection.relief===relief);
    return `<div class="row"><b>${esc(relief)}<small>${r?esc(caption(r)):'Not captured'}</small></b>${zooms.map(zoom=>{const f=r?.frames.find(f=>f.zoom===zoom&&f.tag===tag);return `<div>${f?`<a href="${f.file}"><img src="/${esc(f.file)}"></a>`:'<div class="missing">Pending</div>'}<small>${zoom}</small></div>`;}).join('')}</div>`;
   }).join('');
   const html=`<meta charset="utf-8"><style>body{margin:0;background:#182127;color:white;font:12px system-ui;width:1140px}h2,p{margin:10px}.row{display:grid;grid-template-columns:100px repeat(4,256px);gap:4px;margin:4px}.row b{padding:4px;align-self:center;overflow-wrap:anywhere}img,.missing{display:block;width:256px;height:256px}.missing{background:#243037;padding:50px}small{display:block;padding:4px;font-size:10px;font-weight:400}</style><h2>Generated Neural Earth tiles · ${esc(family)} · ${tag}</h2><p>Same world and centre at all four zooms. Decoder LOD0, physical DEM samples30m. Relief names are requested classes; acceptance below is based on measured native DEM, with visual review kept separate.</p>${rows}`;
   fs.writeFileSync(path.join(out,'sheet.html'),html);await page.goto(`http://127.0.0.1:${server.address().port}/sheet.html`);await page.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth));await page.locator('body').screenshot({path:path.join(out,'sheets',`${family}-${tag}.png`)});count++;
  }
 }finally{if(browser)await browser.close();server.close();}
 const options=(name,vals)=>`<label>${name}<select id="${name}"><option value="">All</option>${vals.map(v=>`<option>${esc(v)}</option>`).join('')}</select></label>`;
 const cards=report.results.flatMap(r=>r.frames.map(f=>`<article data-family="${esc(r.selection.biomeFamily)}" data-relief="${esc(r.selection.relief)}" data-zoom="${f.zoom}" data-style="${f.tag}"><a href="${esc(f.file)}"><img loading="lazy" width="320" height="320" src="${esc(f.file)}"></a><p>${esc(caption(r))}<br>${f.zoom} · ${f.tag}<br>${esc(r.selection.center_m.join(', '))}</p></article>`)).join('');
 fs.writeFileSync(path.join(out,'index.html'),`<!doctype html><meta charset="utf-8"><title>Render v2 · generated tiles review</title><style>body{margin:20px;background:#182127;color:#e9efeb;font:14px system-ui}header{position:sticky;top:0;background:#182127;padding:10px;z-index:1}label{display:inline-flex;gap:5px;margin:5px}main{display:flex;gap:14px;flex-wrap:wrap}article{width:320px}article[hidden]{display:none}a{color:#c0dce9}p{line-height:1.5}</style><header><h1>Render v2 · actual generated tiles</h1><p>24 requested relief/biome crossings, four extents. Every screenshot comes from the live app and decoder30m terrain. Terrain resolution remains30m when zooming. No synthetic texture test plates. <a href="report.json">DEM metrics and provenance</a></p>${options('family',Object.keys(report.selection.families))}${options('relief',report.selection.reliefs)}${options('zoom',zooms)}${options('style',tags)}</header><main>${cards}</main><script>const controls=[...document.querySelectorAll('select')];function filter(){document.querySelectorAll('article').forEach(a=>a.hidden=controls.some(c=>c.value&&a.dataset[c.id]!==c.value))}controls.forEach(c=>c.onchange=filter);document.getElementById('zoom').value='40km';document.getElementById('style').value='v2-july';filter()</script>`);
 console.log(JSON.stringify({sheets:count,sites:report.results.length,frames:report.results.reduce((n,r)=>n+r.frames.length,0),out}));
})().catch(e=>{console.error(e);process.exitCode=1;});
