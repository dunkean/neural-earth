// Only actual generated-world LOD captures, with source stages and extents shown.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const out=path.resolve(__dirname,'../../output/render-v2-40km');
const transitions=process.argv.includes('--transitions'),prefix=transitions?'transitions-lod':'mid-lod',sheetDir=transitions?'transition-sheets':'mid-sheets';
(async()=>{
 const reports=(transitions?[0,3,4,6]:[3,4,6]).map(lod=>({lod,report:JSON.parse(fs.readFileSync(path.join(out,`${prefix}${lod}/report.json`),'utf8'))}));
 fs.mkdirSync(path.join(out,sheetDir),{recursive:true});
 const server=http.createServer((req,res)=>{const f=path.resolve(out,'.'+new URL(req.url,'http://x').pathname);if(!f.startsWith(out+path.sep)||!fs.existsSync(f)){res.statusCode=404;return res.end();}res.setHeader('Content-Type',f.endsWith('.png')?'image/png':'text/html');res.end(fs.readFileSync(f));});await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--disable-gpu']});const page=await browser.newPage({viewport:{width:1150,height:1300}});
  for(const r of reports[0].report.results){
   const rows=reports.map(({lod,report})=>{const entry=report.results.find(p=>p.id===r.id);return `<div class="row"><b>LOD ${lod}<small>${entry.frames[0].state.rendered.sources.join(', ')} · ${(entry.frames[0].extentMetres/1000).toFixed(1)} km<br>quality ${entry.frames[0].state.rendered.qualityLods.join(', ')} · source ${entry.frames[0].state.rendered.sourceResolutions.join(', ')} m</small></b>${['v2-july','render3d-july','v2-january'].map(tag=>{const f=entry.frames.find(f=>f.tag===tag);return `<div><img src="/${prefix}${lod}/${f.file}"><small>${tag} · ${f.state.camera.mpp} m/px</small></div>`;}).join('')}</div>`;}).join('');
   fs.writeFileSync(path.join(out,'mid-sheet.html'),`<meta charset="utf-8"><style>body{background:#182127;color:white;margin:0;font:12px system-ui;width:1100px}h2,p{margin:10px}.row{display:grid;grid-template-columns:110px repeat(3,320px);gap:5px;margin:5px}img{display:block;width:320px;height:320px}small{display:block;padding:5px;font-size:11px;font-weight:400}.row b{align-self:center}</style><h2>Actual generated tiles · ${r.id}</h2><p>${transitions?'Exact same40km extent and centre; source substitution only, no zoom change.':'Same centre, increasing extents at LOD3/4/6; these are not40km decoder plates.'} Source stages, quality LODs and resolutions above are measured. Same live world, shader and lighting.</p>${rows}`);
   await page.goto(`http://127.0.0.1:${server.address().port}/mid-sheet.html`);await page.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth));await page.locator('body').screenshot({path:path.join(out,sheetDir,r.id+'.png')});
  }
  console.log(JSON.stringify({sheets:reports[0].report.results.length,out:path.join(out,sheetDir)}));
 }finally{if(browser)await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
