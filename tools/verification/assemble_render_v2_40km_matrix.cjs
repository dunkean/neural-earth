// Contact sheets retain all four reliefs and zooms; never substitute scattered sites.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const out=path.resolve(__dirname,'../../output/render-v2-40km-matrix');
const report=JSON.parse(fs.readFileSync(path.join(out,'report.json'),'utf8'));
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;');
(async()=>{
 fs.mkdirSync(path.join(out,'sheets'),{recursive:true});
 const server=http.createServer((req,res)=>{const f=path.resolve(out,'.'+new URL(req.url,'http://x').pathname);if(!f.startsWith(out+path.sep)||!fs.existsSync(f)){res.statusCode=404;return res.end();}res.setHeader('Content-Type',f.endsWith('.png')?'image/png':'text/html');res.end(fs.readFileSync(f));});
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;let count=0;
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--disable-gpu']});const page=await browser.newPage({viewport:{width:1160,height:1200}});
  const views=['40km','10km','2.56km','640m','micro320m'].filter(v=>report.frames.some(f=>f.view===v));
  for(const c of report.fixture.cases)for(const altitude of report.fixture.altitudes)for(const mode of ['render-v2','render3d'])for(const season of [0,.5]){
   const name=`${c.id}-${altitude}-${mode}-${season===0?'january':'july'}.png`;
   const rows=report.fixture.reliefs.map(relief=>`<div class="row"><b>${esc(relief)}</b>${views.map(view=>{const f=report.frames.find(f=>f.id===c.id&&f.relief===relief&&f.altitude===altitude&&f.mode===mode&&f.season===season&&f.view===view);if(!f)throw Error(`Missing ${c.id}/${relief}/${altitude}/${mode}/${season}/${view}`);return `<div><img src="/frames/${esc(f.file)}"><small>${esc(view)} · ${f.metresPerPixel.toFixed(3)} m/px</small></div>`;}).join('')}</div>`).join('');
   const html=`<meta charset="utf-8"><style>body{margin:0;background:#182127;color:white;font:12px system-ui;width:${110+views.length*204}px}h2,p{margin:10px}.row{display:grid;grid-template-columns:106px repeat(${views.length},200px);gap:4px;margin:4px}.row b{padding:4px;align-self:center}img{display:block;width:200px;height:200px}small{display:block;height:18px;padding:3px}</style><h2>${esc(c.id.toUpperCase())} · ${esc(c.label)} · base ${altitude} m · ${mode} · ${season===0?'January':'July'}</h2><p>Controlled 30 m DEM. Same geographical centre at every zoom. Actual elevation depends on relief (see report); climate changes with elevation. Micro inset is procedural detail on the same 30 m DEM.</p>${rows}`;
   fs.writeFileSync(path.join(out,'sheet.html'),html);await page.goto(`http://127.0.0.1:${server.address().port}/sheet.html`);await page.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth));await page.locator('body').screenshot({path:path.join(out,'sheets',name)});count++;
  }
  console.log(JSON.stringify({sheets:count,out:path.join(out,'sheets')}));
 }finally{if(browser)await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
