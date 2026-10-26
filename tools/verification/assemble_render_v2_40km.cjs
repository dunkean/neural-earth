const fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {loadPlaywright,browserExecutable}=require('../platform.cjs');
const out=path.resolve(__dirname,'../../output/render-v2-40km'),report=JSON.parse(fs.readFileSync(path.join(out,'report.json'),'utf8'));
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;');
(async()=>{
 fs.mkdirSync(path.join(out,'sheets'),{recursive:true});
 const server=http.createServer((req,res)=>{const file=path.join(out,new URL(req.url,'http://x').pathname.slice(1));if(!file.startsWith(out+path.sep)||!fs.existsSync(file)){res.statusCode=404;return res.end();}res.setHeader('Content-Type',file.endsWith('.png')?'image/png':'text/html');res.end(fs.readFileSync(file));});
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
 try{
  browser=await loadPlaywright().chromium.launch({executablePath:browserExecutable(),headless:true,args:['--disable-gpu']});const page=await browser.newPage({viewport:{width:1100,height:1300}});
  const cases=report.fixture.cases,alts=report.fixture.altitudes;
  for(let offset=0;offset<cases.length;offset+=5)for(const [mode,season,view] of [['render-v2',.5,'plate'],['render-v2',0,'plate'],['render-v2',.5,'detail'],['render3d',.5,'plate']]){
   const group=cases.slice(offset,offset+5),cells=group.map(c=>`<div class="row"><b>${escape(c.id.toUpperCase())}<br>${escape(c.label)}</b>${alts.map(a=>{const f=report.frames.find(f=>f.id===c.id&&f.altitude===a&&f.mode===mode&&f.season===season&&f.view===view);return `<div><img src="/frames/${f.file}"><small>${escape(c.expectedCodesAtAltitudes[a].toUpperCase())} · ${a} m</small></div>`;}).join('')}</div>`).join('');
   const name=`group-${offset/5+1}-${mode}-${season===0?'january':'july'}-${view}.png`,html=`<meta charset="utf-8"><style>body{margin:0;background:#182127;color:white;font:12px system-ui}h2,p{margin:10px}.row{display:grid;grid-template-columns:110px repeat(5,192px);gap:4px;margin:4px}.row b{padding:4px;align-self:center}img{display:block;width:192px;height:192px}small{display:block;height:18px;padding:3px}</style><h2>${mode} · ${season===0?'January':'July'} · ${view==='plate'?'40 × 40 km':'1.28 × 1.28 km detail'}</h2><p>Controlled 30 m DEM, same relief/soil/seed/light. Codes below panels are climate after altitude lapse.</p>${cells}`;
   fs.writeFileSync(path.join(out,'sheet.html'),html);await page.goto(`http://127.0.0.1:${server.address().port}/sheet.html`);await page.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth));await page.locator('body').screenshot({path:path.join(out,'sheets',name)});
  }
  console.log(JSON.stringify({sheets:24,out:path.join(out,'sheets')}));
 }finally{if(browser)await browser.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
