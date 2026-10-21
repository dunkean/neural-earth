// Exercise sketch editing,full CPU inference,downloads and accessible output.
const fs=require('node:fs'),path=require('node:path');
const {loadPlaywright}=require('../tools/platform.cjs');
async function main() {
  const args=process.argv.slice(2),opt=(name,fallback) => args.includes(name) ? args[args.indexOf(name)+1] : fallback;
  const output=opt('--output'),url=opt('--url','http://127.0.0.1:8780/'),cdp=opt('--cdp');
  const decoder=opt('--decoder','decoder16');
  const engine=opt('--engine','browser');
  const provider=opt('--provider',engine==='python'?'auto':'wasm');
  if (!output) throw new Error('External output folder required.');
  fs.mkdirSync(output,{recursive:true});
  const {chromium}=loadPlaywright(),browser=cdp ? await chromium.connectOverCDP(cdp) : await chromium.launch({headless:true});
  const context=await browser.newContext({acceptDownloads:true}),page=await context.newPage(),errors=[];
  page.on('pageerror',error => errors.push(String(error)));
  try {
    await page.goto(url);await page.waitForFunction(() => document.getElementById('bundleSize').textContent.includes('Mo'));
    await page.selectOption('#engine',engine);
    await page.waitForFunction(() => !document.getElementById('generate').disabled);
    if (await page.locator('#preset option').count()!==6) throw new Error('Six input types required.');
    await page.selectOption('#decoder',decoder);
    await page.waitForFunction(()=>!document.getElementById('generate').disabled);
    await page.selectOption('#preset','Crête');await page.selectOption('#climate','cold');await page.selectOption('#provider',provider);
    await page.fill('#row','8');await page.fill('#column','9');await page.fill('#altitude','3150');await page.click('#edit');
    await page.click('#generate');
    await page.waitForFunction(() => !document.getElementById('generate').disabled,{},{timeout:120000});
    const status=await page.locator('#status').innerText();
    const expected=provider==='webgpu'?'GPU':provider==='wasm'?'CPU':null;
    if (!status.startsWith('Tuile calculée sur ') || (expected && !status.includes(expected))) throw new Error(status);
    const dimensions=await page.locator('#result').evaluate(canvas => [canvas.width,canvas.height]);
    if (dimensions.some(v => v!==1334)) throw new Error('Incomplete40km output.');
    const metadata=JSON.parse(await page.locator('#metadata').textContent());
    if (metadata.input.sketch_m[7][8]!==3150 || metadata.input.climate!=='cold' || metadata.side_m!==40000) throw new Error('Edited physical inputs were ignored.');
    if (metadata.selected_models.decoder!==decoder || metadata.selected_models.base!=='base_residual8k') throw new Error('Selected model provenance differs.');
    await page.screenshot({path:path.join(output,'demo.png'),fullPage:true});
    const download=async (button,name) => {
      const next=page.waitForEvent('download');await page.click(button);const item=await next;await item.saveAs(path.join(output,name));
    };
    await download('#dem','terrain40.f32');await download('#parameters','terrain40.json');await download('#image','terrain40.png');
    const raw=fs.readFileSync(path.join(output,'terrain40.f32'));
    if (raw.length!==1334*1334*4) throw new Error('Wrong full DEM export length.');
    let min=Infinity,max=-Infinity;
    for (let i=0;i<raw.length;i+=4) {const value=raw.readFloatLE(i);if (!Number.isFinite(value)) throw new Error('Nonfinite DEM.');min=Math.min(min,value);max=Math.max(max,value);}
    if (Math.abs(min-metadata.summary.minM)>.001 || Math.abs(max-metadata.summary.maxM)>.001 || errors.length) throw new Error('Export/UI mismatch or browser errors:'+errors);
    const report={status:'complete',chromeVersion:await browser.version(),scope:'Browser UI on edited ridge sketch,cold climate,selected Python/browser engine and full40km exports;quality acceptance separate.',engine,metadata,statusText:status,errors};
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify({status:report.status,timing:metadata.timings,output}));
  } finally {await context.close();await browser.close();}
}
main().catch(error => {console.error(error);process.exitCode=1;});
