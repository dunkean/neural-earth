// Actual browser output/timing matrix:6rough sketches x5climates,same seed.
const fs=require('node:fs'),path=require('node:path'),{createHash}=require('node:crypto');
const {loadPlaywright}=require('../tools/platform.cjs');
async function main() {
  const args=process.argv.slice(2),opt=(name,fallback) => args.includes(name) ? args[args.indexOf(name)+1] : fallback;
  const output=opt('--output'),url=opt('--url','http://127.0.0.1:8780/'),cdp=opt('--cdp'),runs=Number(opt('--runs','2'));
  if (!output || !Number.isInteger(runs) || runs<2 || fs.existsSync(path.join(output,'report.json'))) throw new Error('Fresh output and at least2runs required.');
  fs.mkdirSync(output,{recursive:true});
  const {chromium}=loadPlaywright(),browser=cdp ? await chromium.connectOverCDP(cdp) : await chromium.launch({headless:true});
  const context=await browser.newContext({acceptDownloads:true}),page=await context.newPage(),errors=[],rows=[];
  page.on('pageerror',error => errors.push(String(error)));
  const persist=(status,extra={}) => fs.writeFileSync(path.join(output,'report.json'),JSON.stringify({status,rows,errors,...extra},null,2));
  try {
    await page.goto(url);await page.waitForFunction(() => document.getElementById('bundleSize').textContent.includes('Mo'));
    await page.selectOption('#provider','wasm');await page.fill('#seed','9100000');
    const presets=await page.locator('#preset option').evaluateAll(options => options.map(option => option.value));
    const climates=['arid','cold','tropical','temperate','oceanic'];
    const download=async (button,name) => {
      const next=page.waitForEvent('download');await page.click(button);const item=await next;await item.saveAs(path.join(output,name));
    };
    for (let p=0;p<presets.length;p++) for (const climate of climates) {
      await page.selectOption('#preset',presets[p]);await page.selectOption('#climate',climate);
      const measurements=[];let metadata;
      for (let run=0;run<runs;run++) {
        await page.click('#generate');
        await page.waitForFunction(() => !document.getElementById('generate').disabled,{},{timeout:120000});
        const status=await page.locator('#status').innerText();
        if (!status.startsWith('Tuile calculée sur CPU')) throw new Error(status);
        metadata=JSON.parse(await page.locator('#metadata').textContent());
        if (metadata.side_m!==40000 || metadata.pixels!==1334 || metadata.input.climate!==climate || metadata.input.seed!==9100000) throw new Error('Inputs/geometry changed.');
        measurements.push({generationMs:metadata.timings.totalMs,requestMs:metadata.request_ms,previewReadyMs:metadata.preview_ready_ms});
      }
      const prefix=`sketch${p}-${climate}`;
      await download('#dem',prefix+'.f32');await download('#image',prefix+'.png');
      const raw=fs.readFileSync(path.join(output,prefix+'.f32'));
      if (raw.length!==1334*1334*4) throw new Error('Incomplete DEM export.');
      for (let i=0;i<raw.length;i+=4) if (!Number.isFinite(raw.readFloatLE(i))) throw new Error('Nonfinite DEM.');
      fs.writeFileSync(path.join(output,prefix+'.json'),JSON.stringify(metadata,null,2));
      rows.push({preset:presets[p],climate,seed:9100000,prefix,measurements,
        heightSha256:createHash('sha256').update(raw).digest('hex'),summary:metadata.summary,models:metadata.models});
      persist('running');console.log(JSON.stringify({case:rows.length,preset:presets[p],climate,worstGenerationSeconds:Math.max(...measurements.map(m => m.generationMs))/1000}));
    }
    if (presets.length!==6 || rows.length!==30 || errors.length) throw new Error('Incomplete matrix or browser errors.');
    const warm=rows.flatMap((row,index) => row.measurements.filter((_,run) => index || run));
    const worst=Math.max(...warm.map(m => m.generationMs))/1000;
    persist('complete',{chromeVersion:await browser.version(),worstWarmGenerationSeconds:worst,
      worstWarmPreviewReadySeconds:Math.max(...warm.map(m => m.previewReadyMs))/1000,
      latencyUnder10Seconds:worst<10,latencyUnder5Seconds:worst<5,qualityAccepted:false,
      scope:'Real browser CPU:6rough sketches x5climates,all1334²height exports,same seed,first cold run excluded from warm aggregate.Quality acceptance remains separate.'});
  } catch(error) {persist('failed',{error:String(error)});throw error;}
  finally {await context.close();await browser.close();}
}
main().catch(error => {console.error(error);process.exitCode=1;});
