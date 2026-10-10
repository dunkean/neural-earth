const {loadPlaywright, browserExecutable, runtimePath} = require('../platform.cjs');
const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {chromium} = loadPlaywright();
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const output = process.env.TERRAIN_QA_OUTPUT || runtimePath('realtime-qa');
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({
    executablePath: browserExecutable(),
    headless: true,
  });
  const errors = [], views = [], heights = [];
  try {
    const page = await browser.newPage({viewport: {width: 900, height: 650}});
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => {
      if (r.url().includes('/api/view') && r.method() === 'POST') views.push(r.postData());
      if (r.url().includes('/height/')) heights.push(r.url());
    });
    await page.goto('http://127.0.0.1:8765/?seed=42');
    await page.waitForFunction(() => Number(document.getElementById('viewport').dataset.loaded) > 0,
      {}, {timeout: 240000});
    if (await page.locator('#prefetch').count()) await page.locator('#runtimePanel').evaluate(el=>el.open=true);await page.locator('#prefetch').uncheck();
    const meta = await (await page.request.get('http://127.0.0.1:8765/api/world?seed=42')).json();
    assert.match(meta.version, /^natural-/);
    assert.equal(meta.native_resolution, 30);
    const before = await page.evaluate(() => window.terrainDebug?.snapshot());
    assert(before, 'Read-only navigation diagnostics must be available');
    const vp = page.locator('#viewport');
    await vp.focus();
    const responsiveness = await page.evaluate(async () => {
      const start = performance.now();
      let frames = 0;
      await new Promise(resolve => {
        function tick() { frames++; if (performance.now()-start < 500) requestAnimationFrame(tick); else resolve(); }
        requestAnimationFrame(tick);
      });
      return {frames, wallMs: performance.now()-start};
    });
    await page.locator('#native').click();
    await page.waitForFunction(() => document.getElementById('viewport').dataset.lod === '0', {}, {timeout: 10000});
    await vp.focus();
    // A rapid camera burst must be acknowledged while generation is still busy.
    const cameraStart = await page.evaluate(() => window.terrainDebug.snapshot());
    for (let i = 0; i < 8; i++) await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(150);
    const cameraEnd = await page.evaluate(() => window.terrainDebug.snapshot());
    assert.notDeepEqual(cameraStart.camera, cameraEnd.camera, 'Camera must move independently of inference');
    await page.locator('#fit').click();
    await page.waitForFunction(() => document.getElementById('viewport').dataset.pending === '0', {}, {timeout: 240000});
    await page.screenshot({path: path.join(output, 'navigation-desktop.png')});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.locator('#runtimePanel').evaluate(el=>el.open=true);
    const gpuControl = page.locator('#gpuRender');
    if (await gpuControl.isEnabled()) {
      await gpuControl.uncheck();
      await page.waitForFunction(() => document.getElementById('viewport').dataset.render === 'png' &&
        document.getElementById('viewport').dataset.pending === '0', {}, {timeout: 240000});
      const requestsBefore = heights.length;
      await page.locator('#runtimePanel').evaluate(el=>el.open=true);await gpuControl.check();
      await page.waitForFunction(() => document.getElementById('viewport').dataset.render === 'webgpu' &&
        document.getElementById('viewport').dataset.pending === '0', {}, {timeout: 240000});
      assert(heights.length > requestsBefore, 'Enabling WebGPU must fetch physical heights for existing PNG tiles');
    }
    const after = await page.evaluate(() => window.terrainDebug.snapshot());
    assert(after.cache.bytes <= after.cache.budget, 'Host cache must respect its byte budget');
    if (after.renderer) assert(after.renderer.totalBytes <= after.renderer.maxBytes, 'GPU cache must respect its byte budget');
    assert(views.length > 0, 'Browser must publish camera subscriptions to cancel server work');
    assert.deepEqual(errors, []);
    const status = await (await page.request.get('http://127.0.0.1:8765/api/status')).json();
    const report = {passed: true, responsiveness, before, after, cameraStart, cameraEnd,
      cameraPosts: views.length, heightRequests: heights.length, errors, status};
    fs.writeFileSync(path.join(output, 'navigation.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify({passed: true, responsiveness, cameraPosts: views.length,
      heightRequests: heights.length, debug: after}, null, 2));
    await page.setViewportSize({width: 390, height: 844});
    await page.waitForTimeout(300);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.screenshot({path: path.join(output, 'navigation-mobile.png')});
    await page.goto('about:blank');
  } finally {
    for (const page of browser.contexts().flatMap(context => context.pages())) {
      try {
        const session = await page.evaluate(() => window.terrainDebug?.snapshot().session);
        if (session) await page.request.post('http://127.0.0.1:8765/api/view/release', {data: {session}});
      } catch (_) { /* Page may already be closed. The server lease also expires. */ }
    }
    await browser.close();
  }
})().catch(e => { console.error(e); process.exit(1); });
