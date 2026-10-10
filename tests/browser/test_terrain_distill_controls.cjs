const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const root = path.resolve(__dirname, '../..');
process.chdir(root);
const {readRepositoryFile} = require('../../tools/repository-files.cjs');
const {loadPlaywright, browserExecutable, pythonExecutable} = require('../../tools/platform.cjs');
const {chromium} = loadPlaywright();
const schema = JSON.parse(require('node:child_process').execFileSync(pythonExecutable(),
  ['-c', 'import json; from terrain_generation import generator_schema; print(json.dumps(generator_schema()))'], {encoding: 'utf8'}));
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM0LJrHwMDAxAAGAA7JAUW48M0QAAAAAElFTkSuQmCC', 'base64');

(async () => {
  const browser = await chromium.launch({executablePath: browserExecutable(), headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 900}});
    const errors = [], changes = [], worlds = [];
    let models = {selection: {coarse: 'solver4', base: 'teacher', decoder: 'distilled200k'},
      revision: 'initial', switching: false, options: {
        coarse: [{id: 'teacher', label: 'Original'}, {id: 'solver4', label: 'Distillé · 4 étapes'}, {id: 'solver8', label: 'Distillé · 8 étapes'}],
        base: [{id: 'teacher', label: 'Original'}], decoder: [{id: 'teacher', label: 'Original'}, {id: 'distilled200k', label: 'Distillé'}]}};
    page.on('pageerror', error => errors.push(error.message));
    await page.route('https://nn.test/**', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body:
        readRepositoryFile('index.html', 'utf8').replace('</html>', '<script src="/distill/live-controls.js"></script></html>')});
      if (url.pathname === '/distill/live-controls.js') return route.fulfill({contentType: 'application/javascript', body: fs.readFileSync(root + '/distill/live_controls.js', 'utf8')});
      if (url.pathname === '/terrain_renderer.js') return route.fulfill({contentType: 'application/javascript', body: 'window.createTerrainRenderer=async()=>({available:true,clear(){},draw(){return true},deleteTile(){},getStats(){return{}}})'});
      if (url.pathname === '/terrain_styles.js') return route.fulfill({contentType: 'application/javascript', body: 'window.TerrainStyles=' + readRepositoryFile('terrain_styles.json', 'utf8') + ';\n' + readRepositoryFile('terrain_style_rendering.js', 'utf8')});
      if (url.pathname.endsWith('.js')) return route.fulfill({contentType: 'application/javascript', body: readRepositoryFile(url.pathname.slice(1), 'utf8')});
      if (url.pathname === '/api/distill/models') {
        if (route.request().method() === 'POST') {
          const change = route.request().postDataJSON();
          changes.push(change);
          models = {...models, selection: change.selection, revision: 'revision-' + changes.length};
          return route.fulfill({status: 202, json: {restarting: true, revision: models.revision}});
        }
        return route.fulfill({json: models});
      }
      if (url.pathname === '/api/world') {
        const settings = JSON.parse(url.searchParams.get('generation'));
        worlds.push({seed: url.searchParams.get('seed'), profile: url.searchParams.get('world_profile'), settings});
        return route.fulfill({json: {seed: url.searchParams.get('seed'), version: 'test', cache_profile: models.revision,
          world_profile: worlds.at(-1).profile, generation_schema: schema, generation_settings: settings,
          world_bounds: [-20e6, -10e6, 20e6, 10e6], overview_bounds: [-20e6, -10e6, 20e6, 10e6], overview: '/overview.png', gpu: 'Mock'}});
      }
      if (url.pathname === '/api/view') return route.fulfill({json: {accepted: true}});
      if (url.pathname === '/api/status') return route.fulfill({json: {scheduler: {queued: 0, computing: 0, encoding: 0}}});
      return route.fulfill({contentType: 'image/png', body: png});
    });
    await page.goto('https://nn.test/?gpu=0');
    await page.waitForFunction(() => world && rendererReady && !generationPending);
    const firstWorld = worlds[0];
    await page.locator('#distillPanel summary').click();
    await page.locator('#distillCoarse option[value="solver8"]').waitFor({state: 'attached'});
    assert.equal(await page.inputValue('#distillCoarse'), 'solver4');
    await page.selectOption('#distillCoarse', 'solver8');
    await page.selectOption('#distillDecoder', 'teacher');
    await page.click('#distillApply');
    await page.waitForFunction(() => document.querySelector('#distillStatus').textContent.startsWith('NN appliqués'));
    assert.deepEqual(changes[0], {selection: {coarse: 'solver8', base: 'teacher', decoder: 'teacher'}, reset: false});
    assert.deepEqual(worlds.at(-1), firstWorld, 'Applying NN retains the seed and Orogen settings');
    await page.click('#distillReset');
    await page.waitForFunction(() => document.querySelector('#distillStatus').textContent.startsWith('Cache NN réinitialisé'));
    assert.equal(changes[1].reset, true);
    assert.deepEqual(changes[1].selection, changes[0].selection);
    assert.deepEqual(worlds.at(-1), firstWorld);
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#distillPanel').evaluate(element => element.open), false);
    assert.deepEqual(errors, []);
    console.log('NN menu: visible choices, independent models, apply and reset retain the world, Escape closes the panel.');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
