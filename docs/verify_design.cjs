// Read-only document QA; uses the existing local Playwright installation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_PATH || 'E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
  });
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(path.join(__dirname, 'TERRAIN_REALTIME_DESIGN.html')).href);
    const report = await page.evaluate(() => ({
      title: document.title,
      sections: document.querySelectorAll('article h2').length,
      subsections: document.querySelectorAll('article h3').length,
      tables: document.querySelectorAll('article table').length,
      characters: document.querySelector('article').textContent.length,
      brokenAnchors: [...document.querySelectorAll('a')]
        .filter(a => a.getAttribute('href')?.startsWith('#'))
        .filter(a => !document.getElementById(decodeURIComponent(a.hash.slice(1))))
        .map(a => a.hash),
      overflow: document.documentElement.scrollWidth > innerWidth,
      calculator: document.getElementById('calc-output').textContent,
      localLinks: [...document.querySelectorAll('a')].map(a => a.getAttribute('href'))
        .filter(h => h && !h.startsWith('#') && !h.startsWith('http')),
    }));
    assert.equal(report.sections, 16);
    assert.equal(report.brokenAnchors.length, 0);
    assert.equal(report.overflow, false);
    for (const link of report.localLinks) assert.ok(fs.existsSync(path.resolve(__dirname, link)), link);
    await page.screenshot({path: 'E:/TerrainDiffusionRuntime/design-desktop.png'});
    await page.locator('#baseline').fill('324000');
    report.calculatorWithBaseline = await page.locator('#calc-output').innerText();
    assert.ok(report.calculatorWithBaseline.includes('×2,4'));
    await page.locator('article h2').filter({hasText: '11. Site public'}).scrollIntoViewIfNeeded();
    await page.screenshot({path: 'E:/TerrainDiffusionRuntime/design-webgpu.png'});
    await page.setViewportSize({width: 390, height: 844});
    await page.evaluate(() => scrollTo(0, 0));
    report.mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    assert.equal(report.mobileOverflow, false);
    await page.locator('#toc-toggle').click();
    assert.ok(await page.locator('#navigation').isVisible());
    await page.locator('#toc-toggle').click();
    await page.screenshot({path: 'E:/TerrainDiffusionRuntime/design-mobile.png'});
    assert.deepEqual(errors, []);
    delete report.localLinks;
    console.log(JSON.stringify({...report, errors}, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
