const {loadPlaywright, browserExecutable, runtimePath} = require('../tools/platform.cjs');
const { chromium } = loadPlaywright();
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { verifyFixtureManifest, sourceHashes, requestedHashes } = require('./provenance.cjs');

async function main() {
  const model = process.argv[2] || 'coarse_model';
  const useReference = process.argv.includes('--reference');
  const requireGraphCapture = !process.argv.includes('--no-capture');
  const staticBatch = process.argv.includes('--static');
  const resizeUp = process.argv.includes('--resize');
  const url = process.env.TERRAIN_PROBE_URL || 'http://127.0.0.1:8770/webgpu/probe.html';
  const chrome = browserExecutable();
  const browser = await chromium.launch({
    executablePath: chrome,
    headless: true,
    args: ['--enable-unsafe-webgpu'],
  });
  const messages = [];
  const successfulRequests = new Set();
  const page = await browser.newPage();
  page.on('console', entry => messages.push(`console.${entry.type()}: ${entry.text()}`));
  page.on('pageerror', error => messages.push(`pageerror: ${error.stack || error}`));
  page.on('response', response => {
    if (response.status() >= 200 && response.status() < 300) successfulRequests.add(new URL(response.url()).pathname);
  });
  const began = Date.now();
  let result;
  try {
    await page.goto(url, { waitUntil: 'load', timeout: 30000 });
    if (model === 'crop') {
      result = await page.evaluate(async capture => {
        const { runCrop } = await import('/webgpu/crop.mjs');
        const status = document.querySelector('#status');
        status.textContent = '';
        return await runCrop(ort, line => { status.textContent += `${line}\n`; }, capture);
      }, requireGraphCapture);
    } else {
      result = await page.evaluate(async ({ name, reference, capture, staticShape, resize }) => {
        if (!window.terrainWebGpuProbe) throw new Error('Probe script did not load');
        return await window.terrainWebGpuProbe.run(name, reference, capture, staticShape, resize);
      }, { name: model, reference: useReference, capture: requireGraphCapture, staticShape: staticBatch, resize: resizeUp });
    }
  } catch (error) {
    result = {
      model, error: String(error.stack || error),
      ...(model === 'crop' ? { numericalStatus: 'NOT_RUN', l1Status: 'ERROR' } : {}),
      progress: await page.locator('#status').textContent().catch(() => ''),
    };
  }
  const chromeVersion = browser.version();
  const userAgent = await page.evaluate(() => navigator.userAgent).catch(() => null);
  await browser.close();
  const artifacts = runtimePath('webgpu-models');
  const exported = JSON.parse(fs.readFileSync(path.join(artifacts, 'export-report.json')));
  const fixture = JSON.parse(fs.readFileSync(path.join(artifacts, 'crop64-coast-manifest.json')));
  const lock = JSON.parse(fs.readFileSync(path.join(__dirname, 'package-lock.json')));
  const sha256 = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
  const requestedFiles = await requestedHashes(successfulRequests, __dirname, artifacts);
  const modelFiles = requestedFiles.filter(item => /\.(onnx|data)$/.test(item.pathname));
  for (const item of modelFiles) {
    const name = path.basename(item.pathname);
    const known = Object.values(exported.models).flatMap(model => [
      { path: model.path, sha256: model.sha256 }, ...(model.external_files || []),
    ]).find(entry => entry.path === name);
    if (known && known.sha256 !== item.sha256) throw new Error(`Downloaded model artifact differs from export report: ${name}`);
  }
  const report = {
    ...result, wallMs: Date.now() - began, messages,
    identity: {
      checkpointRevision: exported.checkpoint_revision,
      manifestSha256: sha256(path.join(artifacts, 'crop64-coast-manifest.json')),
      probeSourceSha256: sha256(path.join(__dirname, 'probe.js')),
      harnessSourceSha256: sha256(path.join(__dirname, 'verify_browser.cjs')),
      provenanceHelperSha256: sha256(path.join(__dirname, 'provenance.cjs')),
      cropSourceSha256: sha256(path.join(__dirname, 'crop.mjs')),
      ortWebVersion: lock.packages['node_modules/onnxruntime-web'].version,
      chromeVersion, userAgent,
      modelFiles,
      sourceModules: await sourceHashes(__dirname),
      requestedFiles,
      fixtureManifestVerification: await verifyFixtureManifest(artifacts, fixture),
    },
  };
  const outfile = path.join(runtimePath('webgpu-models'), `browser-${model}${useReference ? '-reference' : ''}${requireGraphCapture ? '' : '-nocapture'}${staticBatch ? '-static' : ''}${resizeUp ? '-resize' : ''}.json`);
  const serialized = JSON.stringify(report, null, 2);
  fs.writeFileSync(outfile, serialized);
  const archive = outfile.replace(/\.json$/, `-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
  fs.writeFileSync(archive, serialized);
  console.log(JSON.stringify(report, null, 2));
  if (report.error || (model === 'crop' && report.l1Status !== 'PASS')) process.exitCode = 1;
}

main().catch(error => { console.error(error); process.exitCode = 1; });
