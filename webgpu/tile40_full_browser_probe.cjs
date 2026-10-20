// Time actual input-to-final-height generation in an isolated CPU/GPU browser.
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {createHash} = require('node:crypto');
const {loadPlaywright, browserExecutable} = require('../tools/platform.cjs');

async function main() {
  const args = process.argv.slice(2);
  const option = (name, fallback) => args.includes(name) ? args[args.indexOf(name)+1] : fallback;
  const models = option('--models'), output = option('--output'), provider = option('--provider', 'wasm');
  const threads = Number(option('--threads', '8')), runs = Number(option('--runs', '4')), cdp = option('--cdp');
  if (!models || !output || !['wasm', 'webgpu'].includes(provider) || !Number.isInteger(threads) || threads < 1 || runs < 2) throw new Error('Models,output,provider and positive thread/run counts required.');
  const manifest = JSON.parse(fs.readFileSync(path.join(models, 'manifest.json'), 'utf8'));
  const runtime=fs.existsSync(path.join(models,'runtime-snapshot')) ? path.join(models,'runtime-snapshot') : __dirname;
  const sha = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
  for (const model of Object.values(manifest.models)) {
    if (path.basename(model.file) !== model.file || model.bytes >= 100000000 || sha(path.join(models, model.file)) !== model.sha256) throw new Error('Model identity or size changed.');
  }
  for (const [file, digest] of [[manifest.input_file, manifest.input_sha256], [manifest.expected_file, manifest.expected_sha256]]) {
    if (path.basename(file) !== file || sha(path.join(models, file)) !== digest) throw new Error('Full-tile fixture changed.');
  }
  if (sha(path.join(runtime, 'tile40_runtime.mjs')) !== manifest.provenance.runtime_sha256 ||
      sha(path.join(runtime, 'noise.mjs')) !== manifest.provenance.noise_sha256) throw new Error('Runtime changed;preserve the old bundle and version its manifest.');
  const vendor = path.join(__dirname, 'vendor');
  const server = http.createServer((request, response) => {
    response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    response.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    const route = new URL(request.url, 'http://localhost').pathname;
    if (route === '/') {response.setHeader('Content-Type', 'text/html');response.end('<!doctype html><meta charset="utf-8"><title>Full40km benchmark</title>');return;}
    let filename;
    if (route.startsWith('/ort/') && path.basename(route) === route.slice(5)) filename = path.join(vendor, path.basename(route));
    if (route.startsWith('/model/') && path.basename(route) === route.slice(7)) filename = path.join(models, path.basename(route));
    if (['/runtime/noise.mjs', '/runtime/tile40_runtime.mjs'].includes(route)) filename = path.join(runtime, path.basename(route));
    if (!filename || !fs.existsSync(filename) || !fs.statSync(filename).isFile()) {response.statusCode=404;response.end();return;}
    const ext = path.extname(filename);
    response.setHeader('Content-Type', ext === '.wasm' ? 'application/wasm' : ['.js', '.mjs'].includes(ext) ? 'text/javascript' : ext === '.json' ? 'application/json' : 'application/octet-stream');
    fs.createReadStream(filename).pipe(response);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const {chromium} = loadPlaywright();
  let browser, report;
  const messages = [];
  try {
    browser = cdp ? await chromium.connectOverCDP(cdp) : await chromium.launch({executablePath:browserExecutable(), headless:true, args:['--enable-unsafe-webgpu']});
    const page = await browser.newPage();
    page.on('console', entry => messages.push(`${entry.type()}: ${entry.text()}`));
    page.on('pageerror', error => messages.push(`pageerror:${error.message}`));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    report = await page.evaluate(async ({manifest, provider, threads, runs}) => {
      const adapter = navigator.gpu ? await navigator.gpu.requestAdapter({powerPreference:'high-performance'}) : null;
      const info = adapter ? adapter.info || {} : {};
      const identity = `${info.vendor || ''} ${info.architecture || ''} ${info.description || ''}`.toLowerCase();
      const adapterKind = !adapter ? 'unavailable' : /swiftshader|llvmpipe|software|warp/.test(identity) ? 'software' :
        /nvidia|intel|amd|apple|qualcomm|arm/.test(identity) ? 'hardware' : 'unknown';
      const capabilities = {userAgent:navigator.userAgent, hardwareConcurrency:navigator.hardwareConcurrency,
        crossOriginIsolated, adapterKind, adapter:adapter ? {vendor:info.vendor, architecture:info.architecture, features:[...adapter.features]} : null};
      if (provider === 'webgpu' && adapterKind !== 'hardware') return {status:'unavailable', capabilities, reason:'Identified hardware GPU required.'};
      await new Promise((resolve, reject) => {
        const script=document.createElement('script');script.src='/ort/ort.webgpu.min.js';script.onload=resolve;script.onerror=reject;document.head.append(script);
      });
      const {generateTile} = await import('/runtime/tile40_runtime.mjs');
      ort.env.wasm.wasmPaths = `${location.origin}/ort/`;
      ort.env.wasm.numThreads = provider === 'wasm' ? threads : 1;
      const sessions = {}, loaded = performance.now();
      for (const [stage, model] of Object.entries(manifest.models)) sessions[stage] =
        await ort.InferenceSession.create(`/model/${model.file}`, {executionProviders:[provider]});
      const loadMs = performance.now()-loaded;
      const input = await (await fetch(`/model/${manifest.input_file}`)).json();
      const expected = new Float32Array(await (await fetch(`/model/${manifest.expected_file}`)).arrayBuffer());
      const measurements = [];
      let actual;
      for (let index = 0; index < runs; index++) {
        actual = await generateTile(ort, sessions, manifest, input);
        measurements.push(actual.timings);
      }
      if (actual.height.length !== 1334*1334 || actual.height.length !== expected.length) throw new Error('The output does not cover the contracted40km.');
      let maxAbs=0, sum=0, finite=true, min=Infinity, max=-Infinity;
      for (let i = 0; i < actual.height.length; i++) {
        const value=actual.height[i], delta=value-expected[i];
        finite &&= Number.isFinite(value);maxAbs=Math.max(maxAbs, Math.abs(delta));sum+=delta*delta;
        min=Math.min(min, value);max=Math.max(max, value);
      }
      for (const session of Object.values(sessions)) await session.release();
      const rmse = Math.sqrt(sum/actual.height.length);
      const worstWarmSeconds = Math.max(...measurements.slice(1).map(run => run.totalMs))/1000;
      return {status:finite && maxAbs < 1 && rmse < .05 ? 'complete' : 'numerical_mismatch', capabilities, provider,
        acceptance:{numericalAgreement:finite && maxAbs < 1 && rmse < .05,
          warmLatencyUnder10Seconds:worstWarmSeconds < 10, warmLatencyUnder5Seconds:worstWarmSeconds < 5,
          quality:'unproven', testCoverage:'one fixture'},
        threads:ort.env.wasm.numThreads, loadMs, firstRun:measurements[0], warmRuns:measurements.slice(1),
        maxAbsM:maxAbs, rmseM:rmse, minM:min, maxM:max, pixels:1334, sideM:40000,
        scope:'Real browser sketch/climate/seed to final40km DEM,including noise,conditioning,all neural windows,fusion,reconstruction and output downloads.Quality acceptance remains separate.'};
    }, {manifest, provider, threads, runs});
    report.chromeVersion = await browser.version();
  } catch (error) {
    report={status:'failed', provider, error:String(error)};process.exitCode=1;
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
  report.messages = messages;
  report.identity = {models:manifest.models, totalModelBytes:manifest.total_model_bytes,
    input_sha256:manifest.input_sha256, expected_sha256:manifest.expected_sha256, runtime_sha256:manifest.provenance.runtime_sha256};
  fs.mkdirSync(path.dirname(output), {recursive:true});fs.writeFileSync(output, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  if (report.status !== 'complete') process.exitCode=1;
  else {
    const times = report.warmRuns.map(run => run.totalMs);
    console.log(`METRIC full_tile_seconds=${Math.max(...times)/1000}`);
    console.log(`METRIC full_tile_rmse_m=${report.rmseM}`);
    console.log(`METRIC total_model_bytes=${manifest.total_model_bytes}`);
    console.log(`ARTIFACT browser_full_tile=${output}`);
  }
}
main().catch(error => {console.error(error);process.exitCode=1;});
