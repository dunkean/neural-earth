// Isolated browser CPU/GPU capability and real compact-decoder baseline.
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {createHash} = require('node:crypto');
const {loadPlaywright, browserExecutable} = require('../tools/platform.cjs');

async function main() {
  const args = process.argv.slice(2);
  const option = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
  const output = option('--output');
  const provider = option('--provider', 'capabilities');
  const models = option('--models');
  const cdp = option('--cdp');
  const threads = Number(option('--threads', '4'));
  if (!output || !['capabilities', 'wasm', 'webgpu'].includes(provider) || !Number.isInteger(threads) || threads < 1) {
    throw new Error('Use --output, --provider capabilities|wasm|webgpu, and a positive --threads.');
  }
  if (provider !== 'capabilities' && !models) throw new Error('A decoder export directory is required.');
  const manifest = models ? JSON.parse(fs.readFileSync(path.join(models, 'manifest.json'), 'utf8')) : null;
  const sha = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
  if (manifest) {
    for (const [file, digest] of [[manifest.model, manifest.model_sha256], [manifest.input_file, manifest.input_sha256],
                                [manifest.expected_file, manifest.expected_sha256]]) {
      if (path.basename(file) !== file || sha(path.join(models, file)) !== digest) throw new Error('Export fixture changed.');
    }
  }
  const vendor = path.join(__dirname, 'vendor');
  const server = http.createServer((request, response) => {
    response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    response.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><meta charset="utf-8"><title>Tile40 benchmark</title>');
      return;
    }
    let filename;
    if (pathname.startsWith('/ort/') && path.basename(pathname) === pathname.slice(5)) filename = path.join(vendor, path.basename(pathname));
    if (models && pathname.startsWith('/model/') && path.basename(pathname) === pathname.slice(7)) filename = path.join(models, path.basename(pathname));
    if (!filename || !fs.existsSync(filename) || !fs.statSync(filename).isFile()) {
      response.statusCode = 404; response.end(); return;
    }
    const suffix = path.extname(filename);
    response.setHeader('Content-Type', suffix === '.wasm' ? 'application/wasm' : ['.js', '.mjs'].includes(suffix) ? 'text/javascript' : 'application/octet-stream');
    fs.createReadStream(filename).pipe(response);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const {chromium} = loadPlaywright();
  let browser;
  const messages = [];
  let report;
  try {
    browser = cdp ? await chromium.connectOverCDP(cdp) :
      await chromium.launch({executablePath:browserExecutable(), headless:true, args:['--enable-unsafe-webgpu']});
    const page = await browser.newPage();
    page.on('console', entry => messages.push(`${entry.type()}: ${entry.text()}`));
    page.on('pageerror', error => messages.push(`pageerror: ${error.message}`));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    report = await page.evaluate(async ({provider, manifest, threads}) => {
      const adapter = navigator.gpu ? await navigator.gpu.requestAdapter({powerPreference:'high-performance'}) : null;
      const info = adapter ? adapter.info || await adapter.requestAdapterInfo?.() || {} : {};
      const identity = `${info.vendor || ''} ${info.architecture || ''} ${info.description || ''}`.toLowerCase();
      const adapterKind = !adapter ? 'unavailable' : /swiftshader|llvmpipe|software|warp/.test(identity) ? 'software' :
        /nvidia|intel|amd|apple|qualcomm|arm/.test(identity) ? 'hardware' : 'unknown';
      const capabilities = {userAgent:navigator.userAgent, hardwareConcurrency:navigator.hardwareConcurrency,
        crossOriginIsolated, wasm:typeof WebAssembly === 'object', webgpu:!!adapter,
        adapterKind,
        adapter:adapter ? {vendor:info.vendor, architecture:info.architecture, device:info.device, description:info.description,
                          features:[...adapter.features], maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize} : null};
      if (provider === 'capabilities') return {status:'complete', capabilities};
      if (provider === 'webgpu' && !adapter) return {status:'unavailable', capabilities, reason:'No WebGPU adapter.'};
      if (provider === 'webgpu' && adapterKind !== 'hardware') {
        return {status:'unavailable', capabilities, reason:'Hardware GPU timing requires an identified hardware adapter.'};
      }
      await new Promise((resolve, reject) => {
        const script = document.createElement('script'); script.src = '/ort/ort.webgpu.min.js';
        script.onload = resolve; script.onerror = reject; document.head.append(script);
      });
      ort.env.wasm.wasmPaths = `${location.origin}/ort/`;
      ort.env.wasm.numThreads = provider === 'wasm' ? threads : 1;
      const loaded = performance.now();
      const session = await ort.InferenceSession.create(`/model/${manifest.model}`, {executionProviders:[provider]});
      const loadMs = performance.now()-loaded;
      const input = new Float32Array(await (await fetch(`/model/${manifest.input_file}`)).arrayBuffer());
      const expected = new Float32Array(await (await fetch(`/model/${manifest.expected_file}`)).arrayBuffer());
      const tensor = new ort.Tensor('float32', input, manifest.input_shape);
      const times = [];
      let actual;
      for (let index = 0; index < 4; index++) {
        const started = performance.now();
        const outputs = await session.run({inputs:tensor});
        const result = outputs[manifest.output_name || 'residual'];
        actual = await result.getData();
        times.push(performance.now()-started);
        result.dispose();
      }
      let maxAbs = 0, squares = 0, finite = true;
      if (actual.length !== expected.length) throw new Error('Output length differs from the real fixture.');
      for (let index = 0; index < actual.length; index++) {
        const delta = actual[index]-expected[index];
        finite &&= Number.isFinite(actual[index]); maxAbs = Math.max(maxAbs, Math.abs(delta)); squares += delta*delta;
      }
      await session.release(); tensor.dispose();
      return {status:finite && maxAbs < .001 ? 'complete' : 'numerical_mismatch', provider, capabilities,
              threads:ort.env.wasm.numThreads, loadMs, firstRunMs:times[0], warmRunMs:times.slice(1),
              maxAbs, rmse:Math.sqrt(squares/actual.length),
              scope:manifest.browser_scope || 'One512px compact-decoder window;not full40km latency.'};
    }, {provider, manifest, threads});
    report.chromeVersion = await browser.version();
  } catch (error) {
    report = {status:'failed', provider, error:String(error)};
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
  report.messages = messages;
  if (manifest) report.identity = {model_sha256:manifest.model_sha256, input_sha256:manifest.input_sha256,
                                  expected_sha256:manifest.expected_sha256, model_bytes:manifest.model_bytes};
  fs.mkdirSync(path.dirname(output), {recursive:true});
  fs.writeFileSync(output, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  if (provider !== 'capabilities') {
    if (report.status !== 'complete') process.exitCode = 1;
    else {
      const meanMs = report.warmRunMs.reduce((sum, value) => sum + value, 0) / report.warmRunMs.length;
      const stage = manifest.stage || 'decoder';
      console.log(`METRIC ${stage}_window_seconds=${meanMs / 1000}`);
      console.log(`METRIC ${stage}_export_max_abs=${report.maxAbs}`);
      console.log(`METRIC ${stage}_model_bytes=${manifest.model_bytes}`);
      console.log(`ARTIFACT browser_${stage}=${output}`);
    }
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
