const {loadPlaywright, browserExecutable, runtimePath} = require('../tools/platform.cjs');
const { chromium } = loadPlaywright();
const fs = require('node:fs');

async function main() {
  const browser = await chromium.launch({ executablePath: browserExecutable(), headless: true, args: ['--enable-unsafe-webgpu'] });
  const page = await browser.newPage();
  const messages = [];
  page.on('console', entry => messages.push(`${entry.type()}: ${entry.text()}`));
  await page.goto('http://127.0.0.1:8770/webgpu/probe.html');
  const report = await page.evaluate(async () => {
    const { makeGpuRunner } = await import('/webgpu/io.mjs');
    const models = await (await fetch('/webgpu-models/decoder-diagnostic.json')).json();
    const fixture = await (await fetch('/webgpu-models/crop64-coast-manifest.json')).json();
    const captured = fixture.first_real_forward.decoder_model;
    const load = async spec => new Float32Array(await (await fetch(`/webgpu-models/${spec.file || spec}`)).arrayBuffer());
    const inputs = { x: await load(captured.x), noise_labels: await load(captured.noise_labels) };
    if (!await navigator.gpu.requestAdapter()) throw new Error('WebGPU adapter unavailable');
    ort.env.wasm.wasmPaths = new URL('/webgpu/vendor/', location.href).href;
    ort.env.wasm.numThreads = 1;
    const results = {};
    for (const [index, info] of Object.entries(models)) {
      const session = await ort.InferenceSession.create(`/webgpu-models/${info.model}`, { executionProviders: ['webgpu'], enableGraphCapture: true, freeDimensionOverrides: { batch: 1 } });
      const device = await ort.env.webgpu.device;
      const shape = info.input
        ? { x6: info.input_shape }
        : { x: captured.x.shape, noise_labels: captured.noise_labels.shape };
      const runner = makeGpuRunner(ort, device, session, shape, info.output_shape);
      const output = await runner.run(info.input ? { x6: await load(info.input) } : inputs);
      const expected = await load(info.reference);
      let maxAbs = 0, squares = 0;
      for (let i = 0; i < output.length; i++) {
        const diff = output[i] - expected[i];
        maxAbs = Math.max(maxAbs, Math.abs(diff)); squares += diff * diff;
      }
      results[index] = { node: info.node, maxAbs, rmse: Math.sqrt(squares / output.length), min: Math.min(...output.slice(0, 1000)), max: Math.max(...output.slice(0, 1000)) };
      runner.dispose();
      await session.release();
    }
    return results;
  });
  await browser.close();
  fs.writeFileSync(runtimePath('webgpu-models/decoder-diagnostic-browser.json'), JSON.stringify({ report, messages }, null, 2));
  console.log(JSON.stringify({ report, messages }, null, 2));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
