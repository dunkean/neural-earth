const {loadPlaywright, browserExecutable, runtimePath} = require('../tools/platform.cjs');
const { chromium } = loadPlaywright();
const fs = require('node:fs');

async function main() {
  const browser = await chromium.launch({
    executablePath: browserExecutable(),
    headless: true, args: ['--enable-unsafe-webgpu'],
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(180000);
  const messages = [];
  page.on('console', entry => messages.push(`${entry.type()}: ${entry.text()}`));
  try {
    await page.goto('http://127.0.0.1:8770/webgpu/probe.html');
    const report = await page.evaluate(async () => {
      const { makeGpuRunner } = await import('/webgpu/io.mjs');
      const manifestResponse = await fetch('/webgpu-models/conv-channel-repro.json');
      if (!manifestResponse.ok) throw new Error(`Conv manifest HTTP ${manifestResponse.status}`);
      const corpus = await manifestResponse.json();
      ort.env.wasm.wasmPaths = new URL('/webgpu/vendor/', location.href).href;
      ort.env.wasm.numThreads = 1;
      const read = async (filename, shape) => {
        const response = await fetch(`/webgpu-models/${filename}`);
        if (!response.ok) throw new Error(`Conv tensor HTTP ${response.status}: ${filename}`);
        const bytes = await response.arrayBuffer();
        const expectedBytes = shape.reduce((a, b) => a * b, 1) * 4;
        if (bytes.byteLength !== expectedBytes) throw new Error(`Conv tensor length mismatch: ${filename}`);
        const array = new Float32Array(bytes);
        if (array.some(value => !Number.isFinite(value))) throw new Error(`Non-finite Conv tensor: ${filename}`);
        return array;
      };
      const results = [];
      for (const entry of corpus.entries) {
        for (const layout of (entry.size === 64 ? ['NCHW'] : ['NCHW', 'NHWC'])) {
          let session, runner;
          try {
            session = await ort.InferenceSession.create(`/webgpu-models/${entry.model}`, {
              executionProviders: [{ name: 'webgpu', preferredLayout: layout }],
              enableGraphCapture: false, graphOptimizationLevel: 'all',
            });
            const device = await ort.env.webgpu.device;
            runner = makeGpuRunner(ort, device, session, { x: entry.input_shape }, entry.output_shape);
            const actual = await runner.run({ x: await read(entry.input, entry.input_shape) });
            if (actual.length !== entry.output_shape.reduce((a, b) => a * b, 1) || actual.some(value => !Number.isFinite(value))) throw new Error('Non-finite or truncated Conv output');
            const expected = await read(entry.reference, entry.output_shape);
            let maxAbs = 0, squares = 0, maxIndex = 0;
            for (let i = 0; i < actual.length; i++) {
              const diff = actual[i] - expected[i];
              if (Math.abs(diff) > maxAbs) { maxAbs = Math.abs(diff); maxIndex = i; }
              squares += diff * diff;
            }
            results.push({ size: entry.size, channels: entry.channels, layout,
              maxAbs, rmse: Math.sqrt(squares / actual.length), maxIndex,
              actualAtMax: actual[maxIndex], expectedAtMax: expected[maxIndex] });
          } catch (error) {
            results.push({ size: entry.size, channels: entry.channels, layout, error: String(error.stack || error) });
          } finally {
            if (runner) runner.dispose();
            if (session) await session.release();
          }
        }
      }
      return { sourceModelSha256: corpus.source_model_sha256, results };
    });
    report.chromeVersion = browser.version();
    report.ortWebVersion = '1.30.0';
    report.messages = messages;
    fs.writeFileSync(runtimePath('webgpu-models/conv-channel-browser.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report.results, null, 2));
    if (report.results.some(item => item.error)) process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
