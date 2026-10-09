const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {readRepositoryFile} = require(NEURAL_EARTH_ROOT + '/tools/repository-files.cjs');
// Hardware browser measurement of camera work, display cadence, and coverage.
// GPU execution and presentation latency are not timed here.
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'E:/TerrainDiffusionRuntime/ui-test/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');

const destination = process.env.TERRAIN_BROWSER_REPORT || 'E:/TerrainDiffusionRuntime/audit-implementation/browser-continuous-pan.json';
const quantile = (values, q) => values.length
  ? [...values].sort((a, b) => a - b)[Math.max(0, Math.ceil(q * values.length) - 1)] : null;
const cacheCounts = responses => responses.reduce((counts, response) => {
  const label = response.headers?.['x-terrain-cache'] || 'unspecified';
  counts[label] = (counts[label] || 0) + 1;
  return counts;
}, {});

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true, args: ['--enable-unsafe-webgpu'],
  });
  const page = await browser.newPage({ viewport: { width: 900, height: 650 } });
  const errors = [], transport = [], scenarios = [];
  const headersReady = new WeakMap();
  let activeScenario = 'startup';
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => {
    if (response.url().includes('/height/')) {
      const entry = { scenario: activeScenario, url: response.url(), status: response.status(), headers: null };
      transport.push(entry);
      headersReady.set(entry, response.allHeaders().then(headers => { entry.headers = headers; }));
    }
  });
  async function responsesSince(start) {
    const responses = transport.slice(start);
    await Promise.all(responses.map(entry => headersReady.get(entry)));
    return responses;
  }

  function nativeRendered(snapshot) {
    return snapshot.rendered.patches > 0 && snapshot.rendered.lods.every(lod => lod === 0) &&
      snapshot.rendered.sources.length > 0 && snapshot.rendered.sources.every(source => source === 'decoder');
  }

  function complete(snapshot, minimumEpoch, requireDecoder) {
    return snapshot.cameraEpoch >= minimumEpoch && snapshot.visible.pending === 0 &&
      snapshot.visible.ready === snapshot.visible.wanted &&
      snapshot.visible.lod === snapshot.camera.lod &&
      snapshot.scheduler.inflight === 0 && snapshot.scheduler.queued === 0 &&
      (!requireDecoder || nativeRendered(snapshot));
  }

  async function settled(minimumEpoch = -1, requireDecoder = false) {
    const deadline = Date.now() + 300000;
    while (true) {
      const remaining = deadline - Date.now();
      if (remaining <= 0) throw new Error('Camera settlement timeout after draw drain');
      await page.waitForFunction(({ minimumEpoch, requireDecoder }) => {
        const snapshot = terrainDebug.snapshot();
        return snapshot.cameraEpoch >= minimumEpoch && snapshot.visible.pending === 0 &&
          snapshot.visible.ready === snapshot.visible.wanted &&
          snapshot.visible.lod === snapshot.camera.lod &&
          snapshot.scheduler.inflight === 0 && snapshot.scheduler.queued === 0 &&
          (!requireDecoder || (snapshot.rendered.patches > 0 &&
            snapshot.rendered.lods.every(lod => lod === 0) &&
            snapshot.rendered.sources.length > 0 &&
            snapshot.rendered.sources.every(source => source === 'decoder')));
      }, { minimumEpoch, requireDecoder }, { timeout: remaining });
      const snapshot = await page.evaluate(() => new Promise(resolve =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve(terrainDebug.snapshot())))));
      if (complete(snapshot, minimumEpoch, requireDecoder)) return snapshot;
    }
  }

  async function forceFinalRefresh() {
    return page.evaluate(() => {
      if (timer !== null) { clearTimeout(timer); timer = null; }
      draw();
      refresh();
      return terrainDebug.snapshot().cameraEpoch;
    });
  }

  async function measure(name, action, requireDecoder = true) {
    activeScenario = name;
    const transportStart = transport.length, started = Date.now();
    await action();
    const immediate = await page.evaluate(() => terrainDebug.snapshot());
    const forcedEpoch = await forceFinalRefresh();
    const confirmed = await settled(forcedEpoch, requireDecoder);
    const responses = await responsesSince(transportStart);
    scenarios.push({ name, wall_ms: Date.now() - started, immediate,
      forced_camera_epoch: forcedEpoch, settled: confirmed,
      transport_cache_counts: cacheCounts(responses), transport_response_count: responses.length });
  }

  async function pan(name, direction, seconds) {
    activeScenario = `${name}-baseline`;
    const baselineEpoch = await forceFinalRefresh();
    const baseline = await settled(baselineEpoch, true);
    assert.equal(baseline.camera.lod, 0);
    assert.equal(baseline.visible.pending, 0);
    assert(nativeRendered(baseline));
    activeScenario = name;
    const transportStart = transport.length;
    const sample = await page.evaluate(async ({ direction, seconds }) => {
      const intervals = [], drawFrameCpuMs = [], cameraUpdateCpuMs = [], coverage = [];
      const originalDrawFrame = drawFrame;
      const startCx = cx, startCy = cy, velocityMPerMs = direction * 6;
      let first = null, previous = null, lastCoverage = -Infinity;
      drawFrame = function (...args) {
        const began = performance.now();
        try { return originalDrawFrame.apply(this, args); }
        finally { drawFrameCpuMs.push(performance.now() - began); }
      };
      try {
        await new Promise(resolve => {
          function tick(now) {
            if (first === null) { first = now; previous = now; }
            else intervals.push(now - previous);
            const deltaMs = now - previous;
            previous = now;
            const began = performance.now();
            cx += velocityMPerMs * deltaMs; // 6,000 metres per second.
            draw();
            schedule();
            cameraUpdateCpuMs.push(performance.now() - began);
            if (now - lastCoverage >= 100 || now - first >= seconds * 1000) {
              const snapshot = terrainDebug.snapshot();
              coverage.push({ t_ms: now - first, pending: snapshot.visible.pending,
                ready: snapshot.visible.ready, wanted: snapshot.visible.wanted,
                lod: snapshot.visible.lod, camera_epoch: snapshot.cameraEpoch,
                rendered_lods: snapshot.rendered.lods, rendered_sources: snapshot.rendered.sources,
                rendered_patches: snapshot.rendered.patches,
                fallback_lod_patch_share: visiblePlan.length
                  ? visiblePlan.filter(entry => entry.fallback).length / visiblePlan.length : null });
              lastCoverage = now;
            }
            if (now - first < seconds * 1000) requestAnimationFrame(tick);
            else resolve();
          }
          requestAnimationFrame(tick);
        });
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        return { intervals, drawFrameCpuMs, cameraUpdateCpuMs, coverage,
          elapsedMs: previous - first, startCx, endCx: cx, startCy, endCy: cy,
          snapshot: terrainDebug.snapshot() };
      } finally {
        drawFrame = originalDrawFrame;
      }
    }, { direction, seconds });

    // A final refresh must observe the last camera position before settling.
    const settleStart = Date.now();
    const forcedEpoch = await forceFinalRefresh();
    const confirmed = await settled(forcedEpoch, true);
    const period = quantile(sample.intervals, 0.25);
    const responses = await responsesSince(transportStart);
    scenarios.push({
      name, driver: 'requestAnimationFrame continuous camera pan', target_m_per_s: direction * 6000,
      elapsed_ms: sample.elapsedMs, actual_displacement_m: sample.endCx - sample.startCx,
      cross_axis_displacement_m: sample.endCy - sample.startCy,
      frames: sample.intervals.length,
      headless_compositor_cadence_estimate_ms: period,
      headless_compositor_cadence_estimate_hz: period ? 1000 / period : null,
      frame_interval_median_ms: quantile(sample.intervals, 0.5),
      frame_interval_p95_ms: quantile(sample.intervals, 0.95),
      frame_interval_over_1_5_period_share: period
        ? sample.intervals.filter(value => value > 1.5 * period).length / sample.intervals.length : null,
      draw_frame_cpu_count: sample.drawFrameCpuMs.length,
      draw_frame_cpu_median_ms: quantile(sample.drawFrameCpuMs, 0.5),
      draw_frame_cpu_p95_ms: quantile(sample.drawFrameCpuMs, 0.95),
      draw_frame_cpu_max_ms: sample.drawFrameCpuMs.length ? Math.max(...sample.drawFrameCpuMs) : null,
      camera_update_cpu_p95_ms: quantile(sample.cameraUpdateCpuMs, 0.95),
      coverage_samples: sample.coverage,
      coverage_pending_sample_share: sample.coverage.length
        ? sample.coverage.filter(point => point.pending > 0).length / sample.coverage.length : null,
      coverage_pending_max: sample.coverage.length ? Math.max(...sample.coverage.map(point => point.pending)) : null,
      coverage_fallback_lod_patch_share_max: sample.coverage.length
        ? Math.max(...sample.coverage.map(point => point.fallback_lod_patch_share ?? 0)) : null,
      coverage_settle_after_pan_ms: Date.now() - settleStart,
      baseline, immediate: sample.snapshot, forced_camera_epoch: forcedEpoch, settled: confirmed,
      transport_cache_counts: cacheCounts(responses), transport_response_count: responses.length,
    });
  }

  try {
    const preflightStatus = await (await page.request.get('http://127.0.0.1:8765/api/status')).json();
    if (preflightStatus.scheduler?.sessions > 0) {
      console.warn(`Benchmark has ${preflightStatus.scheduler.sessions} existing server sessions; no exclusive-run claim.`);
    }
    const started = Date.now();
    await page.goto('http://127.0.0.1:8765/?seed=42&profile=natural&coarse_prepare=0');
    await page.waitForFunction(() => terrainDebug?.snapshot().overview && terrainDebug.snapshot().rendererReady,
      null, { timeout: 120000 });
    scenarios.push({ name: 'overview-and-renderer-ready', wall_ms: Date.now() - started,
      snapshot: await page.evaluate(() => terrainDebug.snapshot()) });
    await settled();
    scenarios.push({ name: 'initial-world-settled', wall_ms: Date.now() - started,
      snapshot: await page.evaluate(() => terrainDebug.snapshot()) });
    await page.locator('#runtimePanel').evaluate(el=>el.open=true);await page.locator('#prefetch').uncheck();
    const rendererAdapterInfo = await page.evaluate(() => {
      const info = renderer?.adapter?.info;
      return info ? { vendor: info.vendor, architecture: info.architecture,
        device: info.device, description: info.description } : null;
    });
    assert(rendererAdapterInfo, 'Active renderer adapter info missing');

    // Transport headers, not this scenario name, determine cache warmth.
    await measure('native-region-transport', () => page.evaluate(() => {
      cx = -103680; cy = 80640; mpp = 30; draw(); schedule();
    }));
    const native = await page.evaluate(() => terrainDebug.snapshot());
    assert.equal(native.camera.mpp, 30);
    assert.equal(native.camera.lod, 0);
    assert.equal(native.backend, 'webgpu');
    assert(nativeRendered(native));
    scenarios.push({ name: 'native-reference-region', snapshot: native });
    const nativeScreenshot = destination.replace(/\.json$/, '-native.png');
    await page.screenshot({ path: nativeScreenshot });

    await pan('pan-horizontal-6000-m-per-second', 1, 2);
    await measure('zoom-world-and-native', async () => {
      await page.locator('#fit').click();
      await page.locator('#native').click();
    });
    await pan('pan-west-from-world-centre', -1, 2);
    const finalScreenshot = destination.replace(/\.json$/, '.png');
    await page.screenshot({ path: finalScreenshot });

    const status = await (await page.request.get('http://127.0.0.1:8765/api/status')).json();
    const implementationSha256 = Object.fromEntries(
      ['index.html', 'terrain_lod.js', 'terrain_renderer.js', 'tools/benchmarks/benchmark_browser.cjs']
        .map(name => [name, crypto.createHash('sha256').update(readRepositoryFile(name)).digest('hex')]));
    const screenshotSha256 = path => crypto.createHash('sha256').update(readRepositoryFile(path)).digest('hex');
    const report = { scenarios, errors, transport, status, preflight_status: preflightStatus,
      renderer_adapter_info: rendererAdapterInfo,
      screenshots: { native: { path: nativeScreenshot, sha256: screenshotSha256(nativeScreenshot) },
        final: { path: finalScreenshot, sha256: screenshotSha256(finalScreenshot) } },
      chrome_version: browser.version(),
      implementation_sha256: implementationSha256,
      notes: [
        'One headless Chrome desktop run with a warm local model/cache state; first-ever downloads and sustained cold throughput are unmeasured.',
        'The pan changes cx on every animation frame at 6000 m/s, calls draw() and schedule(), and measures synchronous drawFrame CPU work.',
        'Frame intervals estimate headless compositor cadence and OS scheduling, not physical-screen presentation. The drawFrame CPU wrapper is a main-thread lower bound; it excludes uploads, layout, snapshot cost, GPU execution and presentation latency. The end-to-end <16.7 ms gate remains unmeasured.',
        'Pending target-LOD tiles do not imply a blank screen; rendered LODs, sources, and fallback patch share are sampled separately. Post-pan settlement is a separate measure.',
        'Transport headers determine cache warmth. Background preparation was disabled in this client; other server clients may have been active, as shown by preflight and final server status.',
      ] };
    fs.writeFileSync(destination, JSON.stringify(report, null, 2));
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ report: destination,
      pans: scenarios.filter(item => item.driver).map(item => ({ name: item.name,
        frames: item.frames, headless_compositor_cadence_estimate_ms: item.headless_compositor_cadence_estimate_ms,
        frame_interval_over_1_5_period_share: item.frame_interval_over_1_5_period_share,
        draw_frame_cpu_p95_ms: item.draw_frame_cpu_p95_ms,
        coverage_settle_after_pan_ms: item.coverage_settle_after_pan_ms,
        transport_cache_counts: item.transport_cache_counts })) }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
