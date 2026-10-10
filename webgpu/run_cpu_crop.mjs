// Diagnostic only: run the exact browser crop arithmetic with native ORT CPU.
// Production must never select this backend or treat it as a WebGPU pass.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';
import { runCrop } from './crop.mjs';
import provenance from './provenance.cjs';
import platform from '../tools/platform.cjs';

const { hashFile, sourceHashes, verifyFixtureManifest } = provenance;

const root = process.env.TERRAIN_WEBGPU_MODELS || platform.runtimePath('webgpu-models');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'crop64-coast-manifest.json')));
const exportReport = JSON.parse(fs.readFileSync(path.join(root, 'export-report.json')));
const npmLock = JSON.parse(fs.readFileSync(new URL('./package-lock.json', import.meta.url)));
const sha256 = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');

function loadArray(spec) {
  const bytes = fs.readFileSync(path.join(root, spec.file));
  const raw = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const values = new Float32Array(raw);
  if (values.length !== spec.shape.reduce((a, b) => a * b, 1)) throw new Error(`Shape mismatch: ${spec.file}`);
  return values;
}

const diagnosticBackend = {
  manifest,
  loadArray,
  async createSession(name) {
    const stem = name === 'base_model' ? 'base_model_external' : name;
    return ort.InferenceSession.create(path.join(root, `${stem}.onnx`), {
      executionProviders: ['cpu'], graphOptimizationLevel: 'all', intraOpNumThreads: 4,
    });
  },
  makeRunner(session, shapes, outputShape) {
    return {
      async run(arrays) {
        const feeds = {};
        for (const name of session.inputNames) {
          const shape = shapes[name];
          if (!shape || arrays[name]?.length !== shape.reduce((a, b) => a * b, 1)) throw new Error(`Feed mismatch ${name}`);
          feeds[name] = new ort.Tensor('float32', arrays[name], shape);
        }
        const outputs = await session.run(feeds);
        const data = outputs[session.outputNames[0]].data;
        if (data.length !== outputShape.reduce((a, b) => a * b, 1)) throw new Error('Output shape mismatch');
        return data;
      },
      dispose() {},
    };
  },
};

const started = Date.now();
try {
  const report = await runCrop(null, message => {
    if (message.startsWith('Loading ') || /: \d+\/\d+$/.test(message)) console.log(message);
  }, false, diagnosticBackend);
  report.wallMs = Date.now() - started;
  report.identity = {
    checkpointRevision: manifest.checkpoint_revision,
    manifestSha256: sha256(path.join(root, 'crop64-coast-manifest.json')),
    cropSourceSha256: sha256(new URL('./crop.mjs', import.meta.url)),
    ortNodeVersion: npmLock.packages['node_modules/onnxruntime-node'].version,
    diagnosticHarnessSha256: sha256(new URL('./run_cpu_crop.mjs', import.meta.url)),
    provenanceHelperSha256: sha256(new URL('./provenance.cjs', import.meta.url)),
    sourceModules: await sourceHashes(path.dirname(fileURLToPath(import.meta.url))),
    modelFiles: [],
    fixtureManifestVerification: await verifyFixtureManifest(root, manifest),
  };
  for (const name of ['coarse_model', 'base_model_external', 'base_model_external.data', 'decoder_model']) {
    const file = name.endsWith('.data') ? name : `${name}.onnx`;
    const actual = await hashFile(path.join(root, file));
    const known = name === 'base_model_external.data'
      ? exportReport.models.base_model_external.external_files[0].sha256
      : exportReport.models[name].sha256;
    if (actual.sha256 !== known) throw new Error(`Consumed CPU model file differs from export report: ${file}`);
    report.identity.modelFiles.push({ file, ...actual });
  }
  fs.writeFileSync(path.join(root, 'cpu-crop-diagnostic.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  if (report.numericalStatus !== 'PASS') process.exitCode = 1;
} catch (error) {
  const report = { numericalStatus: 'ERROR', l1Status: 'NOT_APPLICABLE', error: String(error.stack || error), wallMs: Date.now() - started };
  fs.writeFileSync(path.join(root, 'cpu-crop-diagnostic.json'), JSON.stringify(report, null, 2));
  console.error(report.error);
  process.exitCode = 1;
}
