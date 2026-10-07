const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');

const SOURCE_FILES = [
  'probe.html', 'probe.js', 'crop.mjs', 'io.mjs', 'noise.mjs',
  'scheduler.mjs', 'fusion.mjs', 'laplacian.mjs', 'climate.mjs',
];

function hashFile(filename) {
  return new Promise((resolve, reject) => {
    const digest = createHash('sha256');
    const stream = fs.createReadStream(filename, { highWaterMark: 4 * 1024 * 1024 });
    stream.on('data', chunk => digest.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve({ bytes: fs.statSync(filename).size, sha256: digest.digest('hex') }));
  });
}

function collectFixtureSpecs(value, result = new Map()) {
  if (Array.isArray(value)) {
    for (const item of value) collectFixtureSpecs(item, result);
  } else if (value && typeof value === 'object') {
    if (typeof value.file === 'string' && Array.isArray(value.shape) && typeof value.sha256 === 'string') {
      if (path.basename(value.file) !== value.file || !value.file.endsWith('.bin')) throw new Error(`Unsafe fixture file ${value.file}`);
      const previous = result.get(value.file);
      if (previous && JSON.stringify(previous) !== JSON.stringify(value)) throw new Error(`Conflicting fixture spec ${value.file}`);
      result.set(value.file, value);
    }
    for (const nested of Object.values(value)) collectFixtureSpecs(nested, result);
  }
  return result;
}

async function verifyFixtureManifest(modelRoot, manifest) {
  const specs = collectFixtureSpecs(manifest);
  const verified = [];
  for (const [name, spec] of [...specs.entries()].sort(([a], [b]) => a.localeCompare(b))) {
    const actual = await hashFile(path.join(modelRoot, name));
    const expectedBytes = spec.shape.reduce((a, b) => a * b, 1) * 4;
    if (actual.bytes !== expectedBytes || actual.sha256 !== spec.sha256) throw new Error(`Fixture size/hash mismatch ${name}`);
    verified.push({ file: name, ...actual });
  }
  return { verifiedCount: verified.length, files: verified };
}

async function sourceHashes(sourceRoot) {
  return Object.fromEntries(await Promise.all(SOURCE_FILES.map(async name => [name, await hashFile(path.join(sourceRoot, name))])));
}

async function requestedHashes(requestPaths, sourceRoot, modelRoot) {
  const records = [];
  for (const pathname of [...requestPaths].sort()) {
    let root, relative;
    if (pathname.startsWith('/webgpu-models/')) {
      root = modelRoot; relative = decodeURIComponent(pathname.slice('/webgpu-models/'.length));
    } else if (pathname.startsWith('/webgpu/')) {
      root = sourceRoot; relative = decodeURIComponent(pathname.slice('/webgpu/'.length));
    } else continue;
    const full = path.resolve(root, relative);
    if (!full.startsWith(path.resolve(root) + path.sep) || !fs.existsSync(full) || !fs.statSync(full).isFile()) throw new Error(`Requested path escaped or vanished: ${pathname}`);
    records.push({ pathname, ...await hashFile(full) });
  }
  return records;
}

module.exports = { hashFile, verifyFixtureManifest, sourceHashes, requestedHashes };
