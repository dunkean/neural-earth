// Expand test paths in Node so npm scripts work in both Windows and POSIX shells.
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const root = path.resolve(__dirname, '..');
const files = process.argv.slice(2).flatMap(directory =>
  fs.readdirSync(path.join(root, directory)).filter(name =>
    /^(test_.*\.cjs|.*\.test\.mjs)$/.test(name)).sort().map(name => path.join(root, directory, name)));
if (!files.length) throw new Error('No tests found in the requested directories');
const result = spawnSync(process.execPath, ['--test', ...files], {cwd: root, stdio: 'inherit'});
if (result.error) throw result.error;
process.exitCode = result.status ?? 1;
