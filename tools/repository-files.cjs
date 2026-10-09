const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const pythonPaths = [path.join(root, 'backend'), root, process.env.PYTHONPATH].filter(Boolean);
process.env.PYTHONPATH = pythonPaths.join(path.delimiter);
function repositoryFile(file) {
  if (typeof file !== 'string') return file;
  const absolute = path.resolve(file);
  if (fs.existsSync(absolute)) return file;
  const base = path.dirname(absolute);
  const name = path.basename(absolute);
  if (name === 'index.html' || /^terrain_.*\.(js|json)$/.test(name)) return fs.existsSync(path.join(base, 'web')) ? path.join(base, 'web', name) : file;
  if (/^terrain_.*\.py$/.test(name)) return fs.existsSync(path.join(base, 'backend')) ? path.join(base, 'backend', name) : file;
  return file;
}
function readRepositoryFile(file, ...options) { return fs.readFileSync(repositoryFile(file), ...options); }
module.exports = { repositoryFile, readRepositoryFile };
