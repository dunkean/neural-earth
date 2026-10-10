// Shared storage, browser and Python discovery for tests and standalone tools.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const expandHome = value => value.replace(/^~(?=[/\\]|$)/, os.homedir());

function runtimeRoot(env = process.env, platform = process.platform) {
  if (env.TERRAIN_RUNTIME_ROOT) return path.resolve(expandHome(env.TERRAIN_RUNTIME_ROOT));
  return platform === 'win32' ? 'E:/TerrainDiffusionRuntime'
    : path.resolve(expandHome(env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache')), 'neural-earth');
}
function runtimePath(relative = '') { return path.join(runtimeRoot(), relative); }
function browserExecutable(env = process.env, platform = process.platform) {
  if (env.CHROME_PATH) return env.CHROME_PATH;
  const candidates = platform === 'win32' ? ['C:/Program Files/Google/Chrome/Application/chrome.exe']
    : ['/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser'];
  return candidates.find(file => fs.existsSync(file)); // undefined selects Playwright's installed Chromium.
}
function loadPlaywright() {
  const resolved = process.env.PLAYWRIGHT_PATH || require.resolve('playwright',
    {paths: [root, path.join(root, 'webgpu'), runtimePath('ui-test')]});
  const playwright = require(resolved);
  const args = browserArguments();
  if (!args.length) return playwright;
  // Flags are explicit so benchmarks never silently switch to software rendering.
  const chromium = new Proxy(playwright.chromium, {get(target, property) {
    if (property === 'launch') return (options = {}) => target.launch({...options, args: [...(options.args || []), ...args]});
    const value = Reflect.get(target, property);
    return typeof value === 'function' ? value.bind(target) : value;
  }});
  return {...playwright, chromium};
}
function browserArguments(env = process.env) {
  const args = JSON.parse(env.TERRAIN_BROWSER_ARGS || '[]');
  if (!Array.isArray(args) || args.some(arg => typeof arg !== 'string')) {
    throw new Error('TERRAIN_BROWSER_ARGS must be a JSON array of browser flags');
  }
  return args;
}
function pythonExecutable(env = process.env, platform = process.platform) {
  if (env.TERRAIN_PYTHON) return env.TERRAIN_PYTHON;
  const executable = path.join(root, '.venv', platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  return fs.existsSync(executable) ? executable : (platform === 'win32' ? 'python' : 'python3');
}
module.exports = {runtimeRoot, runtimePath, browserExecutable, browserArguments, loadPlaywright, pythonExecutable};
