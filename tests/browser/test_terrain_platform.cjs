const assert = require('node:assert/strict');
const path = require('node:path');
const os = require('node:os');
const {runtimeRoot, browserExecutable, browserArguments, pythonExecutable} = require('../../tools/platform.cjs');

assert.equal(runtimeRoot({}, 'win32'), 'E:/TerrainDiffusionRuntime');
assert.equal(runtimeRoot({}, 'linux'), path.join(os.homedir(), '.cache', 'neural-earth'));
assert.equal(runtimeRoot({XDG_CACHE_HOME: '/tmp/neural-earth-cache'}, 'linux'), path.resolve('/tmp/neural-earth-cache/neural-earth'));
assert.equal(runtimeRoot({TERRAIN_RUNTIME_ROOT: './storage'}, 'linux'), path.resolve('storage'));
assert.equal(runtimeRoot({TERRAIN_RUNTIME_ROOT: '~/storage'}, 'linux'), path.join(os.homedir(), 'storage'));
assert.equal(browserExecutable({CHROME_PATH: '/custom/chrome'}, 'linux'), '/custom/chrome');
assert.equal(browserExecutable({CHROME_PATH: 'C:/custom/chrome.exe'}, 'win32'), 'C:/custom/chrome.exe');
assert.equal(pythonExecutable({TERRAIN_PYTHON: '/custom/python'}, 'linux'), '/custom/python');
assert.deepEqual(browserArguments({}), []);
assert.deepEqual(browserArguments({TERRAIN_BROWSER_ARGS: '["--use-angle=swiftshader"]'}), ['--use-angle=swiftshader']);
assert.throws(() => browserArguments({TERRAIN_BROWSER_ARGS: '{}'}), /JSON array/);
assert.throws(() => browserArguments({TERRAIN_BROWSER_ARGS: '[false]'}), /JSON array/);
console.log('Cross-platform storage, Python and browser overrides passed.');
