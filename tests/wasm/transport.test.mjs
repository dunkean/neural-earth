import {test} from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
test('Default transport preserves normal fetch and releases the correct Python session',async()=>{
  const calls=[],context={window:{},fetch:(...args)=>calls.push(args),navigator:{sendBeacon:(...args)=>calls.push(args)},Blob};
  vm.runInNewContext(await readFile(new URL('../../web/terrain_transport.js',import.meta.url),'utf8'),context);
  context.window.TerrainTransport.fetch('/api/world',{signal:'signal'});
  context.window.TerrainTransport.release('/api/view/release',{session:'python-session'});
  assert.equal(calls[0][0],'/api/world');assert.equal(calls[0][1].signal,'signal');
  assert.deepEqual(JSON.parse(await calls[1][1].text()),{session:'python-session'});
});
