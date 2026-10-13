// Consume the same immutable release and integrity pins as the Python runtime.
import {readFile,writeFile,mkdir,access} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {homedir} from 'node:os';
import {join,resolve} from 'node:path';
const exists=path=>access(path).then(()=>true,()=>false);
export async function preparePackedBase(root, output) {
  const specification=await readFile(join(root,'distill/compact_release.py'),'utf8');
  const release=specification.match(/^RELEASE = '([^']+)'/m)?.[1];
  const hashes=Object.fromEntries([...specification.matchAll(/^    '([^']+)': '([0-9a-f]{64})'/gm)].map(m=>[m[1],m[2]]));
  if(!release||!hashes['manifest.json']||!hashes['weights.bin'])throw Error('Compact base release contract changed');
  const source=resolve(process.env.NEURAL_EARTH_WASM_BASE||join(process.env.DISTILL_ROOT||join(homedir(),'data/distill'),'releases',release));
  await mkdir(source,{recursive:true});
  const files={};
  for(const name of ['manifest.json','weights.bin']) {
    const path=join(source,name);
    let bytes;
    if(await exists(path))bytes=await readFile(path);
    else {
      console.log(`Installing shared distilled base ${release}/${name}…`);
      const response=await fetch(`https://github.com/dunkean/neural-earth/releases/download/${release}/${name}`);
      if(!response.ok)throw Error(`Cannot download distilled base ${name}: ${response.status}`);
      bytes=Buffer.from(await response.arrayBuffer());
    }
    if(createHash('sha256').update(bytes).digest('hex')!==hashes[name])throw Error(`Distilled base integrity mismatch: ${path}`);
    if(!await exists(path))await writeFile(path,bytes);
    files[name]=bytes;
  }
  const manifest=JSON.parse(files['manifest.json']);
  if(manifest.architecture!=='native-weight-quant-scalar-packed'||manifest.scalar_precision!=='anchored-fp32-v1'||manifest.bytes!==files['weights.bin'].length)throw Error('Unsupported distilled base release');
  // Keep the original packed bytes verbatim. Rust performs decoding and MP
  // preparation in the browser; no Python or teacher checkpoint is involved.
  const header=Buffer.from(JSON.stringify(manifest));
  const length=Buffer.alloc(8);length.writeBigUInt64LE(BigInt(header.length));
  await writeFile(output,Buffer.concat([length,header,files['weights.bin']]));
  return {selection:'nf4_b75',release,sourceSha256:hashes['weights.bin'],format:'nf4-anchored-fp32',solverSteps:2};
}
