/** Compare browser decoding with hashes produced by the Python reference. */
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {join} from 'node:path';
import {decodePackedTensor} from './packed_base.mjs';

const [directory, referencePath] = process.argv.slice(2);
if (!directory || !referencePath) throw new Error('Usage: node distill/check_packed_base.mjs DIRECTORY REFERENCE.json');
const manifest = JSON.parse(await readFile(join(directory, 'manifest.json'), 'utf8'));
const reference = JSON.parse(await readFile(referencePath, 'utf8'));
const buffer = await readFile(join(directory, manifest.file));
if (createHash('sha256').update(buffer).digest('hex') !== manifest.sha256) throw new Error('Weight checksum changed');
let count = 0;
for (const [name, expected] of Object.entries(reference.tensors)) {
    const values = decodePackedTensor(manifest, buffer, name);
    const digest = createHash('sha256').update(new Uint8Array(values.buffer)).digest('hex');
    if (digest !== expected) throw new Error(`Python/browser decoding differs: ${name}`);
    count++;
}
if (count !== Object.keys(manifest.tensors).length) throw new Error('Incomplete tensor reference');
console.log(JSON.stringify({status: 'complete', tensors: count, allFloat32DecodedBytesExact: true}));
