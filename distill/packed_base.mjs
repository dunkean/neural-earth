/** Reference reader for native-weight-quant-scalar-packed, usable in a browser.
 * Returns effective BF16 values expanded to Float32Array. NF4 kernels already
 * include MP normalization and fan-in; callers must not normalize them again.
 */
const scratch = new DataView(new ArrayBuffer(4));

function bf16ToFloat(bits) {
    scratch.setUint32(0, bits * 65536, true);
    return scratch.getFloat32(0, true);
}

function roundBf16(value) {
    scratch.setFloat32(0, value, true);
    const bits = scratch.getUint32(0, true);
    return bf16ToFloat(((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) & 0xffff);
}

export function decodePackedTensor(manifest, buffer, name) {
    if (manifest.architecture !== 'native-weight-quant-scalar-packed' ||
        manifest.scalar_precision !== 'anchored-fp32-v1' || manifest.endian !== 'little') {
        throw new Error('Unsupported packed base format');
    }
    const view = buffer instanceof ArrayBuffer ? new DataView(buffer) :
        new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    if (view.byteLength !== manifest.bytes) throw new Error('Truncated weights');
    const row = manifest.tensors[name];
    if (!row) throw new Error(`Unknown tensor: ${name}`);
    const count = row.shape.reduce((a, b) => a * b, 1);
    if (!Number.isSafeInteger(count) || count < 1 || row.offset < 0 ||
        row.offset + row.bytes > view.byteLength) throw new Error('Invalid tensor range');
    const values = new Float32Array(count);
    if (row.dtype === 'bfloat16') {
        if (row.bytes !== count * 2) throw new Error('Invalid BF16 length');
        for (let i = 0; i < count; i++) values[i] = bf16ToFloat(view.getUint16(row.offset + 2 * i, true));
    } else if (row.dtype === 'float32') {
        if (name !== 'net.out_gain' || row.bytes !== 4 || count !== 1) {
            throw new Error('Unexpected FP32 tensor');
        }
        values[0] = view.getFloat32(row.offset, true);
    } else if (row.dtype === 'nf4' || row.dtype === 'int8') {
        const codeCount = row.bytes * (row.dtype === 'nf4' ? 2 : 1);
        if (codeCount !== row.scale_count * manifest.block || codeCount < count ||
            row.scale_offset < 0 || row.scale_offset + 4 * row.scale_count > view.byteLength) {
            throw new Error('Invalid quantization blocks');
        }
        const table = manifest.codebook;
        if (row.dtype === 'nf4' && (!table || table.length !== 16)) throw new Error('Invalid NF4 codebook');
        for (let start = 0; start < count; start += manifest.block) {
            const scale = view.getFloat32(row.scale_offset + 4 * (start / manifest.block), true);
            const end = Math.min(start + manifest.block, count);
            for (let i = start; i < end; i++) {
                const byte = view.getUint8(row.offset + (row.dtype === 'nf4' ? Math.floor(i / 2) : i));
                const unit = row.dtype === 'nf4' ? Math.fround(table[(byte >>> ((i % 2) * 4)) & 15]) :
                    Math.fround((byte - 128) / 127);
                values[i] = roundBf16(Math.fround(unit * scale));
            }
        }
    } else {
        throw new Error(`Unsupported tensor dtype: ${row.dtype}`);
    }
    return values;
}
