// Port of terrain_diffusion.inference.portable_rng plus world_pipeline tile
// mapping. BigInt preserves seed/coordinate semantics;the hot loop uses exact
// pairs of32-bit words to avoid millions of BigInt allocations per terrain tile.
const MASK64 = (1n << 64n) - 1n;
const MULT = 6364136223846793005n;
const INC = 1442695040888963407n;
const TILE_MULT = 0x9e3779b9n;
const INV32 = 1 / 4294967296;

export function tileSeed(seed, tileY, tileX) {
  let h = (BigInt(seed) & MASK64) * TILE_MULT & MASK64;
  h = (h + (BigInt(tileY) & 0xffffffffn)) & MASK64;
  h = (h * TILE_MULT + (BigInt(tileX) & 0xffffffffn)) & MASK64;
  return h;
}

function unsignedProductHigh(a, b) {
  const al = a & 65535, ah = a >>> 16;
  const bl = b & 65535, bh = b >>> 16;
  const low = al * bl, middleA = ah * bl, middleB = al * bh;
  const carry = (low >>> 16) + (middleA & 65535) + (middleB & 65535);
  return (ah * bh + (middleA >>> 16) + (middleB >>> 16) + (carry >>> 16)) >>> 0;
}

export function standardNormal(seed, count) {
  const out = new Float32Array(count);
  const state = BigInt(seed) & MASK64;
  let lo = Number(state & 0xffffffffn), hi = Number(state >> 32n);
  const multLo = Number(MULT & 0xffffffffn), multHi = Number(MULT >> 32n);
  const incLo = Number(INC & 0xffffffffn), incHi = Number(INC >> 32n);
  function next32() {
    const oldLo = lo;
    const lowSum = (Math.imul(oldLo, multLo) >>> 0) + incLo;
    hi = (Math.imul(hi, multLo) + Math.imul(oldLo, multHi) +
          unsignedProductHigh(oldLo, multLo) + incHi + (lowSum >= 4294967296 ? 1 : 0)) >>> 0;
    lo = lowSum >>> 0;
    const xorLo = ((lo >>> 18) | (hi << 14)) ^ lo;
    const xorHi = (hi >>> 18) ^ hi;
    const x = (xorLo >>> 27) | (xorHi << 5);
    const rot = hi >>> 27;
    return ((x >>> rot) | (x << ((32 - rot) & 31))) >>> 0;
  }
  let i = 0;
  while (i < count) {
    const u1 = next32(), u2 = next32();
    const v1 = 2 * (u1 + 1) * INV32 - 1;
    const v2 = 2 * (u2 + 1) * INV32 - 1;
    const s = v1 * v1 + v2 * v2;
    if (s <= 0 || s >= 1) continue;
    const factor = Math.sqrt(-2 * Math.log(s) / s);
    out[i++] = Math.fround(v1 * factor);
    if (i < count) out[i++] = Math.fround(v2 * factor);
  }
  return out;
}

export function gaussianNoisePatch(seed, y0, x0, height, width, channels = 1, tileHeight = 256, tileWidth = 256) {
  const out = new Float32Array(channels * height * width);
  const yStart = Math.floor(y0 / tileHeight);
  const yEnd = Math.floor((y0 + height - 1) / tileHeight);
  const xStart = Math.floor(x0 / tileWidth);
  const xEnd = Math.floor((x0 + width - 1) / tileWidth);
  for (let ty = yStart; ty <= yEnd; ty++) {
    for (let tx = xStart; tx <= xEnd; tx++) {
      const tile = standardNormal(tileSeed(seed, ty, tx), channels * tileHeight * tileWidth);
      const oy0 = Math.max(y0, ty * tileHeight);
      const oy1 = Math.min(y0 + height, (ty + 1) * tileHeight);
      const ox0 = Math.max(x0, tx * tileWidth);
      const ox1 = Math.min(x0 + width, (tx + 1) * tileWidth);
      for (let c = 0; c < channels; c++) {
        for (let y = oy0; y < oy1; y++) {
          const src = c * tileHeight * tileWidth + (y - ty * tileHeight) * tileWidth + (ox0 - tx * tileWidth);
          const dst = c * height * width + (y - y0) * width + (ox0 - x0);
          out.set(tile.subarray(src, src + ox1 - ox0), dst);
        }
      }
    }
  }
  return out;
}
