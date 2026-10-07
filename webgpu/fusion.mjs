const f32 = Math.fround;

export function linearWeightWindow(size) {
  const weights = new Float32Array(size * size);
  const mid = (size - 1) / 2;
  for (let y = 0; y < size; y++) {
    const wy = 1 - (1 - 1e-3) * Math.min(Math.abs(y - mid) / mid, 1);
    for (let x = 0; x < size; x++) {
      const wx = 1 - (1 - 1e-3) * Math.min(Math.abs(x - mid) / mid, 1);
      weights[y * size + x] = f32(wy * wx);
    }
  }
  return weights;
}

// An InfiniteTensor query receives every overlapping output tile, including
// tiles that are not needed merely to avoid zero-weight pixels at the edges.
export function assertFullContributors(tiles, tileSize, stride, y0, x0, height, width) {
  const available = new Set(tiles.map(tile => `${tile.y},${tile.x}`));
  if (available.size !== tiles.length) throw new Error('Duplicate contributing tile');
  for (let iy = Math.floor((y0 - tileSize) / stride) + 1; iy <= Math.floor((y0 + height - 1) / stride); iy++) {
    for (let ix = Math.floor((x0 - tileSize) / stride) + 1; ix <= Math.floor((x0 + width - 1) / stride); ix++) {
      const key = `${iy * stride},${ix * stride}`;
      if (!available.has(key)) throw new Error(`Missing contributing tile ${key}`);
    }
  }
}

// Tiles carry the upstream weighted channels followed by one weight channel.
// All intersecting tiles must be supplied before reading a completed region.
export function fuseWeightedTiles(tiles, channels, tileSize, y0, x0, height, width) {
  const plane = height * width;
  const sourcePlane = tileSize * tileSize;
  const sums = new Float32Array(channels * plane);
  const weights = new Float32Array(plane);
  for (const tile of tiles) {
    if (tile.data.length !== (channels + 1) * sourcePlane) throw new Error('Fusion tile shape mismatch');
    const iy0 = Math.max(y0, tile.y);
    const iy1 = Math.min(y0 + height, tile.y + tileSize);
    const ix0 = Math.max(x0, tile.x);
    const ix1 = Math.min(x0 + width, tile.x + tileSize);
    for (let y = iy0; y < iy1; y++) {
      for (let x = ix0; x < ix1; x++) {
        const src = (y - tile.y) * tileSize + x - tile.x;
        const dst = (y - y0) * width + x - x0;
        weights[dst] = f32(weights[dst] + tile.data[channels * sourcePlane + src]);
        for (let c = 0; c < channels; c++) sums[c * plane + dst] = f32(sums[c * plane + dst] + tile.data[c * sourcePlane + src]);
      }
    }
  }
  if (weights.some(value => value <= 0)) throw new Error('Incomplete fusion: missing contributing tile');
  const result = new Float32Array(channels * plane);
  for (let c = 0; c < channels; c++) {
    for (let p = 0; p < plane; p++) result[c * plane + p] = f32(sums[c * plane + p] / weights[p]);
  }
  return result;
}
