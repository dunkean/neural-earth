/** Browser height transport matching the Python tile header/layout contract.
 * Source fields: height, mean temperature, temperature std, annual rain, rain CV.
 * Climate planes: sea-level temperature, temperature std, rain, rain CV, lapse. */
export function physicalTile(heights, fields, width = 304, halo = 24, climate = false) {
  const count = width * width, n = 33;
  if (heights.length !== count || fields.length !== count * 5) throw Error('Invalid physical tile fields');
  const headers = { 'Content-Type': 'application/octet-stream', 'X-Terrain-Width': String(width), 'X-Terrain-Halo': String(halo) };
  if (!climate) return { body: heights.buffer.slice(heights.byteOffset, heights.byteOffset + heights.byteLength), headers };
  const values = new Float32Array(count + n * n * 5); values.set(heights);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const sx = halo + Math.round(x / (n - 1) * (width - 2 * halo - 1)), sy = halo + Math.round(y / (n - 1) * (width - 2 * halo - 1));
    const index = sy * width + sx;
    for (let layer = 0; layer < 4; layer++) values[count + layer * n * n + y * n + x] = fields[(layer + 1) * count + index];
    values[count + 4 * n * n + y * n + x] = -.006;
  }
  Object.assign(headers, { 'X-Terrain-Climate-Width': String(n), 'X-Terrain-Climate-Height': String(n), 'X-Terrain-Climate-Layers': '5' });
  return { body: values.buffer, headers };
}
