// Point location on a sphere mesh: which Delaunay triangle a point falls in, and its barycentric weights there.
// Used to carry values from a coarse mesh (the 40K-cell sim mesh) to any finer one — the terrain mesh, the map
// raster's pixels — as a smooth, piecewise-linear field instead of blocky nearest-cell patches.
// Relies on the mesh's adjacency listing each cell's neighbours in ring order (js/sphere-mesh.js builds it by
// circulating half-edges), so consecutive neighbours form a triangle with the cell.

/**
 * Locate the unit direction (x, y, z) on `mesh`. Fills `cells` (3 cell ids) and `weights` (3 barycentric weights,
 * ≥ 0, summing to 1). Starts from the nearest cell, which on a Delaunay mesh is almost always a corner of the
 * containing triangle; when it is not, the best of its triangles is used with negative weights clamped.
 * @param {{xyz: Float32Array, adjOffset: Int32Array, adjList: Int32Array, index: {nearest(x,y,z): number}}} mesh
 */
export function locate(mesh, x, y, z, cells, weights) {
    const { xyz, adjOffset, adjList } = mesh;
    const c = mesh.index.nearest(x, y, z);
    const cx = xyz[3 * c], cy = xyz[3 * c + 1], cz = xyz[3 * c + 2];
    const start = adjOffset[c], end = adjOffset[c + 1];
    let bestMin = -Infinity, bA = -1, bB = -1, wc = 1, wa = 0, wb = 0;
    for (let k = start; k < end; k++) {
        const a = adjList[k], b = adjList[k + 1 < end ? k + 1 : start];
        const ax = xyz[3 * a], ay = xyz[3 * a + 1], az = xyz[3 * a + 2], bx = xyz[3 * b], by = xyz[3 * b + 1], bz = xyz[3 * b + 2];
        // signed volumes: (a × b)·p for c's weight, (b × c)·p for a's, (c × a)·p for b's
        const sc = (ay * bz - az * by) * x + (az * bx - ax * bz) * y + (ax * by - ay * bx) * z;
        const sa = (by * cz - bz * cy) * x + (bz * cx - bx * cz) * y + (bx * cy - by * cx) * z;
        const sb = (cy * az - cz * ay) * x + (cz * ax - cx * az) * y + (cx * ay - cy * ax) * z;
        const sum = sc + sa + sb;
        if (!(Math.abs(sum) > 1e-18)) continue;
        const nc = sc / sum, na = sa / sum, nb = sb / sum, lo = Math.min(nc, na, nb);
        if (lo > bestMin) { bestMin = lo; bA = a; bB = b; wc = nc; wa = na; wb = nb; if (lo >= 0) break; }
    }
    if (bA < 0) { cells[0] = cells[1] = cells[2] = c; weights[0] = 1; weights[1] = weights[2] = 0; return; }
    wc = Math.max(0, wc); wa = Math.max(0, wa); wb = Math.max(0, wb);
    const s = wc + wa + wb || 1;
    cells[0] = c; cells[1] = bA; cells[2] = bB;
    weights[0] = wc / s; weights[1] = wa / s; weights[2] = wb / s;
}
