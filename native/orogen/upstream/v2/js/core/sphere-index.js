// O(1) exact nearest-cell lookup on the unit sphere.
// A cube-map table gives a starting cell (the one nearest each texel centre); a greedy
// walk over mesh neighbours then finishes the job. On a Delaunay mesh that walk always
// ends at the true nearest cell, and from the table's guess it is 0–1 hops long.

export class SphereIndex {
    /**
     * @param {Float32Array} xyz   cell positions (unit vectors, 3 per cell)
     * @param {{adjOffset:Int32Array, adjList:Int32Array, numRegions:number}} mesh
     * @param {number} [texelsPerCell=2] linear oversampling of the start-guess table
     */
    constructor(xyz, mesh, texelsPerCell = 2) {
        this.xyz = xyz;
        this.adjOffset = mesh.adjOffset;
        this.adjList = mesh.adjList;
        const n = mesh.numRegions;
        this.res = Math.max(8, Math.ceil(Math.sqrt(n / 6) * texelsPerCell));
        this.table = new Int32Array(6 * this.res * this.res);
        this._build(mesh);
    }

    _build(mesh) {
        const { res, table, xyz } = this;
        const { adjOffset, adjList } = mesh;
        let cur = 0;
        for (let face = 0; face < 6; face++) {
            for (let j = 0; j < res; j++) {
                // Serpentine scan keeps consecutive texels adjacent → short greedy walks.
                for (let ii = 0; ii < res; ii++) {
                    const i = (j & 1) ? res - 1 - ii : ii;
                    const u = (2 * (i + 0.5)) / res - 1, v = (2 * (j + 0.5)) / res - 1;
                    let x, y, z;
                    switch (face) {
                        case 0: x = 1;  y = u; z = v; break;
                        case 1: x = -1; y = u; z = v; break;
                        case 2: x = u; y = 1;  z = v; break;
                        case 3: x = u; y = -1; z = v; break;
                        case 4: x = u; y = v; z = 1;  break;
                        default: x = u; y = v; z = -1;
                    }
                    const inv = 1 / Math.hypot(x, y, z);
                    x *= inv; y *= inv; z *= inv;
                    let best = x * xyz[3 * cur] + y * xyz[3 * cur + 1] + z * xyz[3 * cur + 2];
                    for (let improved = true; improved;) {
                        improved = false;
                        for (let k = adjOffset[cur], e = adjOffset[cur + 1]; k < e; k++) {
                            const nb = adjList[k];
                            const d = x * xyz[3 * nb] + y * xyz[3 * nb + 1] + z * xyz[3 * nb + 2];
                            if (d > best) { best = d; cur = nb; improved = true; }
                        }
                    }
                    table[(face * res + j) * res + i] = cur;
                }
            }
        }
    }

    /** Nearest cell to the (not necessarily normalized) direction (x, y, z). */
    nearest(x, y, z) {
        return this.walk(this.start(x, y, z), x, y, z);
    }

    /**
     * The cube-table guess for (x, y, z): the true nearest cell is within 2 hops of it (measured over millions of
     * lookups on 16K–80K meshes; step/gather.js relies on this, with a ring to spare).
     */
    start(x, y, z) {
        const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
        const res = this.res;
        let face, u, v;
        if (ax >= ay && ax >= az) { face = x > 0 ? 0 : 1; u = y / ax; v = z / ax; }
        else if (ay >= az)        { face = y > 0 ? 2 : 3; u = x / ay; v = z / ay; }
        else                      { face = z > 0 ? 4 : 5; u = x / az; v = y / az; }
        let i = ((u + 1) * 0.5 * res) | 0, j = ((v + 1) * 0.5 * res) | 0;
        if (i >= res) i = res - 1;
        if (j >= res) j = res - 1;
        return this.table[(face * res + j) * res + i];
    }

    /** Greedy walk from cell `cur` to the nearest cell of (x, y, z) — exact on a Delaunay mesh, from any start. */
    walk(cur, x, y, z) {
        const { xyz, adjOffset, adjList } = this;
        let best = x * xyz[3 * cur] + y * xyz[3 * cur + 1] + z * xyz[3 * cur + 2];
        for (let improved = true; improved;) {
            improved = false;
            for (let k = adjOffset[cur], e = adjOffset[cur + 1]; k < e; k++) {
                const nb = adjList[k];
                const d = x * xyz[3 * nb] + y * xyz[3 * nb + 1] + z * xyz[3 * nb + 2];
                if (d > best) { best = d; cur = nb; improved = true; }
            }
        }
        return cur;
    }
}
