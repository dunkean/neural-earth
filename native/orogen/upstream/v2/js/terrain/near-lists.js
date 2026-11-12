// For point features placed on the sim mesh (stripe kernels, volcano cones): which of them can touch a point whose
// nearest sim cell is c. Built once per synthesis as a CSR list per sim cell, so the per-point loop over the terrain
// mesh only visits features within reach.

const R = 6371;

/**
 * @param {{xyz: Float32Array, adjOffset: Int32Array, adjList: Int32Array, numCells: number, edgeKm: number}} simMesh
 * @param {number[]} homes      the sim cell each feature sits in
 * @param {Float32Array|number[]} pos   feature positions, 3 per feature (unit vectors)
 * @param {(i: number) => number} radiusKm   how far feature i reaches
 * @returns {{offset: Int32Array, items: Int32Array}}   items[offset[c]…offset[c+1]) = features that may reach cell c's points
 */
export function nearLists(simMesh, homes, pos, radiusKm) {
    const { xyz, adjOffset, adjList, numCells, edgeKm } = simMesh;
    // (the pairs (cell, feature) in the order found — features in order — then sorted by cell, stably: each cell's list in
    // feature order, as when every cell kept an array of its own)
    let pairCell = new Int32Array(1 << 16), pairItem = new Int32Array(1 << 16), pairs = 0;
    const stamp = new Int32Array(numCells).fill(-1), queue = new Int32Array(numCells);
    for (let i = 0; i < homes.length; i++) {
        const px = pos[3 * i], py = pos[3 * i + 1], pz = pos[3 * i + 2];
        const reach = (radiusKm(i) + edgeKm) / R, chordMax = 2 * Math.sin(Math.min(Math.PI / 2, reach / 2)), cut2 = chordMax * chordMax;
        let tail = 0;
        queue[tail++] = homes[i];
        stamp[homes[i]] = i;
        for (let head = 0; head < tail; head++) {
            const c = queue[head];
            if (pairs === pairCell.length) {
                const pc = new Int32Array(2 * pairs), pi = new Int32Array(2 * pairs);
                pc.set(pairCell); pi.set(pairItem); pairCell = pc; pairItem = pi;
            }
            pairCell[pairs] = c; pairItem[pairs++] = i;
            for (let k = adjOffset[c]; k < adjOffset[c + 1]; k++) {
                const nb = adjList[k];
                if (stamp[nb] === i) continue;
                stamp[nb] = i;
                const dx = xyz[3 * nb] - px, dy = xyz[3 * nb + 1] - py, dz = xyz[3 * nb + 2] - pz;
                if (dx * dx + dy * dy + dz * dz <= cut2) queue[tail++] = nb;
            }
        }
    }
    const offset = new Int32Array(numCells + 1);
    for (let j = 0; j < pairs; j++) offset[pairCell[j] + 1]++;
    for (let c = 0; c < numCells; c++) offset[c + 1] += offset[c];
    const items = new Int32Array(pairs), at = offset.slice(0, numCells);
    for (let j = 0; j < pairs; j++) items[at[pairCell[j]]++] = pairItem[j];
    return { offset, items };
}
