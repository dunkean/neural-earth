// A binary min-heap of (key, id) pairs in typed arrays — for Dijkstra over a mesh and, later, priority-flood drainage.
// Duplicate ids are allowed (lazy deletion: skip an id popped with a key worse than its best).

export class MinHeap {
    constructor(capacity = 1024) {
        this.keys = new Float64Array(capacity);
        this.ids = new Int32Array(capacity);
        this.size = 0;
    }

    push(key, id) {
        if (this.size === this.keys.length) {
            const k = new Float64Array(this.size * 2), i = new Int32Array(this.size * 2);
            k.set(this.keys); i.set(this.ids);
            this.keys = k; this.ids = i;
        }
        const { keys, ids } = this;
        let n = this.size++;
        while (n > 0) {
            const parent = (n - 1) >> 1;
            if (keys[parent] <= key) break;
            keys[n] = keys[parent]; ids[n] = ids[parent];
            n = parent;
        }
        keys[n] = key; ids[n] = id;
    }

    /** Key of the smallest entry (call only when size > 0). */
    peekKey() { return this.keys[0]; }

    /** Remove the smallest entry and return its id (its key is in `lastKey`). */
    pop() {
        const { keys, ids } = this;
        const topId = ids[0];
        this.lastKey = keys[0];
        const n = --this.size;
        if (n > 0) {
            const key = keys[n], id = ids[n];
            let i = 0;
            for (;;) {
                const l = 2 * i + 1;
                if (l >= n) break;
                const r = l + 1, c = r < n && keys[r] < keys[l] ? r : l;
                if (keys[c] >= key) break;
                keys[i] = keys[c]; ids[i] = ids[c];
                i = c;
            }
            keys[i] = key; ids[i] = id;
        }
        return topId;
    }
}

/**
 * Multi-source Dijkstra over a mesh in km. `dist` must be pre-filled: the source distances where known, Infinity
 * elsewhere. `canEnter(from, to)` restricts the spread; `onReach(from, to)` runs when `to` gets a better distance
 * (to carry a label along, e.g. which source it came from). Stops at `maxKm`.
 */
export function spreadKm(mesh, dist, { canEnter = null, onReach = null, maxKm = Infinity } = {}) {
    const { adjOffset, adjList } = mesh, edge = edgeKmOf(mesh);
    const heap = new MinHeap(4096);
    for (let c = 0; c < dist.length; c++) if (dist[c] < Infinity) heap.push(dist[c], c);
    while (heap.size) {
        const c = heap.pop(), dc = heap.lastKey;
        if (dc > dist[c]) continue;
        for (let k = adjOffset[c]; k < adjOffset[c + 1]; k++) {
            const nb = adjList[k];
            if (canEnter && !canEnter(c, nb)) continue;
            const d = dc + edge[k];
            if (d >= dist[nb] || d > maxKm) continue;
            dist[nb] = d;
            if (onReach) onReach(c, nb);
            // (queued at the value STORED: in a Float32Array it is d rounded, and were it queued at d itself, a d rounded
            // down would read as stale when popped — the cell would never pass its distance on, and whole patches were
            // reached late, by detours, or not at all)
            heap.push(dist[nb], nb);
        }
    }
    return dist;
}

/** Each adjacency entry's length in km (the arc from the cell to that neighbour), once per mesh — as spreadKm had it. */
const edgeKms = new WeakMap();
/** Let go of a mesh's cached edge lengths (the finish's 10M mesh: ~240 MB). */
export function forgetEdgeKm(mesh) { edgeKms.delete(mesh); }
function edgeKmOf(mesh) {
    let e = edgeKms.get(mesh);
    if (e) return e;
    const { adjOffset, adjList, xyz } = mesh, n = adjOffset.length - 1, R = 6371;
    e = new Float64Array(adjList.length);
    for (let c = 0; c < n; c++) {
        const cx = xyz[3 * c], cy = xyz[3 * c + 1], cz = xyz[3 * c + 2];
        for (let k = adjOffset[c]; k < adjOffset[c + 1]; k++) {
            const nb = adjList[k], dot = cx * xyz[3 * nb] + cy * xyz[3 * nb + 1] + cz * xyz[3 * nb + 2];
            e[k] = Math.acos(Math.min(1, dot)) * R;
        }
    }
    edgeKms.set(mesh, e);
    return e;
}
