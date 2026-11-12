// Fine detail belongs to the crust, not to the world (developer, 2026-09-25: heights persist through the history, so
// a range should stay recognisably the same range as its continent drifts). A frame stores the world view on the
// fixed sim mesh, but each world cell's crust sits in the raster of a plate that moves. So the terrain:
//   · evaluates its noise where a point lies IN ITS PLATE'S OWN FRAME (the corners' plates, blended by weight where
//     plates meet), and turns plate-frame displacements back into the world;
//   · gives every ridge kernel and volcano a random stream keyed by the crust cell it stands on, and places it where
//     that crust cell is now — never by the order of a scan over the world.
// Then the fine detail moves with the crust, and changes only where the records change (a belt grows, a volcano
// erupts, erosion wears). A plate rifted off keeps its parent's frame and raster, so its detail carries on; at a
// suture the lower plate is resampled into the upper's frame, and its detail re-anchors once, as its outline is redrawn.

import { quatToMat3 } from '../core/rotation.js';

/**
 * Per frame: each plate's rotation (row-major, plate frame → world), and for every world cell the crust cell it
 * shows (`cell`: its index in its plate's raster — an identity that persists while the plate moves) and where that
 * crust cell is now (`home`, a world position within half a cell of the world cell).
 */
export function buildAnchor(frame, simMesh) {
    const n = simMesh.numCells, { xyz, index } = simMesh, owner = frame.owner, P = frame.plates?.length ?? 0;
    const mats = new Float64Array(9 * Math.max(1, P)), moved = new Uint8Array(P), m = new Float64Array(9);
    for (let i = 0; i < P; i++) {
        quatToMat3(frame.plates[i]?.quat ?? [1, 0, 0, 0], m);
        mats.set(m, 9 * i);
        moved[i] = Math.abs(m[0] - 1) + Math.abs(m[4] - 1) + Math.abs(m[8] - 1) > 1e-12 ? 1 : 0;
    }
    const cell = new Int32Array(n), home = new Float32Array(3 * n);
    for (let w = 0; w < n; w++) {
        const o = owner[w], x = xyz[3 * w], y = xyz[3 * w + 1], z = xyz[3 * w + 2];
        if (o < 0 || !moved[o]) { cell[w] = w; home[3 * w] = x; home[3 * w + 1] = y; home[3 * w + 2] = z; continue; }
        const b = 9 * o;
        const r = index.nearest(mats[b] * x + mats[b + 3] * y + mats[b + 6] * z, mats[b + 1] * x + mats[b + 4] * y + mats[b + 7] * z, mats[b + 2] * x + mats[b + 5] * y + mats[b + 8] * z);
        const rx = xyz[3 * r], ry = xyz[3 * r + 1], rz = xyz[3 * r + 2];
        cell[w] = r;
        home[3 * w] = mats[b] * rx + mats[b + 1] * ry + mats[b + 2] * rz;
        home[3 * w + 1] = mats[b + 3] * rx + mats[b + 4] * ry + mats[b + 5] * rz;
        home[3 * w + 2] = mats[b + 6] * rx + mats[b + 7] * ry + mats[b + 8] * rz;
    }
    return { owner, mats, cell, home, moved, rasterWorld: new Map() };
}

/**
 * Every crust cell of a moving plate around the world cells where `wanted(w)` holds — each ONCE, however the world grid
 * happens to land on the plate this frame (a rotated raster leaves some of its cells nobody's nearest and others two
 * world cells' nearest: things placed per world cell came and went, or doubled, as the plate moved) — as
 * visit(recordCell, crustCell, x, y, z): the world cell whose records describe it (the one nearest it, which must be on
 * its plate and pass `wanted` too), and where that crust is now. A plate that has not moved: its world cells themselves.
 * In world-cell order, then raster-neighbour order: the same sites in the same order in every worker.
 */
export function crustSites(A, simMesh, wanted, visit) {
    const { xyz, numCells, adjOffset, adjList } = simMesh, m = A.mats, seen = new Set();
    for (let w = 0; w < numCells; w++) {
        const o = A.owner[w];
        if (o < 0 || !wanted(w)) continue;
        if (!A.moved[o]) { visit(w, w, xyz[3 * w], xyz[3 * w + 1], xyz[3 * w + 2], o); continue; }
        const r0 = A.cell[w], b = 9 * o;
        for (let k = adjOffset[r0] - 1; k < adjOffset[r0 + 1]; k++) {
            const r = k < adjOffset[r0] ? r0 : adjList[k], key = o * numCells + r;
            if (seen.has(key)) continue;
            seen.add(key);
            const w2 = rasterToWorld(A, simMesh, o, r);
            if (w2 < 0 || !wanted(w2)) continue;
            const x = xyz[3 * r], y = xyz[3 * r + 1], z = xyz[3 * r + 2];
            visit(w2, r, m[b] * x + m[b + 1] * y + m[b + 2] * z, m[b + 3] * x + m[b + 4] * y + m[b + 5] * z, m[b + 6] * x + m[b + 7] * y + m[b + 8] * z, o);
        }
    }
}

/**
 * The world cell whose records describe raster cell `r` of plate `o` this frame: the one that shows it (its nearest crust
 * cell), or, for a crust cell no world cell shows, the one nearest where that crust now is — or −1 if that belongs to
 * another plate (the crust is hidden, or at the plate's edge). Cached per plate.
 */
export function rasterToWorld(A, simMesh, o, r) {
    let map = A.rasterWorld.get(o);
    if (!map) {
        map = new Int32Array(simMesh.numCells).fill(-2);
        for (let w = 0; w < simMesh.numCells; w++) if (A.owner[w] === o && map[A.cell[w]] === -2) map[A.cell[w]] = w;
        A.rasterWorld.set(o, map);
    }
    let w = map[r];
    if (w === -2) {
        const m = A.mats, b = 9 * o, { xyz } = simMesh, x = xyz[3 * r], y = xyz[3 * r + 1], z = xyz[3 * r + 2];
        w = simMesh.index.nearest(m[b] * x + m[b + 1] * y + m[b + 2] * z, m[b + 3] * x + m[b + 4] * y + m[b + 5] * z, m[b + 6] * x + m[b + 7] * y + m[b + 8] * z);
        if (A.owner[w] !== o) w = -1;
        map[r] = w;
    }
    return w;
}

/**
 * Terrain point (x, y, z) in the frame of the crust under it: its three sim corners' plates, weighted — exact where
 * one plate owns all three (almost everywhere), a smooth blend across a plate boundary. Written into `out`.
 */
export function plateFramePoint(A, c0, c1, c2, w0, w1, w2, x, y, z, out) {
    const o0 = A.owner[c0], o1 = A.owner[c1], o2 = A.owner[c2];
    if (o0 === o1 && o0 === o2) return toPlate(A.mats, o0, x, y, z, 1, out, false);
    out[0] = 0; out[1] = 0; out[2] = 0;
    toPlate(A.mats, o0, x, y, z, w0, out, true); toPlate(A.mats, o1, x, y, z, w1, out, true); toPlate(A.mats, o2, x, y, z, w2, out, true);
    return out;
}

function toPlate(mats, o, x, y, z, w, out, add) {
    let px = x, py = y, pz = z;
    if (o >= 0) { const b = 9 * o; px = mats[b] * x + mats[b + 3] * y + mats[b + 6] * z; py = mats[b + 1] * x + mats[b + 4] * y + mats[b + 7] * z; pz = mats[b + 2] * x + mats[b + 5] * y + mats[b + 8] * z; }
    if (add) { out[0] += w * px; out[1] += w * py; out[2] += w * pz; } else { out[0] = px; out[1] = py; out[2] = pz; }
    return out;
}

/** A plate-frame vector (a displacement) turned into the world by plate `o`'s rotation, in place. */
export function plateToWorld(A, o, v) {
    if (o < 0) return v;
    const m = A.mats, b = 9 * o, x = v[0], y = v[1], z = v[2];
    v[0] = m[b] * x + m[b + 1] * y + m[b + 2] * z; v[1] = m[b + 3] * x + m[b + 4] * y + m[b + 5] * z; v[2] = m[b + 6] * x + m[b + 7] * y + m[b + 8] * z;
    return v;
}

/**
 * A random stream (0…1) of its own for one crust cell and one purpose (`salt`): the same numbers every frame the cell
 * exists, whatever else changed. (A hash, then mulberry32 — V1's makeRng would give correlated first draws for
 * neighbouring ids.)
 */
export function cellRng(seed, id, salt) {
    let h = fmix((Math.floor(seed) | 0) ^ Math.imul(id + 1, 0x9e3779b1) ^ Math.imul(salt + 1, 0x85ebca77));
    return () => {
        h = (h + 0x6d2b79f5) | 0;
        let t = Math.imul(h ^ (h >>> 15), 1 | h);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function fmix(h) {
    h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16;
    return h | 0;
}
