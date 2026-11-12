// Breadth-first helpers on the sim mesh. hopDistance counts hops (convert with simMesh.edgeKm, and
// remember hops over-estimate straight-line distance by up to ~15 %); withinKm measures true distance.

import { PLANET_RADIUS_KM } from './sim-mesh.js';

/**
 * Every cell whose centre lies within `radiusKm` (great-circle) of the centre of a source cell — exactly. From
 * each source on the EDGE of the source set (a cell outside the set that is within reach of an interior source
 * is nearer still to the edge cell where the way out crosses it) a flood spreads through cells within the
 * radius of that one source; a disc is convex, so the flood reaches everything in it.
 * @returns {Uint8Array} 1 where within reach (sources included)
 */
export function withinKm(simMesh, isSource, radiusKm) {
    const { numCells, adjOffset, adjList, xyz } = simMesh, minDot = Math.cos(radiusKm / PLANET_RADIUS_KM);
    const reached = new Uint8Array(numCells), seen = new Int32Array(numCells).fill(-1);
    const queue = [];
    for (let s = 0; s < numCells; s++) {
        if (!isSource(s)) continue;
        reached[s] = 1;
        let edge = false;
        for (let k = adjOffset[s]; k < adjOffset[s + 1]; k++) if (!isSource(adjList[k])) { edge = true; break; }
        if (!edge) continue;
        const sx = xyz[3 * s], sy = xyz[3 * s + 1], sz = xyz[3 * s + 2];
        queue.length = 0; queue.push(s); seen[s] = s;
        for (let head = 0; head < queue.length; head++) {
            const c = queue[head];
            for (let k = adjOffset[c]; k < adjOffset[c + 1]; k++) {
                const nb = adjList[k];
                if (seen[nb] === s || xyz[3 * nb] * sx + xyz[3 * nb + 1] * sy + xyz[3 * nb + 2] * sz < minDot) continue;
                seen[nb] = s; reached[nb] = 1; queue.push(nb);
            }
        }
    }
    return reached;
}

/**
 * Hop distance from every cell where `isSource(cell)` is true, out to `maxHops`.
 * @returns {Int16Array} distance per cell, −1 where not reached
 */
export function hopDistance(simMesh, isSource, maxHops = Infinity, canEnter = null) {
    const { numCells, adjOffset, adjList } = simMesh;
    const dist = new Int16Array(numCells).fill(-1);
    let frontier = [];
    for (let c = 0; c < numCells; c++) if (isSource(c)) { dist[c] = 0; frontier.push(c); }
    for (let d = 1; frontier.length && d <= maxHops; d++) {
        const next = [];
        for (const c of frontier) {
            for (let k = adjOffset[c]; k < adjOffset[c + 1]; k++) {
                const nb = adjList[k];
                if (dist[nb] >= 0 || (canEnter && !canEnter(nb))) continue;
                dist[nb] = d; next.push(nb);
            }
        }
        frontier = next;
    }
    return dist;
}

/** Cells reachable from `start` through cells where `canEnter` is true. */
export function floodFrom(simMesh, start, canEnter) {
    const { numCells, adjOffset, adjList } = simMesh;
    const seen = new Uint8Array(numCells);
    if (!canEnter(start)) return seen;
    const stack = [start];
    seen[start] = 1;
    while (stack.length) {
        const c = stack.pop();
        for (let k = adjOffset[c]; k < adjOffset[c + 1]; k++) {
            const nb = adjList[k];
            if (!seen[nb] && canEnter(nb)) { seen[nb] = 1; stack.push(nb); }
        }
    }
    return seen;
}

/** True if any neighbour of `cell` satisfies `test`. */
export function touches(simMesh, cell, test) {
    const { adjOffset, adjList } = simMesh;
    for (let k = adjOffset[cell]; k < adjOffset[cell + 1]; k++) if (test(adjList[k])) return true;
    return false;
}

/**
 * Hop distance from a LIST of source cells (all at 0) through cells where `canEnter` is true (default: any), out to
 * `maxHops`; with, for every cell reached, the source it was reached from.
 * @returns {{dist: Int16Array, source: Int32Array}}   −1 where not reached
 */
export function hopsFrom(simMesh, sources, canEnter = null, maxHops = Infinity) {
    const { numCells, adjOffset, adjList } = simMesh;
    const dist = new Int16Array(numCells).fill(-1), source = new Int32Array(numCells).fill(-1);
    let frontier = [];
    for (const c of sources) if (dist[c] < 0) { dist[c] = 0; source[c] = c; frontier.push(c); }
    for (let d = 1; frontier.length && d <= maxHops; d++) {
        const next = [];
        for (const c of frontier) for (let k = adjOffset[c]; k < adjOffset[c + 1]; k++) {
            const nb = adjList[k];
            if (dist[nb] >= 0 || (canEnter && !canEnter(nb))) continue;
            dist[nb] = d; source[nb] = source[c]; next.push(nb);
        }
        frontier = next;
    }
    return { dist, source };
}

/** Is a cell where `isTarget` holds within `maxHops` of `cell`, going only through cells where `canEnter` holds? */
export function reachesWithin(simMesh, cell, maxHops, canEnter, isTarget) {
    const { adjOffset, adjList } = simMesh, seen = new Set([cell]);
    let ring = [cell];
    for (let h = 0; h < maxHops && ring.length; h++) {
        const next = [];
        for (const c of ring) for (let k = adjOffset[c]; k < adjOffset[c + 1]; k++) {
            const nb = adjList[k];
            if (seen.has(nb) || !canEnter(nb)) continue;
            if (isTarget(nb)) return true;
            seen.add(nb); next.push(nb);
        }
        ring = next;
    }
    return false;
}

/** The connected pieces of a set of cells (each piece a list of cells), in the order they are found. */
export function piecesOf(simMesh, cells) {
    const { adjOffset, adjList } = simMesh, inSet = new Set(cells), seen = new Set(), pieces = [];
    for (const start of cells) {
        if (seen.has(start)) continue;
        const piece = [start]; seen.add(start);
        for (let i = 0; i < piece.length; i++) for (let k = adjOffset[piece[i]]; k < adjOffset[piece[i] + 1]; k++) { const nb = adjList[k]; if (inSet.has(nb) && !seen.has(nb)) { seen.add(nb); piece.push(nb); } }
        pieces.push(piece);
    }
    return pieces;
}

/** The biggest connected piece of a set of cells (the first found, on a tie). */
export function largestPieceOf(simMesh, cells) {
    let best = [];
    for (const piece of piecesOf(simMesh, cells)) if (piece.length > best.length) best = piece;
    return best;
}
