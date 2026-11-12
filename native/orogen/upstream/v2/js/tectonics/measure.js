// Measurements on Frames, shared by the lab's stat panels and the check scripts.

import { CrustType, VolcanicType, CrustFlag } from './model/crust.js';
import { BoundaryKind } from './model/world.js';
import { polarFrame } from '../core/sphere-geom.js';

/**
 * What kind of coast a continent's edge is (Phase 7b), for shelves and coastal ranges in Phase 8. The low bits say
 * what the plate does out to sea, the high bits which way the coast faces relative to the plate's motion.
 */
export const Margin = Object.freeze({
    NONE: 0,          // not a coast
    PASSIVE: 1,       // no trench offshore: a rifted, trailing or quiet edge — wide shelves, low coasts
    ACTIVE: 2,        // the plate's own trench offshore: narrow shelf, a coastal range
    TRANSFORM: 3,     // the plate slides past its neighbour here
    COLLISION: 4,     // land meets land
    TYPE: 7,          // mask for the above
    LEADING: 8,       // the coast faces the way the plate moves
    TRAILING: 16,     // the coast faces away from it
    EROSIVE: 32,      // an ACTIVE coast whose trench is erosive: the coast retreats (Phase 7c)
});

/**
 * Classify every coast of a frame. A coast is a land cell next to sea. It is ACTIVE if, going out to sea over its
 * own plate within marginReachKm, the first boundary met carries that plate's trench; PASSIVE if it is
 * a ridge, the lower plate's edge (its ocean going down another plate's trench), or nothing at all; TRANSFORM or
 * COLLISION likewise. Then LEADING / TRAILING by the angle between the coast's seaward normal and the plate's motion.
 * @returns {Uint8Array} Margin bits per world cell
 */
export function classifyMargins(frame, mesh, config) {
    const { numCells, adjOffset, adjList, xyz, edgeKm } = mesh, out = new Uint8Array(numCells);
    const reach = Math.max(1, Math.round(config.marginReachKm / edgeKm)), minSpeed = 0.3 * 10 / 6371;
    const land = w => frame.owner[w] >= 0 && frame.type[w] !== CrustType.OCEAN;
    const seen = new Int32Array(numCells).fill(-1);
    for (let w = 0; w < numCells; w++) {
        if (!land(w)) continue;
        let nx = 0, ny = 0, nz = 0, coast = false;
        for (let k = adjOffset[w]; k < adjOffset[w + 1]; k++) {
            const nb = adjList[k];
            if (land(nb)) continue;
            coast = true;
            nx += xyz[3 * nb] - xyz[3 * w]; ny += xyz[3 * nb + 1] - xyz[3 * w + 1]; nz += xyz[3 * nb + 2] - xyz[3 * w + 2];
        }
        if (!coast) continue;
        // out to sea over this plate: the first boundary met says what the margin is
        const plate = frame.owner[w];
        let type = Margin.PASSIVE;
        seen[w] = w;
        search: for (let ring = [w], hop = 0; hop <= reach && ring.length; hop++) {
            const next = [];
            for (const c of ring) {
                const kind = frame.boundary[c];
                if (kind === BoundaryKind.OVERRIDING || frame.trench[c] > 0) { type = Margin.ACTIVE | (frame.flags[c] & CrustFlag.EROSIVE ? Margin.EROSIVE : 0); break search; }     // a trench is there (converging this instant or not)
                if (kind === BoundaryKind.TRANSFORM) { type = Margin.TRANSFORM; break search; }
                if (kind === BoundaryKind.COLLISION) { type = Margin.COLLISION; break search; }
                if (kind === BoundaryKind.RIDGE || kind === BoundaryKind.SUBDUCTING) break search;          // passive
                for (let k = adjOffset[c]; k < adjOffset[c + 1]; k++) {
                    const nb = adjList[k];
                    if (seen[nb] === w || frame.owner[nb] !== plate || (land(nb) && hop > 0)) continue;      // seaward, on its own plate
                    seen[nb] = w; next.push(nb);
                }
            }
            ring = next;
        }
        // which way the coast faces, relative to the plate's motion
        const om = frame.plates[plate].omega, px = xyz[3 * w], py = xyz[3 * w + 1], pz = xyz[3 * w + 2];
        const vx = om[1] * pz - om[2] * py, vy = om[2] * px - om[0] * pz, vz = om[0] * py - om[1] * px, speed = Math.hypot(vx, vy, vz), nl = Math.hypot(nx, ny, nz);
        let motion = 0;
        if (speed > minSpeed && nl > 0) {
            const cos = (nx * vx + ny * vy + nz * vz) / (nl * speed);
            motion = cos > config.marginMotionCos ? Margin.LEADING : cos < -config.marginMotionCos ? Margin.TRAILING : 0;
        }
        out[w] = type | motion;
    }
    return out;
}

/** A volcanic pile smaller than this (km) is only the hotspot swell between eruptions, not a volcano. */
export const SEAMOUNT_KM = 1;
/** An island is worn down this fast (km/Myr) once it stops erupting — and the sea floor under it keeps sinking. */
export const ISLAND_EROSION_KM_PER_MYR = 0.15;

/** Depth of ocean floor of a given age (km), by half-space cooling (Parsons & Sclater), flattening out on old floor. */
export function seaDepthKm(crustAgeMyr) { return Math.min(6.4, 2.6 + 0.345 * Math.sqrt(Math.max(0, crustAgeMyr))); }

/**
 * Where a hotspot volcano's summit stands relative to sea level now (km; > 0 = an island), or null if there is no
 * volcano here. The pile stands on the sea floor, which deepens as the crust ages; once the volcano stops erupting it
 * is worn down — so volcanoes break the surface near the plume and drown downstream (Hawaii, then the Emperor
 * seamounts). On a continent a volcano is on land: +Infinity.
 */
export function volcanoTopKm(frame, w) {
    if (frame.owner[w] < 0 || frame.volcType[w] !== VolcanicType.TRACK || frame.volcKm[w] < SEAMOUNT_KM) return null;
    if (frame.type[w] !== CrustType.OCEAN) return Infinity;
    const floor = seaDepthKm(frame.time - frame.formedAt[w]), worn = ISLAND_EROSION_KM_PER_MYR * (frame.time - frame.volcAt[w]);
    return frame.volcKm[w] - floor - worn;
}

/** Whole-frame tallies, in cells. */
export function frameTallies(frame) {
    const t = { gaps: 0, landOverOcean: 0, collision: 0, land: 0, newOcean: 0, oldOcean: 0, liveTrench: 0, dormantTrench: 0, provisional: 0 };
    const n = frame.owner.length;
    for (let w = 0; w < n; w++) {
        if (frame.owner[w] < 0) { t.gaps++; continue; }
        if (frame.provisional[w]) t.provisional++;
        if (frame.overlap[w] === 1) t.landOverOcean++;
        else if (frame.overlap[w] === 2) t.collision++;
        if (frame.type[w] !== CrustType.OCEAN) t.land++;
        else if (frame.formedAt[w] < 0) t.oldOcean++;
        else t.newOcean++;
        if (frame.trench[w] === 1) t.liveTrench++; else if (frame.trench[w] === 2) t.dormantTrench++;
    }
    t.cells = n;
    return t;
}

/** Cell selector: cells of plate A or B that touch the other one (their mutual contact, e.g. a ridge). */
export function contactBetween(mesh, idA, idB) {
    const { adjOffset, adjList } = mesh;
    return (frame, w) => {
        const o = frame.owner[w];
        if (o !== idA && o !== idB) return false;
        const other = o === idA ? idB : idA;
        for (let k = adjOffset[w]; k < adjOffset[w + 1]; k++) if (frame.owner[adjList[k]] === other) return true;
        return false;
    };
}

/**
 * Mean azimuth about `pole` (measured from `ref`, see core/sphere-geom.js polarFrame) of the selected
 * cells within ±bandDeg of ref's distance from the pole. Degrees; 0 if none. Default selection: ridge
 * cells. A circular mean, so cells either side of ±180° average correctly.
 */
export function meanAzimuthDeg(frame, xyz, pole, ref, select = (f, w) => f.boundary[w] === BoundaryKind.RIDGE, bandDeg = 10) {
    const toPolar = polarFrame(pole, ref), refColat = toPolar(ref[0], ref[1], ref[2]).colat;
    let sin = 0, cos = 0;
    for (let w = 0; w < frame.owner.length; w++) {
        if (!select(frame, w)) continue;
        const p = toPolar(xyz[3 * w], xyz[3 * w + 1], xyz[3 * w + 2]);
        if (Math.abs(p.colat - refColat) > bandDeg) continue;
        sin += Math.sin(p.azimuth * Math.PI / 180); cos += Math.cos(p.azimuth * Math.PI / 180);
    }
    return sin === 0 && cos === 0 ? 0 : Math.atan2(sin, cos) * 180 / Math.PI;
}

/**
 * Which row of WBP's speed table a plate belongs to right now, judged from the forces on it
 * (FramePlate.dynamics). `min…max` is the expected speed band in cm/yr.
 */
export const SITUATIONS = Object.freeze({
    subducting: { min: 8, max: 20, label: 'subducting ocean plate' },
    colliding: { min: 0, max: 10, label: 'in collision' },
    active: { min: 1.5, max: 5, label: 'active-margin continent (WBP: 2–5; a huge one like Eurasia 1.5)' },
    ringed: { min: 0, max: 2, label: 'pulled from opposite sides — most of the pull cancels' },
    passive: { min: 0, max: 1, label: 'passive — no trench of its own' },
});
const RINGED_TENSION = 0.6;       // more of the pull than this cancels itself → the plate is in a tug of war
                                  // (a continuum really: the two bands overlap between 1.5 and 2 cm/yr)
const MIN_FORCE_KM = 970;         // a margin shorter than this is a sliver, not a margin
const MIN_LOCK_KM2 = 127500;      // a collision has "locked on" once the overlap covers at least this much

/**
 * Has this plate (a FramePlate) settled at the speed its forces call for? Bands are only a fair test
 * then: a plate that is still spinning up, or braking after a collision, is between situations.
 */
export function isSteady(framePlate) {
    const d = framePlate.dynamics;
    if (!d) return true;
    return Math.abs(framePlate.speedCmYr - d.targetCmYr) <= Math.max(STEADY_ABS_CM_YR, STEADY_SHARE * d.targetCmYr);
}
const STEADY_ABS_CM_YR = 0.3, STEADY_SHARE = 0.15;

export function situationOf(dynamics) {
    if (!dynamics) return 'passive';
    const pull = dynamics.forces.find(f => f.name === 'slab pull'), suction = dynamics.forces.find(f => f.name === 'slab suction');
    if (dynamics.collisionKm2 >= MIN_LOCK_KM2) return 'colliding';
    if (pull.km >= MIN_FORCE_KM && pull.torque > suction.torque) return 'subducting';
    if (suction.km >= MIN_FORCE_KM) return dynamics.tension > RINGED_TENSION ? 'ringed' : 'active';
    return 'passive';
}
