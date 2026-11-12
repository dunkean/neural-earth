// TectonicWorld: the single container every step module reads and writes.
// Holds the plates (plate-frame state), the world-frame view of the current instant, and this
// world's copy of the tunables.

import { Plate } from './plate.js';
import { TECTONICS } from '../config.js';
import { makeRng } from '../../../../js/rng.js';
import { CrustType } from './crust.js';

export class TectonicWorld {
    /**
     * @param {import('../../core/sim-mesh.js').SimMesh} simMesh
     * @param {number} seed
     * @param {Partial<typeof TECTONICS>} [overrides]  tunables that differ from the defaults for this world
     */
    constructor(simMesh, seed, overrides = {}) {
        this.mesh = simMesh;
        this.seed = seed;
        /** This world's tunables. Engine code reads these, never the global defaults. */
        this.config = { ...structuredClone(TECTONICS), ...overrides };      // (a deep copy: nested groups like `belts` are this world's own)
        // Seeded random streams, so a history replays exactly — a SEPARATE stream per mechanism: adding or removing one
        // must not re-roll the others. A new mechanism gets a new stream (README: "Adding a mechanism").
        this.rng = makeRng((seed ^ 0x5eed) >>> 0);                   // rifting: when a continent rifts, and along which path (step/rifting.js)
        this.rngSubduction = makeRng((seed ^ 0x51ab) >>> 0);         // which passive margins founder (step/subduction-init.js)
        this.rngMarginFounder = makeRng((seed ^ 0xf0d2) >>> 0);      // which margins inside a plate founder (step/subduction-init.js founderWithinPlates)
        this.rngOceanSplit = makeRng((seed ^ 0x0c5a) >>> 0);         // which big ocean plates split along a new ridge (step/ocean-plates.js)
        this.rngHotspot = makeRng((seed ^ 0x7075) >>> 0);            // plumes: when and where they rise, their names and lives (step/hotspots.js)
        this.rngVolcano = makeRng((seed ^ 0x7a1c) >>> 0);   // the details of eruptions: LIP outlines, a tail's pulses
        this.rngBackArc = makeRng((seed ^ 0xbac4) >>> 0);   // when a rolling-back trench opens a basin behind it (step/backarc.js)
        this.rngMarginIsle = makeRng((seed ^ 0x15e1) >>> 0); // where fore-arc islands rise off an active coast (step/margin-isles.js)
        this.rngCycle = makeRng((seed ^ 0xc1c1) >>> 0);      // the supercontinent cycle: where the mantle's next downwelling lies (step/supercontinent.js)
        /** Mantle plumes, fixed in the MANTLE (world) frame — plates slide over them (step/hotspots.js). */
        this.hotspots = [];
        /** Per world cell, 0…1: how insulated the mantle here has been by continent sitting over it (step/hotspots.js). */
        this.mantleHeat = new Float32Array(simMesh.numCells);
        this.hotspotsReady = false;
        /** The supercontinent cycle (step/supercontinent.js): the mantle's upwelling and downwelling, the landmasses' state; null until first watched. */
        this.cycle = null;
        /** Plume heads that arrived under a continent and may split it — this step, or waiting for it (plumeRiftWaitMyr); step/rifting.js acts on them. */
        this.pendingPlumeRifts = [];
        /** When a continent last rifted (Myr), or null before the first: the pace of breakups is kept from it (step/rifting.js). */
        this.lastRiftAt = null;
        /** @type {Object[]} plumes a scene SCHEDULES: {time, xyz, head?, rift?, …} in the mantle frame, or {time, plate, cell} (6a form); step/hotspots.js brings them up */
        this.plumes = [];
        /** @type {Object<number,string>|null} terrane id → craton letter, when the world was built from the start-world generator */
        this.terraneLetters = null;
        this.time = 0;              // Myr since the start of recorded history
        this.step = 0;
        this.occupancyStamp = 0;                     // counts resolve passes: the mark plates leave in the coarse occupancy bins (step/occupancy.js)
        this.lastDt = 0;            // length of the step just taken (Myr)
        this.slowMyr = 0;           // time since the slow rules last ran (config.slowRulesEveryMyr; kept so a history run on in
                                    // pieces is the one history)
        /** @type {Plate[]} index === plate id; dead plates stay in place */
        this.plates = [];
        /** The one empty raster every dead plate holds (simulate.js), made when the first plate goes */
        this.goneCrust = null;

        const n = simMesh.numCells;
        // ── World view of "now" — rebuilt every step by step/resolve.js (or step/reconstruct.js, read-only) ──
        this.owner = new Int16Array(n).fill(-1);     // plate id owning each world cell, −1 = gap
        this.ownerCell = new Int32Array(n);          // that plate's raster cell
        this.claimCount = new Uint8Array(n);         // how many plates claim the cell (0 = gap)
        this.landClaims = new Uint8Array(n);         // …how many of those claims are continental (≥ 2 = collision)
        this.collider = new Int16Array(n).fill(-1);  // at a collision cell: the other plate whose land is here
        this.oceanOverrider = new Int16Array(n).fill(-1);   // ocean-on-ocean: which plate overrides along this stretch (step/boundaries.js)
        this.consumedAt = new Uint8Array(n);         // 1 where crust was subducted during this step
        // A gap cell that no plate may fill yet (the ridge axis; a jitter hole at a trench) still needs an
        // owner for display. Such cells are PROVISIONAL: shown as crust, but no raster holds crust there.
        this.provisional = new Uint8Array(n);        // Provisional.* (0 = a real cell)
        this.provisionalLike = new Int32Array(n);    // for LIKE_NEIGHBOUR: the owner's raster cell it should look like
        // ── Boundary analysis (step/boundaries.js) ──
        this.boundaryKind = new Uint8Array(n);       // BoundaryKind per world cell
        this.convergence = new Float32Array(n);      // cm/yr across the contact, + = closing (diagnostic)

        /** Cumulative crust budget, in cells: ocean created / consumed, and land lost to erosive margins, underthrusting and arc recycling (Phase 7c). */
        this.budget = { created: 0, consumed: 0, erodedCells: 0, underthrustCells: 0, recycledCells: 0 };
        /** @type {{time:number, kind:string, text:string, data:Object}[]} the history, WBP-style: see log() and EventKind */
        this.events = [];
        /** @type {Map<string, import('../step/collision.js').Collision>} one record per pair of colliding plates, keyed "lo:hi" */
        this.collisions = new Map();
        this.arcScrapes = new Map();                 // "donor:host" → arc crust scraped from one plate onto another so far (step/collision.js)
        this.lastTerraneId = 0;                      // highest terrane id handed out (nextTerraneId)
        this.erosionNoise = null;                    // the field that makes a margin erosive or accretionary (step/margin-isles.js), made on first use…
        this.erosionThreshold = 0;                   // …with its threshold, calibrated on the field for marginErosiveShare
    }

    /**
     * A fresh terrane id (0 = "none"). Terrane ids are unique across the world and never reused: scenes and the start
     * world hand out distinct ids when they build crust, and every id made during a history comes from here — so crust
     * keeps its id whichever plate carries it, and a craton keeps its letter. (The first call starts past every id in use.)
     */
    nextTerraneId() {
        if (this.lastTerraneId === 0) {
            for (const p of this.plates) for (let c = 0; c < p.crust.mask.length; c++) if (p.crust.mask[c] && p.crust.terrane[c] > this.lastTerraneId) this.lastTerraneId = p.crust.terrane[c];
        }
        if (this.lastTerraneId >= 0xffff) throw new Error('terrane ids exhausted (Uint16)');
        return ++this.lastTerraneId;
    }

    addPlate(name) {
        const p = new Plate(this.plates.length, this.mesh.numCells, this.time);
        p.name = name;
        this.plates.push(p);
        return p;
    }

    /** A plate's name from the craton letters it holds (start-world histories), or `fallback`. */
    nameByCratons(plate, fallback) {
        if (!this.terraneLetters) return fallback;
        const held = new Set(), { mask, type, terrane } = plate.crust;
        for (let c = 0; c < mask.length; c++) if (mask[c] && type[c] === CrustType.CRATON && this.terraneLetters[terrane[c]]) held.add(this.terraneLetters[terrane[c]]);
        return held.size ? [...held].sort().join('') : fallback;
    }

    /**
     * Record an event. `kind` is one of EventKind; `text` is for people (the lab shows it) and may be reworded freely;
     * `data` carries the facts — plate ids, sizes, times — and is what code (checks, the lab's counters) reads.
     */
    log(kind, text, data = {}) {
        if (!EVENT_KINDS.has(kind)) throw new Error(`unknown event kind: ${kind} (add it to EventKind in model/world.js)`);
        this.events.push({ time: this.time, kind, text, data });
    }
}

/**
 * Every kind of event the engine logs, with the fields of its `data`. Plates are given by id (world.plates[id], and
 * frame.plates[id] in every frame), so an event stays readable after a plate is renamed. Code must read `kind` and
 * `data`, never parse `text`.
 */
export const EventKind = Object.freeze({
    START: 'start',                     // { plate, cratons }                                     the start world is built
    PLATE_BORN: 'plate-born',           // { plate, areaMkm2 }                                    old passive ocean becomes a plate
    COLLISION: 'collision',             // { upper, lower }                                       two plates' land meets
    SUTURE: 'suture',                   // { upper, lower, afterMyr, frontKm, shorteningKm, maxIndentKm, landLostCells,
                                        //   sealedCells, why: 'shortening' | 'stalled' | 'contact' }   they weld into one
    DOCK: 'dock',                       // { host, donor, arc, areaMkm2, cells, sealed, donorGone } a terrane accretes
    ARC_SCRAPE: 'arc-scrape',           // { host, donor, areaMkm2, donorGone }                   an arc is scraped onto a plate
    ARC_COLLAPSE: 'arc-collapse',       // { plate, areaMkm2 }                                    an arc collapses onto its own coast
    ISLE_MERGE: 'isle-merge',           // { plate, areaKm2 }                                     a fore-arc island grows into the mainland
    ARC_RECYCLED: 'arc-recycled',       // { donor, areaKm2 }                                     a small arc goes down with its slab instead of accreting
    FRAGMENT: 'fragment',               // { from, to, areaMkm2, cells }                          a cut-off piece of land joins another plate
    RIFT: 'rift',                       // { plate, fresh, cause: 'plume' | 'tension' | 'overdue', hotspot, lengthKm, ageMyr (of the piece: since its last rift or weld), weakKm (of it along old sutures,
                                        //   failed arms and rifted margins), failedArmKm }
    SUBDUCTION_INIT: 'subduction-init', // { upper, lower, lengthKm, ageMyr }                     a passive margin founders
    OCEAN_SPLIT: 'ocean-split',         // { plate, fresh, areaMkm2 }                             a new ridge splits an ocean plate
    MICROPLATE: 'microplate',           // { plate, host, areaMkm2 }                              a small ocean plate is captured by its neighbour
    SLAB_FLAT: 'slab-flat',             // { plate }                                              the slab under it flattens (Laramide)
    SLAB_STEEP: 'slab-steep',           // { plate }                                              …and steepens again
    BACKARC_OPEN: 'backarc-open',       // { parent, strip, continental, slabAgeMyr, lengthKm }   a strip rifts off behind a rolling-back trench
    BACKARC_CLOSE: 'backarc-close',     // { strip, parent, rejoined }                            its sea stops spreading (rejoined = moves with its parent again)
    PLUME_RISE: 'plume-rise',           // { hotspot, name, plate (−1 = open ocean), head }       a mantle plume arrives
    PLUME_DEATH: 'plume-death',         // { hotspot, name, lifeMyr }                             …and dies out
    LIP: 'lip',                         // { hotspot, name, onLand, areaMkm2, plate (−1 = open ocean) }   a plume head's flood basalt
    SUPERCONTINENT: 'supercontinent',   // { share, areaMkm2, up, down }                          most continental crust is one landmass: the mantle's upwelling moves under it
    DISPERSED: 'dispersed',             // { share, landmasses }                                  …and after its breakup, its pieces have drifted apart
});
const EVENT_KINDS = new Set(Object.values(EventKind));

export const BoundaryKind = Object.freeze({
    NONE: 0,
    RIDGE: 1,        // diverging
    TRANSFORM: 2,
    SUBDUCTING: 3,   // this cell's plate is going down here
    OVERRIDING: 4,   // this cell's plate is the upper plate of a subduction zone
    COLLISION: 5,    // continent–continent
});

export const Provisional = Object.freeze({
    NO: 0,
    NEWBORN: 1,          // on a ridge axis: looks like crust forming right now
    LIKE_NEIGHBOUR: 2,   // a transient hole at a closing boundary: looks like the plate's crust beside it
});
