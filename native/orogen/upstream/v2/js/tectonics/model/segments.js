// Ridge segments and fracture zones (plans/V2_REALISM_PASS_PLAN.md, phase 6; records only — the history never reads them,
// terrain/records.js and synthesize.js draw them). A mid-ocean ridge is cut into segments offset by transform faults, and
// as the ocean grows each segment leaves its own strip of floor: the strips' edges run across the ocean along its
// spreading, its fracture zones — small circles about the two plates' relative pole, bending where that pole moved.
//
// So every cell of ocean floor records where along its ridge it formed: its distance from the spreading pole (the two
// plates' relative rotation, oriented the same for both flanks), in segment lengths.
// Floor rides rigidly with its plate, so the number never changes; along a flow line it stays the same while the pole
// does — the whole numbers trace the fracture zones, and the whole part says which segment a point is in. Read across
// the terrain mesh it is a smooth field: thin, continuous fracture zones on any sim mesh. (Ids inherited cell by cell
// made jagged blobs whose edges drew a crackle, not fracture zones.)

const R = 6371;
export const SEGMENT_KM = 400;  // a ridge segment's length (first-order segments on Earth: 300–1000 km)

/**
 * The fracture-zone coordinate of floor forming at world point p about the relative rotation `w` (rad/Myr, world frame)
 * of the two plates either side of the ridge: never exactly 0 (0 marks "none").
 */
export function fractureCoordinate(px, py, pz, w) {
    let ax = w[0], ay = w[1], az = w[2];
    const l = Math.hypot(ax, ay, az);
    if (l < 1e-12) return 0.5;
    ax /= l; ay /= l; az /= l;
    // (a ridge lies along meridians of the spreading pole and the floor moves away from it along small circles about it —
    // so a fracture zone keeps its distance from the pole: the coordinate is that distance, in segment lengths)
    const u = Math.acos(Math.max(-1, Math.min(1, px * ax + py * ay + pz * az))) * R / SEGMENT_KM;
    return u === 0 ? 1e-6 : u;
}
