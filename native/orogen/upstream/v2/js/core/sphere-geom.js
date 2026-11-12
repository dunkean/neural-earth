// Small geometry helpers for points on the unit sphere (arrays [x, y, z], z = north).

import { PLANET_RADIUS_KM } from './sim-mesh.js';

export function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
export function normalize(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }

/** Angular distance in radians. */
export function angleBetween(a, b) { return Math.acos(Math.max(-1, Math.min(1, dot(a, b)))); }

/** Unit tangent at `at` pointing from `from` toward `to` (the chord from→to, projected onto the tangent plane at `at`). */
export function tangentToward(at, from, to) {
    const d = [to[0] - from[0], to[1] - from[1], to[2] - from[2]], k = dot(d, at);
    return normalize([d[0] - k * at[0], d[1] - k * at[1], d[2] - k * at[2]]);
}

/** Uniformly random point on the sphere. */
export function randomPoint(rng) {
    const z = rng() * 2 - 1, t = rng() * 2 * Math.PI, r = Math.sqrt(1 - z * z);
    return [r * Math.cos(t), r * Math.sin(t), z];
}

/** Orthonormal tangent basis at p: [east-ish, north-ish]. */
export function tangentBasis(p) {
    const up = Math.abs(p[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
    const e = normalize(cross(up, p));
    return [e, cross(p, e)];
}

/** The point `dist` radians from p along `bearing` (radians, measured in p's tangent basis). */
export function destination(p, bearing, dist) {
    const [e, n] = tangentBasis(p);
    const dx = Math.cos(bearing), dy = Math.sin(bearing), s = Math.sin(dist), c = Math.cos(dist);
    return normalize([
        p[0] * c + (e[0] * dx + n[0] * dy) * s,
        p[1] * c + (e[1] * dx + n[1] * dy) * s,
        p[2] * c + (e[2] * dx + n[2] * dy) * s,
    ]);
}

/** Bearing (radians, in p's tangent basis) from p toward q. */
export function bearingTo(p, q) {
    const [e, n] = tangentBasis(p);
    return Math.atan2(dot(q, n), dot(q, e));
}

/** Area of a spherical cap of angular radius `rad`, as a fraction of the whole sphere. */
export function capFraction(rad) { return (1 - Math.cos(rad)) / 2; }

/**
 * Coordinates of a position in the frame of an axis `pole`: `colat` = angular distance from the
 * pole, `azimuth` = rotation angle about it measured from the reference point `ref` (degrees,
 * positive = counter-clockwise seen from above the pole). Plates turning about an Euler pole
 * move along lines of constant colat, so this is the natural frame for measuring them.
 */
export function polarFrame(pole, ref) {
    const d = dot(pole, ref);
    const u = normalize([ref[0] - d * pole[0], ref[1] - d * pole[1], ref[2] - d * pole[2]]), v = cross(pole, u);
    return (x, y, z) => ({
        colat: Math.acos(Math.max(-1, Math.min(1, x * pole[0] + y * pole[1] + z * pole[2]))) * 180 / Math.PI,
        azimuth: Math.atan2(x * v[0] + y * v[1] + z * v[2], x * u[0] + y * u[1] + z * u[2]) * 180 / Math.PI,
    });
}

/** Great-circle distance (km) between the centres of mesh cells a and b. */
export function cellDistanceKm(xyz, a, b) {
    const d = xyz[3 * a] * xyz[3 * b] + xyz[3 * a + 1] * xyz[3 * b + 1] + xyz[3 * a + 2] * xyz[3 * b + 2];
    return Math.acos(Math.max(-1, Math.min(1, d))) * PLANET_RADIUS_KM;
}

/** The normalised mean position of a list of mesh cells. */
export function centroidOfCells(xyz, cells) {
    let x = 0, y = 0, z = 0;
    for (const c of cells) { x += xyz[3 * c]; y += xyz[3 * c + 1]; z += xyz[3 * c + 2]; }
    const l = Math.hypot(x, y, z) || 1;
    return [x / l, y / l, z / l];
}
