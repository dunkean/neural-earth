// The parallel ridges inside a mountain belt: thrust sheets and folds stacked along the front. A phasor-noise sum
// (Tricard et al. 2019, as V1's phasor ridges) of kernels placed on the belt's sim cells, each oriented ACROSS the
// belt's strike — the direction in which the recorded distance from the front grows — so the stripes run parallel
// to the front and follow its curves. Where the plates met obliquely the kernels are turned by half the obliquity,
// the way en-echelon folds form in a shear zone: stripes cross the belt at a slant, in short staggered segments
// (fully along a transform; only slightly at an oblique trench or collision, where the sideways motion is partitioned).
// Stripes are asymmetric — a gentle back, a steep face toward the belt's far edge (thrusts verge to the foreland).
// Each kernel's wavelength and direction are jittered a little, so spacing and strike wander and ridges fork and end.
// Kernels belong to the crust (anchor.js): each sits where its crust cell is now and draws its jitter and phases from
// that cell's own stream, so a range keeps its ridges as its plate moves — their phases drifting slowly with time
// (detailRenewMyr), each at a rate of its own, so the ridges' forks and ends move on over geological time.
// By default (cfg.stripeCarry) the sum is taken only at the corners of the plate's raster and carried to the points
// between (carriedStripeAt, below): the same ridges to the eye, for about a seventh of the waves.

import { kindType, kindSide } from './records.js';
import { BeltSide } from '../tectonics/model/crust.js';
import { nearLists } from './near-lists.js';
import { cellRng, crustSites, plateToWorld } from './anchor.js';

const R = 6371, TAU = 2 * Math.PI;
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

/**
 * One kernel per belt crust cell of a belt layer (records.js beltLayer) that has relief and a direction — per CRUST cell
 * (anchor.js crustSites), so a moving belt keeps the same kernels frame after frame, and with them its ridges.
 */
export function buildStripeKernels(rec, simMesh, cfg, seed, anchor, timeMyr = 0) {
    const homes = [], pos = [], dir = [], wave = [], phase = [], phase2 = [], d0 = [], acr = [], kW = [], kA = [], kQ = [], kKind = [], kE = [], kR = [];
    const wanted = w => rec.kind[w] && rec.reliefKm[w] >= 0.02 && (rec.across[3 * w] || rec.across[3 * w + 1] || rec.across[3 * w + 2]);
    const renew = cfg.detailRenewMyr > 0 ? timeMyr / cfg.detailRenewMyr : 0;
    crustSites(anchor, simMesh, wanted, (w, crustCell, x, y, z, o) => {
        // which way across: measured on the plate's own raster at this crust cell (records.js rasterAcross), turned into the
        // world with the plate — or, where the raster has none, the world grid's
        let ax = rec.across[3 * w], ay = rec.across[3 * w + 1], az = rec.across[3 * w + 2];
        const R3 = rec.acrossRaster?.get(o);
        if (R3) {
            const lx = R3[3 * crustCell], ly = R3[3 * crustCell + 1], lz = R3[3 * crustCell + 2];
            if (lx || ly || lz) { const m = anchor.mats, b = 9 * o; ax = m[b] * lx + m[b + 1] * ly + m[b + 2] * lz; ay = m[b + 3] * lx + m[b + 4] * ly + m[b + 5] * lz; az = m[b + 6] * lx + m[b + 7] * ly + m[b + 8] * lz; }
        }
        const type = kindType(rec.kind[w]), share = cfg.enEchelonShare[type] ?? cfg.enEchelonShare.other, rng = cellRng(seed, crustCell, 3);
        const jitter = (rng() * 2 - 1) * cfg.stripeAngleJitterDeg;                                  // (no two kernels quite agree)
        const tilt = (share * (rec.shear[w] / 2) + jitter) * Math.PI / 180, c = Math.cos(tilt), s = Math.sin(tilt);      // + dextral → turned counter-clockwise
        const cx = y * az - z * ay, cy = z * ax - x * az, cz = x * ay - y * ax;
        homes.push(w); pos.push(x, y, z);
        dir.push(ax * c + cx * s, ay * c + cy * s, az * c + cz * s);
        wave.push(cfg.stripeKm[type] * (1 + cfg.stripeWaveJitter * (rng() * 2 - 1)));
        // (the drift: up to a whole wave per detailRenewMyr, either way — from a stream of its own, so the draws above
        // and below stay as they were)
        let drift1 = 0, drift2 = 0;
        if (renew > 0) { const r2 = cellRng(seed, crustCell, 9); drift1 = TAU * renew * (2 * r2() - 1); drift2 = TAU * renew * (2 * r2() - 1); }
        phase.push(rng() * TAU + drift1); phase2.push(rng() * TAU + drift2);
        // (for crossAt: the cell's distance from the front and its plain across direction — both negative on the lower
        // side of a collision, so the distance runs on through the suture)
        const sign = kindSide(rec.kind[w]) === BeltSide.LOWER ? -1 : 1;
        d0.push(sign * rec.dKm[w]); acr.push(sign * ax, sign * ay, sign * az);
        kW.push(rec.reach[w]); kA.push(rec.reliefKm[w]); kQ.push(rec.quietMyr[w]); kKind.push(rec.kind[w]);
        kE.push(rec.envRatio ? rec.envRatio[w] : -1); kR.push(rec.rootRatio ? rec.rootRatio[w] : -1);
    });
    const reachKm = cfg.stripeBandKm * 2.5;
    const near = nearLists(simMesh, homes, pos, () => reachKm);
    return { count: homes.length, pos: Float32Array.from(pos), dir: Float32Array.from(dir), wave: Float32Array.from(wave),
        phase: Float32Array.from(phase), phase2: Float32Array.from(phase2), d0: Float32Array.from(d0), acr: Float32Array.from(acr),
        W: Float32Array.from(kW), A: Float32Array.from(kA), Q: Float32Array.from(kQ), kind: Uint8Array.from(kKind), envR: Float32Array.from(kE), rootR: Float32Array.from(kR),
        near, cut2: (reachKm / R) ** 2 };
}

/**
 * The distance across the belt (km from its front; negative on the lower side of a collision, so it runs on through the
 * suture) at the point w: each kernel's own distance, carried on along its own across direction, blended by the kernels'
 * smooth windows — a coordinate that follows the belt's curves without the steps an interpolation over the sim mesh's
 * triangles has where a belt's kind changes. out[0] = the distance, out[1] = the kernels' total weight (0: no belt).
 */
export function crossAt(K, c0, wx, wy, wz, out) {
    const { offset, items } = K.near, { pos, acr, d0, cut2 } = K;
    let s = 0, sum = 0;
    for (let j = offset[c0]; j < offset[c0 + 1]; j++) {
        const k = items[j], dx = wx - pos[3 * k], dy = wy - pos[3 * k + 1], dz = wz - pos[3 * k + 2], d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= cut2) continue;
        const q = 1 - d2 / cut2, g = q * q * q;
        s += g * (d0[k] + (dx * acr[3 * k] + dy * acr[3 * k + 1] + dz * acr[3 * k + 2]) * R);
        sum += g;
    }
    out[0] = sum > 1e-9 ? s / sum : 0;
    out[1] = sum;
}

/**
 * A belt's width, crest height and quiet time at the point w (a sim cell's place: the plan blends them there once, and
 * the terrain reads the blend over the triangles — a field this smooth needs no more), for one kind of belt: that kind's
 * kernels' own values, blended by their windows as crossAt blends the distance — smooth at the window's scale, where the
 * cells' own values stepped from cell to cell (lobed range edges, crests bent into crescents, blobs of crest height:
 * developer, 2026-10-18, "realistic shaping … leading ridges and edges") — and the kernels' across direction, blended.
 * out = [W, A, Q, weight (0: none of this kind here), ax, ay, az (the direction unnormalised, in the world), and the
 * envelope's and the thickened crust's heights against their profiles (synthesize.js preparePlan): weight, value, weight, value].
 */
export function beltAt(K, c0, wx, wy, wz, kind, out) {
    const { offset, items } = K.near, { pos, acr, cut2, W, A, Q, envR, rootR } = K, kinds = K.kind;
    let sW = 0, sA = 0, sQ = 0, sum = 0, ax = 0, ay = 0, az = 0, eS = 0, eW = 0, rS = 0, rW = 0;
    for (let j = offset[c0]; j < offset[c0 + 1]; j++) {
        const k = items[j];
        if (kinds[k] !== kind) continue;
        const dx = wx - pos[3 * k], dy = wy - pos[3 * k + 1], dz = wz - pos[3 * k + 2], d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= cut2) continue;
        const q = 1 - d2 / cut2, g = q * q * q;
        sW += g * W[k]; sA += g * A[k]; sQ += g * Q[k]; sum += g;
        ax += g * acr[3 * k]; ay += g * acr[3 * k + 1]; az += g * acr[3 * k + 2];
        if (envR && envR[k] >= 0) { eS += g * envR[k]; eW += g; }
        if (rootR && rootR[k] >= 0) { rS += g * rootR[k]; rW += g; }
    }
    if (sum > 1e-9) { out[0] = sW / sum; out[1] = sA / sum; out[2] = sQ / sum; } else { out[0] = out[1] = out[2] = 0; }
    out[3] = sum; out[4] = ax; out[5] = ay; out[6] = az;
    out[7] = eW; out[8] = eW > 1e-9 ? eS / eW : 0; out[9] = rW; out[10] = rW > 1e-9 ? rS / rW : 0;
}

/** An asymmetric ridge profile of a phase angle: −1 in the valley, +1 on the crest, the steep face after the crest. */
function ridge(theta) {
    const t = theta / TAU + 0.5, r = 0.65;
    let v = t < r ? t / r : (1 - t) / (1 - r);
    v = v * v * (3 - 2 * v);
    return 2 * v - 1;
}

/**
 * The stripe value (about −1…+1, plus a finer set) at the terrain point p, whose nearest sim cell is c0; w* is p
 * displaced by the meander warp. K.fade1 / K.fade2 (per kernel, set by the caller for its mesh) remove waves of the main
 * and fine set that the terrain mesh is too coarse to show.
 */
export function stripeAt(K, c0, px, py, pz, wx, wy, wz, fineShare, fineRatio) {
    const { offset, items } = K.near, { pos, dir, wave, phase, phase2, cut2, fade1, fade2 } = K;
    let re = 0, im = 0, re2 = 0, im2 = 0, sum = 0, fadeSum = 0, fineSum = 0;
    // how well the kernels agree on a direction (as lines: 0 = every way, 1 = all parallel), in a frame at p
    let rx = 0, ry = 0, rz = 0, qx = 0, qy = 0, qz = 0, c2 = 0, s2 = 0;
    for (let j = offset[c0]; j < offset[c0 + 1]; j++) {
        const k = items[j], kx = pos[3 * k], ky = pos[3 * k + 1], kz = pos[3 * k + 2];
        const dx = px - kx, dy = py - ky, dz = pz - kz, d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= cut2) continue;
        const q = 1 - d2 / cut2, g = q * q * q;                          // compact smooth window
        const lam = wave[k], f1 = fade1[k];
        if (f1 <= 0 && fineShare <= 0) continue;
        const dxk = dir[3 * k], dyk = dir[3 * k + 1], dzk = dir[3 * k + 2];
        if (!(rx || ry || rz)) { rx = dxk; ry = dyk; rz = dzk; qx = py * rz - pz * ry; qy = pz * rx - px * rz; qz = px * ry - py * rx; }
        const ca = dxk * rx + dyk * ry + dzk * rz, sa = dxk * qx + dyk * qy + dzk * qz, n2 = ca * ca + sa * sa || 1;
        c2 += g * (ca * ca - sa * sa) / n2; s2 += g * 2 * ca * sa / n2;
        const s = ((wx - kx) * dxk + (wy - ky) * dyk + (wz - kz) * dzk) * R;   // km along the kernel's direction
        const a = TAU * s / lam + phase[k];
        re += g * Math.cos(a); im += g * Math.sin(a);
        sum += g; fadeSum += g * f1;
        const f2 = fade2[k];
        if (f2 > 0) { const b = TAU * s * fineRatio / lam + phase2[k]; re2 += g * f2 * Math.cos(b); im2 += g * f2 * Math.sin(b); fineSum += g * f2; }
    }
    if (sum <= 1e-9) return 0;
    // where the kernels cancel, or point every which way, the phase means nothing: no stripes there (they would be blobs)
    const agree = Math.hypot(c2, s2) / sum, coherent = Math.min(1, Math.hypot(re, im) / (0.15 * sum)) * smooth(0.35, 0.75, agree);
    let v = (fadeSum / sum) * coherent * ridge(Math.atan2(im, re));
    if (fineSum > 0) v += fineShare * (fineSum / sum) * Math.min(1, Math.hypot(re2, im2) / (0.15 * fineSum)) * smooth(0.35, 0.75, agree) * ridge(Math.atan2(im2, re2));
    return v;
}

// ── Carried ridges (cfg.stripeCarry, the default; developer 2026-09-26: "they look almost identical") ── The same phasor sum, taken only at the corners of the sim mesh as the plate under
// a point carries it (its raster, turned into the world — so the ridges ride with the crust as the kernels do): at each
// corner its sum, its strength, how well its kernels agree, and the wave it follows (its kernels' mean wave, per km). A
// point between corners continues each corner's wave to itself and blends the three: ~3 waves a point instead of ~28.
// Where neighbouring corners disagree their waves cancel in part, and ridges fork and end there, as they did where the
// kernels disagreed.
const FIELD = 11;                                          // re, im, re2, im2, sum, fadeSum, fineSum, agree, wave (3)

/** stripeAt's sum at a point (window and phase both at x, y, z; nearest sim cell c0), and its mean wave, into out[o…o+10]. */
function stripeField(K, c0, x, y, z, fineShare, fineRatio, out, o) {
    const { offset, items } = K.near, { pos, dir, wave, phase, phase2, cut2, fade1, fade2 } = K;
    let re = 0, im = 0, re2 = 0, im2 = 0, sum = 0, fadeSum = 0, fineSum = 0, kx0 = 0, ky0 = 0, kz0 = 0;
    let rx = 0, ry = 0, rz = 0, qx = 0, qy = 0, qz = 0, c2 = 0, s2 = 0;
    for (let j = offset[c0]; j < offset[c0 + 1]; j++) {
        const k = items[j], kx = pos[3 * k], ky = pos[3 * k + 1], kz = pos[3 * k + 2];
        const dx = x - kx, dy = y - ky, dz = z - kz, d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= cut2) continue;
        const q = 1 - d2 / cut2, g = q * q * q;
        const lam = wave[k], f1 = fade1[k];
        if (f1 <= 0 && fineShare <= 0) continue;
        const dxk = dir[3 * k], dyk = dir[3 * k + 1], dzk = dir[3 * k + 2];
        if (!(rx || ry || rz)) { rx = dxk; ry = dyk; rz = dzk; qx = y * rz - z * ry; qy = z * rx - x * rz; qz = x * ry - y * rx; }
        const ca = dxk * rx + dyk * ry + dzk * rz, sa = dxk * qx + dyk * qy + dzk * qz, n2 = ca * ca + sa * sa || 1;
        c2 += g * (ca * ca - sa * sa) / n2; s2 += g * 2 * ca * sa / n2;
        const s = ((x - kx) * dxk + (y - ky) * dyk + (z - kz) * dzk) * R;
        const a = TAU * s / lam + phase[k];
        re += g * Math.cos(a); im += g * Math.sin(a);
        sum += g; fadeSum += g * f1;
        // (the wave, per unit of the sphere: the kernel's direction × 2πR/λ — turned to agree with the first kernel's)
        const kw = g * (ca >= 0 ? 1 : -1) * TAU * R / lam;
        kx0 += kw * dxk; ky0 += kw * dyk; kz0 += kw * dzk;
        const f2 = fade2[k];
        if (f2 > 0) { const b = TAU * s * fineRatio / lam + phase2[k]; re2 += g * f2 * Math.cos(b); im2 += g * f2 * Math.sin(b); fineSum += g * f2; }
    }
    out[o] = re; out[o + 1] = im; out[o + 2] = re2; out[o + 3] = im2; out[o + 4] = sum; out[o + 5] = fadeSum; out[o + 6] = fineSum;
    out[o + 7] = sum > 1e-9 ? Math.hypot(c2, s2) / sum : 0;
    out[o + 8] = sum > 1e-9 ? kx0 / sum : 0; out[o + 9] = sum > 1e-9 ? ky0 / sum : 0; out[o + 10] = sum > 1e-9 ? kz0 / sum : 0;
}

/** The corners of one belt layer's kernels K on plate o's raster (o < 0: the world grid), filled as points ask for them. */
export function carriedField(K, simMesh, A, o, fineShare, fineRatio) {
    const n = simMesh.numCells;
    return { K, simMesh, A, o, fineShare, fineRatio, done: new Uint8Array(n), f: new Float32Array(FIELD * n), p: new Float64Array(3 * n) };
}

function corner(C, r) {
    C.done[r] = 1;
    const { xyz, index } = C.simMesh, v = [xyz[3 * r], xyz[3 * r + 1], xyz[3 * r + 2]];
    if (C.o >= 0) plateToWorld(C.A, C.o, v);
    C.p[3 * r] = v[0]; C.p[3 * r + 1] = v[1]; C.p[3 * r + 2] = v[2];
    stripeField(C.K, index.nearest(v[0], v[1], v[2]), v[0], v[1], v[2], C.fineShare, C.fineRatio, C.f, FIELD * r);
}

/**
 * The carried stripe value at the point whose raster corners are r3 (weights w3) on C's plate; wx, wy, wz: the point in
 * the world, displaced by the meander warp; fineKeep: the share of the finer set kept here (a worn belt's goes first).
 * The same scale as stripeAt.
 */
export function carriedStripeAt(C, r3, w3, wx, wy, wz, fineKeep = 1) {
    const { f, p, fineRatio } = C, fineShare = C.fineShare * fineKeep;
    let re = 0, im = 0, re2 = 0, im2 = 0, sum = 0, fadeSum = 0, fineSum = 0, agree = 0;
    for (let i = 0; i < 3; i++) {
        const r = r3[i], wt = w3[i];
        if (wt <= 0) continue;
        if (!C.done[r]) corner(C, r);
        const o = FIELD * r, cs = f[o + 4];
        if (cs <= 1e-9) continue;
        // the corner's wave, continued from the corner to the point
        const th = f[o + 8] * (wx - p[3 * r]) + f[o + 9] * (wy - p[3 * r + 1]) + f[o + 10] * (wz - p[3 * r + 2]);
        const c = Math.cos(th), s = Math.sin(th);
        re += wt * (f[o] * c - f[o + 1] * s); im += wt * (f[o] * s + f[o + 1] * c);
        if (f[o + 6] > 0) { const c2 = Math.cos(fineRatio * th), s2 = Math.sin(fineRatio * th); re2 += wt * (f[o + 2] * c2 - f[o + 3] * s2); im2 += wt * (f[o + 2] * s2 + f[o + 3] * c2); }
        sum += wt * cs; fadeSum += wt * f[o + 5]; fineSum += wt * f[o + 6]; agree += wt * cs * f[o + 7];
    }
    if (sum <= 1e-9) return 0;
    agree /= sum;
    const coherent = Math.min(1, Math.hypot(re, im) / (0.15 * sum)) * smooth(0.35, 0.75, agree);
    let v = (fadeSum / sum) * coherent * ridge(Math.atan2(im, re));
    if (fineSum > 0) v += fineShare * (fineSum / sum) * Math.min(1, Math.hypot(re2, im2) / (0.15 * fineSum)) * smooth(0.35, 0.75, agree) * ridge(Math.atan2(im2, re2));
    return v;
}
