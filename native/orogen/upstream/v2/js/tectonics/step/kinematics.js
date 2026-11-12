// Plate kinematics: Euler-pole motion ↔ angular-velocity vectors, and time integration.
// Same convention as GPlates: a pole near the plate spins it, a pole 90° away translates it.

import { quatFromAxisAngle, quatIntegrate, quatMul, quatNormalize } from '../../core/rotation.js';
import { CM_PER_YR_TO_RAD_PER_MYR } from '../config.js';

export function latLonToXyz(latDeg, lonDeg) {
    const lat = latDeg * Math.PI / 180, lon = lonDeg * Math.PI / 180;
    return [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
}

export function xyzToLatLon(x, y, z) {
    return {
        lat: Math.asin(Math.max(-1, Math.min(1, z))) * 180 / Math.PI,
        lon: Math.atan2(y, x) * 180 / Math.PI,
    };
}

/**
 * Angular-velocity vector (rad/Myr, world frame) for rotation about an Euler pole.
 * `speedCmYr` is the surface speed 90° from the pole (the fastest point); sign = sense
 * (positive = counter-clockwise seen from above the pole).
 */
export function omegaFromEulerPole(poleXyz, speedCmYr) {
    const rate = speedCmYr * CM_PER_YR_TO_RAD_PER_MYR;
    return [poleXyz[0] * rate, poleXyz[1] * rate, poleXyz[2] * rate];
}

/** Surface velocity (rad/Myr, tangent vector) of a plate at world position p: ω × p. */
export function surfaceVelocity(omega, px, py, pz, out = [0, 0, 0]) {
    out[0] = omega[1] * pz - omega[2] * py;
    out[1] = omega[2] * px - omega[0] * pz;
    out[2] = omega[0] * py - omega[1] * px;
    return out;
}

export function speedCmYrAt(omega, px, py, pz) {
    const v = surfaceVelocity(omega, px, py, pz);
    return Math.hypot(v[0], v[1], v[2]) / CM_PER_YR_TO_RAD_PER_MYR;
}

/** Advance every living plate's orientation by dt Myr using its current omega. */
export function advancePlates(world, dt) {
    for (const p of world.plates) {
        if (p.alive) p.setQuat(quatIntegrate(p.quat, p.omega, dt));
    }
    world.time += dt;
    world.step++;
}

/**
 * Closed-form pose for constant omega: orientation after `t` Myr starting from q0.
 * Used by prescribed-motion test scenes so the timeline can be scrubbed both ways.
 */
export function poseAtConstantOmega(q0, omega, t) {
    const rate = Math.hypot(omega[0], omega[1], omega[2]);
    if (rate < 1e-15) return q0;
    const dq = quatFromAxisAngle(omega[0] / rate, omega[1] / rate, omega[2] / rate, rate * t);
    return quatNormalize(quatMul(dq, q0));
}
