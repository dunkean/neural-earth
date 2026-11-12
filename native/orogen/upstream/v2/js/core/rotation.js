// Rigid rotations on the unit sphere.
// A plate's orientation is a unit quaternion [w, x, y, z] mapping plate frame → world
// frame. The 3×3 matrix form is cached for bulk point transforms.

export function quatIdentity() { return [1, 0, 0, 0]; }

/** Quaternion for a rotation of `angle` radians about unit axis (ax, ay, az). */
export function quatFromAxisAngle(ax, ay, az, angle) {
    const h = angle * 0.5, s = Math.sin(h);
    return [Math.cos(h), ax * s, ay * s, az * s];
}

/** Hamilton product a·b (apply b first, then a). */
export function quatMul(a, b) {
    return [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ];
}

export function quatNormalize(q) {
    const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

/** Write the row-major 3×3 matrix of q into `out` (Float64Array(9)). */
export function quatToMat3(q, out) {
    const [w, x, y, z] = q;
    const xx = x * x, yy = y * y, zz = z * z;
    const xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
    out[0] = 1 - 2 * (yy + zz); out[1] = 2 * (xy - wz);     out[2] = 2 * (xz + wy);
    out[3] = 2 * (xy + wz);     out[4] = 1 - 2 * (xx + zz); out[5] = 2 * (yz - wx);
    out[6] = 2 * (xz - wy);     out[7] = 2 * (yz + wx);     out[8] = 1 - 2 * (xx + yy);
    return out;
}

/**
 * Advance orientation q by angular-velocity vector omega (world frame, rad/Myr) over dt.
 * Returns a new normalized quaternion.
 */
export function quatIntegrate(q, omega, dt) {
    const rate = Math.hypot(omega[0], omega[1], omega[2]);
    if (rate < 1e-12) return q;
    const dq = quatFromAxisAngle(omega[0] / rate, omega[1] / rate, omega[2] / rate, rate * dt);
    return quatNormalize(quatMul(dq, q));
}

/** Get/store a rotation as a GPlates-style Euler pole: { lat, lon, angle } in degrees. */
export function quatToEulerPole(q) {
    const s = Math.hypot(q[1], q[2], q[3]);
    if (s < 1e-12) return { lat: 90, lon: 0, angle: 0 };
    const angle = 2 * Math.atan2(s, q[0]);
    const x = q[1] / s, y = q[2] / s, z = q[3] / s;
    return {
        lat: Math.asin(Math.max(-1, Math.min(1, z))) * 180 / Math.PI,
        lon: Math.atan2(y, x) * 180 / Math.PI,
        angle: angle * 180 / Math.PI,
    };
}
