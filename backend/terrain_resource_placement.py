"""Optional fused CPU placement CDF. No fast-math, parallel reductions or RNG.

The public NumPy distribution remains the reference/fallback. Only a scratch
CDF is accelerated; addressed draws, all deposits and budgets remain unchanged.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def cdf(field, ix, iy, tx, ty, xs, ys, inside_x, inside_y, cx, cy, cosine, sine,
        width, wa, wc, amplitude, background, denominator):
    nx, ny = len(ix), len(iy)
    result = np.empty(nx * ny, np.float64)
    total, any_weight = 0., False
    for row in range(ny):
        j, jp = iy[row], min(iy[row]+1, field.shape[0]-1)
        for col in range(nx):
            i, ip = (ix[col]-1) % field.shape[1], ix[col] % field.shape[1]
            a = field[j,i]*(1-tx[col]) + field[j,ip]*tx[col]
            b = field[jp,i]*(1-tx[col]) + field[jp,ip]*tx[col]
            weight = (a*(1-ty[row]) + b*ty[row]) * (inside_x[col] and inside_y[row])
            intensity = background
            for k in range(len(cx)):
                dx = (xs[col]-cx[k]+width/2) % width-width/2
                dy = ys[row]-cy[k]
                u, v = dx*cosine[k]+dy*sine[k], -dx*sine[k]+dy*cosine[k]
                intensity = intensity + amplitude*math.exp(-.5*(u*u/(wa*wa)+v*v/(wc*wc)))
            weight = weight*(intensity/denominator)
            any_weight = any_weight or weight > 0
            total += weight
            result[row*nx+col] = total
    if not any_weight:
        total = 0.
        for row in range(ny):
            for col in range(nx):
                total += inside_x[col] and inside_y[row]
                result[row*nx+col] = total
    return result
