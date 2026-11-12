"""Magna Urbis' Noise presentation relief on Neural Earth's immutable 30 m DEM.

The ridged, derivative-attenuated fBm is a tensor port of terrain3dNoise.ts.
Native reconstruction retains Neural Earth's continuous coastal correction.
Noise wavelengths/probes never depend on a requested tile or camera LOD.
Physical ground displacement uses the finest supported mesh footprint; the
Satellite material resolves finer ground bands in the pixel shader.
"""
from __future__ import annotations

import math
import torch

import terrain_refinement as refinement

VERSION = f'magna-noise76f9b0bb-grounde501e503-v1+{refinement.VERSION}'
WAVELENGTHS = tuple(240 / 2**k for k in range(11))
SURFACE_CHANNELS = ('flow', 'deposit', 'scree', 'rock')
_MASK = 0xffffffff


def _smooth(a, b, value):
    t = ((value-a)/(b-a)).clamp(0, 1)
    return t*t*(3-2*t)


class _NativeSurface:
    """Prepare one bounded native read, then evaluate fixed physical probes."""

    def __init__(self, world, level, bounds, check):
        a, b, c, d = bounds
        spacing = 30 / 2**level
        radius = refinement.SPLINE_RADIUS
        # 120 m macro probes, four spline taps and the reconstruction FIR.
        pad = radius + 7
        self.y0 = math.floor((a+.5)*spacing/30)-pad
        self.x0 = math.floor((b+.5)*spacing/30)-pad
        y1 = math.ceil((c+.5)*spacing/30)+pad
        x1 = math.ceil((d+.5)*spacing/30)+pad
        # The upstream decoder's arbitrary rectangular reads can differ with
        # shape; reuse the existing immutable 128-cell native block reader.
        from terrain_phasor import _canonical_reader
        native = _canonical_reader(world, check)(self.y0, self.x0, y1, x1)
        q = refinement._signed_sqrt(native)
        self.coefficients = refinement._filter(q, refinement._INTERPOLATE)
        self.raw = refinement._trim(q, radius)
        smooth = refinement._filter(refinement._trim(q, radius-2), refinement.COAST_SMOOTH)
        self.gradient = ((smooth[2:, 1:-1]-smooth[:-2, 1:-1])**2 +
                         (smooth[1:-1, 2:]-smooth[1:-1, :-2])**2).sqrt()*.5
        self.smooth = smooth[1:-1, 1:-1]
        self.y0 += radius
        self.x0 += radius

    def __call__(self, x, y):
        px, py = x/30-.5-self.x0, y/30-.5-self.y0
        ix, iy = px.floor().long(), py.floor().long()
        fx, fy = px-ix, py-iy
        linear = ((1-fy)*((1-fx)*self.raw[iy, ix]+fx*self.raw[iy, ix+1]) +
                  fy*((1-fx)*self.raw[iy+1, ix]+fx*self.raw[iy+1, ix+1]))
        def spline(values):
            out = torch.zeros_like(linear)
            for row, wy in refinement._cubic_taps(iy, fy):
                for col, wx in refinement._cubic_taps(ix, fx):
                    out = out + values[row, col]*wy*wx
            return out
        exact = refinement._limited(linear, spline(self.coefficients)-linear)
        coast, slope = spline(self.smooth), spline(self.gradient)
        near = 1-_smooth(refinement.COAST_NEAR, refinement.COAST_FAR, coast.abs()/(slope+1e-9))
        height = exact+(coast-exact)*near
        return height*height.abs()


def _gradient_noise(x, y, seed):
    """Value and analytic derivatives, matching the upstream unsigned hash."""
    ix, iy = x.floor().long(), y.floor().long()
    fx, fy = x-ix, y-iy
    corners = []
    for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
        h = (((ix+a)*374761393) ^ ((iy+b)*668265263) ^ seed) & _MASK
        h = ((h ^ (h >> 13))*1274126177) & _MASK
        h = h ^ (h >> 16)
        angle = h.double()*(math.tau/4294967296)
        gx, gy = angle.cos(), angle.sin()
        corners.append((gx*(fx-a)+gy*(fy-b), gx, gy))
    u = fx**3*(fx*(fx*6-15)+10)
    v = fy**3*(fy*(fy*6-15)+10)
    du = 30*fx*fx*(fx*(fx-2)+1)
    dv = 30*fy*fy*(fy*(fy-2)+1)
    values = []
    for k in range(3):
        a, b, c, d = (corner[k] for corner in corners)
        value = a+(b-a)*u+(c-a)*v+(a-b-c+d)*u*v
        if k == 1:
            a, b, c, d = (corner[0] for corner in corners)
            value += du*((b-a)+(a-b-c+d)*v)
        elif k == 2:
            a, b, c, d = (corner[0] for corner in corners)
            value += dv*((c-a)+(a-b-c+d)*u)
        values.append(value*1.4)
    return values


def noise_height(base, x, y, seed, strength=1., check=None):
    """The pinned default Noise stack (amplitude .008, H=1, lacunarity 2)."""
    height = base(x, y)
    gx = (base(x+7.5, y)-base(x-7.5, y))/15
    gy = (base(x, y+7.5)-base(x, y-7.5))/15
    slope = torch.hypot(gx, gy)
    n, s, e, w = base(x, y-120), base(x, y+120), base(x+120, y), base(x-120, y)
    relief = torch.stack(torch.broadcast_tensors(*[(height-z).abs()/120 for z in (n, s, e, w)])).amax(0)
    crest = _smooth(.002, .025, (height-(n+s+e+w)*.25)/120)*_smooth(.04, .2, torch.maximum(slope, relief))
    mountain = _smooth(.04, .2, slope)
    ax, ay, residual = torch.zeros_like(height), torch.zeros_like(height), torch.zeros_like(height)
    for k, wavelength in enumerate(WAVELENGTHS):
        if check is not None:
            check()
        angle = .37+k*2.399963229728653
        c, s = math.cos(angle), math.sin(angle)
        v, dx, dy = _gradient_noise((x*c-y*s)/wavelength, (x*s+y*c)/wavelength, (seed+k*7919) & _MASK)
        rounded = (v*v+.0025).sqrt()
        signal = v*.45+(.24-rounded)*.55
        derivative = .45-.55*v/rounded
        dx, dy = (dx*c+dy*s)*derivative, (-dx*s+dy*c)*derivative
        ratio = wavelength/30
        broad = _scalar_smooth(1.4, 3.5, ratio)
        fine = 1-_scalar_smooth(.045, .12, ratio)
        envelope = broad*crest+fine*mountain*3+(1-max(broad, fine))*mountain*.015
        weight = ((torch.maximum(slope, relief) if broad > .5 else slope)+.004).clamp_min(1e-6).sqrt()
        ax, ay = ax+dx*(broad+fine*.15), ay+dy*(broad+fine*.15)
        erosion = 1/(1+.65*(ax*ax+ay*ay))
        direction = 1/(1+1.5*((dx*gx+dy*gy)/slope.clamp_min(1e-6)).abs())
        residual += signal*wavelength*.008*envelope*weight*erosion*direction
    detailed = height+residual*strength*(height/2).clamp(0, 1)
    return torch.where(height > 0, detailed.clamp_min(0), height), slope


def _scalar_smooth(a, b, value):
    t = min(1., max(0., (value-a)/(b-a)))
    return t*t*(3-2*t)


def _ground_height(x, y, seed):
    """8 m bedrock band, the only band resolved at the canonical 0.46875 m grid.

    The 2.2 m compressed joint, gravel and grain bands are entirely unresolved
    at this footprint. Satellite carries their filtered material normals.
    """
    c, s = .8, .6
    turn = (seed >> 29) & 3
    if turn == 1:
        c, s = -s, c
    elif turn == 2:
        c, s = -c, -s
    elif turn == 3:
        c, s = s, -c
    rx, rz = c*x+s*y, -s*x+c*y
    px = (.36*rx+.48*rz)/8+17.137
    py = (.48*rx+.64*rz)/(8*.35)+3.731
    pz = (-.8*rx+.6*rz)/8-11.517
    ix, iy, iz = px.floor().long(), py.floor().long(), pz.floor().long()
    fx, fy, fz = px-ix, py-iy, pz-iz
    ux, uy, uz = [f**3*(f*(f*6-15)+10) for f in (fx, fy, fz)]
    raw = torch.zeros_like(px)
    for k in range(8):
        a, b, c = k & 1, (k >> 1) & 1, k >> 2
        h = (((ix+a)*0x9e3779b9) ^ ((iy+b)*0x85ebca6b) ^ ((iz+c)*0xc2b2ae35) ^ ((seed+0x6c8e9cf5) & _MASK)) & _MASK
        h = ((h ^ (h >> 16))*0x7feb352d) & _MASK
        h = ((h ^ (h >> 15))*0x846ca68b) & _MASK
        h ^= h >> 16
        nx, ny, nz = (h & 1023)/511.5-1, ((h >> 10) & 1023)/511.5-1, ((h >> 20) & 1023)/511.5-1
        weight = (ux if a else 1-ux)*(uy if b else 1-uy)*(uz if c else 1-uz)
        raw += (nx*(fx-a)+ny*(fy-b)+nz*(fz-c))*weight*1.7
    value = .72*raw/torch.sqrt(1+raw*raw)
    visibility = 1-_scalar_quintic((30/64/(8*.35)-.18)/.24)
    return (.3-torch.sqrt(value*value+.02))*.17*visibility


def _scalar_quintic(t):
    t = min(1., max(0., t))
    return t**3*(t*(t*6-15)+10)


def _heights(world, level, bounds, strength, check):
    a, b, c, d = bounds
    spacing = 30/2**level
    x = (torch.arange(b, d, device=world.device, dtype=torch.float64)[None, :]+.5)*spacing
    y = (torch.arange(a, c, device=world.device, dtype=torch.float64)[:, None]+.5)*spacing
    base = _NativeSurface(world, level, bounds, check)
    seed = int(getattr(world, 'seed', 0)) & _MASK
    h, slope = noise_height(base, x, y, seed, strength, check)
    # Native climate is sampled at these same coordinates, independent of view.
    from terrain_climate import sample_coarse_climate
    climate = sample_coarse_climate(world, (x[0]/30).cpu().numpy(), (y[:, 0]/30).cpu().numpy(), check=check)
    temperature = torch.as_tensor(climate[0], device=h.device, dtype=torch.float64)
    rain = torch.as_tensor(climate[2], device=h.device, dtype=torch.float64)
    macro = base(x, y)
    ground_slope = torch.hypot((base(x+.75, y)-base(x-.75, y))/1.5,
                              (base(x, y+.75)-base(x, y-.75))/1.5)
    local = temperature-.0065*macro.clamp_min(0)
    snow = (1-_smooth(-5, 1, local))*(1-_smooth(.55, 1.3, ground_slope))
    exposed = 1-snow
    rock = _smooth(.25, .9, ground_slope)*exposed
    grass = _smooth(250, 1400, rain)*_smooth(-8, 4, local)*(1-_smooth(28, 42, local))*(exposed-rock)
    soil = exposed-rock-grass
    gain = _smooth(.025, .18, ground_slope)*_smooth(0, 2, macro)*(rock+.16*soil+.045*grass+.018*snow)
    ground = _ground_height(x, y, seed)*gain*strength
    height = torch.where(h > 0, (h+ground).clamp_min(0), h).float().contiguous()
    # Noise does not simulate runoff or deposition. Supply substrate guides
    # from this surface, without inventing hydraulic erosion measurements.
    surface = torch.stack((torch.zeros_like(rock), torch.zeros_like(rock),
                           _smooth(.45, 1.1, slope)*exposed, rock)).float().clamp(0, 1).contiguous()
    return height, surface


@torch.inference_mode()
def sample_magna(world, level, i1, j1, i2, j2, *, intensity=1., check=None):
    try:
        strength = float(intensity)
    except (TypeError, ValueError) as exc:
        raise ValueError('Magna intensity must be finite and in 0..2') from exc
    if not math.isfinite(strength) or not 0 <= strength <= 2:
        raise ValueError('Magna intensity must be finite and in 0..2')
    if (type(level) is not int or not 0 <= level <= 6 or
            any(type(v) is not int for v in (i1, j1, i2, j2)) or
            i2 <= i1 or j2 <= j1 or (i2-i1)*(j2-j1) > 1024**2):
        raise ValueError('Invalid or oversized Magna refinement window')
    if check is not None:
        check()
    if strength == 0 or level == 0:
        height = refinement.sample_refined(world, level, i1, j1, i2, j2, check=check)
        return height, torch.zeros((4, i2-i1, j2-j1), device=height.device, dtype=torch.float32)
    return _heights(world, level, (i1, j1, i2, j2), strength, check)
