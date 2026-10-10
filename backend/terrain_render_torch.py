"""CUDA (PyTorch) mirror of ``terrain_render.surface_material``, float32 operation for operation."""

import threading
import numpy as np
import torch
from terrain_device import current_cuda_index
from terrain_render import parse_settings, shore_distance, smooth as _smooth, NOISE_NORM, F

_M = 0xFFFFFFFF
_lock = threading.Lock()
_streams = {}
_consts = {}
_probe = []


def available():
    if not _probe:
        with _lock:
            if not _probe:
                try:
                    ok = torch.cuda.is_available()
                    if ok:
                        with torch.inference_mode():
                            ok = float((torch.ones(4, device=torch.device('cuda', current_cuda_index()))*2).sum().item()) == 8.
                except Exception:
                    ok = False
                _probe.append(bool(ok))
    return _probe[0]


def _stream(dev):
    with _lock:
        if dev not in _streams:
            _streams[dev] = torch.cuda.Stream(device=dev)
        return _streams[dev]


def _cuda(key, make):
    key = (key, torch.cuda.current_device())
    if key not in _consts:
        _consts[key] = make()
    return _consts[key]


def _k(v):
    """Constant colour lists become cached f32 tensors."""
    if isinstance(v, torch.Tensor):
        return v
    v = tuple(float(x) for x in np.asarray(v, F))
    return _cuda(v, lambda: torch.tensor(v, dtype=torch.float32, device='cuda'))


def div(x, d):
    """Division by a scalar via a device scalar: torch folds CPU-scalar divisors into a reciprocal multiply."""
    if not isinstance(d, torch.Tensor):
        d = float(d)
        d = _cuda(('s', d), lambda: torch.tensor(d, dtype=torch.float32, device='cuda'))
    return x/d


def sm(a, b, x):
    if isinstance(a, torch.Tensor):
        t = ((x-a)/(b-a)).clamp(0, 1)
    else:
        t = div(x-float(F(a)), F(b-a)).clamp(0, 1)
    return t*t*(3-2*t)


def mix(a, b, t):
    a, b = _k(a), _k(b)
    if isinstance(t, torch.Tensor):
        t = t.unsqueeze(-1)
        return a*(1-t)+b*t
    t = F(t)
    return a*float(F(1)-t)+b*float(t)


def _hash_terms(c, mult):
    return [(((c+o) & _M)*mult) & _M for o in (0, 1)]


def gradient_noise(p, salt):
    fl = torch.floor(p); f = p-fl
    cx, cy, cz = fl.to(torch.int64).unbind(-1)
    u = f*f*f*(f*(f*6-15)+10)
    ux, uy, uz = u.unbind(-1); fx, fy, fz = f.unbind(-1)
    tx, ty, tz = _hash_terms(cx, 0x9e3779b9), _hash_terms(cy, 0x85ebca6b), _hash_terms(cz, 0xc2b2ae35)
    salt = salt if isinstance(salt, torch.Tensor) else int(salt) & _M

    def corner(x, y, z):
        h = tx[x] ^ ty[y] ^ tz[z] ^ salt
        h = ((h ^ (h >> 16))*0x7feb352d) & _M
        h = ((h ^ (h >> 15))*0x846ca68b) & _M
        h = h ^ (h >> 16)
        g = lambda v: div((v & 1023).float(), F(511.5))-1
        return g(h)*(fx-x if x else fx)+g(h >> 10)*(fy-y if y else fy)+g(h >> 20)*(fz-z if z else fz)
    lerp = lambda a, b, t: a+(b-a)*t
    a = lerp(lerp(corner(0, 0, 0), corner(1, 0, 0), ux), lerp(corner(0, 1, 0), corner(1, 1, 0), ux), uy)
    b = lerp(lerp(corner(0, 0, 1), corner(1, 0, 1), ux), lerp(corner(0, 1, 1), corner(1, 1, 1), ux), uy)
    return lerp(a, b, uz)*float(F(NOISE_NORM))


def _octaves(scale, octaves, footprint, salt):
    """Resolved octaves (scale, weight, salt) and the (kept, total) variance sums."""
    result = []; kept = total = 0.; s = scale; a = 1.
    for i in range(octaves):
        total += a*a
        if footprint < s*.25:
            fade = 1-float(_smooth(s*.125, s*.25, footprint))
            result.append((s, a*fade, salt+i*0x9e37))
            kept += (a*fade)**2
        a *= .5; s *= .5
    return result, kept, total


def fbm_many(specs, footprint, budget=1 << 23):
    """Several ``fbm`` at once. Every resolved octave of every pattern is one
    slice of a single batched noise evaluation (a few hundred kernel launches
    in total rather than per octave); per-element operations are unchanged."""
    plans = [_octaves(scale, octaves, footprint, salt) for _, scale, octaves, salt in specs]
    jobs = [(j, point, s, salt) for j, ((point, *_), (octaves, _, _)) in enumerate(zip(specs, plans))
            for s, _, salt in octaves]
    noise = []
    if jobs:
        step = max(1, budget//jobs[0][1][..., 0].numel())
        for i in range(0, len(jobs), step):
            chunk = jobs[i:i+step]
            p = torch.stack([div(point, F(s)) for _, point, s, _ in chunk])
            salts = torch.tensor([int(salt) & _M for *_, salt in chunk], dtype=torch.int64,
                                 device=p.device).view(-1, *([1]*(p.dim()-2)))
            noise.extend(gradient_noise(p, salts).unbind(0))
    noise = iter(noise); result = []
    for (point, *_), (octaves, kept, total) in zip(specs, plans):
        value = point.new_zeros(point.shape[:-1])
        for _, weight, _ in octaves:
            value = value+float(F(weight))*next(noise)
        result.append(((div(value, F(np.sqrt(kept))) if kept > 0 else value), F(kept/total)))
    return result


def fbm(point, scale, octaves, footprint, salt):
    return fbm_many([(point, scale, octaves, salt)], footprint)[0]


def cover(fraction, pattern, sharpness=.35):
    value, detail = pattern
    c = fraction.clamp(0, 1)
    v = value/torch.sqrt(1+value*value)*float(np.sqrt(detail))
    return c+float(F(np.clip(sharpness, 0, 1)))*c*(1-c)*v


def _ocean(height, color, t_now, m1, var, snow_amount, phase):
    d = (-height).clamp(min=0)
    water = mix([.10, .30, .36], [.045, .16, .27], sm(5, 120, d))
    water = mix(water, [.018, .065, .15], sm(150, 3000, d))
    ice = (sm(-1.5, -6, t_now+F(.4)*m1)*F(snow_amount)).clamp(0, 1)
    water = mix(water, [.80, .84, .88], ice*F(.9))*(1+(F(.04)*var)*m1).unsqueeze(-1)
    return torch.where((height < 0).unsqueeze(-1), water, color).clamp(0, 1)


def _up(x, shape, dev, dtype=torch.float32):
    """Upload without materializing broadcast (zero-stride) axes."""
    a = np.asarray(x, np.float64 if dtype == torch.float64 else F)
    if a.size > 1:
        a = a[tuple(slice(0, 1) if st == 0 and n > 1 else slice(None) for n, st in zip(a.shape, a.strides))]
    a = np.ascontiguousarray(a)
    t = torch.from_numpy(a if a.flags.writeable else a.copy()).to(dev)
    return t if tuple(t.shape) == shape else t.expand(shape)


def surface_material(height, gradient, tpi, point, north, footprint, soil, seasons,
                     salt, settings=None, bare=False, pedology=None, shore=None, *, device=None):
    dev = torch.device('cuda', current_cuda_index()) if device is None else torch.device(device)
    wide = np.asarray(north).dtype != np.float32
    stream = _stream(dev)
    with torch.inference_mode(), torch.cuda.device(dev), torch.cuda.stream(stream):
        out = _surface_material(height, gradient, tpi, point, north, footprint, soil, seasons,
                                int(salt), settings, bare, pedology, shore, dev, wide)
        stream.synchronize()
        res = out.cpu().numpy()
        stream.synchronize()
    return res


def _surface_material(height, gradient, tpi, point, north, footprint, soil, seasons, salt, settings, bare, pedology, shore, dev, wide):
    s = parse_settings(settings)
    if shore is None:
        shore = shore_distance(height, float(footprint))
    var, forest_amount, rock_tan = F(s['variation']), F(s['forest']), F(np.tan(np.deg2rad(s['rock_slope'])))
    fp = float(footprint)
    shape = tuple(np.shape(height))
    up = lambda x: _up(x, shape, dev)
    material_detail = F(float(_smooth(30, 60, fp))*(1-float(_smooth(120, 240, fp))))
    phase = F(.5-.5*np.cos(2*np.pi*s['season']))
    height = up(height)
    h = height.clamp(min=0)
    point = _up(point, (*shape, 3), dev)
    macro, macro2, patch, crag, fine, province = fbm_many([
        (point, 64000., 4, salt+1), (point+_k([5101., -2297., 3313.]), 48000., 4, salt+2),
        (point, 4800., 4, salt+3), (point, 700., 4, salt+4), (point, 240., 4, salt+5),
        (point, 900000., 3, salt+6)], fp)
    sq = lambda f: f[0]*float(np.sqrt(f[1]))
    m1 = sq(macro); m2 = sq(macro2); pv = sq(province)
    ts0, ps, ls, tw0, pw, lw = (up(x) for x in seasons)
    tpi = up(tpi)
    ts = (ts0-ls*h).clamp(-45, 45); tw = (tw0-lw*h).clamp(-45, 45)
    locality = F(1-float(_smooth(600, 6000, fp)))
    terrain = sm(.015, .12, tpi.abs())
    jitter = F(.8)*m2*var*locality*(F(.25)+F(.75)*terrain)
    t_hot = torch.maximum(ts, tw)+jitter; t_cold = torch.minimum(ts, tw)+jitter
    t_now = tw+(ts-tw)*phase
    pth = (F(20.)*(ts+tw)*F(.5)+F(280.)).clamp(min=120.)
    wet_shift = torch.exp(F(.6)*F(s['moisture'])+F(.10)*var*m1*locality)
    valley = sm(0, -.12, tpi); ridge = sm(0, .12, tpi)
    humid = (ps+pw).clamp(min=0)/pth*wet_shift
    humid = humid*(1+F(.7)*valley*(1-sm(1.5, 3, humid))-F(.18)*ridge)
    humid_now = (pw+(ps-pw)*phase).clamp(min=0)*F(2.)/pth*wet_shift*(1+F(.7)*valley-F(.18)*ridge)
    gx, gy = (up(gradient[..., i]) for i in (0, 1))
    gnorm = torch.sqrt(gx*gx+gy*gy)
    grade = gnorm*(F(max(fp, 30.)/30.)**F(.3))
    gr = div(grade, rock_tan)
    arid = 1-sm(.35, 1., humid)
    w_trop = sm(13, 19, t_cold); w_boreal = 1-sm(-14, -3, t_cold)

    chans = lambda src, a, b: torch.stack([up(src[i]) for i in range(a, b)], -1)
    ground = chans(soil if bare or pedology is None else pedology, 0, 3)
    if bare:
        sand = mix([.80, .66, .46], [.70, .50, .33], sm(-.3, .5, pv))
        desert = mix([.55, .46, .36], sand, sm(-.4, .4, m2+F(1.5)*up(soil[6])-F(.5)-div(F(.8)*grade, rock_tan)))
        ground = mix(ground, desert, arid*F(.8))
        ground = mix(ground, [.50, .29, .17], w_trop*sm(1, 2, humid)*F(.6))
        steppe = (1-arid)*(1-w_trop)*(1-w_boreal)*sm(.8, 1.2, humid)*(1-sm(1.6, 2.4, humid))
        ground = mix(ground, [.25, .21, .16], steppe*F(.5))
    # Non-bare mixes above are weighted by exactly zero in the reference, so they are skipped.
    ground = ground*(1-F(.12)*valley+F(.08)*ridge).unsqueeze(-1)
    ground = ground*(1+var*(F(.035)*m1*locality+F(.025)*patch[0]*float(np.sqrt(patch[1]))+F(.02)*fine[0]*float(np.sqrt(fine[1])))).unsqueeze(-1)
    rock = mix([.50, .48, .45], [.30, .29, .28], sm(.15, .6, pv))
    rock = mix(rock, [.66, .62, .55], sm(-.15, -.6, pv))
    rock = mix(rock, mix([.36, .30, .25], [.60, .38, .25], sm(.1, .6, m2)), arid*F(.85))
    rock = mix(rock, chans(soil, 3, 6), F(.35))
    if not bare:
        neutral = (rock*_k([.2126, .7152, .0722])).unbind(-1)
        rock = ((neutral[0]+neutral[1])+neutral[2]).unsqueeze(-1)*_k([1.04, 1., .94])
    rock = rock*_k(s['rock_tint'])
    strata = torch.sin(div(F(2*np.pi)*(h+F(120.)*patch[0]), F(90.)))*F(1-float(_smooth(25, 45, fp)))
    rock = rock*(1+var*F(.10)*strata*(F(.15)+F(.85)*arid)+var*F(.07)*crag[0]*float(np.sqrt(crag[1]))).unsqueeze(-1)
    rock_tan_eff = rock_tan*(F(.55)+F(.45)*sm(.3, 1.3, humid))
    alpine = 1-sm(2, 9, t_hot)
    exposure = (sm(rock_tan_eff*.5, rock_tan_eff*1.3, grade)+F(.45)*alpine*sm(rock_tan_eff*.15, rock_tan_eff*.7, grade)
                +F(.35)*ridge*sm(.35, .9, gr)-F(.3)*valley)
    outcrop = cover(exposure.clamp(0, 1), (F(.75)*crag[0]+F(.25)*patch[0]*float(np.sqrt(patch[1])), crag[1]))
    color = mix(ground, rock, outcrop)
    if bare:
        return _ocean(height, color, t_now, m1, var, s['snow'], phase)

    tint = _k(s['vegetation_tint'])
    green = sm(.5, 1.4, humid_now)*sm(3, 12, t_now)
    grass = mix([.60, .53, .34], [.29, .37, .16], green)
    grass = mix(grass, [.40, .38, .27], 1-sm(7, 14, t_hot))*tint
    herbs = sm(.2, .85, humid)*sm(0, 7, t_hot)*(1-sm(rock_tan*.8, rock_tan*1.4, grade))
    sward = fine if fine[1] > 0 else patch
    color = mix(color, grass, cover(herbs*(1-outcrop), sward, .12*float(var)))
    dry_forest = 1-sm(1.3, 3, humid)
    canopy = mix([.20, .28, .14], [.16, .255, .13], w_trop)
    canopy = mix(canopy, [.15, .22, .16], w_boreal)
    canopy = mix(canopy, [.27, .29, .17], dry_forest*F(.75))
    deciduous = (1-w_trop)*(1-F(.7)*w_boreal)*(1-F(.5)*dry_forest)
    local = (float(F(s['season']))+torch.where(ts >= tw, 0., .5)) % 1
    autumn = sm(.66, .76, local)*(1-sm(.84, .92, local))
    canopy = mix(canopy, [.42, .27, .11], autumn*deciduous*F(.6))
    leafless = sm(8, 2, t_now)*deciduous
    canopy = mix(canopy, [.36, .33, .28], leafless*F(.75))
    highland = (1-sm(10, 17, t_hot))*material_detail
    canopy_detail = (F(.12)*highland*crag[0]*float(np.sqrt(crag[1]))+F(.025)*fine[0]*float(np.sqrt(fine[1]))).clamp(-.18, .18)
    canopy = canopy*tint*((1-F(.18)*highland)*(1+var*canopy_detail)).unsqueeze(-1)
    trees = (sm(.65, 2.1, humid)*sm(5.5, 13.5, t_hot+F(.8)*valley)*forest_amount*
             (1-sm(rock_tan*.9, rock_tan*1.6, grade))*(1-outcrop))
    trees = trees*(1-F(.22)*ridge)+F(.14)*valley*sm(2, 10, t_hot)*sm(.4, 1.2, humid)*(1-outcrop)*forest_amount
    woods = cover(trees.clamp(0, 1), patch, .65*float(var))
    color = mix(color, canopy, woods)
    flats = (1-sm(1.5, 6, height))*(1-sm(.05, .2, grade))
    reach = fbm(point, 20000., 3, fp, salt+7)
    width = F(25)+F(255)*sm(-.4, .6, sq(reach))
    band = div(width-up(shore)+float(F(fp)), F(fp)).clamp(0, 1)
    beach = band*(1-sm(4, 12, height))*(1-sm(.2, .5, grade))
    shingle = torch.maximum(1-sm(4, 14, t_hot), sm(.04, .22, grade)*terrain)
    sediment = mix([.80, .75, .61], [.76, .63, .43], sm(-.2, .9, pv))
    sediment = mix(sediment, [.86, .84, .76], w_trop*sm(0, -.8, pv)*F(.7))
    sediment = mix(sediment, rock, shingle*F(.9))
    sediment = mix(sediment, mix(ground, rock, F(.4)), sm(-.9, -1.5, m2)*F(.85))
    sediment = mix(sediment, [.27, .26, .25], sm(1.25, 1.75, m1+F(.35)*shingle)*F(.9))
    color = mix(color, sediment*F(.9), flats*F(1-float(_smooth(240, 960, fp)))*F(.8))
    color = mix(color, sediment, beach*F(.8))

    py = point[..., 1]
    pp = point*point
    hemisphere = py/torch.sqrt(py*py+F(.0064)*((pp[..., 0]+pp[..., 1])+pp[..., 2])).clamp(min=1)
    if wide:  # float64 aspect when the reference's north is not float32
        nx, ny = (_up(np.asarray(north, np.float64)[..., i], shape, dev, torch.float64) for i in (0, 1))
        aspect = (gx.double()*nx+gy.double()*ny)/gnorm.clamp(min=1e-5).double()*hemisphere.double()
        sun = (aspect*gnorm.clamp(max=1).double()).float()
    else:
        nx, ny = (up(north[..., i]) for i in (0, 1))
        sun = (gx*nx+gy*ny)/gnorm.clamp(min=1e-5)*hemisphere*gnorm.clamp(max=1)
    t_snow = t_now+F(2.2)*sun+F(.45)*m1*var*locality+F(.25)*ridge-F(.35)*valley
    supply = sm(60, 400, (ps+pw).clamp(min=0)*wet_shift*(1+F(.5)*valley-F(.25)*ridge))
    perennial = sm(2.5, -2.5, torch.maximum(ts, tw)+F(1.5)*sun+F(.35)*m1*var*locality)*supply
    seasonal = sm(1, -4, t_snow)*sm(.15, .5, humid)*supply
    snow = (torch.maximum(seasonal, perennial)*F(s['snow'])).clamp(0, 1)
    slip = sm(F(1.25)+F(.65)*perennial, F(3.5)+perennial, gnorm)
    snow = snow*(1-(F(.88)-F(.18)*perennial)*slip)*(1-F(.3)*woods*(1-perennial)*(1-F(.7)*leafless))
    color = mix(color, _k(s['snow_color']), cover(snow, crag, .35*float(var)))
    color = color*(1-F(.06)*valley).unsqueeze(-1)
    return _ocean(height, color, t_now, m1, var, s['snow'], phase)
