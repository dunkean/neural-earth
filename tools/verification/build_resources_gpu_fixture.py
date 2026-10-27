"""Build tests/fixtures/resources_gpu_parity.json from the NumPy mirror.

Synthetic packages (continuous, seasonal, bounded, realized geology, cyclic, deposits)
and per-sample cases with their expected values. The WGSL parity check
(verify_resources_gpu_parity.cjs) evaluates the same cases on the GPU;
test_terrain_resources.py checks that this file matches the mirror.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
import terrain_resource_gpu as G  # noqa: E402
import terrain_resources as R  # noqa: E402

F16 = np.float16


def _f16(a):
    return np.asarray(a, np.float32).astype(F16).astype(np.float32)


def _bits(a):
    """Half-float bit patterns, as uploaded to r16float textures."""
    return np.asarray(a, np.float32).astype(F16).view(np.uint16).ravel().tolist()


def build():
    rng = np.random.default_rng(20261026)
    i0, j0, W, H = -8, -4, 16, 8
    lods = [11, 10, 7, 3, 0, -3]
    world_bounds = [-7.73 * 30720, -3.865 * 30720, 7.73 * 30720, 3.865 * 30720]

    def samples(n):
        out = []
        for _ in range(n):
            lod = int(rng.choice(lods))
            span = 2 ** (10 - lod)
            gi = int(rng.integers((i0 + 2) * span, (i0 + W - 2) * span)) if lod <= 10 else int(rng.integers(i0 + 2, i0 + W - 2) // 2)
            gj = int(rng.integers((j0 + 2) * span, (j0 + H - 2) * span)) if lod <= 10 else int(rng.integers(j0 + 2, j0 + H - 2) // 2)
            out.append((gi, gj, lod))
        out += [(-8 * (2 ** (10 - l)), 0, l) for l in (10, 7, 0)]
        out += [(8 * (2 ** (10 - l)), 0, l) for l in (10, 7, 0)]
        return out

    def header(kind, layers, **extra):
        h = dict(mode='resources-fixture', kind=G.KINDS[kind], lattice=[i0, j0, W, H], layers=layers, classes=0, seasonal=False,
                 range=[0., 1.], bounds=[0., 1.], bounded=False, log=False, profile=0, coef=[-1 / 6, 1., -.8, .3, .012, .3], sigma=.2,
                 seed32=0x5eed1234, layer=7, groups=[], order=[], thresholds=0, palette=[[.1, .2, .3], [.9, .8, .7]],
                 deposits=0, world_bounds=world_bounds, field_shape=[layers, H, W])
        h.update(extra)
        return h

    configs = []
    # Continuous: positive, bounded, seasonal.
    for name, kind, bounded, seasonal, rng_ in [('positive', 'ramp', False, False, [0., 2000.]),
                                                 ('bounded', 'ramp', True, False, [0., .3]),
                                                 ('seasonal', 'ramp', False, True, [0., 250.]),
                                                 ('potential-copper', 'log', False, False, [5e-4, 5.]),
                                                 ('potential-iron', 'log', False, False, [.05, 500.])]:
        k = 12 if seasonal else 1
        field = _f16(rng.uniform(.05, .95, (k, H, W)) * (rng_[1] if not bounded else 1))
        if name.startswith('potential-'):
            prospect = {'prospect_' + family: rng.uniform(0, 1, (H, W)) for family in {**R.FAMILIES, **R.PLACERS}}
            field = _f16(R.metal_potential(prospect, name.split('-')[1])[None])
        bounds = [0., 1.] if bounded else rng_
        if name.startswith('potential-'):
            bounded = True
            bounds = [0., sum(R.metal_potential_weights(name.split('-')[1]).values())]
        h = header(kind, k, bounded=bounded, seasonal=seasonal, range=rng_, bounds=bounds, log=kind == 'log')
        if name.startswith('potential-'):      # per-metal potential: the mineral profile, log scale
            h.update(coef=list(G.PROFILES['mineral'][0]), sigma=G.PROFILES['mineral'][1])
        pkg = dict(seed32=h['seed32'], layer=h['layer'], field=G.Field(field, i0, j0, world_bounds), seasonal=seasonal, season=.37,
                   profile='arable', range=tuple(rng_), bounded=bounded, bounds=tuple(bounds))
        G.PROFILES['fixture'] = (tuple(h['coef']), h['sigma'])
        coef_used = tuple(h['coef'])
        pkg['profile'] = 'fixture'
        cases = []
        for gi, gj, lod in samples(10):
            terms = float(rng.uniform(-1, 1))
            feat = dict(slope=np.float32(0), valley=np.float32(0), ridge=np.float32(0), north=np.float32(0), dz=np.float32(0),
                        river=np.float32(0))
            # Terms through the valley coefficient only; the GPU case receives the summed terms.
            feat['valley'] = np.float32(terms)
            v = G.evaluate_continuous(pkg, np.array(gi), np.array(gj), lod, feat)
            cases.append(dict(gi=gi, gj=gj, lod=lod, terms=float(G.terrain_terms(feat, coef_used)), expected=float(v)))
        configs.append(dict(name=name, header=h, season=.37, field=_bits(field), cases=cases))
    # Realized geology (rocks) and dominant class (contexts), on the same package.
    kcls = 8
    expected = rng.dirichlet(np.ones(kcls) * .7, (H, W)).transpose(2, 0, 1)
    theta = rng.uniform(0, np.pi, (H, W))
    logs = np.log(np.maximum(expected, 1e-4)) + rng.normal(0, .3, kcls)[:, None, None]
    field = _f16(np.concatenate((logs, np.cos(2 * theta)[None], np.sin(2 * theta)[None])))
    aligned = [False, True, True, True, False, True, True, True]
    groups, group_aligned = [0, 3, 3, 3, 1, 2, 2, 3], [False, False, True, False]
    h = header('mix', kcls + 2, classes=kcls, realized=True, beta=list(G.REALIZED_BETA),
               order=[int(a) << 4 | g for a, g in zip(aligned, groups)] + [int(a) for a in group_aligned],
               layer=4001, palette=[[c / 8, .5, 1 - c / 8] for c in range(8)])
    pkg = dict(seed32=h['seed32'], layer=4001, field=G.Field(field, i0, j0, world_bounds), classes=kcls, aligned=aligned,
               groups=groups, group_aligned=group_aligned, beta=G.REALIZED_BETA)
    cases = [dict(gi=gi, gj=gj, lod=lod, expected=int(G.realized_class(pkg, np.array([gi]), np.array([gj]), lod)[0]),
                  dominant=int(G.evaluate_dominant(pkg, np.array([gi]), np.array([gj]), lod)[0]))
             for gi, gj, lod in samples(32)]
    configs.append(dict(name='geology', header=h, field=_bits(field), cases=cases))
    # Cyclic (one neighbouring cell by bilinear weight).
    field = _f16(rng.integers(-1, 12, (1, H, W)))
    h = header('cyclic', 1)
    pkg = dict(seed32=h['seed32'], layer=h['layer'], field=G.Field(field, i0, j0, world_bounds))
    cases = [dict(gi=gi, gj=gj, lod=lod, expected=float(G.evaluate_cyclic(pkg, np.array([gi]), np.array([gj]), lod)[0]))
             for gi, gj, lod in samples(12)]
    configs.append(dict(name='cyclic', header=h, field=_bits(field), cases=cases))
    # Deposits: shapes and kernels around cell centres.
    deps = []
    for k in range(6):
        ci, cj = int(rng.integers(i0 + 3, i0 + W - 3)), int(rng.integers(j0 + 3, j0 + H - 3))
        deps.append(dict(di=ci * 1024 + int(rng.integers(0, 1024)), dj=cj * 1024 + int(rng.integers(0, 1024)), shape=k % 4,
                         color=k, a=float(rng.uniform(60, 3000)), b=float(rng.uniform(5, 400)), theta=float(rng.uniform(0, 3.1)),
                         access=float(rng.uniform(0, 1)), gossan=float(k % 2), symbol=float(rng.uniform(1.2, 4)),
                         sigma=float(rng.uniform(100, 2000)), mass=float(rng.uniform(10, 1e5))))
    deps.sort(key=lambda d: ((d['dj'] >> 10) - j0) * W + ((d['di'] >> 10) - i0))
    words = np.zeros((len(deps), 12), np.float32)
    ints = words.view(np.int32)
    for n, d in enumerate(deps):
        ints[n, :4] = [d['di'], d['dj'], d['shape'], d['color']]
        words[n, 4:] = [d['a'], d['b'], d['theta'], d['access'], d['gossan'], d['symbol'], d['sigma'], d['mass']]
        for key in ('a', 'b', 'theta', 'access', 'gossan', 'symbol', 'sigma', 'mass'):
            d[key] = float(words[n, 4 + ('a', 'b', 'theta', 'access', 'gossan', 'symbol', 'sigma', 'mass').index(key)])
    buckets = np.zeros((H, W, 2), np.uint32)
    cells = [((d['dj'] >> 10) - j0) * W + ((d['di'] >> 10) - i0) for d in deps]
    for n, c in enumerate(cells):
        if buckets.reshape(-1, 2)[c, 1] == 0:
            buckets.reshape(-1, 2)[c, 0] = n
        buckets.reshape(-1, 2)[c, 1] += 1
    palette = [[(c * 37 % 100) / 100, (c * 59 % 100) / 100, (c * 83 % 100) / 100] for c in range(20)]
    far = np.zeros((1, H, W), np.float32)       # stocks use deposit kernels at every LOD
    for kind in ('points', 'points-screen', 'stock', 'points-clusters'):
        if kind == 'points-clusters':
            previous_rng_state = rng.bit_generator.state
            deps = [dict(d, gossan=-count) for d, count in zip(deps, (2, 12, 108, 1658, 3, 25))]
            words[:, 8] = [d['gossan'] for d in deps]
        screen_mpp = 600. if kind == 'points-screen' else 0.
        h = header('points' if kind.startswith('points') else kind, 1, screen_mpp=screen_mpp,
                   palette=palette, deposits=len(deps), range=[1e-3, 1e5], max_footprint=max(max(d['a'], d['b']) * 2 for d in deps))
        stock_pkg = dict(field=G.Field(far, i0, j0, world_bounds))
        cases = []
        for n in range(36):
            d = deps[n % len(deps)]
            lod = ([10, 7, 6, 5, 4, 3, 2, 0, -2] if kind == 'stock' else [10, 7, 4, 2, 0, -2])[n % (9 if kind == 'stock' else 6)]
            r = 30. * 2. ** lod
            off = rng.normal(0, (2*r if kind == 'points-clusters' else max(d['a'], 3 * r) * .6), 2)
            gi = int(np.floor((d['di'] * 30. + 15 + off[0]) / r))
            gj = int(np.floor((d['dj'] * 30. + 15 + off[1]) / r))
            valley = float(rng.uniform(0, 1))
            if kind == 'stock':
                expected = float(G.stock_display(stock_pkg, deps, np.array(gi), np.array(gj), lod))
                cases.append(dict(gi=gi, gj=gj, lod=lod, valley=valley, expected=expected))
            else:
                rgb = G.deposit_color(deps, np.array(gi), np.array(gj), lod, np.float32(valley), palette, world_bounds, screen_mpp=screen_mpp)
                cases.append(dict(gi=gi, gj=gj, lod=lod, valley=valley, expected=[float(x) for x in rgb]))
        configs.append(dict(name=kind, header=dict(h, field_shape=[1, H, W]), field=_bits(far), deposits=words.view(np.uint32).ravel().tolist(),
                            buckets=buckets.ravel().tolist(), cases=cases))
        if kind == 'points-clusters':
            rng.bit_generator.state = previous_rng_state
    # Features and terms.
    feats = []
    for _ in range(8):
        e = float(rng.uniform(0, 3000))
        g, gl = rng.normal(0, .3, 2), rng.normal(0, .1, 2)
        small, large, r, river = e + float(rng.normal(0, 20)), e + float(rng.normal(0, 40)), float(rng.choice([30, 240, 3840])), float(rng.uniform(0, 1))
        f = G.features(np.float32(e), (np.float32(g[0]), np.float32(g[1])), (np.float32(gl[0]), np.float32(gl[1])),
                       np.float32(small), np.float32(large), r, np.float32(river))
        terms = G.terrain_terms(f, (-1 / 6, 1., -.8, .3, .012, .3))
        feats.append(dict(height=e, g=list(map(float, g)), gl=list(map(float, gl)), small=small, large=large, r=r, river=river,
                          expected=float(terms)))
    return dict(version=1, reference='backend/terrain_resource_gpu.py', configs=configs, features=feats,
                coef=[-1 / 6, 1., -.8, .3, .012, .3])


if __name__ == '__main__':
    out = ROOT / 'tests/fixtures/resources_gpu_parity.json'
    out.write_text(json.dumps(build(), indent=1), encoding='utf-8')
    print(out)
