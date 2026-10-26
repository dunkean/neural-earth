"""Phasor comparison sheets, matched MSE views and warm CUDA timings.

Run from the repository root with .venv/Scripts/python.exe. Analytic reliefs
and decoder stubs are imported read-only from the MSE benchmark. Reference
PNG heightfields, when present, are not georeferenced survey DEMs: their
provenance and scale convention are in tests/fixtures/amplification/README.md.
"""
from pathlib import Path
import sys
import os
# Review jobs must never select or initialize the live server's GPU 1.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()

import argparse
from contextlib import nullcontext
import json
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from amplification_bench import RELIEFS, analytic_detail, fake_world, real_world, synchronize
import terrain_phasor as phasor


def fixture_world(path, device):
    values = np.asarray(Image.open(path)).astype(np.float64)
    if values.ndim == 3:
        values = values[..., :3] @ np.array([.2126, .7152, .0722])
    ranges = {'montserrat_256': (30, 1230), 'mountains': (100, 2500), 'new_zealand': (-40, 2160)}
    low, high = ranges[path.stem]
    values = low+(values-values.min())/max(float(np.ptp(values)), 1)*(high-low)
    dem = torch.tensor(values, device=device, dtype=torch.float32)
    h, w = dem.shape

    def get(i1, j1, i2, j2, *, with_climate):
        y = (torch.arange(i1, i2, device=device)+h//2).clamp(0, h-1)
        x = (torch.arange(j1, j2, device=device)+w//2).clamp(0, w-1)
        return {'elev': dem[y[:, None], x[None, :]]}

    return SimpleNamespace(device=device, get=get, seed=17, kwargs={'residual_std': 1.1678})


def hillshade(height, cell):
    """Four oblique lights; normalize multiple terrain panels together later."""
    dy, dx = np.gradient(height.astype(np.float64), cell)
    normal = np.sqrt(1+dx*dx+dy*dy)
    result = np.zeros_like(dx)
    for azimuth in (225, 270, 315, 360):
        angle = np.deg2rad(azimuth)
        result += np.clip((.7071-.7071*(np.sin(angle)*dx+np.cos(angle)*dy))/normal, 0, 1)
    return result/4


def shared_shades(heights, cell):
    shades = [hillshade(h, cell) for h in heights]
    low, high = np.percentile(np.concatenate([s.ravel() for s in shades]), [1, 99])
    return [np.repeat(np.clip(.08+.86*(s-low)/max(high-low, .01), 0, 1)[..., None], 3, -1) for s in shades]


def write_panels(path, panels, labels, title):
    h, w = panels[0].shape[:2]
    canvas = Image.new('RGB', (len(panels)*w, h+58), '#17212c')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill='white')
    for k, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(Image.fromarray((np.clip(panel, 0, 1)*255).astype(np.uint8)), (k*w, 58))
        draw.text((k*w+8, 31), label, fill='white')
    canvas.save(path)


def write_contact_sheet(paths, output):
    panels = [Image.open(path).convert('RGB') for path in paths]
    canvas = Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),'#17212c')
    top = 0
    for panel in panels:
        canvas.paste(panel,(0,top))
        top += panel.height
    canvas.save(output)


def sheet(path, before, after, surface, level, title):
    delta = np.abs(after-before)
    high = max(float(np.percentile(delta, 99)), .001)
    intensity = np.clip(delta/high, 0, 1)
    rgb = np.stack((surface[3]+.6*surface[2], surface[1]+.4*surface[2], surface[0]), -1)
    panels = shared_shades([before, after], 30/2**level)+[
        np.stack((intensity, intensity*.45, intensity*.12), -1), np.clip(rgb, 0, 1)]
    write_panels(path, panels, ['C2 + decoder base', 'Phasor ravines', f'|delta| p99={high:.2f} m',
                               'Flow B / deposit G / rock R + scree'], title)


def measure(call, device, repeats=1):
    synchronize(device)
    start = perf_counter()
    result = None
    for _ in range(repeats):
        result = call()
    synchronize(device)
    return result, (perf_counter()-start)/repeats


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64)**2)))


def appearance_panels(heights, surfaces, bounds, seed, profile):
    """Same Render material/lighting functions as the server, in-process on GPU 0."""
    from terrain_server import biome_atlas
    from terrain_render import colorize_surface
    from terrain_lighting import relief_intensity
    atlas, settings = biome_atlas(seed, profile)
    level, y0, x0, y1, x1 = bounds
    cell = 30/2**level
    xs = (np.arange(x0, x1)+.5)*cell
    ys = (np.arange(y0, y1)+.5)*cell
    panels = []
    for height, surface in zip(heights, surfaces):
        rgb = colorize_surface(atlas, xs, ys, height, cell, settings, surface=surface)
        rgb *= np.where(height < 0, 1, relief_intensity(height, cell, None))[..., None]
        panels.append(np.clip(rgb, 0, 1))
    return panels


def zoom_strip(path, world, title, centre=None):
    """Same 240 m square at -2/-4/-6; also measure all adjacent LODs."""
    arrays, transitions = {}, []
    previous = None
    for level in range(1, 7):
        half = 4*2**level
        # Anchor to the coarsest (15 m) grid once: independently rounding the
        # site at each LOD would shift the footprint and corrupt continuity.
        cy, cx = (0, 0) if centre is None else (round(centre[1]/15)*2**(level-1), round(centre[0]/15)*2**(level-1))
        args = (level, cy-half, cx-half, cy+half, cx+half)
        before = phasor._base(world, *args, None)
        after, _ = phasor.sample_phasor(world, *args)
        delta = after-before
        if previous is not None:
            down = lambda value: F.avg_pool2d(value[None, None], 2)[0, 0]
            drms = rms(delta.cpu().numpy())
            transitions.append(dict(lod=-level,
                                    height_rms_m=rms((down(after)-previous[1]).cpu().numpy()),
                                    base_rms_m=rms((down(before)-previous[0]).cpu().numpy()),
                                    detail_rms_m=rms((down(delta)-previous[2]).cpu().numpy()),
                                    detail_relative=rms((down(delta)-previous[2]).cpu().numpy())/max(drms, 1e-12)))
        if level in (2, 4, 6):
            arrays[level] = before.cpu().numpy(), after.cpu().numpy()
        previous = before, after, delta
    # All six panels share one percentile range and the same physical scale.
    raw = {(level, row): hillshade(arrays[level][row], 30/2**level)
           for level in arrays for row in range(2)}
    lo, hi = np.percentile(np.concatenate([value.ravel() for value in raw.values()]), [1, 99])
    pixels = 512
    canvas = Image.new('RGB', (3*pixels, 2*pixels+94), '#17212c')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title+' / same 240 m x 240 m footprint; shared shade percentiles', fill='white')
    for col, level in enumerate((2, 4, 6)):
        draw.text((col*pixels+8, 28), f'LOD -{level}: before (top) / phasor (bottom)', fill='white')
        for row in range(2):
            value = np.clip(.08+.86*(raw[level, row]-lo)/max(hi-lo, .01), 0, 1)
            image = Image.fromarray((value*255).astype(np.uint8)).convert('RGB')
            canvas.paste(image.resize((pixels, pixels), Image.Resampling.NEAREST), (col*pixels, 58+row*(pixels+18)))
    canvas.save(path)
    return transitions


def _surface_rgb(surface):
    return np.clip(np.stack((surface[3]+.6*surface[2],surface[1]+.4*surface[2],surface[0]),-1),0,1)


def hybrid_zoom_strip(path, world, title, centre=None):
    """Off / unchanged v3 / hybrid over exactly the same 240 m square."""
    from terrain_hybrid import sample_hybrid
    import terrain_refinement as refinement
    arrays, previous, transitions = {}, {}, []
    for level in range(1,7):
        half = 4*2**level
        cy,cx = (0,0) if centre is None else (round(centre[1]/15)*2**(level-1),round(centre[0]/15)*2**(level-1))
        bounds = level,cy-half,cx-half,cy+half,cx+half
        off = refinement.sample_refined(world,*bounds)
        p,_ = phasor.sample_phasor(world,*bounds,drainage=False)
        h,_ = sample_hybrid(world,*bounds)
        direct,_ = phasor.sample_phasor(world,*bounds,drainage=True)
        if not torch.equal(h,direct):
            raise RuntimeError(f'Hybrid halo crop differs from direct heights at LOD -{level}')
        base = phasor._base(world,*bounds,None)
        hybrid_base = phasor._drainage_base(world,*bounds)
        for mode,value in (('phasor-v3',p),('hybrid',h)):
            delta = value-(hybrid_base if mode == 'hybrid' else base)
            if mode in previous:
                down = lambda x: F.avg_pool2d(x[None,None],2)[0,0]
                d = max(rms(delta.cpu().numpy()),1e-12)
                transitions.append(dict(mode=mode,lod=-level,
                    height_rms_m=rms((down(value)-previous[mode][0]).cpu().numpy()),
                    detail_relative=rms((down(delta)-previous[mode][1]).cpu().numpy())/d))
            previous[mode] = value,delta
        if level in (2,4,6):
            arrays[level] = [v.cpu().numpy() for v in (off,p,h)]
    raw = {(level,row):hillshade(arrays[level][row],30/2**level) for level in arrays for row in range(3)}
    lo,hi = np.percentile(np.concatenate([v.ravel() for v in raw.values()]),[1,99])
    pixels = 512
    canvas = Image.new('RGB',(3*pixels,3*(pixels+22)+58),'#17212c')
    draw = ImageDraw.Draw(canvas)
    draw.text((8,5),title+' / same 240 m square / rows: off, phasor-v3, hybrid',fill='white')
    for col,level in enumerate((2,4,6)):
        draw.text((col*pixels+8,29),f'LOD -{level}',fill='white')
        for row in range(3):
            v = np.clip(.08+.86*(raw[level,row]-lo)/max(hi-lo,.01),0,1)
            image = Image.fromarray((v*255).astype(np.uint8)).convert('RGB')
            canvas.paste(image.resize((pixels,pixels),Image.Resampling.NEAREST),(col*pixels,58+row*(pixels+22)))
    canvas.save(path)
    return transitions


def run_hybrid_benchmark(args):
    """Recommended amplification comparison, using final-height materials."""
    import terrain_hybrid as hybrid
    import terrain_amplification as amplification
    import terrain_refinement as refinement
    phasor.USE_NATIVE_DRAINAGE = False
    args.out = args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    sites,skips = [],[]
    cases = [] if args.real_only else [(name,None,None) for name in args.reliefs]
    if args.seed is not None:
        sites = json.loads(args.sites.read_text())['real_sites']
        world = real_world(args.seed,args.real_profile)
        cases += [(f'real-{args.seed}-{site["name"]}',world,(site['x'],site['y'])) for site in sites]
    surface_available = callable(getattr(amplification,'sample_surface',None))
    if not surface_available:
        skips.append('Public sample_surface is unavailable: hybrid uses its drainage-phasor surface fallback.')
    rows,zooms,conservation = [],[],[]
    torch.set_num_threads(2)
    with torch.inference_mode():
        for name,supplied,centre in cases:
            world = supplied or fake_world(args.device,name)
            device = world.device
            context = nullcontext() if supplied is not None else analytic_detail(device,zero=name in ('plateau','glacier'))
            with context:
                for level in args.levels:
                    cy,cx = (0,0) if centre is None else (round(centre[1]/30*2**level),round(centre[0]/30*2**level))
                    y0,x0 = cy-args.size//2,cx-args.size//2
                    bounds = level,y0,x0,y0+args.size,x0+args.size
                    off,base_s = measure(lambda:refinement.sample_refined(world,*bounds),device)
                    (p,ps),pcold = measure(lambda:phasor.sample_phasor(world,*bounds,drainage=False),device)
                    (_, _),pwarm = measure(lambda:phasor.sample_phasor(world,*bounds,drainage=False),device,args.repeats)
                    (h,s),cold = measure(lambda:hybrid.sample_hybrid(world,*bounds),device)
                    (h,s),warm = measure(lambda:hybrid.sample_hybrid(world,*bounds),device,args.repeats)
                    direct,_ = phasor.sample_phasor(world,*bounds,drainage=True)
                    if not torch.equal(h,direct):
                        raise RuntimeError(f'{name} LOD -{level}: enlarged hybrid height read is inconsistent')
                    # Actual runtime overlap check, including A's materials.
                    part = hybrid.sample_hybrid(world,level,y0+args.size//3,x0+args.size//3,
                        y0+args.size//3+32,x0+args.size//3+32)
                    q = args.size//3
                    if not torch.equal(part[0],h[q:q+32,q:q+32]) or not torch.equal(part[1],s[:,q:q+32,q:q+32]):
                        raise RuntimeError(f'{name} LOD -{level}: real hybrid overlap check failed')
                    b,pn,hn,sn,psn = [v.cpu().numpy() for v in (off,p,h,s,ps)]
                    coherent = phasor._drainage_base(world,*bounds).cpu().numpy()
                    legacy_base = phasor._base(world,*bounds,None).cpu().numpy()
                    path = args.out/f'{name}_lod-{level}_hybrid.png'
                    write_panels(path,shared_shades([b,pn,hn],30/2**level)+[_surface_rgb(sn)],
                        ['Off','phasor-v3','Hybrid','Hybrid: flow B / deposit G / rock R + scree'],
                        f'{name}, LOD -{level}, {hybrid.VERSION}')
                    render_path = None
                    if supplied is not None:
                        panels = appearance_panels([b,pn,hn],[None,psn,sn],bounds,args.seed,args.real_profile)
                        render_path = args.out/f'{name}_lod-{level}_hybrid-render.png'
                        write_panels(render_path,panels+[_surface_rgb(sn)],
                                     ['Off','phasor-v3','Hybrid','Hybrid surface'],f'{name}, Render LOD -{level}')
                    np.savez_compressed(args.out/f'{name}_lod-{level}_hybrid.npz',off=b,phasor=pn,hybrid=hn,surface=sn)
                    rows.append(dict(relief=name,lod=-level,centre_metres=centre,
                        base_ms=base_s*1000,phasor_cold_ms=pcold*1000,phasor_warm_ms=pwarm*1000,
                        hybrid_cold_ms=cold*1000,hybrid_warm_ms=warm*1000,
                        hybrid_max_delta_m=float(np.abs(hn-b).max()),
                        hybrid_mean_delta_m=float((hn-b).mean()),hybrid_delta_rms_m=rms(hn-b),
                        hybrid_detail_mean_m=float((hn-coherent).mean()),hybrid_detail_rms_m=rms(hn-coherent),
                        legacy_native_frame_difference_rms_m=rms(coherent-legacy_base),
                        classic_vs_coherent_base_rms_m=rms(coherent-b),
                        surface_fraction={key:float((sn[k]>.5).mean()) for k,key in enumerate(hybrid.SURFACE_CHANNELS)},
                        sheet=str(path),render=str(render_path) if render_path else None))
                    print(f'{name:30} LOD {-level}: hybrid cold {cold*1000:.2f}, warm {warm*1000:.2f} ms',flush=True)
                path = args.out/f'{name}_hybrid-zoom.png'
                metrics = hybrid_zoom_strip(path,world,name,centre)
                zooms.append(dict(relief=name,sheet=str(path),transitions=metrics,
                    centre_metres=None if centre is None else [round(c/15)*15 for c in centre]))
                print('Zoom strip:',path.name,flush=True)
                # Native-aligned 3.84 km square: validate actual production
                # drainage, projection and material-halo composition together.
                cy,cx = (0,0) if centre is None else (round(centre[1]/30)*4,round(centre[0]/30)*4)
                large = 2,cy-256,cx-256,cy+256,cx+256
                large_h,_ = hybrid.sample_hybrid(world,*large)
                detail = large_h-phasor._drainage_base(world,*large)
                power = max(rms(detail.cpu().numpy()),1e-12)
                mean_ratio = abs(float(detail.double().mean()))/power
                means = F.avg_pool2d(detail[None,None],4)[0,0]
                native_ratio = rms(means.cpu().numpy())/power
                if mean_ratio >= .02 or native_ratio >= .15:
                    raise RuntimeError(f'{name}: hybrid conservation failed ({mean_ratio=}, {native_ratio=})')
                conservation.append(dict(relief=name,area_metres=3840,mean_over_rms=mean_ratio,
                                         native_cell_mean_over_rms=native_ratio))
    contacts = []
    for name in dict.fromkeys(row['relief'] for row in rows):
        group = [row for row in rows if row['relief']==name]
        path = args.out/f'{name}_hybrid-contact.png'
        write_contact_sheet([row['sheet'] for row in group],path)
        item = dict(relief=name,hillshade=str(path))
        if all(row['render'] for row in group):
            path = args.out/f'{name}_hybrid-render-contact.png'
            write_contact_sheet([row['render'] for row in group],path)
            item['render'] = str(path)
        contacts.append(item)
    report = dict(version=hybrid.VERSION,phasor_version=phasor.VERSION,
        drainage_version=phasor.PHASOR_DRAINAGE_VERSION,surface_version=hybrid.SURFACE_VERSION,
        surface_fallback=not surface_available,device=args.device,physical_gpu=0,
        exact_runtime_overlaps=True,exact_direct_vs_halo_heights=True,
        native_sampling='Globally anchored 128-cell square native reads; legacy v3 is unchanged.',
        gpu=torch.cuda.get_device_name(args.device) if torch.device(args.device).type=='cuda' else None,
        size=args.size,repeats=args.repeats,real_sites=sites,timings=rows,zooms=zooms,
        conservation=conservation,contacts=contacts,skips=skips,
        note='Off is exact sample_refined. Hybrid uses explicit drainage=True and final-height surface diagnostics. '
             'Cold is the first hybrid call in this process and may reuse persisted native/model caches; '
             'warm includes height halos and materials. No server requests.')
    if args.seed is not None:
        from distill.live_runtime import current
        report['models'] = current().identity()
    (args.out/'timings.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Hybrid sheets and JSON:',args.out,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('tmp/phasor'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--size', type=int, default=304)
    parser.add_argument('--levels', type=int, nargs='+', choices=(2, 4, 6), default=[2, 4, 6])
    parser.add_argument('--reliefs', nargs='+', default=list(RELIEFS))
    parser.add_argument('--no-fixtures', action='store_true')
    parser.add_argument('--keep-procedural', action='store_true')
    parser.add_argument('--seed', type=int, help='Also try a full installed runtime world')
    parser.add_argument('--real-only', action='store_true')
    parser.add_argument('--real-profile', default='orogen')
    parser.add_argument('--sites', type=Path, default=Path('tmp/amplification/timings.json'))
    parser.add_argument('--drainage', choices=('proxy', 'native', 'compare'), default='proxy')
    parser.add_argument('--hybrid', action='store_true', help='Compare off / unchanged phasor-v3 / hybrid with materials')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if not 8 <= args.size <= 1024 or args.repeats < 1:
        parser.error('size must be 8..1024; repeats must be positive')
    if args.real_only and args.seed is None:
        parser.error('--real-only requires --seed')
    unknown = set(args.reliefs)-set(RELIEFS)
    if unknown:
        parser.error(f'unknown reliefs: {unknown}')
    torch.set_num_threads(2)
    phasor.KEEP_PROCEDURAL_BASE = args.keep_procedural
    if args.hybrid:
        run_hybrid_benchmark(args)
        return
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    fixtures = Path('tests/fixtures/amplification')
    fixture_paths = {('montserrat' if p.stem == 'montserrat_256' else p.stem): p
                     for p in sorted(fixtures.glob('*.png'))} if fixtures.exists() else {}
    cases = [] if args.real_only else [(name, None, fixture_paths.get(name)) for name in args.reliefs]
    if fixtures.exists() and not args.no_fixtures and not args.real_only:
        cases += [(name, fixture_world(p, args.device), p) for name, p in fixture_paths.items()
                  if name not in RELIEFS]
    if args.no_fixtures:
        cases = [case for case in cases if case[2] is None]
    skips = []
    sites = []
    centres = {}
    if args.seed is not None:
        try:
            # In-process: do not contact/restart/reconfigure the live server.
            # Coordinates are the exact sites already verified by A and B.
            sites = json.loads(args.sites.read_text())['real_sites']
            runtime = real_world(args.seed, args.real_profile)
            for site in sites:
                name = f'real-{args.seed}-{site["name"]}'
                centres[name] = (site['x'], site['y'])
                cases.append((name, runtime, 'runtime'))
        except Exception as exc:
            skips.append(f'Runtime skipped: {type(exc).__name__}: {exc}')
            print(skips[-1], flush=True)
            if args.real_only:
                raise
    modes = (False, True) if args.drainage == 'compare' else (args.drainage == 'native',)
    cases = [(name+('-drainage' if mode else '-proxy') if args.drainage == 'compare' else name,
              world, source, centres.get(name), mode)
             for name, world, source in cases for mode in modes]
    start, stop = -args.size//2, -args.size//2+args.size
    rows, zooms, drainage_pairs = [], [], {}
    with torch.inference_mode():
        for name, supplied, source, centre, drainage in cases:
            phasor.USE_NATIVE_DRAINAGE = drainage
            world = supplied or fake_world(args.device, name)
            for level in args.levels:
                context = nullcontext() if source == 'runtime' else analytic_detail(args.device, zero=name == 'plateau')
                cy, cx = (0, 0) if centre is None else (round(centre[1]/30*2**level), round(centre[0]/30*2**level))
                bounds = (level, cy+start, cx+start, cy+stop, cx+stop)
                with context:
                    if torch.device(args.device).type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(args.device)
                    before, base_s = measure(lambda: phasor._base(world, *bounds, None), args.device)
                    (after, surface), cold_s = measure(lambda: phasor.sample_phasor(world, *bounds), args.device)
                    # One untimed warm-up after compilation/allocator growth.
                    phasor.sample_phasor(world, *bounds)
                    (after, surface), warm_s = measure(lambda: phasor.sample_phasor(world, *bounds), args.device, args.repeats)
                    gpu_mib = (torch.cuda.max_memory_allocated(args.device)/2**20
                               if torch.device(args.device).type == 'cuda' else 0)
                    b, a, s = [v.detach().cpu().numpy() for v in (before, after, surface)]
                    if args.drainage == 'compare':
                        key = (name.removesuffix('-proxy').removesuffix('-drainage'), level)
                        drainage_pairs.setdefault(key, {})[drainage] = (b, a, s, bounds, source)
                    path = args.out/f'{name}_lod-{level}.png'
                    sheet(path, b, a, s, level, f'{name} / LOD -{level} / {phasor.VERSION}')
                    # Explicitly inspect >1 depth scaling on the fine LODs.
                    if level in (4, 6):
                        variants = [phasor.sample_phasor(world, *bounds, intensity=value) for value in (.5, 2.)]
                        vh = [value[0].cpu().numpy() for value in variants]
                        write_panels(args.out/f'{name}_lod-{level}_intensity.png',
                                     shared_shades([b, vh[0], a, vh[1]], 30/2**level),
                                     ['Before', 'Intensity .5', 'Intensity 1', 'Intensity 2 (soft depth)'],
                                     f'{name}, LOD -{level}, {phasor.VERSION}')
                        if source == 'runtime':
                            panels = appearance_panels([b, vh[0], a, vh[1]],
                                [None, variants[0][1].cpu().numpy(), s, variants[1][1].cpu().numpy()],
                                bounds, args.seed, args.real_profile)
                            write_panels(args.out/f'{name}_lod-{level}_render-intensity.png', panels,
                                         ['Before', 'Intensity .5', 'Intensity 1', 'Intensity 2 (soft depth)'],
                                         f'{name}, Render LOD -{level}, {phasor.VERSION}')
                    comparison = None
                    # Always regenerate from A's CURRENT module. Existing
                    # PNGs neither gate nor supply any part of this column.
                    try:
                        from terrain_amplification import sample_amplified
                        if not hasattr(world, 'tile_store'):
                            from infinite_tensor import MemoryTileStore
                            world.tile_store = MemoryTileStore()
                            world._terrain_profile = SimpleNamespace(gpu_windows=True)
                        (mse, mse_surface), _ = measure(lambda: sample_amplified(world, *bounds), args.device)
                        comparison = args.out/f'{name}_lod-{level}_three-way.png'
                        write_panels(comparison, shared_shades([b, mse.cpu().numpy(), a], 30/2**level),
                                     ['C2 + decoder base', 'Multi-scale erosion (current module)', 'Phasor'],
                                     f'{name}, LOD -{level}; shared multidirectional shade percentiles')
                        if source == 'runtime':
                            panels = appearance_panels([b, mse.cpu().numpy(), a],
                                [None, mse_surface.cpu().numpy(), s], bounds, args.seed, args.real_profile)
                            write_panels(args.out/f'{name}_lod-{level}_render.png', panels,
                                         ['Before', 'Current MSE', phasor.VERSION], f'{name}, Render LOD -{level}')
                    except Exception as exc:
                        skips.append(f'MSE comparison {name}/{level}: {type(exc).__name__}: {exc}')
                        print(skips[-1], flush=True)
                row = dict(relief=name, lod=-level, base_ms=1000*base_s, cold_ms=1000*cold_s,
                           warm_ms=1000*warm_s, peak_gpu_mib=gpu_mib,
                           max_delta_m=float(np.abs(a-b).max()), mean_delta_m=float(np.abs(a-b).mean()),
                           signed_mean_delta_m=float((a-b).mean()), delta_rms_m=rms(a-b),
                           sheet=str(path), three_way=str(comparison) if comparison else None,
                           source=str(source) if source else 'analytic')
                row.update(centre_metres=centre, native_drainage=drainage)
                rows.append(row)
                print(f'{name:17} LOD {-level}: warm {row["warm_ms"]:7.2f} ms, '
                      f'max delta {row["max_delta_m"]:6.2f} m', flush=True)
            context = nullcontext() if source == 'runtime' else analytic_detail(args.device, zero=name == 'plateau')
            with context:
                path = args.out/f'{name}_zoom.png'
                metrics = zoom_strip(path, world, f'{name} / {phasor.VERSION}', centre)
            zooms.append(dict(relief=name, sheet=str(path), transitions=metrics,
                              centre_metres=None if centre is None else [round(c/15)*15 for c in centre]))
            print(f'{name:17} zoom strip: {path.name}', flush=True)
    comparisons = []
    for (name, level), pair in drainage_pairs.items():
        before, proxy, proxy_surface, bounds, source = pair[False]
        _, drainage, drainage_surface, _, _ = pair[True]
        path = args.out/f'{name}_lod-{level}_drainage-comparison.png'
        write_panels(path, shared_shades([before, proxy, drainage], 30/2**level),
                     ['Before', 'Proxy (default)', 'Native drainage'], f'{name}, LOD -{level}, {phasor.VERSION}')
        comparisons.append(str(path))
        if source == 'runtime':
            with torch.inference_mode():
                panels = appearance_panels([before, proxy, drainage], [None, proxy_surface, drainage_surface],
                                           bounds, args.seed, args.real_profile)
            path = args.out/f'{name}_lod-{level}_drainage-render.png'
            write_panels(path, panels, ['Before', 'Proxy (default)', 'Native drainage'],
                         f'{name}, Render LOD -{level}, {phasor.VERSION}')
            comparisons.append(str(path))
    report = dict(version=phasor.VERSION, device=args.device, size=args.size,
                  gpu=torch.cuda.get_device_name(args.device) if torch.device(args.device).type == 'cuda' else None,
                  fused=phasor._kernel(args.device) is not None,
                  procedural_base=phasor.KEEP_PROCEDURAL_BASE, wavelengths_m=phasor.WAVELENGTHS,
                  repeats=args.repeats, timings=rows, zooms=zooms, skips=skips,
                  project_native_mean=phasor.PROJECT_NATIVE_MEAN,
                  parameters=dict(frequency_ratio_max=1.25, orientation_jitter_degrees=18,
                                  gully_shares=phasor.GULLY_SHARE, profile_epsilons=phasor.PROFILE_EPSILONS,
                                  profile_cosine_shares=phasor.PROFILE_ROUNDING,
                                  patch_wavelengths=(3, 1.3), intensity_two_depth=phasor._intensity_gain(2)),
                  native_drainage=args.drainage, real_sites=sites,
                  drainage_comparisons=comparisons,
                  native_drainage_note='Optional public sample_drainage: area >= 28800 m2; capped 240 m channel distance.',
                  mse_version=getattr(sys.modules.get('terrain_amplification'), 'VERSION', None),
                  mse_note='All three-way MSE columns are evaluated from the imported current module; no PNG input.',
                  note='Warm includes native controls, C2/decoder base and all patterns; no persistent phasor cache. '
                       'Reference PNGs are not georeferenced DEMs. Decoder is an analytic stub except runtime cases.')
    if args.seed is not None:
        from distill.live_runtime import current
        report['models'] = current().identity()
    report['physical_gpu'] = 0
    (args.out/'timings.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('Sheets and JSON:', args.out)


if __name__ == '__main__':
    main()
