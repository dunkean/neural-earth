"""Time cached multi-scale erosion and render reproducible comparison sheets."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()

import argparse
from contextlib import nullcontext
import gc
import json
import os
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import torch

from test_terrain_amplification import (RELIEFS,analytic_detail,fake_world,lattice_spectrum,
                                       resolved_band_ratio,radial_spectrum,orientation_coherence)
from terrain_amplification import (ITERATIONS, GEOMETRY, DRAIN_ITERATIONS, VERSION,
                                   DETAIL_GAIN, DETAIL_CAP_METRES, PROJECTION_RADIUS, PROJECTION_SIGMA,
                                   TRANSFER_SIGMA, TRANSFER_ORDER,
                                   PROJECTION_CUTOFF, DETAIL_SMOOTH_RADIUS,
                                   AREA_REFERENCE_M2, EROSION_RATE,
                                   kernel_status, sample_amplified)
import terrain_refinement as refinement
import terrain_amplification as amplification


def surface_sheet(path, height, surface, level, title):
    """Fixed-opacity material overlays; no per-channel contrast normalisation."""
    h = height.detach().cpu().numpy()
    fields = surface.detach().cpu().numpy()
    raw = hillshade(h,30/2**level)
    lo,hi = np.percentile(raw,[2,98])
    gray = .5+.30*np.arctan((raw-(lo+hi)/2)/max((hi-lo)/2,.01))
    gray = np.repeat(gray[...,None],3,axis=2)
    colors = ((.1,.55,1.),(.15,.85,.25),(.95,.70,.2),(.95,.2,.3))
    panels = [gray]+[gray*(1-.7*f[...,None])+np.array(color)*(.7*f[...,None])
                     for f,color in zip(fields,colors)]
    rows,cols = h.shape
    crop = min(96,rows,cols)
    zoom = min(cols,crop*3)
    canvas = Image.new('RGB',(5*cols,rows+zoom+86),'#18212b')
    draw = ImageDraw.Draw(canvas)
    draw.text((8,5),title,fill='white')
    land = h>0
    fractions = [float(((f>.5)&land).sum()/max(1,land.sum())) for f in fields]
    labels = ['Final heights']+[f'{name}: >0.5 on {100*f:.2f}% land'
              for name,f in zip(amplification.SURFACE_CHANNELS,fractions)]
    for k,(panel,label) in enumerate(zip(panels,labels)):
        im = Image.fromarray((np.clip(panel,0,1)*255).astype(np.uint8))
        canvas.paste(im,(k*cols,56))
        draw.text((k*cols+8,29),label,fill='white')
        y,x = (rows-crop)//2,(cols-crop)//2
        canvas.paste(im.crop((x,y,x+crop,y+crop)).resize((zoom,zoom),Image.Resampling.NEAREST),
                     (k*cols,rows+86))
    draw.text((8,rows+64),f'{crop}-cell crop, {30/2**level:.4f} m/cell, 3x nearest; fixed 70% overlay',fill='white')
    canvas.save(path)


def benchmark_surfaces(world, level, bounds, name, out):
    """Callback time is measured separately; native drainage stays reusable."""
    modes = [('refined',lambda a,b,c,d: refinement.sample_refined(world,level,a,b,c,d),refinement.VERSION)]
    try:
        import inspect
        import terrain_phasor
        from terrain_phasor import sample_phasor, VERSION as phasor_version
        modes.append(('phasor',lambda a,b,c,d: sample_phasor(world,level,a,b,c,d)[0],phasor_version))
        if 'drainage' in inspect.signature(sample_phasor).parameters:
            modes.append(('phasor-drainage',lambda a,b,c,d: sample_phasor(world,level,a,b,c,d,drainage=True)[0],
                          terrain_phasor.PHASOR_DRAINAGE_VERSION))
    except ImportError as exc:
        print('Phasor surface comparison unavailable:',exc,flush=True)
    rows = []
    for mode,provider,height_version in modes:
        if torch.device(world.device).type=='cuda':
            torch.cuda.reset_peak_memory_stats(world.device)
        callback_seconds = [0.]
        def height_fn(a,b,c,d):
            value,seconds = timed(lambda: provider(a,b,c,d),world.device)
            callback_seconds[0] += seconds
            return value
        height = provider(*bounds)
        callback_seconds[0] = 0
        field,cold = timed(lambda: amplification.sample_surface(world,level,*bounds,height_fn),world.device)
        cold_height = callback_seconds[0]
        samples = []
        for _ in range(3):
            callback_seconds[0] = 0
            field,total = timed(lambda: amplification.sample_surface(world,level,*bounds,height_fn),world.device)
            samples.append((total-callback_seconds[0],total,callback_seconds[0]))
        _,warm,warm_height = sorted(samples)[1]
        path = out/f'{name}_surface-{mode}_lod-{level}.png'
        surface_sheet(path,height,field,level,f'{name}, {mode}, LOD -{level}, {amplification.SURFACE_VERSION}')
        data = out/f'{name}_surface-{mode}_lod-{level}.npz'
        np.savez_compressed(data,height=height.cpu().numpy(),surface=field.cpu().numpy())
        land = height>0
        fractions = {channel:float(((field[k]>.5)&land).sum().item()/max(1,land.sum().item()))
                     for k,channel in enumerate(amplification.SURFACE_CHANNELS)}
        row = dict(relief=name,lod=-level,height_mode=mode,height_version=height_version,
                   cold_total_s=cold,warm_total_s=warm,
                   cold_height_callback_s=cold_height,warm_height_callback_s=warm_height,
                   cold_surface_s=max(0,cold-cold_height),warm_surface_s=max(0,warm-warm_height),
                   warm_surface_samples_s=[s[0] for s in samples],warm_statistic='median of 3 surface overheads',
                   fractions_above_half=fractions,sheet=str(path),fields=str(data),
                   peak_gpu_mib=(torch.cuda.max_memory_allocated(world.device)/2**20
                                 if torch.device(world.device).type=='cuda' else 0),
                   callback_margin_cells=amplification._surface_margin(level))
        if mode=='phasor-drainage' and min(bounds[2]-bounds[0],bounds[3]-bounds[1])>48:
            inner = bounds[0]+24,bounds[1]+24,bounds[2]-24,bounds[3]-24
            shared = amplification.sample_surface(world,level,*inner,provider)
            row['real_overlap_exact'] = torch.equal(field[:,24:-24,24:-24],shared)
            row['real_overlap_max_error'] = float((field[:,24:-24,24:-24]-shared).abs().max())
        rows.append(row)
        print('  surface:',json.dumps(row),flush=True)
    return rows


def surface_overview(path, rows):
    names = list(dict.fromkeys(r['relief'] for r in rows))
    modes = list(dict.fromkeys(r['height_mode'] for r in rows))
    width,height = 600,180
    canvas = Image.new('RGB',(3*len(modes)*width,len(names)*(height+22)),'#18212b')
    draw = ImageDraw.Draw(canvas)
    for i,name in enumerate(names):
        y = i*(height+22)
        draw.text((6,y+3),name+' | '+' / '.join(modes)+' (-2/-4/-6 each) | height / flow / deposit / scree / rock',fill='white')
        for row in rows:
            if row['relief']!=name:
                continue
            col = 3*modes.index(row['height_mode'])+{-2:0,-4:1,-6:2}[row['lod']]
            with Image.open(row['sheet']) as im:
                canvas.paste(im.resize((width,height),Image.Resampling.LANCZOS),(col*width,y+22))
    canvas.save(path)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def timed(call, device):
    synchronize(device)
    start = perf_counter()
    result = call()
    synchronize(device)
    return result, perf_counter()-start


def rms(values):
    return values.double().square().mean().sqrt().item()


def continuity_metrics(world,centre):
    """Same block/footprint conventions as the independent phasor tests."""
    cy,cx = (0,0) if centre is None else (round(centre[1]/30),round(centre[0]/30))
    previous = None
    transitions = []
    for level in range(1,7):
        scale = 2**level
        y0,x0,y1,x1 = ((cy-4)*scale,(cx-4)*scale,(cy+4)*scale,(cx+4)*scale)
        h,_ = sample_amplified(world,level,y0,x0,y1,x1)
        base = refinement.sample_refined(world,level,y0,x0,y1,x1)
        delta = h-base
        if previous is not None:
            coarse = torch.nn.functional.avg_pool2d(h[None,None],2)[0,0]
            coarse_delta = torch.nn.functional.avg_pool2d(delta[None,None],2)[0,0]
            coarse_base = torch.nn.functional.avg_pool2d(base[None,None],2)[0,0]
            transitions.append(dict(level=level,delta_rms_m=rms(delta),
                delta_restriction_rms_m=rms(coarse_delta-previous[1]),
                delta_restriction_relative=rms(coarse_delta-previous[1])/max(rms(delta),1e-9),
                full_height_transition_rms_m=rms(coarse-previous[0]),
                baseline_quadrature_rms_m=rms(coarse_base-previous[2])))
        previous = h,delta,base
    # Native-aligned 3.84 km square at LOD -2; this is a fixed physical area,
    # not a per-request recentering or quantile calibration of the algorithm.
    level = 2
    bounds = ((cy-64)*4,(cx-64)*4,(cy+64)*4,(cx+64)*4)
    h,_ = sample_amplified(world,level,*bounds)
    base = refinement.sample_refined(world,level,*bounds)
    delta = h-base
    return dict(area_size_metres=3840,mean_delta_m=delta.double().mean().item(),delta_rms_m=rms(delta),
        mean_over_rms=abs(delta.double().mean().item())/max(rms(delta),1e-9),
        native_block_mean_relative=rms(torch.nn.functional.avg_pool2d(delta[None,None],4))/max(rms(delta),1e-9),
        native_resolved_lowpass_relative=resolved_band_ratio(delta,4),
        transitions=transitions)


def hillshade(height, cell):
    """Multidirectional diffuse light; physical gradients, gentle fine-LOD z gain."""
    z_factor = float(np.clip(np.sqrt(cell/7.5), .5, 1.5))
    dy, dx = np.gradient(height.astype(np.float64)*z_factor, cell)
    norm = np.sqrt(1+dx*dx+dy*dy)
    shade = np.zeros_like(height, dtype=np.float64)
    for azimuth, weight in ((315, .4), (45, .3), (135, .15), (225, .15)):
        angle = np.deg2rad(azimuth)
        dot = (-np.sin(angle)*dx*.766-np.cos(angle)*dy*.766+.643)/norm
        shade += weight*(dot+1)**2
    return shade


def sheet(path, before, after, surface, level, title):
    before, after, surface = [x.detach().cpu().numpy() for x in (before, after, surface)]
    delta = np.abs(after-before)
    p99 = float(np.percentile(delta,99))
    high = max(p99,.001)
    intensity = np.clip(delta/high, 0, 1)
    # Flow blue, deposition green, exposed rock red; scree adds warm colour.
    rgb = np.stack((surface[3]+.6*surface[2], surface[1]+.4*surface[2], surface[0]), -1)
    raw_before, raw_after = hillshade(before, 30/2**level), hillshade(after, 30/2**level)
    lo, hi = np.percentile(raw_before, [2, 98])
    centre, scale = (lo+hi)/2, max((hi-lo)/2, .01)
    # Shared before-derived percentile mapping, with arctan tails instead of
    # clipped highlights/shadows. After cannot choose its own contrast scale.
    normalize = lambda raw: np.repeat((.5+np.arctan((raw-centre)/scale)*.30)[..., None], 3, axis=2)
    panels = [normalize(raw_before), normalize(raw_after),
              np.stack((intensity, intensity*.45, intensity*.12), -1), np.clip(rgb, 0, 1)]
    h, w = before.shape
    crop_size = min(96, h, w)
    crop_pixels = min(w, crop_size*3)
    canvas = Image.new('RGB', (4*w, h+crop_pixels+86), '#18212b')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill='white')
    labels = ['Before (intensity 0)', 'After (intensity 1)', f'|delta| p99={p99:.3g} m; scale={high:.3g} m',
              'Surface: rock R / deposit G / flow B + scree']
    for k, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(Image.fromarray((np.clip(panel, 0, 1)*255).astype(np.uint8)), (k*w, 56))
        draw.text((k*w+8, 29), label, fill='white')
        y0, x0 = (h-crop_size)//2, (w-crop_size)//2
        crop = Image.fromarray((np.clip(panel[y0:y0+crop_size, x0:x0+crop_size], 0, 1)*255).astype(np.uint8))
        canvas.paste(crop.resize((crop_pixels, crop_pixels), Image.Resampling.NEAREST), (k*w, h+86))
    draw.text((8, h+64), f'Unfiltered cell crop: {crop_size} cells, {30/2**level:.4f} m/cell, enlarged 3x', fill='white')
    canvas.save(path)


def spectrum_sheet(path, delta, level, title, before=None):
    """Signed-delta spectrum: distinguish an octave from narrow lattice lines."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = delta.detach().double().cpu().numpy()
    z = (z-z.mean())*np.outer(np.hanning(z.shape[0]),np.hanning(z.shape[1]))
    power = np.abs(np.fft.fft2(z))**2
    maximum = max(float(power.max()),1e-30)
    db = 10*np.log10(np.maximum(power/maximum,1e-10))
    fig,(image,axis,radial_axis) = plt.subplots(1,3,figsize=(15,4),layout='constrained')
    image.imshow(np.fft.fftshift(db),origin='lower',extent=(-.5,.5,-.5,.5),vmin=-80,vmax=0,cmap='magma')
    image.set(xlabel='x cycles / fine cell',ylabel='y cycles / fine cell',title='Signed delta power (dB)')
    for label,p in (('x',power.sum(0)),('y',power.sum(1))):
        n = len(p)
        axis.plot(np.arange(n//2+1)/n,10*np.log10(np.maximum(p[:n//2+1]/max(p.max(),1e-30),1e-10)),label=label)
    for parent in sorted(set(s for s in (0,2,4,level-1) if s<level)):
        axis.axvline(2**(parent-level),color='gray',alpha=.5,linestyle=':')
    metric = lattice_spectrum(delta,level)
    axis.set(xlim=(0,.5),ylim=(-80,2),xlabel='cycles / fine cell',ylabel='relative axial power (dB)',
             title=f"Parent-line excess max {metric['max_peak_ratio']:.2f}x; gate 4x")
    axis.legend()
    n = min(z.shape)
    radius = np.rint(np.hypot(np.fft.fftfreq(z.shape[0])[:,None],
                             np.fft.fftfreq(z.shape[1])[None,:])*n).astype(int)
    radial = np.bincount(radius.ravel(),power.ravel())/np.maximum(1,np.bincount(radius.ravel()))
    radial_axis.plot(np.arange(n//2+1)/n,10*np.log10(np.maximum(
        radial[:n//2+1]/max(radial.max(),1e-30),1e-10)),color='darkorange')
    peak = radial_spectrum(delta,before)
    if 0<peak['relative_power_floor']<1:
        radial_axis.axhline(10*np.log10(peak['relative_power_floor']),color='gray',linestyle=':',
                            label='diagnostic / quantisation floor')
        radial_axis.legend(fontsize=8)
    radial_axis.set(xlim=(0,.5),ylim=(-80,2),xlabel='radial cycles / fine cell',
                    ylabel='relative radial mean power (dB)',
                    title=f"Narrow radial peak {peak['max_peak_neighbour_ratio']:.2f}x; gate 2x (above floor)")
    fig.suptitle(title)
    fig.savefig(path,dpi=130)
    plt.close(fig)


def native_sheet(path, height, title):
    """Unscaled 30 m DEM context, with the LOD -2 comparison footprint marked."""
    raw = hillshade(height.detach().cpu().numpy(), 30.)
    lo, hi = np.percentile(raw, [2, 98])
    gray = .5+.30*np.arctan((raw-(lo+hi)/2)/max((hi-lo)/2, .01))
    rgb = np.repeat((gray*255).astype(np.uint8)[..., None], 3, axis=2)
    canvas = Image.new('RGB', (height.shape[1], height.shape[0]+42), '#18212b')
    canvas.paste(Image.fromarray(rgb), (0,42))
    draw = ImageDraw.Draw(canvas)
    draw.text((5,4), title, fill='white')
    draw.text((5,21), '30 m/cell, native pixels; red = -2 footprint', fill='white')
    centre = height.shape[0]//2
    draw.rectangle((centre-38,42+centre-38,centre+38,42+centre+38), outline='#ff6655')
    canvas.save(path)


def comparison_overview(path, rows, size):
    names = list(dict.fromkeys(row['relief'] for row in rows))
    thumb_w, thumb_h = 2*size, (size+56)//2
    canvas = Image.new('RGB', (3*thumb_w, len(names)*(thumb_h+22)), '#18212b')
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(names):
        top = index*(thumb_h+22)
        draw.text((6,top+4), name+'   |   LOD -2 / -4 / -6; before / after / delta / surface', fill='white')
        for row in rows:
            if row['relief'] == name:
                with Image.open(row['sheet']) as panel:
                    thumb = panel.crop((0,0,4*size,size+56)).resize((thumb_w,thumb_h),Image.Resampling.LANCZOS)
                column = {-2:0,-4:1,-6:2}[row['lod']]
                canvas.paste(thumb,(column*thumb_w,top+22))
    canvas.save(path)


def real_world(seed, profile='orogen'):
    """Optional runtime path: never download assets or build native adapters."""
    from terrain_paths import OUTPUT_ROOT, model_snapshot
    snapshot = model_snapshot('9ef8030cb805b433b98ec25c5dddefbac07a9e26')
    required = [snapshot/'config.json'] + [snapshot/stage/name for stage in
                ('coarse_model', 'base_model', 'decoder_model') for name in
                ('config.json', 'diffusion_pytorch_model.safetensors')]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f'Installed model snapshot missing: {snapshot}')
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    # terrain_app has explicit raster/checkpoint preflight; errors are reported
    # as a skipped real case while the analytic benchmarks remain available.
    os.environ.setdefault('TERRAIN_PREWARM', '0')
    try:
        from terrain_server import get_world
        world = get_world(seed, profile, foreground=False)
    finally:
        # terrain_app changes cwd even when a later runtime import fails.
        activate()
    return world


def real_sites(seed, profile):
    """Revalidate historical screenshots against the current source identity.

    Old screenshot locations can become ocean when generator provenance
    changes. Choose equivalent land regimes from this seed's current atlas,
    and retain both the old coordinates and physical selection evidence.
    """
    from terrain_conditioning import make_conditioning_factory
    from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter
    factory = make_conditioning_factory(seed, profile)
    atlas = factory.heightmap
    h, w = atlas.height_m.shape
    x0, y0, x1, y1 = atlas.bounds
    xs = x0+(np.arange(w)+.5)*(x1-x0)/w
    ys = y0+(np.arange(h)+.5)*(y1-y0)/h
    fields = factory.sample(xs, ys)
    height, rain = fields[0], fields[3]
    relief = maximum_filter(height, 5)-minimum_filter(height, 5)
    distance = distance_transform_edt(height > 0)
    middle = np.broadcast_to((np.abs(ys[:, None]) < 7_000_000), height.shape).copy()
    middle[:8] = False; middle[-8:] = False; middle[:, :8] = False; middle[:, -8:] = False
    regimes = {
        'high-mountains': ((height > 2500)&(height < 5000), relief+.15*height),
        'coastal-mountains': ((height > 200)&(height < 2200)&(distance < 2.5), relief+.2*height),
        'desert-escarpment': ((height > 400)&(height < 3000)&(rain < 300), relief*(1-np.minimum(rain, 300)/600)),
    }
    historical = json.loads(Path('docs/images/screenshots.json').read_text())['sites']
    result = []
    for (name, (mask, score)), old_name in zip(regimes.items(), ('snow', 'coast', 'desert')):
        valid = mask&middle
        if not valid.any():
            raise RuntimeError(f'No {name} region in seed {seed}, profile {profile}')
        row, col = np.unravel_index(np.argmax(np.where(valid, score, -np.inf)), height.shape)
        item = dict(name=name, x=float(xs[col]), y=float(ys[row]), source_height_m=float(height[row,col]),
                    source_rain_mm=float(rain[row,col]), regional_relief_m=float(relief[row,col]),
                    source_identity=atlas.metadata.get('height_sha256', atlas.cache_key),
                    historical_site=historical[old_name])
        result.append(item)
        print('Real site:', json.dumps(item), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('tmp/amplification'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, help='Also try a real runtime world')
    parser.add_argument('--real-profile', default='orogen')
    parser.add_argument('--real-only', action='store_true')
    parser.add_argument('--surface-only', action='store_true',help='Skip experimental MSE heights; compare final-height surfaces')
    parser.add_argument('--size', type=int, default=304)
    parser.add_argument('--reliefs', nargs='+', choices=RELIEFS, default=list(RELIEFS))
    parser.add_argument('--levels', nargs='+', type=int, choices=(2, 4, 6), default=[2, 4, 6])
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not 8 <= args.size <= 1024:
        parser.error('--size must be in 8..1024')
    torch.set_num_threads(args.threads)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device
    if torch.device(device).type == 'cuda':
        torch.zeros(1, device=device)
        print('GPU:', torch.cuda.get_device_name(device))
    rows, skips, selected_sites, native_sheets, continuity, surfaces = [], [], [], [], [], []
    cases = [] if args.real_only else [(name, None, None) for name in args.reliefs]
    if args.seed is not None:
        try:
            world = real_world(args.seed, args.real_profile)
            selected_sites = real_sites(args.seed, args.real_profile)
            for site in selected_sites:
                cases.append((f'real-{args.seed}-{site["name"]}', world, (site['x'], site['y'])))
        except Exception as exc:
            skips.append(f'Real world skipped: {type(exc).__name__}: {exc}')
            print(skips[-1])
    print('relief      LOD  base(s) cold(s) warm(s) peak GPU(MiB) max|delta|(m)')
    torch.set_num_threads(args.threads)  # Model loader may set its own CPU budget.
    with torch.inference_mode():
        for name, real, centre in cases:
            world = real or fake_world(device,name)
            for level in args.levels:
                # Each stage timing starts with a fresh store: cold includes
                # all required coarser erosion windows and baseline sampling.
                case_device = world.device
                if level == args.levels[0] and (real is not None or name in ('montserrat','mountains','new_zealand')):
                    ny, nx = (0,0) if centre is None else (round(centre[1]/30),round(centre[0]/30))
                    with nullcontext() if real is not None else analytic_detail(case_device):
                        native, _ = sample_amplified(world, 0, ny-128, nx-128, ny+128, nx+128, intensity=0)
                    native_path = args.out/f'{name}_native-30m.png'
                    native_sheet(native_path, native, name)
                    native_sheets.append(str(native_path))
                    del native
                for tensor in getattr(world,'_terrain_amplification_fields',{}).values():
                    tensor.clear_cache()
                context = nullcontext() if real is not None else analytic_detail(device, zero=name in ('plateau','glacier'))
                cy, cx = (0, 0) if centre is None else (round(centre[1]/30*2**level), round(centre[0]/30*2**level))
                y0, x0 = cy-args.size//2, cx-args.size//2
                read = lambda strength: sample_amplified(world, level, y0, x0, y0+args.size, x0+args.size,
                                                         intensity=strength)
                if args.surface_only:
                    with context:
                        surfaces.extend(benchmark_surfaces(world,level,(y0,x0,y0+args.size,x0+args.size),name,args.out))
                    continue
                if torch.device(case_device).type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(case_device)
                with context:
                    (before, _), base_seconds = timed(lambda: read(0), case_device)
                    (after, surface), cold_seconds = timed(lambda: read(1), case_device)
                    (_, _), warm_seconds = timed(lambda: read(1), case_device)
                memory = (torch.cuda.max_memory_allocated(case_device)/2**20
                          if torch.device(case_device).type == 'cuda' else 0)
                path = args.out / f'{name}_lod-{level}.png'
                sheet(path, before, after, surface, level, f'{name}, LOD -{level}, {VERSION}')
                spectrum_path = args.out/f'{name}_spectrum_lod-{level}.png'
                spectrum_sheet(spectrum_path,after-before,level,f'{name}, LOD -{level}, {VERSION}',before)
                delta = (after-before).abs()
                land = before>0
                fractions = {channel: ((surface[k]>.5)&land).sum().item()/max(1,land.sum().item())
                             for k,channel in enumerate(('flow','deposit','scree','rock'))}
                support = {channel: ((surface[k]>.1)&land).sum().item()/max(1,land.sum().item())
                           for k,channel in enumerate(('flow','deposit','scree','rock'))}
                with nullcontext() if real is not None else analytic_detail(device,zero=name in ('plateau','glacier')):
                    inherited = (amplification._upsample(world,level-2,level,y0,x0,y0+args.size,x0+args.size)[0]
                                 if level>2 else torch.zeros_like(before))
                increment = after-before-inherited
                fields_path = args.out/f'{name}_fields_lod-{level}.npz'
                np.savez_compressed(fields_path,before=before.cpu().numpy(),after=after.cpu().numpy(),
                                    inherited=inherited.cpu().numpy(),surface=surface.cpu().numpy())
                row = dict(relief=name, lod=-level, base_s=base_seconds, cold_s=cold_seconds,
                           warm_s=warm_seconds, peak_gpu_mib=memory, max_delta_m=delta.max().item(),
                           mean_delta_m=(after-before).double().mean().item(),
                           mean_abs_delta_m=delta.mean().item(), delta_rms_m=rms(after-before),
                           p99_delta_m=torch.quantile(delta, .99).item(),
                           centre_metres=centre, device=str(case_device),
                           height_min_m=before.min().item(), height_max_m=before.max().item(),
                           surface_fraction_above_half=fractions,
                           surface_fraction_above_point_one=support,
                           increment_p99_m=torch.quantile(increment.abs(),.99).item(),
                           increment_rms_m=rms(increment),increment_max_m=increment.abs().max().item(),
                           lattice_spectrum=lattice_spectrum(after-before,level),
                           radial_spectrum=radial_spectrum(after-before,before),
                           orientation=orientation_coherence(after-before,before+inherited,30/2**level),
                           spectrum_sheet=str(spectrum_path.resolve()),
                           fields=str(fields_path.resolve()),
                           sheet=str(path.resolve()))
                rows.append(row)
                print(f'{name:11} {-level:3} {base_seconds:7.3f} {cold_seconds:7.3f} '
                      f'{warm_seconds:7.3f} {memory:13.1f} {row["max_delta_m"]:13.4f}', flush=True)
                print('  surface >0.5, land fraction:',json.dumps(fractions),flush=True)
                print('  lattice peak/background:',row['lattice_spectrum']['max_peak_ratio'],flush=True)
                print('  radial peak / orientation:',row['radial_spectrum'],row['orientation'],flush=True)
                with nullcontext() if real is not None else analytic_detail(device,zero=name in ('plateau','glacier')):
                    surfaces.extend(benchmark_surfaces(world,level,(y0,x0,y0+args.size,x0+args.size),name,args.out))
                del before, after, surface, delta, read, inherited, increment
                gc.collect()
            if not args.surface_only:
                context = nullcontext() if real is not None else analytic_detail(device,zero=name in ('plateau','glacier'))
                with context:
                    metrics = continuity_metrics(world,centre)
                continuity.append(dict(relief=name,**metrics))
                print('  continuity:',json.dumps(dict(relief=name,mean_over_rms=metrics['mean_over_rms'],
                      native_block_mean_relative=metrics['native_block_mean_relative'])),flush=True)
            del world
            gc.collect()
    report = dict(version=VERSION, device=device, size=args.size,
                  surface_version=amplification.SURFACE_VERSION,surfaces=surfaces,
                  surface_only=args.surface_only,
                  surface_parameters=dict(final_gaussian_sigma_cells=1.1,broad_sigma_cells=3.,
                      native_source_read_square_cells=128,
                      native_geometry_cache_blocks=32,native_geometry_cache_max_mib=2,
                      native_log_area_reference_m2=900,native_anti_image_sigma_cells=.45,
                      native_interpolation='positive C2 B-spline, no inverse prefilter',
                      native_flow_log_gate=[3.3,6.],native_flow_contrast_gate=[.08,.45],
                      fine_concavity_gate='[.001+.0005*cell, .020+.008*cell]',
                      cross_slope_ridge_gate=[.0003,.006],longitudinal_averages=3,
                      flow_confidence_gate='[.10+.012*cell, .36+.045*cell]',
                      water_log_gate=[1.8,4.7],deposit_slope_gate=[.45,.85],
                      rock_slope_gate=[.55,.95],rock_smooth_slope_jitter=.07,
                      rock_noise_metres=[32,96],rock_noise_weights=[.65,.35],
                      rock_patch_gate=[-.15,.35],apron_steps=6,apron_step='max(cell,2 metres)',
                      apron_decay=.85,scree_apron_gate=[.09,.48],
                      scree_slope_rise=[.40,.53],scree_slope_fall=[.75,.93],
                      native_datums_define_land=True,final_mask_cache=False),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  gpu=torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else None,
                  geometry=GEOMETRY, iterations=ITERATIONS, drainage_iterations=DRAIN_ITERATIONS,
                  stencil_reach_cells={s:amplification._reach(s) for s in GEOMETRY},
                  kernels=kernel_status(),
                  parameters=dict(flow_exponent=1.3,area_exponent=.5,slope_exponent=1.6,
                      area_reference_square_metres=AREA_REFERENCE_M2,
                      routing_exponent_smooth_jitter=.18,parent_direction_prior_gain=8,
                      incision_coefficient=EROSION_RATE,
                      relief_share={2:.38,4:.28,6:.22},cap_cell_widths=3,
                      noise_metres=[180,48,12],noise_weights=[.5,.35,.15],
                      strata='aperiodic height/physical-noise value field',strata_vertical_metres=12,
                      hydraulic_erodibility_floor=.04,hydraulic_inherited_channel_share=.96,
                      repose_tangent_mean=.72,repose_noise_gain=.16,thermal_rate=.035,
                      sediment_capacity=.025,settling_rate=.45,coast_fade_metres=12,
                      curvature_gate=[.0003,.006],increment_slope_per_axis=.5,
                      slope_envelope_neighbours=8,binomial_rounding_passes=2,
                      projection='one final radial Blackman-jinc parent-band subtraction, after positive Gaussian LP; no residual gain',
                      projection_sigma_cells=PROJECTION_SIGMA,projection_radius=PROJECTION_RADIUS,
                      projection_cutoff_cycles_per_cell=PROJECTION_CUTOFF,
                      detail_lowpass_radius=DETAIL_SMOOTH_RADIUS,
                      process_slope_gate_tangents=[.20,.65],channel_slope_gate_tangents=[.04,.20],
                      downhill_positive_smoothing_steps=8,
                      projection_post_extrema_sweeps=3,
                      transfer_anti_image_sigma_source_cells=TRANSFER_SIGMA,
                      transfer_anti_image_order=TRANSFER_ORDER,
                      flow_transfer='notched C2 interpolation with confidence clamped to [0,1]',
                      detail_gain=DETAIL_GAIN,
                      detail_cap_metres=DETAIL_CAP_METRES,
                      sediment_amount_scale=8,deposit_valley_position=[-.03,.09],
                      deposit_slope_gate=[.45,1.05],parent_deposit_retention=.75,
                      material_gaussian_sigma_min_metres=2,material_gaussian_sigma_min_cells=.8,
                      scree_foot_position=[-.06,.12],flow_classification_grid_metres=30,
                      flow_log_gate=[.38,.64],flow_ridge_gate=[.001,.025],
                      fine_flow_concavity_metres=6,fine_flow_concavity_gate=[.016,.11],
                      fine_flow_elongation_gate=[.15,.75]),
                  timings=rows, skips=skips, real_sites=selected_sites,
                  continuity=continuity,
                  morphology_test_limits=dict(parent_line_peak_ratio=4.,radial_peak_neighbour_ratio=2.,
                      radial_peak_definition='Hann FFT radial mean; one-bin peak / adjacent bins; >=8 cycles; >1e-5 max power and 4x estimated FP32/cache quantisation noise power',
                      orientation='(Jcross-Jalong)/trace weighted on inherited slopes >15 degrees',
                      isotropic_expected_structure_tensor_score=0.,orientation_minimum_qualified_cells=64),
                  continuity_test_limits=dict(mean_over_rms=.02,native_box_relative=.30,
                      native_resolved_relative=.15,adjacent_delta_relative={2:.85,3:.30,4:.8,5:.30,6:.8},
                      full_height_rms_metres={2:1.5,3:.5,4:.9,5:.25,6:.3},
                      full_height_reliefs=['alpine','montserrat','new_zealand']),
                  native_sheets=native_sheets,
                  note=('Cold starts empty amplification windows after the intensity-0 baseline read, '
                        'retains neural/source caches and includes recursive parents. Warm includes '
                        'sample_refined and cached field fusion. Surface fractions use each shown '
                        'footprint (2.28 km, 570 m, 142.5 m at the default size), not a coverage quota.'))
    if selected_sites:
        from distill.live_runtime import current
        report['real_models'] = current().identity()
        report['real_profile'] = args.real_profile
    (args.out/'timings.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    if rows:
        comparison_overview(args.out/'overview.png', rows, args.size)
    if surfaces:
        surface_overview(args.out/'overview_surface.png',surfaces)
    print('Sheets and JSON:', args.out.resolve())


if __name__ == '__main__':
    main()
