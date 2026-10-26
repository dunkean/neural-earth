"""Deterministic erosion windows, scale inheritance, coast safety and morphology."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'terrain-diffusion'))

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from infinite_tensor import MemoryTileStore
import terrain_refinement as refinement
import terrain_amplification as amplification


ANALYTIC_RELIEFS = ('alpine', 'cliff', 'hills', 'coast', 'plateau', 'glacier', 'valley')
DEM_RELIEFS = ('montserrat', 'mountains', 'new_zealand')
RELIEFS = ANALYTIC_RELIEFS + DEM_RELIEFS
FIXTURES = Path(__file__).resolve().parents[1]/'fixtures'/'amplification'


def fbm(y, x, *, seed=137, spacing=64., octaves=5, ridged=False):
    result = 0
    norm = 0
    for octave in range(octaves):
        n = refinement._value_noise(seed, y, x, spacing/2**octave, octave+51)
        if ridged:
            n = (1-n.abs()).square()
        weight = .52**octave
        result = result + weight*n
        norm += weight
    return result/norm


def relief(name, y, x):
    """Analytic native DEMs in metres, coordinates in 30 m cells."""
    y, x = y.double(), x.double()
    if name == 'alpine':
        wx = x + 18*fbm(y, x, seed=73, spacing=96., octaves=3)
        wy = y + 18*fbm(y, x, seed=29, spacing=96., octaves=3)
        massif = torch.exp(-(x*x+y*y)/(2*160**2))
        return 100 + 1250*massif*fbm(wy, wx, spacing=48., ridged=True)
    if name == 'cliff':
        rim = 8*fbm(y, x, seed=37, spacing=40., octaves=3)
        return 150 + 280*torch.sigmoid((x-rim)/1.1) + 35*fbm(y, x, spacing=35.)
    if name == 'hills':
        return 220 + 180*fbm(y, x, spacing=48., octaves=6)
    if name == 'coast':
        shore = x+5*fbm(y, x, seed=71, spacing=70., octaves=3)
        return 2.8*shore + 35*fbm(y, x, spacing=22.)*refinement._smoothstep(0, 20, shore)
    if name == 'plateau':
        return torch.full_like(x+y, 500.)
    if name == 'glacier':
        return 1400 + 5*y + 100*torch.exp(-(x*x+y*y)/(2*600**2))
    if name == 'valley':
        return 300 + 2.8*x.square() + 9*y + 2*fbm(y, x, spacing=15., octaves=3)
    raise ValueError(name)


def fake_world(device='cpu', name='alpine', seed=17):
    raster = None
    if name in DEM_RELIEFS:
        filename = 'montserrat_256.png' if name == 'montserrat' else name+'.png'
        with Image.open(FIXTURES/filename) as image:
            pixels = np.asarray(image.convert('L') if image.mode == 'RGBA' else image).copy()
        values = pixels.astype(np.float32)
        values = (values-values.min())/max(float(values.max()-values.min()), 1.)
        low, high = {'montserrat': (30., 1230.), 'mountains': (100., 2500.),
                     'new_zealand': (-40., 2160.)}[name]
        raster = torch.tensor(low+(high-low)*values, device=device)
    def get(i1, j1, i2, j2, *, with_climate):
        if with_climate:
            raise AssertionError('Amplification needs only heights')
        y = torch.arange(i1, i2, device=device)[:, None]
        x = torch.arange(j1, j2, device=device)[None, :]
        if raster is not None:
            h, w = raster.shape
            return {'elev': raster[(y+h//2).clamp(0, h-1), (x+w//2).clamp(0, w-1)]}
        return {'elev': relief(name, y, x).expand(i2-i1, j2-j1).float()}
    return SimpleNamespace(device=device, get=get, seed=seed,
                           kwargs={'residual_std': 1.1678}, tile_store=MemoryTileStore(),
                           _terrain_profile=SimpleNamespace(gpu_windows=True))


@contextmanager
def analytic_detail(device, *, zero=False):
    class DetailField:
        def __getitem__(self, key):
            _, rows, cols = key
            y = torch.arange(rows.start, rows.stop, device=device).double()[:, None]
            x = torch.arange(cols.start, cols.stop, device=device).double()[None, :]
            # The coastal tests' decoder stub deliberately contains a strong
            # checkerboard. Here isotropic coherent residuals make comparison
            # hillshades useful for erosion rather than Nyquist diagnostics.
            detail = sum(weight*refinement._value_noise(97, y, x, spacing, stream)
                         for weight, spacing, stream in ((1., 4., 2), (.6, 9., 3), (.3, 19., 4)))
            if zero:
                detail = torch.zeros_like(detail)
            detail = detail.float()
            return torch.stack((detail, torch.ones_like(detail)))
    with patch('terrain_refinement._detail_field', return_value=DetailField()):
        yield


def cheap_reference(world, level, i1, j1, i2, j2, *, check=None):
    """Cheap continuous valley for CPU window/cache tests; identity uses the real pipeline."""
    if check is not None:
        check()
    device = world.device
    y = (torch.arange(i1, i2, device=device).double()[:, None]+.5)/2**level-.5
    x = (torch.arange(j1, j2, device=device).double()[None, :]+.5)/2**level-.5
    phase = refinement._hash_int(int(world.seed))* (2*np.pi/2**32)
    seed_detail = 24*refinement._value_noise(int(world.seed),y,x,2.,77)
    return (900+2*y+.035*x*x+.03*y*y+1.5*torch.cos(.05*x+.09*y+phase)+seed_detail).float()


def cpu_reference(device):
    return patch('terrain_refinement.sample_refined', side_effect=cheap_reference) if device == 'cpu' else nullcontext()


def valley_reference(world, level, i1, j1, i2, j2, *, check=None):
    """A real open channel for CPU flow tests; the cache fixture is a broad bowl."""
    if check is not None:
        check()
    y = (torch.arange(i1,i2,device=world.device).double()[:,None]+.5)/2**level-.5
    x = (torch.arange(j1,j2,device=world.device).double()[None,:]+.5)/2**level-.5
    return relief('valley',y,x).float()


def rilled_reference(world, level, i1, j1, i2, j2, *, check=None):
    """Inclined watershed with aperiodic longitudinal ribs, in native units.

    A directional process fixture must contain longitudinal relief. The cheap
    window/cache surrogate is a broad isotropic bowl, with no such guarantee.
    """
    if check is not None:
        check()
    y = (torch.arange(i1,i2,device=world.device).double()[:,None]+.5)/2**level-.5
    x = (torch.arange(j1,j2,device=world.device).double()[None,:]+.5)/2**level-.5
    return (700+18*y+16*fbm(y/6,x,seed=int(world.seed),spacing=2.,octaves=3)).float()


def extrema_count(height, prominence):
    neighbours = amplification._neighbours(height[None])
    low, high = neighbours[0], neighbours[0]
    for value in neighbours[1:]:
        low, high = torch.minimum(low, value), torch.maximum(high, value)
    extrema = (height[None] < low-prominence) | (height[None] > high+prominence)
    return extrema[:, 6:-6, 6:-6].count_nonzero().item()


def lattice_spectrum(delta, level):
    """Hann-windowed axis power at parent-lattice lines versus nearby bins.

    Integrating the perpendicular axis keeps individual random Fourier-bin
    variance small. A three-bin line is compared with log-quadratic backgrounds
    three to six bins away; Nyquist uses the inward background. At least
    eight cycles must fit in the patch. A broad physical octave is not a line.
    """
    z = delta.detach().double().cpu().numpy()
    z = z-z.mean()
    if np.mean(z*z)<1e-16:
        return dict(max_peak_ratio=0.,worst=None,lines=[])
    # Use full FFT for both axes so their perpendicular integration agrees.
    full = np.abs(np.fft.fft2(z*np.outer(np.hanning(z.shape[0]),np.hanning(z.shape[1]))))**2
    lines = []
    parents = sorted(set(s for s in (0,2,4,level-1) if s<level))
    frequencies = sorted(set(round(h/2**(level-s),12) for s in parents
                             for h in range(1,2**(level-s-1)+1)))
    for axis,p in (('x',full.sum(0)),('y',full.sum(1))):
        n = len(p)
        for frequency in frequencies:
            k = round(frequency*n)
            if k<8 or k>n//2:
                continue
            centre = np.arange(k-1,min(k+2,n//2+1))
            neighbours = np.array([j for d in range(3,7) for j in (k-d,k+d) if 1<=j<n//2])
            if len(neighbours)<4:
                continue
            if k==n//2:
                background = p[neighbours].mean()
            else:
                fit = np.polyfit(neighbours-k,np.log(np.maximum(p[neighbours],1e-30)),2)
                background = np.exp(np.polyval(fit,centre-k)).mean()
            ratio = float(p[centre].mean()/max(background,1e-30))
            lines.append(dict(axis=axis,frequency_cycles_per_cell=frequency,peak_ratio=ratio))
    worst = max(lines,key=lambda row:row['peak_ratio']) if lines else None
    return dict(max_peak_ratio=worst['peak_ratio'] if worst else 0.,worst=worst,lines=lines)


def resolved_band_ratio(delta, scale):
    """Ideal radial anti-alias restriction; box means also contain sidelobes."""
    z = delta.detach().double().cpu().numpy()
    z = z-z.mean()
    h,w = z.shape
    power = np.abs(np.fft.fft2(z*np.outer(np.hanning(h),np.hanning(w))))**2
    fy,fx = np.fft.fftfreq(h)[:,None],np.fft.fftfreq(w)[None,:]
    mask = fy*fy+fx*fx<=(.5/scale)**2
    return float(np.sqrt(power[mask].sum()/max(power.sum(),1e-30)))


def radial_spectrum(delta, height=None):
    """Radial mean power: peak divided by its two neighbouring frequencies.

    Omit DC/patch leakage (<8 cycles), powers below 1e-5 of the maximum and
    four times the estimated quantisation power (FP32 metre addition plus the
    1/65536 m window cache). A sub-millimetre field at kilometre altitude must
    not classify rounding noise as a terrain wavelength. Compare with bins
    The neighbouring radial bins are one Fourier frequency away. Broad octaves
    are necessarily present in a pyramid; this detects a narrow radial line.
    Use arithmetic power, not a log fit across an inherited spectral cutoff.
    The injected-ring test admits this diagnostic's finite-width sensitivity.
    """
    z = delta.detach().double().cpu().numpy()
    z -= z.mean()
    h,w = z.shape
    n = min(h,w)
    power = abs(np.fft.fft2(z*np.outer(np.hanning(h),np.hanning(w))))**2
    radius = np.hypot(np.fft.fftfreq(h)[:,None],np.fft.fftfreq(w)[None,:])
    bins = np.rint(radius*n).astype(int)
    count = np.bincount(bins.ravel())
    radial = np.bincount(bins.ravel(),power.ravel())/np.maximum(count,1)
    ratios = []
    maximum = max(radial.max(),1e-30)
    quantum = 1/65536
    if height is not None:
        base = height.detach().float()
        ulp = (torch.nextafter(base,torch.full_like(base,float('inf')))-base).abs().max().item()
        quantum = np.hypot(quantum,ulp)
    power_floor = max(maximum*1e-5,4*quantum**2/12*(np.outer(np.hanning(h),np.hanning(w))**2).sum())
    for k in range(8,n//2-8):
        peak = radial[k]
        if peak<power_floor:
            continue
        background = (radial[k-1]+radial[k+1])/2
        ratios.append((float(peak/max(background,1e-30)),k/n))
    worst = max(ratios) if ratios else (0.,None)
    return dict(max_peak_neighbour_ratio=worst[0],frequency_cycles_per_cell=worst[1],
                qualified_bins=len(ratios),quantisation_quantum_metres=float(quantum),
                relative_power_floor=float(power_floor/maximum))


def orientation_coherence(delta, inherited_height, cell):
    """Structure-tensor rill alignment on inherited slopes >15 degrees.

    Elongated downslope rills have cross-slope delta gradients. Score is
    (J_cross-J_along)/trace(J), weighted by delta gradient energy. A perfectly
    isotropic tensor has score zero; contour-parallel washboard has negative
    score. No preferred global compass direction enters the diagnostic.
    """
    from scipy.ndimage import gaussian_filter
    z = delta.detach().double().cpu().numpy()
    height = inherited_height.detach().double().cpu().numpy()
    gy,gx = np.gradient(gaussian_filter(height,2.),cell)
    dy,dx = np.gradient(gaussian_filter(z,.6),cell)
    xx,yy,xy = [gaussian_filter(v,3.) for v in (dx*dx,dy*dy,dx*dy)]
    trace = xx+yy
    norm = gx*gx+gy*gy
    along = (gx*gx*xx+gy*gy*yy+2*gx*gy*xy)/np.maximum(norm,1e-24)
    mask = norm>np.tan(np.deg2rad(15))**2
    mask[:8] = False; mask[-8:] = False; mask[:,:8] = False; mask[:,-8:] = False
    energy = float(trace[mask].sum())
    anisotropy = np.sqrt((xx-yy)**2+4*xy*xy)
    return dict(aligned_coherence=float((trace-2*along)[mask].sum()/energy) if energy>1e-20 else None,
                anisotropy=float(anisotropy[mask].sum()/energy) if energy>1e-20 else None,
                slope_area_fraction=float(mask.mean()),isotropic_control=0.,
                qualified=bool(energy>1e-20 and mask.sum()>=64))


class AmplificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def devices(self):
        return ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']

    def test_parse_intensity(self):
        for raw in (None, '', '0', 'off', ' OFF ', -1):
            self.assertEqual(amplification.parse_intensity(raw), 0)
        for raw, expected in [('1.234', 1.23), (3, 2), (.005, .01)]:
            self.assertEqual(amplification.parse_intensity(raw), expected)
        for raw in ('garbage', float('inf'), float('nan'), object()):
            with self.assertRaises(ValueError):
                amplification.parse_intensity(raw)

    def test_passthrough_and_validation(self):
        for device in self.devices():
            world = fake_world(device)
            with analytic_detail(device):
                for level, intensity in [(0, 1), (1, 2), (2, 0), (6, 0)]:
                    base = refinement.sample_refined(world, level, -13, -17, 19, 21)
                    actual, surface = amplification.sample_amplified(world, level, -13, -17, 19, 21,
                                                                     intensity=intensity)
                    self.assertTrue(torch.equal(base, actual))
                    self.assertEqual(surface.count_nonzero().item(), 0)
                self.assertFalse(hasattr(world, '_terrain_amplification_fields'))
                for args in [(7, 0, 0, 1, 1), (2, 0., 0, 1, 1), (2, 0, 0, 0, 1),
                             (2, 0, 0, 1025, 1025)]:
                    with self.assertRaises(ValueError):
                        amplification.sample_amplified(world, *args)

    def test_determinism_seed_and_intensity_cache(self):
        for device in self.devices():
            with analytic_detail(device), cpu_reference(device):
                world = fake_world(device)
                a = amplification.sample_amplified(world, 2, 20, 30, 70, 90)
                b = amplification.sample_amplified(fake_world(device), 2, 20, 30, 70, 90)
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
                different = amplification.sample_amplified(fake_world(device, seed=18), 2, 20, 30, 70, 90)
                self.assertFalse(torch.equal(a[0], different[0]))
                with patch('terrain_amplification._erode', side_effect=AssertionError('cache missed')):
                    again = amplification.sample_amplified(world, 2, 20, 30, 70, 90)
                    half = amplification.sample_amplified(world, 2, 20, 30, 70, 90, intensity=.5)
                self.assertTrue(torch.equal(a[0], again[0]))
                self.assertTrue(torch.equal(a[1], half[1]))
                base = refinement.sample_refined(world, 2, 20, 30, 70, 90)
                # Comparing two already-rounded metre heights can differ by
                # one FP32 ULP at altitude; the cached delta is unchanged.
                ulp = (torch.nextafter(base,torch.full_like(base,float('inf')))-base).max().item()
                self.assertLessEqual((half[0]-(base+(a[0]-base)*.5)).abs().max().item(),max(1e-5,ulp))

    def test_mosaic_halos_negative_coordinates_and_scale_interpolation(self):
        for device in self.devices():
            with analytic_detail(device), cpu_reference(device):
                whole_world, world = fake_world(device), fake_world(device)
                for level in (2, 3, 4, 5, 6):
                    # Separate worlds and reversed window arrival order stress
                    # the store's FP64 exact sums and variable batch lengths.
                    whole = amplification.sample_amplified(whole_world, level, -56, -62, 84, 78)
                    for y, x in ((8, 2), (-40, 2), (8, -46), (-40, -46)):
                        tile = amplification.sample_amplified(world, level, y-8, x-8, y+56, x+56)
                        ys, xs = slice(y+48, y+112), slice(x+54, x+118)
                        with self.subTest(device=device, level=level, origin=(y, x)):
                            self.assertTrue(torch.equal(tile[0], whole[0][ys, xs]))
                            self.assertTrue(torch.equal(tile[1], whole[1][:, ys, xs]))

    def test_reliefs_sign_bounds_and_flatness(self):
        for device in self.devices():
            with analytic_detail(device, zero=True):
                for name in RELIEFS:
                    world = fake_world(device, name)
                    for level in (2, 4, 6):
                        if device == 'cpu':
                            # Exercise full-budget tensor physics on bounded
                            # native fixtures; large real window trees run on
                            # CUDA and the CPU seam/cache cases above.
                            y = (torch.arange(-48,48).double()[:,None]+.5)/2**level-.5
                            x = (torch.arange(-48,48).double()[None,:]+.5)/2**level-.5
                            if name in ANALYTIC_RELIEFS:
                                base = relief(name, y, x).float()
                            else:
                                native = world.get(-48,-48,48,48,with_climate=False)['elev']
                                yy, xx = torch.broadcast_tensors(y, x)
                                grid = torch.stack(((xx+48.5)/48-1,(yy+48.5)/48-1),-1).float()
                                base = F.grid_sample(native[None,None],grid[None],align_corners=False)[0,0]
                            parent = amplification._native_drainage(base[None])
                            fields = amplification._erode(base[None], parent, [(-48,-48)], world.seed, level)[0]
                            h, surface = base+2*fields[0], fields[1:5]
                        else:
                            base = refinement.sample_refined(world, level, -48, -48, 48, 48)
                            h, surface = amplification.sample_amplified(world, level, -48, -48, 48, 48,
                                                                       intensity=2)
                        with self.subTest(device=device, name=name, level=level):
                            self.assertTrue(torch.equal(h.sign(), base.sign()))
                            self.assertTrue(torch.equal(h[base <= 0], base[base <= 0]))
                            self.assertTrue(torch.isfinite(h).all() and torch.isfinite(surface).all())
                            self.assertTrue(((surface >= 0) & (surface <= 1)).all())
                            self.assertLessEqual((h-base).abs().max().item(), 60.001)
                            self.assertTrue(h.is_contiguous() and surface.is_contiguous())
                            self.assertEqual(h.dtype, torch.float32)
                            if name in ('plateau', 'glacier'):
                                difference = (h-base).abs()
                                if device == 'cpu':
                                    # Direct CPU physics has artificial block
                                    # edges; production windows discard them.
                                    difference = difference[12:-12,12:-12]
                                self.assertLess(difference.max().item(), .01)
                            elif name == 'coast':
                                # This near-shore analytic fixture is a gentle
                                # ~5 degree slope. Impose no minimum cosmetic
                                # change; even the CPU's unprojected, artificial-
                                # boundary transport stays below five centimetres.
                                self.assertLess((h-base).abs().max().item(),.05)
                            elif level == 2:
                                self.assertGreater((h-base).abs().max().item(), .001)
                            before_extrema = extrema_count(base, .1)
                            after_extrema = extrema_count((base+h)*.5, .1)
                            self.assertLessEqual(after_extrema, before_extrema+max(4, before_extrema//4))

    def test_ridges_lower_valleys_deposit_and_receive_flow(self):
        for device in self.devices():
            with analytic_detail(device, zero=True):
                world = fake_world(device, 'valley')
                y = (torch.arange(-64,64,device=device).double()[:,None]+.5)/4-.5
                x = (torch.arange(-64,64,device=device).double()[None,:]+.5)/4-.5
                base = relief('valley', y, x).float()
                parent = amplification._native_drainage(base[None])
                fields = amplification._erode(base[None],parent,[(-64,-64)],world.seed,2)[0]
                # Transport lowers ridges and delivers sediment before the
                # required projection redistributes its coarse displacement.
                self.assertLess(fields[0,:,:20].mean().item(),-.01)
                self.assertGreater(fields[0,:,56:72].max().item(),.001)
                height,surface = base+fields[0],fields[1:5]
                if device != 'cpu':
                    base = refinement.sample_refined(world, 2, -64, -64, 64, 64)
                    height, surface = amplification.sample_amplified(world, 2, -64, -64, 64, 64)
                delta = height-base
                # Local cuts survive, but a smooth parabolic flank may not
                # acquire a net offset at scales already resolved by its parent.
                self.assertLess(delta[:, :20].min().item(), 0)
                self.assertGreater(delta[:, 56:72].max().item(), .001)
                self.assertGreater(surface[1, :, 56:72].max().item(), .01)
                valley_flow = surface[0, :, 56:72].max().item()
                self.assertGreater(valley_flow, surface[0, :, :20].max().item())
                self.assertGreater(surface[2].max().item(), .01)
                self.assertGreater(surface[3].max().item(), .01)

    def test_cancellation_and_scheduler(self):
        from terrain_window_scheduler import WorldWindowScheduler
        world = fake_world()
        world.coarse = world.latents = world.residual = SimpleNamespace(args=[], _terrain_scheduled=True)
        world._terrain_window_scheduler = WorldWindowScheduler(world)
        calls = []
        def check():
            calls.append(1)
            if len(calls) == 15:
                raise RuntimeError('cancelled')
        def fast_reference(world, level, i1, j1, i2, j2, *, check=None):
            # Isolate cancellation inside erosion iterations, rather than
            # spending this test's callback budget in decoder dependencies.
            if check is not None:
                check()
            return cheap_reference(world, level, i1, j1, i2, j2)
        with patch('terrain_refinement.sample_refined', side_effect=fast_reference):
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                amplification.sample_amplified(world, 2, 0, 0, 32, 32, check=check)
            h, _ = amplification.sample_amplified(world, 2, 0, 0, 32, 32)
            self.assertTrue(torch.isfinite(h).all())
        self.assertTrue(any('amplification_'+amplification.VERSION in k for k in world._terrain_window_scheduler.status()))

    def test_native_drainage_reaches_beyond_sixteen_cells(self):
        for device in self.devices():
            y = torch.arange(384, device=device)[None, :, None]
            x = torch.arange(-32, 32, device=device)[None, None, :]
            base = (1000+2*y+.08*x*x).float()
            long = amplification._native_drainage(base)
            with patch('terrain_amplification.DRAIN_ITERATIONS', 16):
                short = amplification._native_drainage(base)
            self.assertEqual(long[:, 0].count_nonzero().item(), 0)
            self.assertGreater(long[0, 1, 200:320, 28:36].mean().item(),
                               short[0, 1, 200:320, 28:36].mean().item()+.1)
        for stage, (_, _, margin) in amplification.GEOMETRY.items():
            self.assertGreaterEqual(margin, amplification._reach(stage))

    def test_public_native_drainage_reader(self):
        for device in self.devices():
            with cpu_reference(device):
                world = fake_world(device,'valley')
                native = refinement.sample_refined(world,0,-32,-40,32,40)
                whole = amplification.sample_drainage(world,-32,-40,32,40)
                again = amplification.sample_drainage(fake_world(device,'valley'),-32,-40,32,40)
                self.assertTrue(torch.equal(whole,again))
                overlap = amplification.sample_drainage(fake_world(device,'valley'),-5,-10,17,23)
                self.assertTrue(torch.equal(overlap,whole[27:49,30:63]))
                self.assertEqual(whole.dtype,torch.float32)
                self.assertTrue(whole.is_contiguous())
                self.assertTrue(torch.isfinite(whole).all())
                self.assertTrue((whole>=0).all())
                self.assertEqual(whole[native<=0].count_nonzero().item(),0)
                for y,x in ((-32,-40),(-32,0),(0,-40),(0,0)):
                    tile = amplification.sample_drainage(world,y,x,y+32,x+40)
                    self.assertTrue(torch.equal(tile,whole[y+32:y+64,x+40:x+80]))
                self.assertTrue(torch.equal(native,refinement.sample_refined(world,0,-32,-40,32,40)))
                self.assertGreater(whole.max().item(),900)
                for bounds in ((0.,0,1,1),(0,0,0,1),(0,0,1025,1025)):
                    with self.assertRaises(ValueError):
                        amplification.sample_drainage(world,*bounds)
            # One native cell's rainfall is 30*30 m², not a normalised guide
            # or a count of sub-30 m cells. A flat has no outgoing receivers.
            flat = amplification.sample_drainage(fake_world(device,'plateau'),-2,-2,2,2)
            self.assertLess((flat-900).abs().max().item(),.12)

    def test_large_area_centered_displacement_and_native_means(self):
        def rms(x): return x.double().square().mean().sqrt().item()
        for device in self.devices():
            with analytic_detail(device),cpu_reference(device):
                world = fake_world(device)
                for level in (2,4):
                    h,_ = amplification.sample_amplified(world,level,-256,-256,256,256)
                    base = refinement.sample_refined(world,level,-256,-256,256,256)
                    delta = h-base
                    # Gain-one physics may honestly produce only millimetres
                    # after parent-band removal; never impose a cosmetic gain.
                    self.assertGreater(rms(delta),.0001)
                    self.assertLess(abs(delta.double().mean().item()),.02*rms(delta))
                    native = F.avg_pool2d(delta[None,None],2**level)
                    # A box is not an anti-alias filter: its sidelobes retain
                    # up to ~27% of a sub-cell sinusoid. Bound that quadrature
                    # residual separately from the truly native-resolved band.
                    self.assertLess(rms(native),.30*rms(delta))
                    self.assertLess(resolved_band_ratio(delta,2**level),.15)

    def test_material_filter_batch_invariance_and_native_flow_lines(self):
        for device in self.devices():
            y = torch.arange(-24,24,device=device).double()[:,None]
            x = torch.arange(-28,28,device=device).double()[None,:]
            values = torch.stack((refinement._value_noise(17,y,x,3.,77).float(),
                                  refinement._value_noise(29,y,x,5.,78).float()))
            for stage in amplification.STAGE_LEVELS:
                cell = 30/2**stage
                together = amplification._material_smooth(values,cell)
                separate = torch.cat([amplification._material_smooth(v[None],cell) for v in values])
                self.assertTrue(torch.equal(together,separate))
            cell = 30/64
            yy,xx = y[None]*cell,x[None]*cell
            plane = 500+.4*yy+torch.zeros_like(xx)
            gx,gy = torch.zeros_like(plane),torch.full_like(plane,.4)
            line = amplification._channel_shape(plane+.02*xx.square(),gx,gy,cell)
            pit = amplification._channel_shape(plane+.02*(xx.square()+yy.square()),gx,gy,cell)
            self.assertGreater(line[:,6:-6,6:-6].mean().item(),.99)
            self.assertLess(pit[:,6:-6,6:-6].max().item(),.01)
            # Native confidence is carried by C2 interpolation: no fine-grid
            # derivative of a quantised guide can invent cross/dash marks.
            context = patch('terrain_refinement.sample_refined',side_effect=valley_reference) if device=='cpu' else nullcontext()
            with analytic_detail(device),context:
                world = fake_world(device,'valley')
                for stage in (4,6):
                    cy = cx = 2**(stage-1)
                    bounds = (cy-24,cx-28,cy+24,cx+28)
                    height,surface = amplification.sample_amplified(world,stage,*bounds)
                    parent = amplification._upsample(world,stage-2,stage,*bounds)
                    # Cached process confidence still inherits the old guide.
                    # The public diagnostic now classifies FINAL heights in
                    # common units for all modes, rather than copying that mask.
                    process = amplification._read_stage(world,stage,*bounds)
                    self.assertTrue((process[1]>=parent[1]-2e-5).all())
                    fn = lambda *args:amplification._amplified_height(world,stage,*args)
                    shared = amplification.sample_surface(world,stage,*bounds,fn)
                    self.assertTrue(torch.equal(surface,shared*(height>0)))
                    self.assertGreater(surface[0].max().item(),.5)

    def test_projection_commutes_with_one_cell_translation_and_rotation(self):
        for device in self.devices():
            y = torch.arange(-72,72,device=device).double()[:,None]
            x = torch.arange(-76,76,device=device).double()[None,:]
            values = 8*fbm(y,x,spacing=17.,octaves=4)
            base = (1000+5*y+3*x).float()
            parent = torch.zeros_like(base)
            a = amplification._project_increment(values,(-72,-76),32,64,base,parent,2)
            b = amplification._project_increment(values[1:-1,1:-1],(-71,-75),31,64,
                                                 base[1:-1,1:-1],parent[1:-1,1:-1],2)
            self.assertTrue(torch.equal(a,b))
            h = amplification._high_pass(values)
            rotated = amplification._high_pass(values.T).T
            torch.testing.assert_close(h,rotated,atol=1e-12,rtol=1e-12)
            dc = amplification._high_pass(torch.ones_like(values))
            self.assertLess(dc.abs().max().item(),1e-12)
            torch.testing.assert_close(amplification._high_pass(2*values),2*h,atol=1e-12,rtol=1e-12)
            if device=='cuda':
                # A non-contiguous reference crop must use its actual rows in
                # the fused guard, just as in the tensor implementation.
                rugged = (1000+12*fbm(y,x,spacing=3.,octaves=3)).float()
                fast = amplification._project_increment(values,(-72,-76),32,64,rugged,parent,2)
                with patch('terrain_amplification._cuda_pack',return_value=None):
                    slow = amplification._project_increment(values,(-72,-76),32,64,rugged,parent,2)
                torch.testing.assert_close(fast,slow,atol=.00025,rtol=0)

    def test_delta_spectrum_has_no_parent_lattice_lines(self):
        for device in self.devices():
            with analytic_detail(device),cpu_reference(device):
                world = fake_world(device)
                for level in (2,4,6):
                    h,_ = amplification.sample_amplified(world,level,-128,-128,128,128)
                    base = refinement.sample_refined(world,level,-128,-128,128,128)
                    delta = h-base
                    spectrum = lattice_spectrum(delta,level)
                    with self.subTest(device=device,level=level,worst=spectrum['worst']):
                        self.assertLess(spectrum['max_peak_ratio'],4.)
                    if level==4:
                        # A coherent parent-lattice carrier must fail the
                        # detector; otherwise a smooth spectrum could pass vacuously.
                        x = torch.arange(256,device=device).double()[None,:]
                        amplitude = delta.double().square().mean().sqrt()*2
                        injected = delta+amplitude*torch.cos(2*np.pi*x/4)
                        self.assertGreater(lattice_spectrum(injected,level)['max_peak_ratio'],4.)

    def test_radial_peak_and_downhill_orientation(self):
        for device in self.devices():
            y = torch.arange(256,device=device).double()[:,None]
            x = torch.arange(256,device=device).double()[None,:]
            plane = (1000+.4*y+torch.zeros_like(x)).float()
            isotropic = fbm(y,x,spacing=8.,octaves=3)
            # Elongated rills vary across the slope; contour-parallel bands
            # give the opposite sign. Test the diagnostic before using it.
            rills = torch.cos(2*np.pi*x/13)+torch.zeros_like(y)
            across = torch.cos(2*np.pi*y/13)+torch.zeros_like(x)
            self.assertGreater(orientation_coherence(rills,plane,1)['aligned_coherence'],.99)
            self.assertLess(orientation_coherence(across,plane,1)['aligned_coherence'],-.99)
            self.assertLess(abs(orientation_coherence(isotropic,plane,1)['aligned_coherence']),.2)
            self.assertGreater(radial_spectrum(rills)['max_peak_neighbour_ratio'],2.)
            ring = sum(torch.cos(2*np.pi/13*(np.cos(a)*x+np.sin(a)*y)+k*1.618)
                       for k,a in enumerate(np.linspace(0,np.pi,24,endpoint=False)))
            self.assertGreater(radial_spectrum(ring)['max_peak_neighbour_ratio'],2.)
            self.assertGreater(radial_spectrum(.1*ring,plane+2000)['max_peak_neighbour_ratio'],2.)
            self.assertLess(radial_spectrum(isotropic)['max_peak_neighbour_ratio'],2.)
            with analytic_detail(device),patch('terrain_refinement.sample_refined',side_effect=rilled_reference):
                world = fake_world(device)
                for stage in (2,4,6):
                    base = refinement.sample_refined(world,stage,-128,-128,128,128)
                    h,_ = amplification.sample_amplified(world,stage,-128,-128,128,128)
                    parent = amplification._upsample(world,stage-2,stage,-128,-128,128,128)[0]
                    metric = orientation_coherence(h-base,base+parent,30/2**stage)
                    with self.subTest(device=device,stage=stage,orientation=metric):
                        self.assertTrue(metric['qualified'])
                        self.assertGreater(metric['aligned_coherence'],metric['isotropic_control'])
                        self.assertLess(radial_spectrum(h-base,base)['max_peak_neighbour_ratio'],2.)

    def test_final_band_filter_rejects_nyquist_and_flat_dry_activity(self):
        for device in self.devices():
            y = torch.arange(128,device=device).double()[:,None]
            x = torch.arange(128,device=device).double()[None,:]
            wave = torch.cos(2*np.pi*x*.45)+torch.zeros_like(y)
            smooth = refinement._filter(wave,amplification._gaussian_taps(
                amplification.PROJECTION_SIGMA,amplification.DETAIL_SMOOTH_RADIUS))
            self.assertLess(amplification._high_pass(smooth).square().mean().sqrt().item(),.008)
            coarse = torch.cos(2*np.pi*x/8)+torch.zeros_like(y)
            self.assertLess(amplification._high_pass(coarse).abs().max().item(),.0003)
            base = (500+.08*y*30/64+torch.zeros_like(x)).float()[None]
            parent = torch.zeros((1,6,128,128),device=device)
            activity,_,_ = amplification._process_strength(base,parent,30/64)
            self.assertEqual(activity[:,12:-12,12:-12].count_nonzero().item(),0)
            fields = amplification._erode(base,parent,[(0,0)],17,6)
            self.assertLess(fields[:,0,24:-24,24:-24].abs().max().item(),.001)

    def test_zoom_continuity_same_physical_footprint(self):
        def rms(x): return x.double().square().mean().sqrt().item()
        for device in self.devices():
            with analytic_detail(device),cpu_reference(device):
                # cpu_reference is a single analytic window fixture; repeating
                # it with DEM labels would test the identical field three times.
                names = ('alpine',) if device == 'cpu' else ('alpine','montserrat','new_zealand')
                for name in names:
                    world = fake_world(device,name)
                    previous = None
                    for level in range(1,7):
                        half = 4*2**level
                        h,_ = amplification.sample_amplified(world,level,-half,-half,half,half)
                        base = refinement.sample_refined(world,level,-half,-half,half,half)
                        delta = h-base
                        if previous is not None:
                            coarse_delta = F.avg_pool2d(delta[None,None],2)[0,0]
                            # Stage births add a new octave; a 2-cell box is
                            # not an anti-alias restriction for that octave.
                            # Interpolated LODs get no new erosion increment.
                            bound = {2:.85,3:.30,4:.8,5:.30,6:.8}[level]
                            with self.subTest(device=device,relief=name,level=level):
                                self.assertLess(rms(coarse_delta-previous[1]),bound*rms(delta)+1e-4)
                                metres = {2:1.5,3:.5,4:.9,5:.25,6:.3}[level]
                                self.assertLess(rms(F.avg_pool2d(h[None,None],2)[0,0]-previous[0]),metres)
                        previous = h,delta

    def test_slope_limiter_does_not_make_diamond_cones(self):
        for device in self.devices():
            delta = torch.zeros((1,33,33),device=device)
            delta[0,16,16] = -3
            rounded = amplification._smooth_increment(amplification._slope_limit(delta,1.),1.)
            # Near-equal Euclidean radii (4 and sqrt(18)) should agree. A
            # Manhattan envelope instead has a half-metre axial/diagonal gap.
            self.assertLess(abs(rounded[0,16,20].item()-rounded[0,19,19].item()),.2)
            self.assertLessEqual((rounded[:,:,1:]-rounded[:,:,:-1]).abs().max().item(),.50001)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA only')
    def test_fused_passes_match_tensor_fallback(self):
        if amplification._cuda_pack('cuda') is None:
            self.skipTest('Optional NVRTC compiler unavailable')
        y = torch.arange(128,device='cuda')[None,:,None]
        x = torch.arange(128,device='cuda')[None,None,:]
        base = (800+.3*y+.008*(x-60).square()+2*torch.cos(y*.1)).float()
        parent = torch.zeros((1,5,128,128),device='cuda')
        parent[:,0] = .2
        parent[:,1] = .35
        for stage in amplification.STAGE_LEVELS:
            # Admit the individual passes numerically. Long FP32 drainage
            # trajectories can choose different receivers at near-flat ties;
            # full-budget determinism/safety are tested separately per engine.
            with patch.dict(amplification.ITERATIONS,{stage:(1,1,1)}):
                fused = amplification._erode(base,parent,[(0,0)],17,stage)
                with patch('terrain_amplification._cuda_pack',return_value=None):
                    tensor = amplification._erode(base,parent,[(0,0)],17,stage)
                # As in production, compare retained interiors. A 32-cell
                # collar exceeds the complete one-pass stencil reach (29).
                fused, tensor = fused[...,32:-32,32:-32], tensor[...,32:-32,32:-32]
                torch.testing.assert_close(fused[:,:1],tensor[:,:1],atol=.001,rtol=.001)
                # Normalised sub-metre sediment guides magnify height ULPs.
                torch.testing.assert_close(fused[:,1:],tensor[:,1:],atol=.005,rtol=.001)

    def test_parent_delta_and_physical_flow_survive_finer_stage(self):
        for device in self.devices():
            cell = 30/64
            x = torch.arange(64, device=device)[None, None, :]
            base = (500 + x*cell*.35).expand(1, 64, 64).contiguous()
            parent = torch.zeros((1, 5, 64, 64), device=device)
            parent[:, 0] = 2
            parent[:, 1] = .5
            inherited = amplification._erode(base, parent, [(0, 0)], 17, 6)
            parent[:, 1] = 0
            unseeded = amplification._erode(base, parent, [(0, 0)], 17, 6)
            other_seed = amplification._erode(base, parent, [(0, 0)], 18, 6)
            centre = (0, slice(20, 44), slice(20, 44))
            # This uniform plane has no active new process. Seed variation
            # must not manufacture relief merely to make outputs different;
            # active-fixture seed dependence is tested through the public API.
            self.assertTrue(torch.equal(unseeded[:,0][centre],other_seed[:,0][centre]))
            torch.testing.assert_close(unseeded[:,0][centre],torch.full((24,24),2.,device=device),atol=.0001,rtol=0)
            self.assertGreater(inherited[:, 0][centre].min().item(), 1.)
            self.assertGreater(inherited[:, 1][centre].mean().item(), .45)
            self.assertLess(unseeded[:, 1][centre].mean().item(), .02)


def surface_fixture(world, level, i1, j1, i2, j2):
    """A sloping escarpment with a sinuous valley, in physical metres."""
    cell = 30/2**level
    y = (torch.arange(i1,i2,device=world.device).double()[:,None]+.5)*cell
    x = (torch.arange(j1,j2,device=world.device).double()[None,:]+.5)*cell
    rim = 15*torch.sin(y/120)
    channel = x-35*torch.sin(y/190)
    rill = x+70-8*torch.sin(y/100)
    return (700+.50*y+70*torch.tanh((x-rim)/35)-14*torch.exp(-channel.square()/(2*24**2))
            -5*torch.exp(-rill.square()/(2*6**2)))


def surface_world(device='cpu',seed=17):
    world = fake_world(device,'plateau',seed)
    world.get = lambda a,b,c,d,with_climate: {'elev':surface_fixture(world,0,a,b,c,d).float()}
    return world


@contextmanager
def surface_cpu_drainage(device):
    # Native drainage itself is tested end-to-end above. Bound CPU surface
    # tests to diagnostics rather than repeatedly solving its 160-step tree.
    def area(world,a,b,c,d):
        y = torch.arange(a,c,device=device).double()[:,None]
        x = torch.arange(b,d,device=device).double()[None,:]
        channel = x-1.2*torch.sin(y/6)
        return (900+300000*torch.exp(-channel.square()/2)).float()
    with patch('terrain_amplification.sample_drainage',side_effect=area) if device=='cpu' else nullcontext():
        yield


class SurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def devices(self):
        return ['cpu','cuda'] if torch.cuda.is_available() else ['cpu']

    def test_contract_determinism_seed_offset_and_input_immutability(self):
        for device in self.devices():
            with surface_cpu_drainage(device):
                for level in (2,4,6):
                    world = surface_world(device)
                    margin = amplification._surface_margin(level)
                    bounds = (-40,-48,40,48)
                    padded = (-40-margin,-48-margin,40+margin,48+margin)
                    storage = surface_fixture(world,level,*padded)
                    snapshot = storage.clone()
                    calls = []
                    def fn(a,b,c,d):
                        calls.append((a,b,c,d))
                        return storage[a-padded[0]:c-padded[0],b-padded[1]:d-padded[1]]
                    actual = amplification.sample_surface(world,level,*bounds,fn)
                    self.assertEqual(calls,[padded])
                    self.assertTrue(torch.equal(storage,snapshot))
                    self.assertEqual(actual.shape,(4,80,96))
                    self.assertEqual(actual.dtype,torch.float32)
                    self.assertEqual(actual.device,storage.device)
                    self.assertTrue(actual.is_contiguous() and torch.isfinite(actual).all())
                    self.assertTrue(((actual>=0)&(actual<=1)).all())
                    fn2 = lambda a,b,c,d: surface_fixture(world,level,a,b,c,d)
                    again = amplification.sample_surface(surface_world(device),level,*bounds,fn2)
                    self.assertTrue(torch.equal(actual,again))
                    offset = amplification.sample_surface(world,level,*bounds,lambda *args:fn2(*args)+1024)
                    torch.testing.assert_close(actual,offset,atol=2e-6,rtol=0)
                    other = amplification.sample_surface(surface_world(device,29),level,*bounds,fn2)
                    self.assertFalse(torch.equal(actual,other))

    def test_exact_negative_overlaps_and_intermediate_levels(self):
        for device in self.devices():
            with surface_cpu_drainage(device):
                world = surface_world(device)
                for level in (1,2,3,4,5,6):
                    fn = lambda *args:surface_fixture(world,level,*args)
                    big = amplification.sample_surface(world,level,-64,-68,80,88,fn)
                    for a,b,c,d in ((-48,-52,16,20),(-16,-8,64,72),(0,4,48,56)):
                        tile = amplification.sample_surface(world,level,a,b,c,d,fn)
                        with self.subTest(device=device,level=level,bounds=(a,b,c,d)):
                            self.assertTrue(torch.equal(tile,big[:,a+64:c+64,b+68:d+68]))

    def test_loose_fixture_coverage_and_quiet_flats(self):
        for device in self.devices():
            with surface_cpu_drainage(device):
                world = surface_world(device)
                level = 4
                fn = lambda *args:surface_fixture(world,level,*args)
                fields = amplification.sample_surface(world,level,-128,-128,128,128,fn)
                fractions = (fields>.5).double().mean((1,2))
                # Broad bounds admit fixed physical masks, never per-tile
                # quantiles. Each process must have support and avoid saturation.
                for k in range(4):
                    with self.subTest(device=device,channel=amplification.SURFACE_CHANNELS[k],fraction=float(fractions[k])):
                        self.assertGreater(float(fractions[k]),.0005)
                        self.assertLess(float(fractions[k]),.6)
            world = fake_world(device,'plateau')
            flat = lambda a,b,c,d:torch.full((c-a,d-b),500.,device=device)
            fields = amplification.sample_surface(world,6,-32,-32,32,32,flat)
            self.assertEqual(fields.count_nonzero().item(),0)

    def test_validation_and_cancellation(self):
        world = surface_world()
        fn = lambda *args:surface_fixture(world,2,*args)
        for level,bounds in ((7,(0,0,1,1)),(2,(0.,0,1,1)),(2,(0,0,0,1)),(2,(0,0,1025,1025))):
            with self.assertRaises(ValueError):
                amplification.sample_surface(world,level,*bounds,fn)
        with self.assertRaises(ValueError):
            amplification.sample_surface(world,2,0,0,1,1,lambda *args:torch.zeros(2,2))
        def stop():
            raise RuntimeError('surface cancelled')
        with self.assertRaisesRegex(RuntimeError,'surface cancelled'):
            amplification.sample_surface(world,2,0,0,1,1,fn,check=stop)

    def test_large_callback_reads_stay_within_height_budget(self):
        world = fake_world('cpu','plateau')
        calls = []
        def fn(a,b,c,d):
            calls.append((a,b,c,d))
            self.assertLessEqual((c-a)*(d-b),1024**2)
            return torch.full((c-a,d-b),500.)
        with surface_cpu_drainage('cpu'):
            fields = amplification.sample_surface(world,6,-500,-8,524,8,fn)
        self.assertEqual(fields.shape,(4,1024,16))
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0][0],-500-amplification._surface_margin(6))
        self.assertEqual(calls[-1][2],524+amplification._surface_margin(6))

    def test_fine_flow_follows_a_channel_and_rejects_a_round_bowl(self):
        for device in self.devices():
            world = fake_world(device,'plateau')
            def guides(world,level,bounds):
                a,b,c,d = bounds
                z = torch.zeros((c-a,d-b),device=device,dtype=torch.float64)
                return z,z,z,torch.ones_like(z,dtype=torch.bool)
            def height(kind,a,b,c,d):
                y = (torch.arange(a,c,device=device).double()[:,None]+.5)*30/16
                x = (torch.arange(b,d,device=device).double()[None,:]+.5)*30/16
                return (500+.55*y-3*torch.exp(-x.square()/72) if kind=='line'
                        else 500+.55*y+.001*(x*x+y*y))
            with patch('terrain_amplification._surface_guides',side_effect=guides):
                line = amplification.sample_surface(world,4,-64,-64,64,64,lambda *args:height('line',*args))
                bowl = amplification.sample_surface(world,4,-64,-64,64,64,lambda *args:height('bowl',*args))
            self.assertGreater(line[0].max().item(),.6)
            self.assertLess(bowl[0].max().item(),1e-4)

    def test_native_geometry_uses_fixed_square_reads(self):
        world = surface_world('cpu')
        get = world.get
        shapes = []
        def shape_sensitive(a,b,c,d,*,with_climate):
            shapes.append((c-a,d-b))
            value = get(a,b,c,d,with_climate=with_climate)['elev']
            return {'elev':value if c-a==d-b else value*1.15}
        world.get = shape_sensitive
        fn = lambda *args:surface_fixture(world,4,*args)
        with surface_cpu_drainage('cpu'):
            big = amplification.sample_surface(world,4,-48,-72,64,88,fn)
            tile = amplification.sample_surface(world,4,-24,-40,32,40,fn)
        # sample_refined may add its own coherent-coast halo to the 128-cell
        # request; the underlying decoder still sees one fixed square shape.
        self.assertTrue(shapes and len(set(shapes))==1 and all(h==w for h,w in shapes))
        self.assertTrue(torch.equal(tile,big[:,24:80,32:112]))

    def test_native_geometry_cache_is_versioned_bounded_and_reused(self):
        world = fake_world('cpu','plateau')
        for tile in range(35):
            amplification._surface_native(world,tile*128,0,(tile+1)*128,1)
        cache = world._terrain_surface_native_blocks
        self.assertEqual(len(cache),32)
        self.assertTrue(all(key[0]==amplification.SURFACE_VERSION for key in cache))
        self.assertLessEqual(sum(v.numel()*v.element_size() for v in cache.values()),2*1024**2)
        with patch('terrain_amplification._reference',side_effect=AssertionError('native cache missed')):
            cached = amplification._surface_native(world,34*128,0,35*128,1)
        self.assertTrue(torch.equal(cached,torch.full((128,1),500.,dtype=torch.float64)))


if __name__ == '__main__':
    unittest.main()
