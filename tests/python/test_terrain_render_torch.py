"""CUDA surface material mirrors the numpy reference."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import unittest
import numpy as np
import torch
import terrain_render as R
import terrain_render_torch as T

TEMPERATE = [22, 500, .0065, 2, 500, .0065]
FOOTPRINTS = [3.75, 30, 60, 240, 1920, 7680, 19531]
SETTINGS = [None, {'variation': 2, 'forest': .2}, {'moisture': -.5, 'season': 0}, {'moisture': .7, 'season': .25, 'snow': .3},
            {'rock_slope': 65, 'season': .9, 'vegetation_tint': [.9, 1, .8], 'rock_tint': [1, .9, .95], 'snow_color': [1, 1, .9]}, {'season': .6}]
SOIL = np.array([.42, .32, .22, .46, .44, .40, .3, .3, .4], np.float32)


def random_case(seed, shape=(96, 128), spread=1):
    rng = np.random.default_rng(seed)
    u = lambda lo, hi: rng.uniform(lo, hi, shape).astype(np.float32)
    lat, lon = rng.uniform(-np.pi/2, np.pi/2, shape), rng.uniform(-np.pi, np.pi, shape)
    point = (6.37e6*np.stack((np.cos(lat)*np.cos(lon), np.sin(lat), np.cos(lat)*np.sin(lon)), -1)).astype(np.float32)
    north = rng.normal(size=(*shape, 2)); north = (north/np.linalg.norm(north, axis=-1, keepdims=True)).astype(np.float32)
    lapse = u(.004, .008)
    seasons = np.stack((u(-30, 35), u(0, 3000), lapse, u(-30, 35), u(0, 3000), lapse))
    gradient = (rng.normal(size=(*shape, 2))*rng.choice([.05, .3, 1.2], shape)[..., None]).astype(np.float32)
    return dict(height=u(-3000, 6000), gradient=gradient, tpi=u(-.2, .2), point=point, north=north,
                soil=rng.uniform(0, 1, (9, *shape)).astype(np.float32), seasons=seasons,
                pedology=rng.uniform(0, 1, (3, *shape)).astype(np.float32))


def direct_case(shape=(64, 64), origin=(6.3e6, 0, 0)):
    """Smooth near-surface field like a real tile."""
    rng = np.random.default_rng(7)
    y, x = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    point = np.stack((np.full(shape, origin[0]) + x*30, y*30 - 2e6, np.full(shape, origin[2]) + x*5), -1).astype(np.float32)
    height = (1500 + 1200*np.sin(x*.07)*np.cos(y*.05) + rng.normal(size=shape)*5).astype(np.float32)
    gradient = np.stack(np.gradient(height, 30.)[::-1], -1).astype(np.float32)
    seasons = np.broadcast_to(np.array(TEMPERATE, np.float32)[:, None, None], (6, *shape))
    return dict(height=height, gradient=gradient, tpi=(np.sin(y*.3)*.1).astype(np.float32), point=point,
                north=np.broadcast_to(np.array([0, -1], np.float32), (*shape, 2)),
                soil=np.broadcast_to(SOIL[:, None, None], (9, *shape)), seasons=seasons,
                pedology=np.broadcast_to(np.array([.62, .41, .25], np.float32)[:, None, None], (3, *shape)))


@unittest.skipUnless(T.available(), 'CUDA unavailable')
class TorchRenderTests(unittest.TestCase):
    def setUp(self):
        self.dev = torch.device('cuda', 0)

    def noise_points(self, rng, n=50000):
        p = rng.uniform(-2e7, 2e7, (n, 3)).astype(np.float32)/30
        p[:100] = rng.integers(-5, 6, (100, 3))  # exact cell corners
        p[100:200] = rng.uniform(-3, 3, (100, 3))
        return p

    def test_hash_noise_and_fbm_match_numpy(self):
        rng = np.random.default_rng(1)
        p = self.noise_points(rng)
        c = rng.integers(-2**31, 2**31, (50000, 3)).astype(np.int32)
        with torch.inference_mode():
            tp = torch.from_numpy(p).to(self.dev)
            for salt in (0, 5, 0xdeadbeef + 12, 2**40 + 9):
                a = R.gradient_noise(p, salt)
                b = T.gradient_noise(tp, salt).cpu().numpy()
                self.assertLess(float(np.abs(a - b).max()), 1e-5)
            tc = torch.from_numpy(c.astype(np.int64)).to(self.dev)
            ref = R._hash(c, 77)
            got = ((((tc[..., 0] & T._M)*0x9e3779b9) & T._M) ^ (((tc[..., 1] & T._M)*0x85ebca6b) & T._M) ^ (((tc[..., 2] & T._M)*0xc2b2ae35) & T._M) ^ 77)
            got = ((got ^ (got >> 16))*0x7feb352d) & T._M
            got = ((got ^ (got >> 15))*0x846ca68b) & T._M
            got = (got ^ (got >> 16)).cpu().numpy()
            np.testing.assert_array_equal(ref.astype(np.int64), got)
            for scale, octaves, fp in ((64000., 4, 30.), (240., 4, 3.75), (700., 4, 100.), (900000., 3, 19531.), (4800., 4, 1500.)):
                big = rng.uniform(-6.4e6, 6.4e6, (20000, 3)).astype(np.float32)
                a, ad = R.fbm(big, scale, octaves, fp, 9)
                b, bd = T.fbm(torch.from_numpy(big).to(self.dev), scale, octaves, fp, 9)
                self.assertEqual(float(ad), float(bd))
                self.assertLess(float(np.abs(a - b.cpu().numpy()).max()), 1e-5)

    def compare(self, case, footprint, settings, bare, pedology, label):
        args = (case['height'], case['gradient'], case['tpi'], case['point'], case['north'], footprint, case['soil'], case['seasons'], 42, settings, bare)
        ped = case['pedology'] if pedology else None
        ref = R.surface_material(*args, ped)
        got = T.surface_material(*args, ped)
        self.assertEqual(got.dtype, np.float32); self.assertEqual(got.shape, ref.shape)
        diff = np.abs(ref - got).max(-1)
        close = float((diff < 1e-4).mean())
        eight = float((np.abs(np.round(ref*255) - np.round(got*255)).max(-1) <= 1).mean())
        info = f'{label} fp={footprint} bare={bare} ped={pedology}: max={diff.max():.2e} close={close:.5f} 8bit={eight:.5f}'
        self.assertLessEqual(diff.max(), 2e-3, info)
        self.assertGreaterEqual(close, .999, info)
        self.assertGreaterEqual(eight, .999, info)
        return diff.max(), (diff >= 1e-4).sum()

    def test_random_fields_match_numpy(self):
        worst = flips = 0
        for fp in FOOTPRINTS:
            for index, settings in enumerate(SETTINGS):
                case = random_case(100 + index, (64, 96))
                for bare in (False, True):
                    m, f = self.compare(case, fp, settings, bare, bool(index % 2), f'random{index}')
                    worst = max(worst, m); flips += int(f)
        print(f'random fields: max diff {worst:.2e}, pixels over 1e-4: {flips}')

    def test_tile_like_fields_match_numpy(self):
        worst = flips = 0
        case = direct_case()
        for fp in FOOTPRINTS:
            for settings in SETTINGS:
                for bare in (False, True):
                    m, f = self.compare(case, fp, settings, bare, not bare, 'tile')
                    worst = max(worst, m); flips += int(f)
        print(f'tile fields: max diff {worst:.2e}, pixels over 1e-4: {flips}')

    def test_coastal_lowlands_match_numpy(self):
        case = direct_case()
        y, x = np.mgrid[:64, :64].astype(np.float32)
        case['height'] = (x*.6-12+np.sin(y*.4)*3).astype(np.float32)
        case['gradient'] = np.stack(np.gradient(case['height'], 30.)[::-1], -1).astype(np.float32)
        worst = flips = 0
        for fp in FOOTPRINTS:
            for settings in (None, {'season': .6}):
                m, f = self.compare(case, fp, settings, False, False, 'coast')
                worst = max(worst, m); flips += int(f)
        self.assertEqual(flips, 0)

    def test_cold_and_rugged_coasts_match_numpy(self):
        case = direct_case((16, 32))
        x = np.arange(32, dtype=np.float32)
        for fp in (15., 30., 60., 120., 240.):
            case['height'] = np.broadcast_to(np.where(x < 8, -5., 1.), (16, 32)).copy()
            case['gradient'] = np.zeros((16, 32, 2), np.float32)
            case['gradient'][..., 0] = .14
            case['tpi'] = np.full((16, 32), -.12, np.float32)
            for climate in ([4, 650, 0, -15, 650, 0], [22, 500, 0, -20, 500, 0]):
                case['seasons'] = np.broadcast_to(np.array(climate, np.float32)[:, None, None], (6, 16, 32))
                for settings in ({'season': .5}, {'season': 0}, {'snow': 0, 'variation': 2}):
                    self.compare(case, fp, settings, False, True, 'cold/rugged coast')

    def test_inputs_may_be_integer_north_scalar_tpi_and_flat_shapes(self):
        case = direct_case((2, 300))
        flat = dict(height=case['height'][0], gradient=case['gradient'][0], point=case['point'][0],
                    soil=case['soil'][:, 0], seasons=case['seasons'][:, 0])
        north = np.broadcast_to([0, -1], (300, 2))
        args = (flat['height'], flat['gradient'], np.float32(.05), flat['point'], north, 30., flat['soil'], flat['seasons'], 42, {'season': 0})
        ref = R.surface_material(*args); got = T.surface_material(*args)
        self.assertEqual(got.shape, (300, 3))
        self.assertLess(float(np.abs(ref - got).max()), 2e-3)
        self.assertGreater(float((np.abs(ref - got).max(-1) < 1e-4).mean()), .999)

    def test_float64_north_matches_numpy_promotion(self):
        case = random_case(5, (64, 96)); case['north'] = case['north'].astype(np.float64)
        self.compare(case, 30, None, False, True, 'north64')

    def test_available_is_cached_bool(self):
        self.assertIs(T.available(), True)


if __name__ == '__main__':
    unittest.main()
