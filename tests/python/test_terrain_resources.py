"""Resource layers: keyed draws, exact budgets across LODs, order independence."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import hashlib
import json
import unittest

import numpy as np

import terrain_resources as R


def atlas(h=64, w=128, seed=3):
    """Small synthetic Orogen atlas: two continents, a subduction margin,
    a collision belt, a continental rift and a hotspot."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:h, :w]
    lat = 90 - (y + .5) / h * 180
    height = np.where((x > 20) & (x < 90) & (np.abs(lat) < 60), 600., -3000.)
    height[(x > 100) & (x < 120) & (np.abs(lat) < 40)] = 300.
    height += rng.normal(0, 150, height.shape) * (height > 0)
    crust = (height < 0).astype(np.float32)
    boundaries = np.zeros((h, w), np.float32)
    boundaries[:, 20:22] = 1     # ocean-continent: subduction
    boundaries[:, 55] = 1        # inside the continent: collision
    boundaries[:, 75] = 2        # continental rift
    layers = dict(crust=crust, boundaries=boundaries,
                  convergence=np.where(boundaries == 1, 1.5, .2).astype(np.float32),
                  hotspot=np.where((np.abs(x - 110) < 3) & (np.abs(y - 32) < 3), .6, 0.).astype(np.float32),
                  backArc=np.where(np.abs(x - 30) < 3, -.05, 0.).astype(np.float32),
                  tecActivity=np.full((h, w), .5, np.float32), uplift=np.where(np.abs(x - 55) < 3, .3, 0.).astype(np.float32),
                  temperature_summer=((np.clip(30 - .4 * np.abs(lat), -40, 40) + 45) / 90).astype(np.float32),
                  temperature_winter=((np.clip(10 - .6 * np.abs(lat), -40, 40) + 45) / 90).astype(np.float32),
                  precip_summer=(.15 + .45 * (x % 40) / 40).astype(np.float32),
                  precip_winter=(.10 + .35 * (y % 32) / 32).astype(np.float32))
    for k, name in enumerate(('sandy', 'calcareous', 'clayey', 'ferrallitic', 'organic', 'podzolic', 'mineral')):
        layers['pedology_' + name] = np.full((h, w), 1 / 7, np.float32)
    return layers, height.astype(np.float32)


def digest(f, deposits):
    h = hashlib.sha256()
    for name in sorted(k for k, v in f.items() if isinstance(v, np.ndarray)):
        h.update(name.encode()); h.update(np.ascontiguousarray(f[name]).tobytes())
    h.update(json.dumps(deposits, sort_keys=True).encode())
    return h.hexdigest()


class ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers, cls.height = atlas()
        # A 1,000 km sphere keeps repeated population/determinism tests bounded;
        # the live captures and benchmark validate the full 4,000 km world.
        cls.bounds = (-np.pi*5e5, -np.pi*2.5e5, np.pi*5e5, np.pi*2.5e5)
        cls.grid, cls.f, cls.deposits = R.compile_world(cls.layers, cls.height, 42, lod=12, balance=False, bounds=cls.bounds)

    def test_keys_match_chronos_rust(self):
        # Values printed by chronos-core keys.rs (rustc 1.99) for the same inputs.
        self.assertEqual(R.key(42, ['resources', 'deposit.count', 'lod10']), 9210688585695448675)
        self.assertEqual(R.key(0, []), 6566800829925814604)
        self.assertEqual(R.key(7985403534069300490, ['a', 'bc']), 18197790663204608793)
        self.assertAlmostEqual(float(R.to_unit(np.uint64(R.key(42, ['x'])))), 0.08454580298607017, places=15)
        prefix = R.key(42, ['resources', 'deposit.count', 'lod10'])
        self.assertEqual(int(R.splitmix_np(np.uint64(prefix) ^ np.uint64(0x123456789abcdef0))), 515893553741813438)

    def test_spherical_areas_close_and_nest(self):
        for lod in (10, 12, 14):
            g = R.Lattice(lod)
            self.assertAlmostEqual(g.area_m2.sum() / (4 * np.pi * R.RADIUS_M ** 2), 1, places=12)
        parent = R.Lattice(10, (0, 0, 30720, 30720))
        children = R.Lattice(7, (0, 0, 30720, 30720))
        self.assertEqual(children.shape, (8, 8))
        self.assertAlmostEqual(children.area_m2.sum() / parent.area_m2.sum(), 1, places=12)

    def test_resized_world_bounds_and_cache_identity(self):
        from types import SimpleNamespace
        import terrain_resource_layers as L
        from terrain_generation import _defaults
        from terrain_geometry import world_bounds
        bounds = world_bounds(_defaults('orogen'))
        self.assertEqual(_defaults('orogen')['world_diameter_km'], 4000.)
        g = R.Lattice(10, bounds=bounds)
        self.assertEqual(g.shape, (206, 410))
        self.assertAlmostEqual(g.area_m2.sum() / (4 * np.pi * 2e6 ** 2), 1, places=12)
        a = SimpleNamespace(seed=42, height_m=self.height, bounds=bounds)
        b = SimpleNamespace(seed=42, height_m=self.height, bounds=R.BOUNDS)
        self.assertNotEqual(L._identity(a), L._identity(b))
        self.assertNotEqual(L._v2_identity(a), L._v2_identity(b))
        field = np.arange(np.prod(g.shape)).reshape(g.shape)
        eps = .001
        v = R.sample_lattice(g, field, [bounds[0] + eps, bounds[2] - eps], [0])
        np.testing.assert_allclose(v[0, 0], v[0, 1], rtol=0, atol=.001)

    def test_cell_ids_roundtrip_and_parents(self):
        i, j = np.array([-5, 0, 7, -(1 << 20)]), np.array([3, -1, -9, 1 << 19])
        for lod in (-6, 0, 3, 10, 15):
            lods, ii, jj = R.decode_id(R.cell_id(lod, i, j))
            np.testing.assert_array_equal(ii, i); np.testing.assert_array_equal(jj, j)
            self.assertTrue((lods == lod).all())
        pi, pj = R.parent_index(i, j, 3)
        np.testing.assert_array_equal(pi, np.floor(i / 8).astype(int))
        np.testing.assert_array_equal(pj, np.floor(j / 8).astype(int))

    def test_apportion_exact_and_order_independent(self):
        rng = np.random.default_rng(0)
        ids = R.cell_id(3, np.arange(50), np.zeros(50, int))
        w = R.quantize(rng.random(50) ** 3)
        a = R.apportion(123457, w, ids)
        self.assertEqual(a.sum(), 123457)
        perm = rng.permutation(50)
        b = R.apportion(123457, w[perm], ids[perm])
        np.testing.assert_array_equal(b, a[perm])
        self.assertEqual(R.apportion(9, np.zeros(4, np.int64), ids[:4]).sum(), 9)
        np.testing.assert_array_equal(R.apportion(0, w, ids), 0)

    def test_world_is_deterministic_and_seed_addressed(self):
        g, f, d = R.compile_world(self.layers, self.height, 42, lod=12, balance=False, bounds=self.bounds)
        self.assertEqual(digest(f, d), digest(self.f, self.deposits))
        _, f2, d2 = R.compile_world(self.layers, self.height, 43, lod=12, balance=False, bounds=self.bounds)
        self.assertNotEqual([x['id'] for x in d2], [x['id'] for x in self.deposits])
        # Seed only drives keyed parts: climate-derived layers do not move.
        np.testing.assert_array_equal(f2['npp_g_m2'], self.f['npp_g_m2'])

    def test_parallel_compile_matches_serial_and_reports_stages(self):
        # Worker processes (Render v2 months, deposit placement) change nothing.
        import terrain_resource_workers as W
        stages = []
        g, f, d = R.compile_world(self.layers, self.height, 42, lod=12, balance=False, bounds=self.bounds, parallel=W.parallel_map,
                                  progress=lambda stage, fields: stages.append((stage, 'land' in fields)))
        self.assertEqual(digest(f, d), digest(self.f, self.deposits))
        self.assertEqual(stages, [('climate', True), ('tectonics', True), ('environment', True), ('geology', True), ('deposits', True)])

    def test_cached_render_v2_environment_gives_the_same_world(self):
        v2 = {k: self.f[k] for k in R.V2_FIELDS}
        g, f, d = R.compile_world(self.layers, self.height, 42, lod=12, balance=False, bounds=self.bounds, v2_fields=v2)
        self.assertEqual(digest(f, d), digest(self.f, self.deposits))

    def test_bounds_and_partitions(self):
        f, land = self.f, self.f['land']
        ctx = sum(f['context_' + c] for c in R.CONTEXTS)
        rock = sum(f['rock_' + r] for r in R.ROCKS)
        np.testing.assert_allclose(ctx, 1, atol=1e-5)
        np.testing.assert_allclose(rock, 1, atol=1e-5)
        for name in ('cereal_suitability', 'tree_cover', 'transhumance', 'stone_ashlar', 'clay', 'sand_gravel'):
            self.assertTrue(((f[name] >= 0) & (f[name] <= 1 + 1e-6)).all(), name)
        self.assertTrue((f['npp_g_m2'] <= 3000).all() and (f['npp_g_m2'] >= 0).all())
        np.testing.assert_allclose(f['npp_month_g_m2'].sum(axis=0)[land], f['npp_g_m2'][land], rtol=1e-4)
        self.assertTrue((f['arable_area_m2'] <= self.grid.area_m2 + 1e-6).all())
        for name in R.EXTENSIVE:
            self.assertTrue((np.asarray(f[name]) >= 0).all(), name)
        self.assertTrue((f['harvest_month'][f['growing_months'] > 0] < 12).all())
        # Tectonic contexts land where the synthetic plates put them.
        col = self.grid.x < 0
        self.assertGreater(f['context_collision'][land].max(), .3)
        self.assertGreater(f['context_rift'][land].max(), .3)

    def test_deposits_follow_lithology_and_context(self):
        self.assertGreater(len(self.deposits), 0)
        f = self.f
        index = {int(c): (j, i) for (j, i), c in np.ndenumerate(self.grid.ids)}
        for d in self.deposits:
            j, i = index[d['owner_cell']]
            spec = {**R.FAMILIES, **R.PLACERS}[d['family']]
            self.assertTrue(f['land'][j, i])
            self.assertGreater(f['prospect_' + d['family']][j, i], 0)
            if 'hosts' in spec:
                self.assertGreater(sum(f['rock_' + r][j, i] + f['rock_expected_' + r][j, i]
                                       for r in spec['hosts']), 0)
            # The position cell is a descendant of the owner cell.
            lod, ci, cj = R.decode_id(np.uint64(d['owner_cell']))
            self.assertEqual(d['position_lod'], R.placement_lod(int(lod)))
            pi, pj = R.parent_index(d['position_i'], d['position_j'], int(lod) - d['position_lod'])
            self.assertEqual((int(pi), int(pj)), (int(ci), int(cj)))
            for m, v in d['contained'].items():
                self.assertGreaterEqual(v, d['medieval_recoverable'][m])
                self.assertGreaterEqual(d['medieval_recoverable'][m], 0)
        evap = [d for d in self.deposits if d['family'] == 'evaporite_salt']
        for d in evap:
            self.assertGreater(f['rock_evaporite'][index[d['owner_cell']]], 0)

    def test_stock_layers_equal_deposit_sums(self):
        for metal in ('iron', 'copper', 'gold', 'tin', 'lead', 'silver'):
            if 'stock_' + metal not in self.f:
                continue
            total = sum(d['contained'].get(metal, 0) for d in self.deposits)
            self.assertEqual(int(self.f['stock_' + metal].sum()), total)

    def test_budget_descent_conserves_and_ignores_request_order(self):
        f, g = self.f, self.grid
        committed = R.committed(f, 'arable_area_m2')
        cells = [tuple(c) for c in np.argwhere(committed > 0)[:3]]
        self.assertTrue(cells)
        first = {c: R.descend_budget(g, f, 'arable_area_m2', *c, lod=8) for c in cells}
        second = {c: R.descend_budget(g, f, 'arable_area_m2', *c, lod=8) for c in reversed(cells)}
        for c in cells:
            ids, amounts = first[c]
            self.assertEqual(int(amounts.sum()), int(committed[c]))
            np.testing.assert_array_equal(amounts, second[c][1])
            # Coarser LOD = sum of children: LOD 8 -> LOD 11 parents within the cell.
            lods, ii, jj = R.decode_id(ids.ravel())
            pi, pj = R.parent_index(ii, jj, 3)
            sums = R.aggregate(amounts.ravel(), R.cell_id(11, pi, pj))
            self.assertEqual(sum(sums.values()), int(committed[c]))

    def test_region_aggregates_are_exact(self):
        regions = (np.arange(self.grid.shape[1])[None, :] // 7 + 100 * (np.arange(self.grid.shape[0])[:, None] // 5))
        summary = R.region_summary(self.grid, self.f, self.deposits, regions)
        for name in R.EXTENSIVE:
            total = sum(v['natural'][name]['value'] for v in summary.values() if name in v['natural'])
            self.assertEqual(total, int(R.committed(self.f, name).sum()), name)
        self.assertEqual(sum(len(v['deposits']) for v in summary.values()), len(self.deposits))

    def test_fantasy_is_isolated(self):
        _, f, d = R.compile_world(self.layers, self.height, 42, lod=12, balance=False, bounds=self.bounds, fantasy=True)
        plain = [x for x in d if not x['family'].startswith('fantasy')]
        self.assertEqual(plain, self.deposits)
        for name, value in self.f.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(f[name], value, name)
        self.assertTrue(all(x['status'] == 'synthetic' for x in d if x['family'].startswith('fantasy')))

    def test_history_delta_is_monotone_and_exact(self):
        rng = np.random.default_rng(1)
        ids = R.cell_id(3, np.arange(64), np.zeros(64, int))
        areas = rng.uniform(5e4, 6e4, 64)
        order = R.clearing_order(42, ids, rng.random(64), parent_id=7)
        previous = np.zeros(64)
        for cleared in (0, 1e5, 7.7e5, 2e6, areas.sum()):
            frac = R.cleared_cells(order, areas, cleared)
            self.assertAlmostEqual(float((frac * areas).sum()), cleared, delta=1e-6 * max(cleared, 1))
            self.assertTrue((frac >= previous - 1e-12).all())
            previous = frac
        dep = self.deposits[0]
        metal = next(iter(dep['contained']))
        left = R.deposit_remaining(dep, {metal: dep['contained'][metal] + 5})
        self.assertEqual(left[metal], 0)
        path = R.forest_stock(100., 100., .05, [10] * 30)
        self.assertTrue((path >= 0).all() and path[-1] < 100)

    def test_render_v2_fields_are_consumed_not_recomputed(self):
        from terrain_render_v2 import evaluate
        from terrain_render import terrain_derivatives
        from terrain_soil import seed_value
        f = self.f
        land = np.argwhere(f['land'] & (f['tree_cover'] > 0))
        self.assertTrue(len(land))
        # Woody cover is the evaluator's, minus perennial snow, on land only.
        np.testing.assert_allclose(f['tree_cover'], f['v2_woody'] * (1 - f['v2_perennial_snow']) * f['land'], rtol=1e-6)
        # Snow-covered months carry less grazing than the same month without snow.
        herb = f['v2_grass_month']
        self.assertTrue(((herb >= 0) & (herb <= 1)).all())
        snowy = f['v2_snow_month'] > .9
        if snowy.any():
            self.assertLess(float(herb[snowy].max()), .2)

    def test_value_noise_is_scale_independent(self):
        x = np.array([[1234567.]]); y = np.array([[-765432.]])
        a = R.value_noise(42, 'lithology-granite', 13, x, y)
        b = R.value_noise(42, 'lithology-granite', 13, x + 0., y + 0.)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(-1 <= float(np.ravel(a)[0]) <= 1)


class ResourceDistrictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bounds = (-np.pi * 5e5, -np.pi * 2.5e5, np.pi * 5e5, np.pi * 2.5e5)
        layers, height = atlas()
        cls.grid, cls.fields, cls.deposits = R.compile_world(layers, height, 42, bounds=cls.bounds, balance=False)

    def test_inspector_keeps_the_selected_resource_after_diagnostic_values(self):
        import terrain_resource_layers as L
        from unittest.mock import patch
        state=L._World(self.grid,self.fields,self.deposits,42,'inspector-test')
        n=next(i for i,d in enumerate(self.deposits) if d['medieval_recoverable'].get('iron',0)>0)
        x,y=(state.dep_di[n]+.5)*30,(state.dep_dj[n]+.5)*30
        # Put an unrelated diagnostic last, as in the real menu.
        spec=dict(kind='ramp',value=lambda f,g:np.zeros(g.shape),label='Diagnostic',unit='indice',status='derived',proxy=False)
        with patch.object(L,'world_resources',return_value=(self.grid,self.fields,self.deposits,0,'fixture')), \
             patch.object(L,'world_state',return_value=state), \
             patch.object(L,'specs',return_value={'resources-dep-potential-tungsten':spec}):
            data=L.inspect(object(),x,y,radius_m=1000,mode='resources-dep-iron')
        self.assertTrue(data['deposits'])
        self.assertEqual(data['deposits'][0]['distance_m'],0)
        self.assertTrue(all(d['medieval_recoverable'].get('iron',0)>0 for d in data['deposits']))

    def test_district_marks_preserve_counts_and_tonnage(self):
        g, f = self.grid, self.fields
        actual = {d['id']: d for d in self.deposits}
        expected = {}
        for name, spec in {**R.FAMILIES, **R.PLACERS}.items():
            domain = ('resources', 'deposit', name)
            lam = spec['density'] * np.clip(f['prospect_' + name], 0, 1) * g.area_m2 / 1e10
            count = R._poisson(lam, R.to_unit(R.cell_keys(42, domain + ('count', 'lod10'), g.ids)))
            for j, i in zip(*np.nonzero(count)):
                for k in range(int(count[j, i])):
                    d = R._deposit(42, domain, name, spec, g, f, j, i, k, position=R.cell_id(3, 0, 0))
                    expected[d['id']] = d
        self.assertTrue(actual)
        self.assertEqual(set(actual), set(expected))
        for ident, d in actual.items():
            self.assertEqual(d['contained'], expected[ident]['contained'])
            self.assertEqual(d['ore_t'], expected[ident]['ore_t'])
            self.assertIn('district_id', d)
            x, y = (d['position_i'] + .5) * 240, (d['position_j'] + .5) * 240
            self.assertTrue(g.bounds[0] <= x < g.bounds[2] and g.bounds[1] <= y < g.bounds[3])
            lod, i, j = R.decode_id(np.uint64(d['owner_cell']))
            self.assertEqual((d['position_i'] >> 7, d['position_j'] >> 7), (int(i), int(j)))
        for metal in R.METALS:
            for prefix, key in (('stock_', 'contained'), ('recoverable_', 'medieval_recoverable')):
                self.assertEqual(int(f.get(prefix + metal, np.array(0)).sum()), sum(d[key].get(metal, 0) for d in actual.values()))

    def test_district_placement_is_query_order_independent(self):
        g, f = self.grid, self.fields
        family = 'porphyry_cu'
        a = R.district_centres(42, family, 10, g.bounds)
        b = R.district_centres(42, family, 7, g.bounds)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        jj, ii = np.nonzero(f['prospect_' + family] > 0)
        jj, ii = jj[:6], ii[:6]
        kk = np.arange(len(jj))
        common = dict(seed=42, domain=('resources', 'deposit', family), lod=10,
                      field=f['prospect_' + family], bounds=g.bounds, name=family, centres=a, strike=f['structural_strike'])
        first = R.place_positions(**common, jj=jj, ii=ii, kk=kk)
        second = R.place_positions(**common, jj=jj[::-1], ii=ii[::-1], kk=kk[::-1])
        self.assertEqual(first, second[::-1])
        row, col, _ = np.argwhere(a[4])[0]
        intensity = R.district_intensity(a[0][row, col, 0], a[1][row, col, 0], family, a, f['structural_strike'], g)
        self.assertGreater(float(intensity), 1.)

    def test_sites_stay_in_fertile_family_tracts(self):
        index = {int(cid): ij for ij, cid in np.ndenumerate(self.grid.ids)}
        for d in self.deposits:
            p = self.fields['prospect_' + d['family']]
            self.assertGreaterEqual(p[index[d['owner_cell']]], .15*float(p.max())-1e-8)

    def test_family_potential_requires_its_tectonic_tract(self):
        f = dict(self.fields)
        f['tract_arc'] = np.zeros(self.grid.shape)
        f['tract_backarc'] = np.zeros(self.grid.shape)
        f['tract_collision'] = np.zeros(self.grid.shape)
        p = R.prospectivity(f, R.FAMILIES['porphyry_cu'])
        np.testing.assert_array_equal(p, 0)
        f['tract_arc'] = np.ones(self.grid.shape)
        self.assertGreater(float(R.prospectivity(f, R.FAMILIES['porphyry_cu']).max()), 0)

    def test_every_metal_potential_is_bounded_and_fantasy_isolated(self):
        import terrain_resource_gpu as G
        rng = np.random.default_rng(18)
        f = {'prospect_' + k: rng.uniform(-1, 2, (4, 8)) for k in {**R.FAMILIES, **R.PLACERS, **R.FANTASY}}
        families = {**R.FAMILIES, **R.PLACERS, **R.FANTASY}
        for metal in (*R.METALS, 'star_metal'):
            p = R.metal_potential(f, metal, fantasy=True)
            upper = sum(s['density'] * s['tonnage'][0] * s['metals'].get(metal, 0) for s in families.values()) / 1e4
            upper *= 1000 if metal in R.GRAM_METALS else 1
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(((p >= 0) & (p <= upper * (1 + 1e-12))).all(), metal)
            pkg = dict(seed32=42, layer=17, field=G.Field(np.full((1, 4, 8), upper, np.float32), -4, -2),
                       profile='mineral', seasonal=False, bounded=True, bounds=(0., upper), range=(0., upper))
            v = G.evaluate_continuous(pkg, np.arange(-400, 400), np.zeros(800, int), 3, None)
            self.assertTrue(((v >= 0) & (v <= np.float32(upper))).all(), metal)
            if metal != 'star_metal':
                np.testing.assert_array_equal(p, R.metal_potential(f, metal, fantasy=False))
        with self.assertRaises(KeyError):
            R.metal_potential(f, 'star_metal', fantasy=False)

    def test_stock_kernel_integral_at_every_lod_and_polar_clip(self):
        from scipy.integrate import quad
        import terrain_resource_gpu as G
        radius = 2e6
        bounds = (-np.pi * radius, -np.pi * radius / 2, np.pi * radius, np.pi * radius / 2)
        for lod in (10, 7, 6, 5, 4, 3, 2, 0, -2):
            su, sv = G.stock_radii(1200, 35, lod)
            theta = .83
            sy = np.hypot(float(su) * np.sin(theta), float(sv) * np.cos(theta))
            for y in (0, radius * 1.1, bounds[3] - 300):
                dj = int(y / 30)
                center = (dj + .5) * 30
                lo, hi = max(-1, (bounds[1] - center) / sy), min(1, (bounds[3] - center) / sy)
                integral = 16 / (5 * np.pi) * quad(lambda t: max(0, 1 - t*t)**2.5 * np.cos((center + sy*t) / radius), lo, hi, epsabs=1e-12)[0]
                norm = G.stock_normalization(dj, theta, su, sv, bounds)
                self.assertAlmostEqual(integral / float(norm), 1, delta=2e-4)

    def test_organic_stock_integral_at_every_lod_and_polar_clip(self):
        from scipy.integrate import quad
        import terrain_resource_gpu as G
        radius = 2e6
        bounds = (-np.pi*radius, -np.pi*radius/2, np.pi*radius, np.pi*radius/2)
        errors = []
        for lod in (10, 7, 6, 5, 4, 3, 2, 0, -2):
            for y in (0, radius*1.1, bounds[3]-300):
                d = dict(di=100, dj=int(y/30), theta=.83, a=1200, b=35, access=-.7, mass=123456)
                integral = 0.
                for ox, oy, a, b, weight in G.stock_lobes(d, lod, bounds):
                    cy = (d['dj']+.5)*30+float(oy)
                    sy = np.hypot(float(a)*np.sin(d['theta']), float(b)*np.cos(d['theta']))
                    lo, hi = max(-1, (bounds[1]-cy)/sy), min(1, (bounds[3]-cy)/sy)
                    exact = 16/(5*np.pi)*quad(lambda t: max(0,1-t*t)**2.5*np.cos((cy+sy*t)/radius), lo, hi, epsabs=1e-13)[0]
                    integral += float(weight)*exact/float(G.stock_normalization(d['dj'], d['theta'], a, b, bounds, oy))
                errors.append(abs(integral-1))
                self.assertAlmostEqual(integral, 1, delta=2e-4)
        print('organic stock max integral relative error:', max(errors), flush=True)


class ResourcePresentationTests(unittest.TestCase):
    def test_partial_deposit_stage_is_not_served_as_an_empty_map(self):
        import terrain_resource_layers as L
        from types import SimpleNamespace
        from unittest.mock import patch
        partial=SimpleNamespace(deposits=[])
        with patch.object(L,'world_state',return_value=partial), patch.object(L,'compile_status',return_value={'state':'compiling'}):
            with self.assertRaises(L.ResourcesPending):
                L.gpu_package(object(),'resources-dep-copper')

    def test_menu_contains_resources_and_diagnostics_without_stock_views(self):
        import terrain_resource_layers as L
        from unittest.mock import patch
        with patch.object(L, 'fantasy_enabled', return_value=False):
            specs = dict(L._specs())
        self.assertEqual({m.removeprefix('resources-dep-') for m, s in specs.items() if s['kind']=='points'},
                         {*R.METALS, 'points'})
        self.assertFalse(any('-stock-' in m or '-recoverable-' in m for m in specs if m.startswith('resources-dep-')))
        diagnostic = [s['group']=='diagnostic' for s in specs.values()]
        self.assertEqual(diagnostic, sorted(diagnostic))
        self.assertTrue(all(s['group']=='diagnostic' for m,s in specs.items() if '-potential-' in m or '-prospect-' in m))
        with patch.object(L, 'fantasy_enabled', return_value=True):
            self.assertIn('resources-dep-star-metal', dict(L._specs()))

    def test_symbols_use_recoverable_physical_mass_and_buckets(self):
        import terrain_resource_layers as L
        from types import SimpleNamespace
        grid = R.Lattice(10, bounds=(-30720., -30720., 30720., 30720.))
        state = SimpleNamespace(grid=grid, dep_ore=np.array([1e9,1e9,1e9]), dep_di=np.array([-1,1,2]), dep_dj=np.array([-1,1,2]),
            dep_shape=np.zeros(3),dep_color=np.zeros(3),dep_a=np.full(3,10.),dep_b=np.full(3,2.),dep_theta=np.zeros(3),
            dep_bend=np.zeros(3),dep_access=np.ones(3),dep_gossan=np.zeros(3),
            dep_mass={('medieval_recoverable','copper'):np.array([0,1,1e8]),
                      ('medieval_recoverable','gold'):np.array([0,1000,1e11])})
        cu, buckets = L._deposit_buffers(state,'resources-dep-copper','points')
        au, _ = L._deposit_buffers(state,'resources-dep-gold','points')
        self.assertEqual(len(cu),2)
        self.assertEqual(int(buckets[...,1].sum()),2)
        np.testing.assert_array_equal(cu[:,9], au[:,9])  # kilograms are converted to physical tonnes
        np.testing.assert_allclose(cu[:,9]**2,[9,49],rtol=1e-6)
        np.testing.assert_array_equal(cu[:,11],[1,1e8])

    def test_symbol_overlap_and_crossfade_are_order_independent(self):
        import terrain_resource_gpu as G
        small=dict(di=0,dj=0,shape=1,color=0,a=1.,b=1.,theta=0.,symbol=3.,mass=1)
        large=dict(small,color=1,mass=100,symbol=5.)
        palette=[(.8,.2,.1),(.1,.8,.3)]
        for lod in (10,7,4,2,0):
            a=G.deposit_color([small,large],np.array(0),np.array(0),lod,np.float32(0),palette)
            b=G.deposit_color([large,small],np.array(0),np.array(0),lod,np.float32(0),palette)
            np.testing.assert_array_equal(a,b)
        # Moving the footprint across the symbol transition cannot cause a pop.
        d=dict(small,a=120.,b=120.)
        a=G.deposit_contribution(d,np.array(3),np.array(0),0,np.float32(0))[0]
        b=G.deposit_contribution(dict(d,a=120.01,b=120.01),np.array(3),np.array(0),0,np.float32(0))[0]
        self.assertLess(abs(float(a-b)),.002)

    def test_dense_coarse_poisson_is_not_capped(self):
        self.assertGreater(int(R._poisson(np.array(100.),np.array(.5))),40)

    def test_industrial_mineral_grades_cannot_exceed_the_ore(self):
        from types import SimpleNamespace
        grid=SimpleNamespace(ids=np.array([[R.cell_id(10,0,0)]]),lod=10)
        f={'rugged_m':np.zeros((1,1)), 'temperature_c':np.zeros((12,1,1))}
        for name in ('gypsum_quarry','salt_spring','coal','evaporite_salt'):
            for k in range(20):
                d=R._deposit(42,('resources','deposit',name),name,R.FAMILIES[name],grid,f,0,0,k,position=R.cell_id(3,0,0))
                self.assertLessEqual(sum(d['contained'].values()), d['ore_t']+len(d['contained']))


class ResourceGpuMirrorTests(unittest.TestCase):
    """NumPy mirror of the WGSL evaluator: fixture, addressing, realized geology, display tolerance.

    Seed of the reference world's GPU key, synthetic 32 x 20 LOD-10 lattice at
    (-16, -10); rock shares come from terrain_resources.realize_rocks and are
    compared with the display at finer LODs (a display tolerance; the canonical
    budgets stay in terrain_resources and are tested exactly above).
    """

    @classmethod
    def setUpClass(cls):
        import terrain_resource_gpu as G
        cls.G = G
        rng = np.random.default_rng(5)
        cls.i0, cls.j0, cls.W, cls.H, cls.K = -16, -10, 32, 20, len(R.ROCKS)
        base = rng.dirichlet(np.ones(cls.K) * 2, (cls.H // 8 + 2, cls.W // 8 + 2)).transpose(2, 0, 1)
        y, x = np.mgrid[:cls.H, :cls.W] // 8
        expected = base[:, y, x]
        cls.expected = expected / expected.sum(axis=0)
        cls.strike = .5 + np.linspace(0, 1.5, cls.W)[None] + np.zeros((cls.H, 1))
        cls.seed32 = 0x1234abcd
        cls.bias = np.zeros(cls.K)
        cls.rocks = dict(seed32=cls.seed32, layer=R.ROCK_LAYER, classes=cls.K, aligned=R.ROCK_ALIGNED,
                         groups=R.ROCK_GROUPS, group_aligned=R.ROCK_GROUP_ALIGNED,
                         field=G.Field(R.realized_rock_field(cls.expected, cls.strike, cls.bias), cls.i0, cls.j0))

    def cell(self, i, j, lod):
        n = 2 ** (10 - lod)
        gi, gj = np.meshgrid((self.i0 + i) * n + np.arange(n), (self.j0 + j) * n + np.arange(n))
        return gi, gj

    def test_parity_fixture_matches_mirror(self):
        import importlib.util
        path = _REPO_ROOT / 'tools/verification/build_resources_gpu_fixture.py'
        spec = importlib.util.spec_from_file_location('build_resources_gpu_fixture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stored = json.loads((_REPO_ROOT / 'tests/fixtures/resources_gpu_parity.json').read_text(encoding='utf-8'))
        rebuilt = json.loads(json.dumps(module.build()))
        self.assertEqual(stored, rebuilt, 'tests/fixtures/resources_gpu_parity.json is stale: rerun the builder')

    def test_noise_is_keyed_by_position_not_tile(self):
        G = self.G
        gi, gj = np.meshgrid(np.arange(-300, -100), np.arange(40, 160))
        full = G.fbm(gi, gj, 3, 0x5eed, 9, 4, 10, .8)
        part = G.fbm(gi[30:70, 50:90], gj[30:70, 50:90], 3, 0x5eed, 9, 4, 10, .8)
        np.testing.assert_array_equal(full[30:70, 50:90], part)
        a = G.realized_class(self.rocks, gi[::7, ::7], gj[::7, ::7], 3)
        b = G.realized_class(self.rocks, gi[::7, ::7][::-1], gj[::7, ::7][::-1], 3)[::-1]
        np.testing.assert_array_equal(a, b)

    def shares(self, i, j, lod):
        c = self.G.realized_class(self.rocks, *self.cell(i, j, lod), lod).ravel()
        return np.bincount(c, minlength=self.K) / c.size

    def test_display_geology_matches_cell_shares_at_every_lod(self):
        # Cell shares as terrain_resources.realize_rocks computes them (LOD 8,
        # 4 x 4), against the display at LOD 6 and 3; measured TV 0.008 mean
        # on the reference world.
        tv = {6: [], 3: []}
        for j in range(2, self.H - 2, 3):
            for i in range(2, self.W - 2, 3):
                ref = self.shares(i, j, 8)
                for lod in tv:
                    tv[lod].append(.5 * np.abs(self.shares(i, j, lod) - ref).sum())
        for lod, values in tv.items():
            self.assertLess(float(np.mean(values)), .05, lod)

    def test_geology_forms_provinces_not_confetti(self):
        # LOD 3 (240 m) over 12 x 8 cells (370 km x 245 km): few class changes,
        # and each present class is a few large connected units.
        from scipy.ndimage import label
        n = 2 ** 7
        gi, gj = np.meshgrid(np.arange((self.i0 + 10) * n, (self.i0 + 22) * n, 4),
                             np.arange((self.j0 + 6) * n, (self.j0 + 14) * n, 4))
        c = self.G.realized_class(self.rocks, gi, gj, 3)
        changes = (c[1:] != c[:-1]).mean() + (c[:, 1:] != c[:, :-1]).mean()
        self.assertLess(float(changes), .02)
        for k in np.unique(c):
            labels, count = label(c == k)
            sizes = np.bincount(labels.ravel())[1:]
            self.assertGreater(float(sizes.max() / sizes.sum()), .5, f'class {k} is scattered')

    def test_realized_rock_shares_are_partitions_and_deterministic(self):
        # World shares are calibrated by a per-class bias; on this 1 000 km
        # domain one province noise (123-983 km) dominates, so only validity
        # and determinism are checked here (world shares: docs §5.1).
        import terrain_resources as RR
        land = np.ones((self.H, self.W), bool)
        grid = type('Grid', (), dict(lod=10, i=np.arange(self.W) + self.i0, j=np.arange(self.H) + self.j0,
                                     area_m2=np.ones((self.H, self.W))))()
        shares, bias = RR.realize_rocks(self.expected, self.strike, land, grid, 42)
        np.testing.assert_allclose(shares.sum(axis=0), 1, atol=1e-9)
        self.assertTrue(((shares >= 0) & (shares <= 1)).all())
        again, bias2 = RR.realize_rocks(self.expected, self.strike, land, grid, 42)
        np.testing.assert_array_equal(shares, again)
        np.testing.assert_array_equal(bias, bias2)

    def test_continuous_display_keeps_parent_means(self):
        G = self.G
        pkg = dict(seed32=0x1234abcd, layer=5, field=G.Field(np.full((1, self.H, self.W), 100., np.float32), self.i0, self.j0),
                   seasonal=False, profile='arable', range=(0., 1000.), bounded=False)
        ratio = np.array([G.evaluate_continuous(pkg, *self.cell(i, j, 4), 4, None).mean() / 100.
                          for j in range(2, self.H - 2) for i in range(2, self.W - 2)])
        self.assertAlmostEqual(float(ratio.mean()), 1., delta=.03)
        self.assertLess(float(np.percentile(np.abs(ratio - 1), 90)), .2)


if __name__ == '__main__':
    unittest.main()
