"""Canonical coarse profiles retain station identity and shared fixed water."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'backend'))
from terrain_hydrology_profiles import build_profiles


def reach(rid, xy, water, downstream=None, **extra):
    return dict(id=rid, xy=np.asarray(xy, float), water_m=np.asarray(water, float),
                downstream_id=downstream, basin_id=0, **extra)


def world(reaches, lakes=()):
    return SimpleNamespace(reaches=reaches, lakes=lakes, topology='plane', bounds=(0, 0, 100000, 100000),
                           sample=lambda x, y: dict(basin_id=np.zeros(np.shape(x), int)))


class CanonicalProfilesTest(unittest.TestCase):
    def test_downstream_bump_cannot_create_rise_or_exceed_source_envelope(self):
        raw = reach(0, [[0, 0], [16000, 0], [20000, 0]], [111, 110, 109])
        calls = []
        def sample(x, y):
            calls.append(len(x))
            return 80 + 65 * np.exp(-((x - 16000) / 5000) ** 2)
        built = build_profiles(world([raw]), [0], sample)
        p = built.profiles[0]
        self.assertEqual(len(calls), 1)
        self.assertTrue(np.all(np.diff(p['water_m']) <= 0))
        self.assertTrue(np.all(p['water_m'] <= np.interp(p['station'], [0, 1, 2], raw['water_m'])))
        self.assertEqual(built.level(0, 2), 109.)
        self.assertIn(1., p['station'])  # original vertex, not arc-length index
        self.assertGreater(built.metadata['max_incision_m'], 30.)
        np.testing.assert_array_equal(raw['water_m'], [111, 110, 109])
        with self.assertRaises(ValueError):
            p['water_m'][0] = 999
        with self.assertRaises(TypeError):
            built.profiles[0] = {}

    def test_downstream_closure_is_order_independent_and_confluences_equal(self):
        reaches = [reach(0, [[0, 0], [10000, 0]], [130, 110], 2),
                   reach(1, [[0, 10000], [10000, 0]], [120, 110], hydraulic_downstream_id=2),
                   reach(2, [[10000, 0], [20000, 0]], [110, 90])]
        w = world(reaches)
        sample = lambda x, y: 125 - x / 1000 + .001 * y
        first = build_profiles(w, [0], sample)
        second = build_profiles(w, [1, 0], sample)
        self.assertEqual(set(first.profiles), {0, 2})
        for rid in first.profiles:
            np.testing.assert_array_equal(first.profiles[rid]['water_m'], second.profiles[rid]['water_m'])
        self.assertEqual(second.level(0, 1), second.level(2, 0))
        self.assertEqual(second.level(1, 1), second.level(2, 0))

    def test_lake_entry_exit_keep_fixed_level_despite_conflicting_ground(self):
        reaches = [reach(0, [[0, 0], [10000, 0]], [130, 100], 1, enters_lake=0),
                   reach(1, [[12000, 0], [20000, 0]], [100, 70], leaves_lake=0)]
        built = build_profiles(world(reaches, [dict(level_m=100.)]), [0],
                               lambda x, y: 180 - x / 2000)
        self.assertEqual(built.level(0, 1), 100.)
        self.assertEqual(built.level(1, 0), 100.)
        self.assertTrue(all(np.all(np.diff(p['water_m']) <= 0) for p in built.profiles.values()))
        self.assertGreater(built.metadata['max_incision_m'], 30.)

    def test_barrier_cuts_instead_of_damming_upstream_and_normal_case_is_unchanged(self):
        # Real seed42 failure: a Coarse ridge 30+ m above the source line held
        # every upstream reach flat at the source level over a lowland.
        reaches = [reach(0, [[0, 0], [30000, 0]], [140, 120], 1),
                   reach(1, [[30000, 0], [60000, 0]], [120, 60])]
        w = world(reaches)
        lowland = lambda x: 100 - x / 3000
        ridge = lambda x, y: lowland(x) + 60 * np.exp(-((x - 45000) / 4000) ** 2)
        built = build_profiles(w, [0], ridge)
        up, down = built.profiles[0]['water_m'], built.profiles[1]['water_m']
        self.assertTrue(np.all(np.diff(np.r_[up, down]) <= 0))
        self.assertEqual(up[-1], down[0])
        # The tributary lowland bounds the junction: no fill above its fine valley.
        self.assertLessEqual(up.max(), lowland(0) - 15 + 2 + 1e-9)
        cut = ridge(45000, 0) - built.level(1, .5)
        self.assertGreater(cut, 30.)
        self.assertLessEqual(cut, 60. + 1e-9)
        self.assertGreater(built.metadata['reaches'][1]['conflict_lowering_m'], 0.)
        # Without a barrier the preferred 30 m rule is used exactly.
        plain = build_profiles(w, [0], lambda x, y: lowland(x))
        p = plain.profiles[1]
        valley = lowland(np.interp(p['station'], [0, 1], [30000, 60000]))
        expected = np.maximum(np.minimum(np.maximum.accumulate((valley - 30)[::-1])[::-1],
                                         np.interp(p['station'], [0, 1], [120, 60])), 60.)
        np.testing.assert_allclose(p['water_m'], expected)
        self.assertEqual(plain.metadata['reaches'][1]['conflict_lowering_m'], 0.)

    def test_known_profiles_are_reused_not_resampled_or_returned(self):
        reaches = [reach(0, [[0, 0], [10000, 0]], [130, 110], 1),
                   reach(1, [[10000, 0], [20000, 0]], [110, 90])]
        w = world(reaches)
        sample = lambda x, y: 125 - x / 1000
        full = build_profiles(w, [0], sample)
        seen = []
        reused = build_profiles(w, [0, 1], lambda x, y: seen.append(x.copy()) or sample(x, y),
                                known={1: full.profiles[1]})
        self.assertEqual(set(reused.profiles), {0})
        self.assertTrue(np.all(np.concatenate(seen) <= 10000 + 1e-6))
        np.testing.assert_array_equal(reused.profiles[0]['water_m'], full.profiles[0]['water_m'])


if __name__ == '__main__':
    unittest.main()
