import unittest

from distill.native_optimization_audit import morphology_guard


def report(drops):
    rows = []
    for seed, drop in enumerate(drops):
        value = dict(valley_retention=.6-drop, tpi_correlation=.9)
        rows.append(dict(file=f'case-{seed}', seed=seed, coarse_variant='coarse4',
            case_kind='temperate-plain', profile='test', morphology=dict(fine=value, native=value)))
    return dict(rows=rows)


class OptimizationGuardTests(unittest.TestCase):
    def test_consistent_raw_regression_names_the_affected_group(self):
        reasons = morphology_guard(report([.05]*4), report([0.]*4))
        self.assertTrue(reasons)
        self.assertTrue(all(r['group'] == 'coarse4:temperate-plain' for r in reasons))
        self.assertTrue(all(r['lower95_drop'] > r['tolerance'] for r in reasons))

    def test_uncertain_small_changes_do_not_stop_the_pilot(self):
        self.assertEqual(morphology_guard(report([.02, -.02, -.02, .02]), report([0.]*4)), [])

    def test_undefined_candidate_morphology_is_not_discarded(self):
        candidate = report([0.]*4); candidate['rows'][0]['morphology']['fine'] = None
        reasons = morphology_guard(candidate, report([0.]*4))
        self.assertTrue(reasons)
        self.assertIn('undefined', reasons[0]['reason'])


if __name__ == '__main__':
    unittest.main()
