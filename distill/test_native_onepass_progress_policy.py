"""Training continuation cannot reward better loss with worse terrain evidence."""
import copy
import unittest

from distill.native_onepass_progress_policy import KINDS, evidence, compare, select


def reports(scale=1., step=20000, selection='ema'):
    result = []
    for coarse in ('coarse2', 'coarse4'):
        rows = []
        for index, kind in enumerate(sorted(KINDS)):
            for lod in (0, 3):
                rows.append(dict(kind=kind, seed=index, coarse=coarse, lod=lod,
                    mae_m=100*scale, coast_disagreement=.1*scale,
                    high_frequency_rms_ratio=2.**scale,
                    shape=dict(topographic_position=[dict(available=True, tpi_mae_m=10*scale)])))
        result.append(dict(status='complete', representation='packed-nf4', weight_bytes=84593780,
                           gpu='4090', step=step, selected_weights=selection, rows=rows))
    return result


class ProgressTests(unittest.TestCase):
    def test_matched_terrain_improvement_continues(self):
        result = compare(evidence(reports(.9, 25000)), evidence(reports()))
        self.assertTrue(result['improved'])
        self.assertAlmostEqual(result['relative_score'], .9)

    def test_identical_or_worse_terrain_does_not_extend(self):
        reference = evidence(reports())
        self.assertFalse(compare(reference, reference)['improved'])
        self.assertFalse(compare(evidence(reports(1.2)), reference)['eligible'])

    def test_coast_regression_blocks_otherwise_better_average(self):
        actual = reports(.7, 25000)
        for report in actual:
            for row in report['rows']:
                if row['kind'] == 'wet-complex-coast':
                    row['coast_disagreement'] = .2
        result = compare(evidence(actual), evidence(reports()))
        self.assertLess(result['relative_score'], 1)
        self.assertFalse(result['improved'])
        self.assertTrue(result['coast_regressions'])

    def test_valley_regression_blocks_better_height_error(self):
        actual = reports(.7, 25000)
        for report in actual:
            for row in report['rows']:
                if row['kind'] == 'wet-high-mountain':
                    row['shape']['topographic_position'][0]['tpi_mae_m'] = 20.
        result = compare(evidence(actual), evidence(reports()))
        self.assertFalse(result['eligible'])

    def test_selection_prefers_valid_ema_over_regressing_raw(self):
        selected, result = select([evidence(reports(1.2, 25000, 'raw')),
                                   evidence(reports(.9, 25000, 'ema'))], evidence(reports()))
        self.assertEqual(selected['selection'], 'ema')
        self.assertTrue(result['improved'])

    def test_missing_scenes_different_gpu_or_bad_metrics_cannot_signal_progress(self):
        actual = reports()
        actual[0]['rows'].pop()
        with self.assertRaises(ValueError):
            evidence(actual)
        actual = evidence(reports())
        actual['gpu'] = '5090'
        with self.assertRaises(ValueError):
            compare(actual, evidence(reports()))
        actual = reports()
        actual[0]['rows'][0]['mae_m'] = float('nan')
        with self.assertRaises(ValueError):
            evidence(actual)

    def test_three_mountains_do_not_overweight_the_mountain_climate(self):
        actual, previous = reports(.9), reports()
        for bundle in (actual, previous):
            for report in bundle:
                mountains = [row for row in report['rows'] if row['kind'] == 'wet-high-mountain']
                for seed in (100, 101):
                    report['rows'].extend(copy.deepcopy(row) | dict(seed=seed) for row in mountains)
        self.assertAlmostEqual(compare(evidence(actual), evidence(previous))['relative_score'], .9)


if __name__ == '__main__':
    unittest.main()
