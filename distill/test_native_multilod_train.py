import unittest

import numpy as np

from distill.native_mixed_bank import KINDS,VARIANTS
from distill.native_multilod_train import choose_record


class RareSamplerTests(unittest.TestCase):
    def records(self):
        return [dict(file=f'{variant}-{kind}',coarse_variant=variant,case_kind=kind)
                for variant in VARIANTS for kind in (None,*KINDS)]

    def test_rare_draw_covers_each_case_and_each_coarse(self):
        generator = np.random.default_rng(123)
        drawn = [choose_record(self.records(),generator,1.) for _ in range(600)]
        self.assertEqual({(r['coarse_variant'],r['case_kind']) for r in drawn},
                         {(v,k) for v in VARIANTS for k in KINDS})
        self.assertTrue(all(r['case_kind'] for r in drawn))

    def test_general_draw_does_not_double_count_rare_locations(self):
        generator = np.random.default_rng(123)
        self.assertTrue(all(choose_record(self.records(),generator,0.)['case_kind'] is None for _ in range(50)))

    def test_resume_step_seed_reproduces_the_same_source(self):
        seed = np.random.SeedSequence([20261013,500])
        a = choose_record(self.records(),np.random.default_rng(seed))
        b = choose_record(self.records(),np.random.default_rng(seed))
        self.assertEqual(a,b)

    def test_missing_rare_stratum_refuses_a_silent_fallback(self):
        records = [r for r in self.records() if r['case_kind'] is None]
        with self.assertRaisesRegex(ValueError,'stratum is missing'):
            choose_record(records,np.random.default_rng(1),1.)


if __name__=='__main__':
    unittest.main()
