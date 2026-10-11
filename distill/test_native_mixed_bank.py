import unittest

from distill.native_mixed_bank import KINDS, validate_inventory


class MixedSplitTests(unittest.TestCase):
    def rows(self):
        return [dict(metadata=dict(seed=start+i, split=split, profile='natural', y=0, x=32*i),
                     case_kind=kind)
                for split,start in [('train',10000),('val',200000)]
                for i,kind in enumerate(KINDS)]

    def test_independent_worlds_cover_all_critical_cases(self):
        worlds = validate_inventory(self.rows())
        self.assertFalse(worlds['train'] & worlds['val'])
        self.assertEqual(len(worlds['train']), 4)

    def test_same_world_cannot_cross_splits_even_with_other_profile(self):
        rows = self.rows()
        rows[-1]['metadata'].update(seed=rows[0]['metadata']['seed'], profile='orogen')
        with self.assertRaisesRegex(ValueError, 'worlds overlap'):
            validate_inventory(rows)

    def test_reserved_final_test_world_is_rejected(self):
        rows = self.rows()
        rows[0]['metadata']['seed'] = 42
        with self.assertRaisesRegex(ValueError, 'worlds overlap'):
            validate_inventory(rows)

    def test_duplicate_location_cannot_inflate_case_coverage(self):
        rows = self.rows()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, 'more than once'):
            validate_inventory(rows)

    def test_missing_wet_coast_validation_is_rejected(self):
        rows = [r for r in self.rows() if not (r['metadata']['split']=='val' and
                                               r['case_kind']=='wet-complex-coast')]
        with self.assertRaisesRegex(ValueError, 'all four'):
            validate_inventory(rows)


if __name__ == '__main__':
    unittest.main()
