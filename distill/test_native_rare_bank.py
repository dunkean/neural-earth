"""Rare native proposals keep aligned geometry and independent worlds."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from distill.native_rare_bank import BUCKETS, core_candidates, select_candidates
from distill.native_decoded_eval import qualified_validation


class NativeRareBankTests(unittest.TestCase):
    def test_local_low_coast_can_qualify_in_a_large_mixed_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scene.npz'
            y, x = -32, 32
            yy, xx = np.mgrid[:256, :256]
            height = np.where(xx+x < 120, -4., 20.)
            q = np.sign(height)*np.sqrt(np.abs(height))
            target = np.zeros((5, 256, 256), np.float32)
            target[4] = (q+31.4)/38.6
            coarse = np.zeros((6, 12, 12), np.float32)
            coarse[2] = 18.; coarse[4] = 180.
            np.savez(path, target=target, coarse=coarse,
                metadata=json.dumps(dict(y=y, x=x, crop=256, coarse_y=-3, coarse_x=-1,
                                         seed=12345, profile='natural', split='val')))
            candidates = core_candidates(path, {'low-plain-sea'})
            self.assertTrue(candidates)
            for row in candidates:
                self.assertEqual(row['metadata']['y']%96, 0)
                self.assertEqual(row['metadata']['x']%96, 0)
                self.assertEqual(row['metadata']['split'], 'val')
                self.assertGreater(row['proposal_lowfreq']['land'], .1)
                self.assertLess(row['proposal_lowfreq']['land'], .9)

    def test_selects_distinct_worlds_and_rejects_cross_split_leakage(self):
        rows = []
        for split, start in (('train', 1000), ('val', 2000)):
            for i, (kind, archetype) in enumerate(BUCKETS):
                for j in range(2):
                    rows.append(dict(metadata=dict(seed=start+i*10+j, profile='natural', split=split,
                                                    y=0, x=0), proposal_kind=kind, selection_score=j))
        selected, counts = select_candidates(rows, {'train': 2, 'val': 2})
        self.assertEqual(len(selected), 16)
        self.assertTrue(all(row['selected'] == 2 for row in counts.values()))
        rows[-1]['metadata']['seed'] = rows[0]['metadata']['seed']
        # Keep the conflicting proposal spatially eligible: near duplicates
        # are deliberately excluded before the final world-separation check.
        rows[-1]['metadata'].update(y=960, x=960)
        with self.assertRaises(ValueError):
            select_candidates(rows, {'train': 2, 'val': 2})

    def test_validation_excludes_failed_qualifications_and_reserved_worlds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'status.json').write_text(json.dumps(dict(status='complete')))
            rows = [dict(file=f'val-{i:04d}.npz', split='val', seed=2000+i,
                         requested=kind, physical_qualified=True)
                    for i, (kind, _) in enumerate(BUCKETS)]
            rejected = dict(rows[0], file='val-0004.npz', physical_qualified=False)
            rows.append(rejected)
            (root/'qualification.json').write_text(json.dumps(dict(rows=rows)))
            files, selected = qualified_validation(root)
            self.assertEqual(len(files), 4)
            self.assertNotIn(rejected['file'], selected)
            rows[0]['seed'] = 42
            (root/'qualification.json').write_text(json.dumps(dict(rows=rows)))
            with self.assertRaises(ValueError):
                qualified_validation(root)


if __name__ == '__main__':
    unittest.main()
