"""Mountain qualification rejects easy hills, dry mountains and world leakage."""
import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from distill.wet_mountains import check_splits, mountain_stats, qualifies


class WetMountainTests(unittest.TestCase):
    def test_altitude_and_rain_alone_do_not_qualify_a_plateau(self):
        stats = mountain_stats(np.full((64, 64), 3500.), 3)
        self.assertFalse(qualifies(stats, dict(rain=1800.)))

    def test_steep_wet_mountain_requires_decoded_valley_amplitude(self):
        yy, xx = np.mgrid[:64, :64]
        height = 3500.+900.*np.sin(xx*.18)+600.*np.sin(yy*.13)
        stats = mountain_stats(height, 3)
        self.assertTrue(qualifies(stats, dict(rain=1800.)))
        self.assertFalse(qualifies(stats, dict(rain=600.)))
        self.assertFalse(qualifies(stats | dict(tpi8_rms_m=0.), dict(rain=1800.), decoded=True))
        self.assertTrue(qualifies(stats, dict(rain=1800.), decoded=True))

    def test_reserved_and_cross_split_worlds_cannot_enter_training(self):
        existing = [dict(metadata=dict(seed=1234, split='train'))]
        with self.assertRaises(ValueError):
            check_splits([dict(metadata=dict(seed=1234, split='val'))], existing)
        with self.assertRaises(ValueError):
            check_splits([dict(metadata=dict(seed=42, split='train'))], [])
        with self.assertRaises(ValueError):
            check_splits([dict(metadata=dict(seed=777, split='final'))], [])
        check_splits([dict(metadata=dict(seed=42, split='final')),
                      dict(metadata=dict(seed=999, split='val'))], existing)

    def test_progress_selection_cannot_include_a_mislabeled_final_scene(self):
        from distill.wet_mountain_progress import selected_validation
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(file='final-0000-coarse4.npz', coarse_variant='coarse4',
                       metadata=dict(seed=42, split='final'))
            (root/'manifest.json').write_text(json.dumps(dict(scenes=[row])))
            (root/'status.json').write_text(json.dumps(dict(status='complete')))
            (root/'qualification.json').write_text(json.dumps(dict(rows=[dict(
                file=row['file'], coarse_variant='coarse4', seed=42,
                split='val', physical_qualified=True)])))
            with self.assertRaisesRegex(ValueError, 'identities differ'):
                selected_validation(root, 'coarse4')


if __name__ == '__main__':
    unittest.main()
