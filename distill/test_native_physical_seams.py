from pathlib import Path
import unittest

import numpy as np

from distill.check_physical_seams import partition_stats, stage_paths


class NativePhysicalSeamsTests(unittest.TestCase):
    def test_native_base_can_share_application_coarse_and_decoder(self):
        paths, shared = stage_paths(native_base=Path('native.pt'),
            conditioning_coarse=Path('coarse4.pt'), conditioning_decoder=Path('decoder.pt'))
        self.assertEqual(paths, {})
        self.assertEqual(shared, dict(coarse=Path('coarse4.pt'), decoder=Path('decoder.pt')))

    def test_teacher_only_accepts_the_same_shared_stages(self):
        _, shared = stage_paths(conditioning_coarse=Path('coarse2.pt'))
        self.assertEqual(shared, dict(coarse=Path('coarse2.pt')))

    def test_per_candidate_stage_cannot_override_shared_conditioning(self):
        with self.assertRaises(ValueError):
            stage_paths(native_base=Path('native.pt'), coarse=Path('coarse4.pt'))
        with self.assertRaises(ValueError):
            stage_paths(base=Path('base.pt'), coarse=Path('coarse.pt'), decoder=Path('decoder.pt'),
                conditioning_coarse=Path('other.pt'))

    def test_partition_metrics_detect_disagreement_on_both_axes(self):
        size, halo = 16, 8
        y, x = np.mgrid[-halo:2*size+halo, -halo:2*size+halo]
        field = (100 + 3*y + x).astype(np.float32)
        full = field[halo:halo+2*size, halo:halo+2*size]
        tiles = {(y,x):field[y*size:y*size+size+2*halo,
                            x*size:x*size+size+2*halo].copy()
                 for y in range(2) for x in range(2)}
        exact, _ = partition_stats(full, tiles, halo)
        self.assertEqual(exact['max_m'], 0)
        self.assertEqual(exact['overlap_max_m'], 0)
        for tile in ((0,1), (1,0)):
            changed = {key:value.copy() for key,value in tiles.items()}
            changed[tile] += 7
            metrics, _ = partition_stats(full, changed, halo)
            self.assertEqual(metrics['max_jump_error_m'], 7)
            self.assertEqual(metrics['overlap_max_m'], 7)


if __name__ == '__main__':
    unittest.main()
