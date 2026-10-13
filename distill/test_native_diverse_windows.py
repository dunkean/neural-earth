import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from distill.native_diverse_windows import choose_windows, read_window
from distill.native_mixed_bank import KINDS, VARIANTS


class DiverseWindowTests(unittest.TestCase):
    def test_batch_has_eight_worlds_balanced_phases_and_rare_cases(self):
        records = [dict(coarse_variant=variant, case_kind=kind,
                        metadata=dict(seed=seed, profile='test'))
                   for variant in VARIANTS for kind in (None, *KINDS) for seed in range(16)]
        for step in range(12):
            batch = choose_windows(records, np.random.default_rng(step), step)
            self.assertEqual(len({r['metadata']['seed'] for _, r in batch}), 8)
            self.assertEqual(sum(r['case_kind'] is not None for _, r in batch), 2)
            for phase in (0, 1):
                subset = [r for p, r in batch if p == phase]
                self.assertEqual(len(subset), 4)
                self.assertEqual({r['coarse_variant'] for r in subset}, set(VARIANTS))
                self.assertEqual(sum(r['case_kind'] is not None for r in subset), 1)
            self.assertEqual(batch, choose_windows(records, np.random.default_rng(step), step))

    def test_missing_world_diversity_is_rejected(self):
        records = [dict(coarse_variant=variant, case_kind=kind,
                        metadata=dict(seed=1, profile='test'))
                   for variant in VARIANTS for kind in (None, *KINDS)]
        with self.assertRaises(ValueError):
            choose_windows(records, np.random.default_rng(7), 0)

    def test_direct_loader_keeps_the_selected_phase_and_window_target_aligned(self):
        with tempfile.TemporaryDirectory() as folder:
            contexts, labels = [], []
            for label, offsets in ((1.5625, (-64, -32, 0, 32, 64)), (.609375, (-32, 0, 32))):
                for y in offsets:
                    for x in offsets:
                        contexts.append([0, y//32, x//32]); labels.append(label)
            n = len(labels)
            inputs = np.arange(n, dtype=np.float16)[:, None, None, None]*np.ones((n, 5, 64, 64), dtype=np.float16)
            condition = np.zeros((n, 58), dtype=np.float16); condition[:, 0] = np.arange(n)
            path = Path(folder)/'scene.npz'
            np.savez(path, metadata=json.dumps(dict(y=0, x=0, native_size=128)),
                     contexts=np.array(contexts), labels=np.array(labels, dtype=np.float16),
                     inputs=inputs, condition=condition, target=inputs+100)
            for phase, label in enumerate((1.5625, .609375)):
                window = read_window(path, phase, np.random.default_rng(7), 'cpu')
                self.assertEqual(float(window['labels'][0]), label)
                self.assertEqual(float(window['inputs'][0, 0, 0, 0]), float(window['condition'][0, 0]))
                self.assertEqual(float(window['target'][0, 0, 0, 0]), float(window['condition'][0, 0])+100)


if __name__ == '__main__':
    unittest.main()
