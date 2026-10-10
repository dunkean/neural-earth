"""Regression tests for field invariance, deterministic resume and masked losses."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from distill.common import atomic_json, atomic_write, HOLDOUT_SEEDS
from distill.dataset import StepBatches
from distill.features import base_features, noise
from distill.jobs import proc_identity
from distill.student import Student, StudentConfig
from distill.train import losses
from distill.evaluate import audit


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_global_noise_is_independent_of_negative_crop_partition(self):
        for seed in (9281, 9282):
            whole = noise(seed, -96, -32, 128, 128, 5)
            upper = torch.cat([noise(seed, -96, -32, 64, 64, 5),
                               noise(seed, -96, 32, 64, 64, 5)], -1)
            lower = torch.cat([noise(seed, -32, -32, 64, 64, 5),
                               noise(seed, -32, 32, 64, 64, 5)], -1)
            torch.testing.assert_close(whole, torch.cat([upper, lower], -2), rtol=0, atol=0)

    def test_features_match_on_overlapping_global_regions(self):
        coarse = np.random.default_rng(7).normal(size=(6, 16, 16)).astype(np.float32)
        a = base_features(coarse, -8, -8, 9001, -128, -128, 256, [0]*5)
        b = base_features(coarse, -8, -8, 9001, -64, -64, 256, [0]*5)
        torch.testing.assert_close(a[:, 64:, 64:], b[:, :192, :192], rtol=0, atol=0)

    def test_student_valid_halo_removes_tile_boundary_effects(self):
        torch.manual_seed(17)
        model = Student(StudentConfig('base', 16, 2, (1,))).eval()
        # A nonzero head makes this a real network test, not the zero initializer.
        torch.nn.init.normal_(model.head.weight, std=.1)
        halo, core = model.halo, 64
        full = torch.randn(1, 32, 256, 256)
        offset = 96
        with torch.no_grad():
            expected = model(full)[..., offset:offset+core, offset:offset+core]
            patch = full[..., offset-halo:offset+core+halo, offset-halo:offset+core+halo]
            actual = model(patch)[..., halo:halo+core, halo:halo+core]
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)

    def test_prefetch_and_resume_do_not_change_sample_addresses(self):
        all_steps = list(StepBatches(51, 3, 0, 17, 333))
        resumed = list(StepBatches(51, 3, 8, 17, 333))
        self.assertEqual(all_steps[8:], resumed)
        self.assertNotEqual(all_steps[0], all_steps[1])

    def test_masked_pixels_do_not_affect_any_loss(self):
        torch.manual_seed(11)
        target = torch.randn(1, 5, 32, 32)
        mask = torch.ones(1, 1, 32, 32)
        mask[..., :4, :] = 0
        prediction = target.clone()
        prediction[..., :4, :] = 1000
        value, terms = losses(prediction, target, mask)
        self.assertEqual(float(value), 0.)
        self.assertTrue(all(float(term) == 0 for term in terms.values()))

    def test_loss_has_finite_gradients_for_an_untrained_student(self):
        prediction = torch.zeros(1, 1, 32, 32, requires_grad=True)
        loss, _ = losses(prediction, torch.randn_like(prediction), torch.ones(1, 1, 32, 32))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_atomic_writer_preserves_previous_checkpoint_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.json'
            atomic_json(path, {'step': 9})
            before = path.read_bytes()
            def fail(handle):
                handle.write(b'incomplete')
                raise RuntimeError('interruption')
            with self.assertRaises(RuntimeError):
                atomic_write(path, fail)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_live_process_identity_is_verified(self):
        import os
        self.assertIsNotNone(proc_identity(os.getpid()))
        self.assertIsNone(proc_identity(2147483647))

    def test_acceptance_rejects_missing_evidence_and_bad_spectral_band(self):
        sites = [dict(name=f's{i}') for i in range(7)]
        def rows(student=False):
            return [dict(site=site['name'], lod=lod, slope_mean=4., slope_p90=9.,
                         psd=[1.]*5, vs_reference=dict(mae=8. if student else 10., coast=.01))
                    for site in sites for lod in (3, 0)]
        report = dict(sites=sites, variants=dict(reference=rows(), fp32base=rows(), student=rows(True)))
        self.assertTrue(audit(report, 'student')['physical_passed'])
        report['variants']['student'][0]['psd'][4] = 1.08
        self.assertFalse(audit(report, 'student')['physical_passed'])
        report['variants']['student'].pop()
        with self.assertRaises(ValueError):
            audit(report, 'student')


if __name__ == '__main__':
    unittest.main()
