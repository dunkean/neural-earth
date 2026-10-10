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
from distill.resume_teacher import validate_resume_plan
from distill.student import Student, StudentConfig
from distill.train import coarse_delta_loss, coarse_height_mae, losses, lowfreq_height_mae, spectral_band_loss
from distill.evaluate import artifact_audit, audit, pipeline_speed_audit
from distill.rare_cases import additional_evidence, load_sites, local_errors, qualifies, sampling_policy, KINDS
from distill.common import source_digest
from distill.coarse_solver import CoarseSolver


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

    def test_weighted_sampling_resumes_exactly_and_rejects_invalid_weights(self):
        all_steps = list(StepBatches(4, 8, 0, 12, 333, [0., 0., .1, .9]))
        resumed = list(StepBatches(4, 8, 5, 12, 333, [0., 0., .1, .9]))
        self.assertEqual(all_steps[5:], resumed)
        self.assertTrue(all(i in (2, 3) for batch in all_steps for i, _ in batch))
        for weights in ([1., 2.], [0.] * 4, [1., -1., 1., 1.], [1., float('nan'), 1., 1.]):
            with self.assertRaises(ValueError):
                StepBatches(4, 1, 0, 10, 333, weights)

    def test_short_solver_backpropagates_without_mutating_weights_or_autocast_drift(self):
        class SmallDenoiser(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(11, 6, 1)
            def forward(self, x, noise_labels, conditional_inputs):
                return self.conv(x)
        solver = CoarseSolver.__new__(CoarseSolver)
        torch.nn.Module.__init__(solver)
        solver.net, solver.steps, solver.delta_ratio = SmallDenoiser(), 4, .044
        x = torch.randn(1, 16, 64, 64)
        versions = [p._version for p in solver.parameters()]
        p = solver(x)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = solver(x)
        torch.testing.assert_close(actual, p, rtol=0, atol=0)
        p.square().mean().backward()
        self.assertEqual([p._version for p in solver.parameters()], versions)
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in solver.parameters()))
        self.assertGreater(sum(float(v.grad.abs().sum()) for v in solver.parameters()), 0.)

    def test_rare_sampling_policy_keeps_validation_and_holdout_out(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'coverage.json', Path(directory) / 'policy.json'
            rows = [dict(file=f'train-{i}.npz', split='train', seed=10000+i, kinds=[kind])
                    for i, kind in enumerate(KINDS)]
            rows.append(dict(file='val-0.npz', split='val', seed=10500, kinds=list(KINDS)))
            atomic_json(source, dict(dataset_manifest_digest='digest', rows=rows))
            sampling_policy(source, output)
            policy = json.loads(output.read_text())
            self.assertEqual(policy['train_files'], [f'train-{i}.npz' for i in range(4)])
            self.assertAlmostEqual(sum(policy['probabilities']), 1.)
            rows[0]['seed'] = 42
            atomic_json(source, dict(dataset_manifest_digest='digest', rows=rows))
            with self.assertRaises(ValueError):
                sampling_policy(source, output)

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

    def test_spectral_bands_detect_small_high_frequency_changes(self):
        x = torch.arange(128, dtype=torch.float32)
        low = torch.sin(x*2*torch.pi/128).view(1, 1, 1, 128).expand(1, 1, 128, 128)
        high = torch.sin(x*2*torch.pi*.375).view(1, 1, 1, 128)
        target = low+.001*high
        mask = torch.ones(1, 1, 128, 128)
        self.assertEqual(float(spectral_band_loss(target, target, mask)), 0.)
        changed = (low+.002*high).requires_grad_()
        value = spectral_band_loss(changed, target, mask)
        self.assertGreater(float(value.detach()), .1)
        value.backward()
        self.assertTrue(torch.isfinite(changed.grad).all())
        zeros = torch.zeros_like(target, requires_grad=True)
        spectral_band_loss(zeros, zeros.detach(), mask).backward()
        self.assertTrue(torch.isfinite(zeros.grad).all())

    def test_coarse_relief_loss_distinguishes_offset_from_relief_change(self):
        target = torch.zeros(1, 6, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        common_offset = target.clone()
        common_offset[:, :2] = .01
        self.assertEqual(float(coarse_delta_loss(common_offset, target, mask)), 0.)
        changed_relief = common_offset.clone().requires_grad_()
        changed_relief = changed_relief + torch.tensor([.01, 0., 0., 0., 0., 0.]).view(1, 6, 1, 1)
        value = coarse_delta_loss(changed_relief, target, mask)
        self.assertGreater(float(value.detach()), .04)
        gradients, = torch.autograd.grad(value, changed_relief)
        self.assertTrue(torch.isfinite(gradients).all())
        self.assertLess(float((gradients[:, 0]+gradients[:, 1]).abs().max()), 1e-7)

    def test_coarse_metres_detect_common_height_offsets(self):
        target = torch.zeros(1, 6, 2, 2)
        target[:, 0], target[:, 1] = 10., 9.
        prediction = target.clone()
        prediction[:, :2] += 1.
        mask = torch.ones(1, 1, 2, 2)
        self.assertEqual(float(coarse_delta_loss(prediction, target, mask)), 0.)
        mask[..., 0, 0] = 0.
        prediction[..., 0, 0] = 1000.
        prediction.requires_grad_()
        value = coarse_height_mae(prediction, target, mask, ([0.]*6, [1.]*6))
        self.assertEqual(float(value.detach()), 20.)
        value.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[..., 0, 0].abs().max()), 0.)

    def test_autocast_preserves_output_precision_and_backward(self):
        model = Student(StudentConfig('coarse', 16, 2, (1,)))
        torch.nn.init.normal_(model.head.weight, std=.1)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            prediction = model(torch.randn(1, 16, 32, 32))
            loss = prediction.square().mean()
        self.assertEqual(prediction.dtype, torch.float32)
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters() if parameter.grad is not None))

    def test_height_loss_measures_metres_and_ignores_masked_pixels(self):
        target = torch.zeros(1, 5, 2, 2)
        target[:, 4] = (10.+31.4)/38.6
        prediction = target.clone()
        prediction[:, 4] = (11.+31.4)/38.6
        mask = torch.ones(1, 1, 2, 2)
        mask[..., 0, 0] = 0
        prediction[..., 0, 0] = 1000
        prediction.requires_grad_()
        loss = lowfreq_height_mae(prediction, target, mask)
        self.assertAlmostEqual(float(loss.detach()), 21., places=3)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[..., 0, 0].abs().max()), 0.)

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

    def test_rare_cases_require_actual_plain_and_coast_at_each_scale(self):
        climate = dict(rain=200., temp=15.)
        flat = dict(land=1., land_height_std=30., slope_land_p90=2.,
                    land_height_p90=80., coast_complexity=0.,
                    land_largest_component_fraction=1., sea_largest_component_fraction=1.)
        self.assertTrue(qualifies('desert-plain', flat, climate))
        self.assertFalse(qualifies('desert-plain', flat | dict(slope_land_p90=12.), climate))
        self.assertFalse(qualifies('low-plain-sea', flat, climate))
        coast = flat | dict(land=.5, coast_complexity=3.)
        self.assertTrue(qualifies('low-plain-sea', coast, climate))
        self.assertFalse(qualifies('low-plain-sea', coast | dict(sea_largest_component_fraction=.1), climate))
        self.assertFalse(qualifies('wet-complex-coast', coast, climate))
        self.assertTrue(qualifies('wet-complex-coast', coast, dict(rain=1500., temp=15.)))

    def test_local_coast_metrics_expose_small_lowland_errors(self):
        ref = np.full((32, 32), 500., dtype=np.float32)
        self.assertEqual(local_errors(ref, ref, 0)['coast_300m']['pixels'], 0)
        ref[:, :2] = -1.
        ref[:, 2:4] = 1.
        candidate = ref.copy()
        candidate[:, 2:4] = -1.
        result = local_errors(candidate, ref, 0)
        self.assertEqual(result['low_land_0_20m']['mae_m'], 2.)
        self.assertEqual(result['low_land_0_20m']['sign_flip'], 1.)
        self.assertGreater(result['coast_300m']['pixels'], 0)
        self.assertGreater(result['axis_error']['column_mean_std_m'], 0)

    def test_rare_bank_excludes_training_worlds_and_cannot_replace_original_bank(self):
        self.assertFalse(additional_evidence({}, 'student_all')['passed'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sites.json'
            site = dict(name='plain', seed=101, kind='desert-plain', x=0., y=0.)
            manifest = dict(teacher_sources=source_digest(), sites=[site])
            atomic_json(path, manifest)
            self.assertEqual(load_sites(path), [site])
            manifest['sites'][0]['seed'] = 10001
            atomic_json(path, manifest)
            with self.assertRaises(ValueError):
                load_sites(path)
        with self.assertRaises(ValueError):
            audit(dict(sites=[dict(name='plain')]), 'student')

    def test_acceptance_rejects_mismatched_weights_and_unreviewed_seams(self):
        sources = {'code:'+name: 'source-hash' for name in
                   ('distill/student.py', 'distill/features.py', 'distill/inference.py')}
        fingerprint = dict(base='trained-model', coarse='coarse-model', decoder='decoder-model', **sources)
        report = dict(gpu='gpu', checkpoint_digests=dict(student_all=fingerprint))
        benchmark = dict(gpu='gpu', step=100, checkpoint_digest='trained-model', student_source_digests=sources)
        seams = dict(gpu='gpu', split='val', rows=[dict(passed=True)]*12,
                     numerical_passed=True, checkpoint_digest='trained-model', student_source_digests=sources,
                     visual_review=dict(passed=True, checkpoint_digests=fingerprint))
        pipeline = dict(benchmark)
        self.assertTrue(artifact_audit(report, 'student_all', benchmark, seams, pipeline)['passed'])
        benchmark['checkpoint_digest'] = 'architecture-only'
        self.assertFalse(artifact_audit(report, 'student_all', benchmark, seams, pipeline)['passed'])
        benchmark['checkpoint_digest'] = 'trained-model'
        seams['visual_review'] = dict(passed=False)
        self.assertFalse(artifact_audit(report, 'student_all', benchmark, seams, pipeline)['passed'])

    def test_pipeline_speed_rejects_cached_partial_or_incomplete_work(self):
        benchmark = dict(status='complete', stage='base', includes_feature_construction=True,
                         includes_transfers=True, fresh_world_per_sample=True, whole_system_idle=True,
                         sizes=[1024], repeats=3, rows=[])
        for repeat in range(3):
            benchmark['rows'].extend([
                dict(variant='reference', size=1024, repeat=repeat, warmup=False, seconds=12., base_windows=2300),
                dict(variant='student', size=1024, repeat=repeat, warmup=False, seconds=1., base_windows=0,
                     student_counts=dict(base=dict(calls=4, output_pixels=1024**2)))])
        self.assertTrue(pipeline_speed_audit(benchmark)['passed'])
        benchmark['rows'][-1]['student_counts']['base']['calls'] = 0
        self.assertFalse(pipeline_speed_audit(benchmark)['passed'])
        benchmark['rows'].pop()
        self.assertFalse(pipeline_speed_audit(benchmark)['passed'])

    def test_seed_replacement_growth_never_changes_existing_examples(self):
        manifest = dict(seed_base=10000, samples_per_world=32)
        current = dict(policy=dict(stride=1000000), replacements={'10001':dict(seed=1010001)})
        validate_resume_plan({}, current, ['train-0000000.npz'], manifest, True)
        with self.assertRaises(ValueError):
            validate_resume_plan({}, current, ['train-0000032.npz'], manifest, True)
        with self.assertRaises(ValueError):
            validate_resume_plan({}, current, ['train-0000000.npz'], manifest, False)
        changed = dict(policy=current['policy'], replacements={'10001':dict(seed=2010001)})
        with self.assertRaises(ValueError):
            validate_resume_plan(current, changed, ['train-0000000.npz'], manifest, True)


if __name__ == '__main__':
    unittest.main()
