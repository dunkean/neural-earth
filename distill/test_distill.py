"""Regression tests for field invariance, deterministic resume and masked losses."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from distill.common import atomic_json, atomic_write, HOLDOUT_SEEDS
from distill.dataset import StepBatches
from distill.features import base_features, coarse_features, decoder_features, noise
from distill.inference import base_inputs, coarse_inputs, decoder_inputs
from distill.jobs import proc_identity
from distill.resume_teacher import validate_resume_plan
from distill.student import Student, StudentConfig, load_student
from distill.train import coarse_delta_loss, coarse_height_mae, losses, lowfreq_height_mae, spectral_band_loss
from distill.evaluate import artifact_audit, audit, pipeline_speed_audit
from distill.rare_cases import additional_evidence, augment, load_sites, local_errors, qualifies, sampling_policy, KINDS
from distill.common import source_digest
from distill.coarse_solver import CoarseSolver
from distill.widen import expand_state, expanded_config
from distill.check_physical_seams import partition_stats
from distill.export import export_bundle
from distill.decoded_loss import PairedCrops, decode, reconstructed_height
from distill.benchmark_candidates import wait_seam_diagnostic
from distill.bench_pipeline import StageCount, release_counter_graphs
from distill.verify_equivalence import compare as compare_physical_optimization
from distill.grid_artifacts import axis_grid_rms


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_seam_rejection_can_be_benchmarked_but_crash_or_stale_report_cannot(self):
        from distill.common import REPO
        import os
        import time
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'seams.json'
            sources = {'code:'+name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                       for name in ('distill/student.py', 'distill/features.py', 'distill/inference.py', 'distill/coarse_solver.py')}
            report = dict(stage='base', numerical_passed=False, student_source_digests=sources,
                          rows=[dict(passed=False, max_abs=.001, mean_abs=.0001, max_jump_error=.001)]*12)
            atomic_json(path, report)
            job = dict(live=False, returncode=1, started_at=time.time()-1)
            with patch('distill.benchmark_candidates.wait_job', side_effect=RuntimeError('check failed')), \
                 patch('distill.benchmark_candidates.state', return_value=job):
                wait_seam_diagnostic('seams', path)
                job['returncode'] = -9
                with self.assertRaises(RuntimeError):
                    wait_seam_diagnostic('seams', path)
                job['returncode'] = 1
                os.utime(path, (0, 0))
                with self.assertRaises(RuntimeError):
                    wait_seam_diagnostic('seams', path)
                atomic_json(path, report | dict(rows=report['rows'][:11]))
                with self.assertRaises(RuntimeError):
                    wait_seam_diagnostic('seams', path)

    def test_stage_counter_preserves_production_embedding_methods(self):
        class Network(torch.nn.Module):
            config = {'stage': 'coarse'}

            def compute_embeddings(self, labels, conditions):
                return labels + conditions

            def forward_with_embeddings(self, inputs, embeddings):
                return inputs + embeddings

            def forward(self, inputs):
                return inputs*2

        model = Network().eval()
        counted = StageCount(model)
        inputs = torch.ones(3, 2)
        embeddings = counted.compute_embeddings(torch.tensor(2.), torch.tensor(4.))
        torch.testing.assert_close(counted.forward_with_embeddings(inputs, embeddings),
                                   model.forward_with_embeddings(inputs, embeddings), rtol=0, atol=0)
        self.assertEqual(counted.config, model.config)
        self.assertFalse(counted.training)
        self.assertEqual((counted.calls, counted.windows), (1, 3))
        torch.testing.assert_close(counted(inputs), model(inputs), rtol=0, atol=0)
        self.assertEqual((counted.calls, counted.windows), (2, 6))
        counted.reset()
        # Solver capture forwards must not inflate the real replay count.
        counted(inputs)
        counted._terrain_note_graph_forwards(inputs, 20)
        self.assertEqual((counted.calls, counted.windows), (20, 60))
        counted._terrain_note_graph_forwards(inputs, 20)
        self.assertEqual((counted.calls, counted.windows), (40, 120))
        counted.reset()
        self.assertEqual((counted.calls, counted.windows), (0, 0))
        with self.assertRaises(AttributeError):
            counted.missing_method

    def test_benchmark_releases_owned_solver_graphs_without_clearing_borrowed_model(self):
        borrowed = Mock()
        owned = Mock()
        pool = Mock()
        counter = StageCount(torch.nn.Identity())
        counter.model.__dict__['_terrain_solver_graph_cache'] = {'borrowed': borrowed}
        counter.__dict__['_terrain_solver_graph_cache'] = {'owned': owned}
        counter.__dict__['_terrain_stream_pool'] = ('key', pool)
        release_counter_graphs({'coarse': counter})
        owned._clear_buckets.assert_called_once_with()
        pool.close.assert_called_once_with()
        borrowed._clear_buckets.assert_not_called()
        self.assertNotIn('_terrain_solver_graph_cache', counter.__dict__)
        self.assertNotIn('_terrain_stream_pool', counter.__dict__)
        release_counter_graphs({'coarse': counter})
        owned._clear_buckets.assert_called_once_with()

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

    def test_inference_feature_assembly_preserves_training_schema_exactly(self):
        coarse = np.random.default_rng(8).normal(size=(6, 16, 16)).astype(np.float32)
        for y, x in [(-128, -64), (64, 32), (100000000, -100000000)]:
            args = (coarse, y//32, x//32, 8173, y, x, 128, [.1, .2, .3, .4, .5])
            torch.testing.assert_close(base_inputs(*args, 'cpu'), base_features(*args), rtol=0, atol=0)
        latents = torch.randn(4, 64, 64)
        for y, x in [(-384, 384), (768, -768)]:
            torch.testing.assert_close(decoder_inputs(latents, 8173, y, x, 512, 'cpu'),
                                       decoder_features(latents, 8173, y, x), rtol=0, atol=0)
        condition, labels = torch.randn(5, 64, 64), torch.randn(5)
        torch.testing.assert_close(coarse_inputs(condition, labels, 8173, -48, 96, 'cpu'),
                                   coarse_features(condition, labels, 8173, -48, 96), rtol=0, atol=0)

    def test_optimization_equivalence_requires_same_weights_and_complete_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            before, after = [Path(directory)/n for n in ('before', 'after')]
            fingerprint = {n:n+'-sha' for n in ('base', 'coarse', 'decoder')}
            report = dict(gpu='gpu', torch='version', sites=['site'], source_digests={},
                          checkpoint_digests={'student_all':fingerprint},
                          variants={'student_all':[dict(site='site', lod=0)]})
            key = 'student_all|site|0'
            for path in (before, after):
                path.mkdir()
                atomic_json(path/'report.json', report)
                np.savez(path/'arrays.npz', **{key:np.zeros((4, 4), np.float32)})
            self.assertTrue(compare_physical_optimization(before, after)['exact_passed'])
            np.savez(after/'arrays.npz', **{key:np.ones((4, 4), np.float32)})
            self.assertFalse(compare_physical_optimization(before, after)['exact_passed'])
            atomic_json(after/'report.json', report | dict(checkpoint_digests={
                'student_all':fingerprint | dict(base='different-sha')}))
            with self.assertRaises(ValueError):
                compare_physical_optimization(before, after)
            atomic_json(after/'report.json', report)
            np.savez(after/'arrays.npz', **{key:np.zeros((4, 4), np.float32),
                                          'student_all|extra|0':np.zeros((4, 4), np.float32)})
            with self.assertRaises(ValueError):
                compare_physical_optimization(before, after)

    def test_grid_metric_detects_coherent_axes_without_counting_other_frequencies(self):
        x = np.arange(256, dtype=np.float64)
        field = 2*np.cos(2*np.pi*x[:, None]/32)+np.sin(2*np.pi*x[None, :]/64)
        self.assertAlmostEqual(axis_grid_rms(field), np.sqrt(2.5), places=12)
        other = np.broadcast_to(np.cos(2*np.pi*x[:, None]/16), (256, 256))
        self.assertLess(axis_grid_rms(other), 1e-12)
        self.assertEqual(axis_grid_rms(np.zeros((256, 256))), 0.)

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

    def test_widening_preserves_learned_field_and_unet_skip_layout(self):
        torch.manual_seed(88)
        old = Student(StudentConfig('base', 48, 3, (1, 2))).eval()
        torch.nn.init.normal_(old.head.weight, std=.1)
        torch.nn.init.normal_(old.direct.weight, std=.01)
        new = Student(expanded_config(old.config, 64)).eval()
        new.load_state_dict(expand_state(old, new, old.state_dict(), jitter=0))
        x = torch.randn(1, 32, 64, 64)
        with torch.no_grad():
            expected, actual = old(x), new(x)
        self.assertGreater(float(expected.std()), .01)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(new.direct.weight, old.direct.weight, rtol=0, atol=0)
        self.assertEqual(new.halo, old.halo)
        with self.assertRaises(ValueError):
            expanded_config(old.config, 32)

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
        self.assertFalse(qualifies('desert-plain', flat, climate | dict(temp=5.), 'warm-arid'))
        self.assertTrue(qualifies('desert-plain', flat, dict(rain=200., temp=25.), 'warm-arid'))
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

    def test_final_terrain_partition_detects_horizontal_and_vertical_cracks(self):
        size, halo = 32, 8
        y, x = np.indices((2*size+2*halo,)*2)
        field = (x*.5 + y*.2 - 25).astype(np.float32)
        full = field[halo:halo+2*size, halo:halo+2*size]
        tiles = {(y,x):field[y*size:y*size+size+2*halo, x*size:x*size+size+2*halo].copy()
                 for y in range(2) for x in range(2)}
        values, mosaic = partition_stats(full, tiles, halo)
        np.testing.assert_array_equal(mosaic, full)
        self.assertEqual(values['max_jump_error_m'], 0.)
        self.assertEqual(values['overlap_max_m'], 0.)
        tiles[(0, 1)][halo:halo+size, halo] += 2
        tiles[(1, 0)][halo, halo:halo+size] -= 3
        values, _ = partition_stats(full, tiles, halo)
        self.assertGreaterEqual(values['max_jump_error_m'], 3.)
        self.assertGreaterEqual(values['overlap_max_m'], 3.)

    def test_inference_export_preserves_ema_predictions_and_bundle_identity(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as directory:
            root, sources, expected = Path(directory), {}, {}
            for stage in ('base', 'coarse', 'decoder'):
                config = StudentConfig(stage, 16, 2, (1,))
                model = Student(config).eval()
                raw = {k:v.clone() for k,v in model.state_dict().items()}
                torch.nn.init.normal_(model.head.weight, std=.1)
                inputs = torch.randn(1, config.in_channels, 64, 64)
                with torch.no_grad():
                    expected[stage] = (inputs, model(inputs))
                source = root/(stage+'-source.pt')
                torch.save(dict(config=asdict(config), model=raw, ema=model.state_dict(), step=1,
                                optimizer={'state':'must not ship'}), source)
                sources[stage] = source
            bundle = export_bundle(sources, root/'bundle')
            self.assertFalse(bundle['accepted'])
            for stage in sources:
                model, saved = load_student(root/'bundle'/(stage+'.pt'))
                self.assertNotIn('optimizer', saved)
                self.assertNotIn('ema', saved)
                with torch.no_grad():
                    torch.testing.assert_close(model(expected[stage][0]), expected[stage][1], atol=0, rtol=0)
            self.assertEqual(export_bundle(sources, root/'bundle')['exports'], bundle['exports'])
            saved = torch.load(sources['base'], weights_only=True)
            saved['step'] = 2
            torch.save(saved, sources['base'])
            with self.assertRaises(ValueError):
                export_bundle(sources, root/'bundle')

    def test_decoded_supervision_backpropagates_into_latents_with_decoder_frozen(self):
        decoder = Student(StudentConfig('decoder', 16, 2, (1,))).eval().requires_grad_(False)
        torch.nn.init.normal_(decoder.head.weight, std=.03)
        latents = torch.randn(1, 5, 64, 64, requires_grad=True)
        output = decode(decoder, latents, torch.randn(1, 1, 512, 512))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(latents.grad).all())
        self.assertGreater(float(latents.grad[:, :4].abs().sum()), 0.)
        self.assertEqual(float(latents.grad[:, 4].abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))

    def test_reconstructed_height_keeps_metres_and_lowfrequency_gradients(self):
        latents = torch.zeros(1,5,64,64)
        latents[:,4] = (2.+31.4)/38.6
        latents.requires_grad_(True)
        height = reconstructed_height(torch.zeros(1,1,512,512), latents)
        torch.testing.assert_close(height, torch.full_like(height, 4.), rtol=1e-5, atol=1e-5)
        height.mean().backward()
        self.assertTrue(torch.isfinite(latents.grad).all())
        self.assertGreater(float(latents.grad[:,4].abs().sum()), 0.)

    def test_paired_dataset_keeps_negative_coordinates_and_rejects_changed_latents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'base').mkdir(); (root/'decoder').mkdir()
            name = 'train-0000000.npz'
            meta = dict(y=-96, x=96, seed=10000, split='train', coarse_y=-4, coarse_x=2)
            target = np.random.default_rng(2).normal(size=(5, 256, 256)).astype(np.float16)
            coarse = np.zeros((6, 20, 20), np.float32)
            np.savez(root/'base'/name, metadata=json.dumps(meta), target=target, mask=np.ones((1,256,256)),
                     coarse=coarse, histogram=np.zeros(5))
            dmeta = dict(y=-768, x=768, seed=10000, split='train')
            def write_decoder(latents):
                np.savez(root/'decoder'/name, metadata=json.dumps(dmeta), latents=latents,
                         target=np.zeros((1,512,512)), mask=np.ones((1,512,512)))
            write_decoder(target[:4,:64,:64])
            audit = root/'audit.json'
            atomic_json(audit, dict(dataset=str(root), mismatches=[], rows=[dict(file=name, seed=10000,
                        split='train', latent_pair_exact=True)]))
            paired = PairedCrops(root, 'train', 32, audit)
            inputs, actual, _, gaussian, _, _ = paired[0]
            self.assertEqual(inputs.shape, (32,128,128))
            torch.testing.assert_close(actual, torch.from_numpy(target[:,:64,:64].astype(np.float32)), rtol=0, atol=0)
            torch.testing.assert_close(gaussian, noise(15819,-768,768,512,512,1,tile=512), rtol=0, atol=0)
            write_decoder(np.zeros((4,64,64), np.float16))
            with self.assertRaises(ValueError):
                paired[0]

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

    def test_warm_plain_augmentation_preserves_each_original_rare_site(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, addition, output = [root / (name+'.json') for name in ('original', 'addition', 'complete')]
            sites = [dict(name=kind, seed=101, kind=kind, x=0., y=0.) for kind in KINDS]
            warm = [dict(name=kind+'-warm', seed=202, kind=kind, x=10., y=10., climate_archetype=archetype)
                    for kind, archetype in [('desert-plain', 'warm-arid'), ('temperate-plain', 'mild-temperate')]]
            base = dict(frozen=True, teacher_sources=source_digest())
            atomic_json(original, base | dict(sites=sites))
            atomic_json(addition, base | dict(sites=warm))
            augment(original, addition, output)
            self.assertEqual(load_sites(output), sites+warm)
            parents = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (original, addition)}
            atomic_json(output, base | dict(sites=sites[1:]+warm, parent_banks=parents))
            rejected = additional_evidence({}, 'student_all', output, root/'unused')
            self.assertFalse(rejected['passed'])
            self.assertIn('original rare cases', rejected['reason'])
            atomic_json(output, base | dict(sites=sites+warm, parent_banks=parents))
            atomic_json(original, base | dict(sites=sites[1:]))
            self.assertIn('missing or changed', additional_evidence({}, 'student_all', output, root/'unused')['reason'])

    def test_acceptance_rejects_mismatched_weights_and_unreviewed_seams(self):
        sources = {'code:'+name: 'source-hash' for name in
                   ('distill/student.py', 'distill/features.py', 'distill/inference.py', 'distill/coarse_solver.py')}
        fingerprint = dict(base='trained-model', coarse='coarse-model', decoder='decoder-model', **sources)
        report = dict(gpu='gpu', checkpoint_digests=dict(student_all=fingerprint))
        benchmark = dict(gpu='gpu', step=100, checkpoint_digest='trained-model', student_source_digests=sources)
        seams = dict(gpu='gpu', split='val', rows=[dict(passed=True)]*12,
                     numerical_passed=True, checkpoint_digest='trained-model', student_source_digests=sources,
                     visual_review=dict(passed=True, checkpoint_digests=fingerprint))
        pipeline = dict(benchmark)
        self.assertTrue(artifact_audit(report, 'student_all', benchmark, seams, pipeline)['passed'])
        old_fingerprint = {k:v for k,v in fingerprint.items() if k != 'code:distill/coarse_solver.py'}
        self.assertFalse(artifact_audit(dict(report, checkpoint_digests=dict(student_all=old_fingerprint)),
                                      'student_all', benchmark, seams, pipeline)['passed'])
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


class FinalEvidenceTests(unittest.TestCase):
    def test_equal_budget_inspection_rejects_smoke_checkpoint(self):
        from distill.inspect_candidate import validate_candidate
        saved = dict(config=dict(stage='base'), arguments=dict(steps=32), step=32)
        status = dict(step=20000)
        with self.assertRaises(ValueError):
            validate_candidate(saved, status, 'base', 20000, selection='latest')
        # The best validation candidate can be earlier, with its step disclosed.
        validate_candidate(saved, status, 'base', 20000, selection='best')
        completed = dict(saved, step=20000, arguments=dict(steps=20000))
        validate_candidate(completed, status, 'base', 20000, selection='latest')

    def test_final_plates_reject_changed_reference_and_context(self):
        from distill.final_plates import validate_banks
        report = {key: 'same' for key in ('gpu', 'sites', 'lods', 'teacher_sources',
                                         'site_manifest_digest', 'tile_size', 'halo')}
        reference = np.zeros((4, 4), dtype=np.float32)
        bank = (report, {('site', 0, 'reference'): reference}, {})
        validate_banks([bank, bank])
        with self.assertRaises(ValueError):
            validate_banks([bank, (dict(report, gpu='other'), bank[1], {})])
        changed = reference.copy()
        changed[0, 0] = 1
        with self.assertRaises(ValueError):
            validate_banks([bank, (report, {('site', 0, 'reference'): changed}, {})])

    def test_extra_plate_bank_requires_same_weights_and_new_views(self):
        from distill.final_plates import combine_banks
        report = dict(gpu='same', lods=[0, 3], checkpoint_digests={'base':'trained'},
                      teacher_sources={'teacher':'same'}, tile_size=256, halo=24,
                      sites=[{'name':'plain'}], rows=[], site_manifest_digest='rare-bank')
        first = (report, {('plain',0,'reference'):np.zeros((4,4))}, {})
        extra_report = dict(report, sites=[{'name':'snow'}], site_manifest_digest='original-sites')
        extra = (extra_report, {('snow',0,'reference'):np.zeros((4,4))}, {})
        merged = combine_banks(first, extra)
        self.assertEqual(len(merged[0]['sites']), 2)
        with self.assertRaises(ValueError):
            combine_banks(first, first)
        with self.assertRaises(ValueError):
            combine_banks(first, (dict(extra_report, checkpoint_digests={'base':'other'}),extra[1],{}))


if __name__ == '__main__':
    unittest.main()
