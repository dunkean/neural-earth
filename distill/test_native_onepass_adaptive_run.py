"""The durable follower extends improving terrain and stops a confirmed plateau."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

from distill.native_onepass_adaptive_run import Controller
from distill.native_onepass_progress_policy import evidence
from distill.test_native_onepass_progress_policy import reports


class AdaptiveTests(unittest.TestCase):
    def run_controller(self, root, scales):
        args = SimpleNamespace(root=Path(root), gpu_lock=Path(root)/'gpu.lock',
                               preparation=Path(root)/'preparation', initial_job='initial')
        with patch('distill.native_onepass_adaptive_run.signal.signal'):
            controller = Controller(args)
        stages = []

        def stage(previous, start, target):
            stages.append((start, target))
            return Path(root)/f'stage-{target}'

        controller.stage = stage
        with patch('distill.native_onepass_adaptive_run.state', return_value=dict(live=False, status='complete', returncode=0)), \
             patch('distill.native_onepass_adaptive_run.reports', side_effect=lambda step, selection:
                   evidence(reports(scales[step], step, selection))), \
             patch('builtins.print'):
            controller.run()
        return stages, json.loads((Path(root)/'adaptive/run-status.json').read_text())

    def test_extension_has_no20k_cap_and_waits_for_two_flat_blocks(self):
        scales = {10000:1., 15000:.9, 20000:.8, 25000:.7, 30000:.7, 35000:.7}
        with tempfile.TemporaryDirectory() as root:
            stages, status = self.run_controller(root, scales)
            self.assertEqual(stages, [(20000, 25000), (25000, 30000), (30000, 35000)])
            self.assertEqual(status['reason'], 'plateau')
            self.assertEqual(status['best_step'], 25000)

    def test_material_regression_retains_the_best_and_prevents_extension(self):
        with tempfile.TemporaryDirectory() as root:
            stages, status = self.run_controller(root, {10000:1., 15000:.9, 20000:1.2})
            self.assertFalse(stages)
            self.assertEqual(status['reason'], 'terrain_regression')
            self.assertEqual(status['best_step'], 15000)

    def test_failed_initial_run_cannot_start_new_training(self):
        with tempfile.TemporaryDirectory() as root, patch('distill.native_onepass_adaptive_run.signal.signal'):
            args = SimpleNamespace(root=Path(root), initial_job='initial')
            controller = Controller(args)
            with patch('distill.native_onepass_adaptive_run.state', return_value=dict(live=False, status='failed', returncode=1)), \
                 self.assertRaisesRegex(ValueError, 'Initial20k run failed'):
                controller.run()


if __name__ == '__main__':
    unittest.main()
