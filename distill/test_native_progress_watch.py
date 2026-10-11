"""Publication must pin checkpoints even while the trainer replaces best.pt."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from distill import native_progress_watch as watch


class PublicationTests(unittest.TestCase):
    def test_atomic_replacement_cannot_change_retained_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory)/'ckpt', Path(directory)/'images'
            root.mkdir(); output.mkdir()
            torch.save(dict(step=20000, weight=torch.tensor([1.])), root/'best.pt')
            expected = watch.checksum(root/'best.pt')
            torch.save(dict(step=25000, weight=torch.tensor([2.])), root/'step25000.pt')

            def render(command, **kwargs):
                snapshot = Path(command[command.index('--checkpoints')+1])
                step = torch.load(snapshot, weights_only=True)['step']
                if step == 20000:
                    replacement = root/'replacement.pt'
                    torch.save(dict(step=25000, weight=torch.tensor([9.])), replacement)
                    os.replace(replacement, root/'best.pt')
                    self.assertEqual(watch.checksum(snapshot), expected)
                (output/f'{step:06d}.json').write_text(json.dumps(dict(
                    checkpoint=str(snapshot), checkpoint_sha256=watch.checksum(snapshot))))

            argv = ['watch', '--job', 'training', '--checkpoint-root', str(root),
                    '--target-step', '25000', '--output', str(output)]
            states = [dict(live=True), dict(live=False, status='complete', returncode=0)]
            with patch('sys.argv', argv), patch.object(watch, 'state', side_effect=states), \
                    patch.object(watch.subprocess, 'run', side_effect=render), patch('threading.Event.wait'):
                watch.main()
            self.assertEqual(watch.checksum(root/'progress-snapshots/step20000.pt'), expected)
            self.assertFalse(list((root/'progress-snapshots').glob('.publication-*')))
            proof = json.loads((root/'progress-publication.json').read_text())
            self.assertEqual(proof['status'], 'complete')
            self.assertEqual([r['step'] for r in proof['publications']], [20000, 25000])

    def test_completed_job_must_have_requested_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['watch', '--job', 'training', '--checkpoint-root', directory,
                    '--target-step', '50000', '--output', str(Path(directory)/'images')]
            with patch('sys.argv', argv), patch.object(watch, 'state',
                    return_value=dict(live=False, status='complete', returncode=0)):
                with self.assertRaises(FileNotFoundError):
                    watch.main()

    def test_same_step_of_different_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory), Path(directory)/'images'
            output.mkdir()
            torch.save(dict(step=50000), root/'step50000.pt')
            (output/'050000.json').write_text(json.dumps(dict(checkpoint_sha256='other-model')))
            argv = ['watch', '--job', 'training', '--checkpoint-root', directory,
                    '--target-step', '50000', '--output', str(output)]
            with patch('sys.argv', argv), patch.object(watch, 'state',
                    return_value=dict(live=False, status='complete', returncode=0)), \
                    patch.object(watch.subprocess, 'run') as render:
                with self.assertRaises(ValueError):
                    watch.main()
                render.assert_not_called()
            self.assertFalse(list((root/'progress-snapshots').glob('.publication-*')))


if __name__ == '__main__':
    unittest.main()
