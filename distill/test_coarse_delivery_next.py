"""Exercise the complete delivery commit/push in an isolated local Git repo."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from distill import coarse_delivery_next as delivery
from distill.common import REVISION


class CoarseDeliveryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.repo, self.data, self.remote = root/'repo', root/'data', root/'remote.git'
        self.repo.mkdir(); self.data.mkdir()
        subprocess.run(['git', 'init', '-q', '--bare', str(self.remote)], check=True)
        self.git('init', '-q', '-b', 'distill')
        self.git('config', 'user.name', 'Test')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('remote', 'add', 'origin', str(self.remote))
        weights = self.repo/'distill/weights'; weights.mkdir(parents=True)
        for n in (1, 2):
            (weights/f'coarse{n}.pt').write_bytes(b'partial')
        manifest = dict(teacher_revision=REVISION, exports=dict(decoder=dict(sha256='preserved')))
        (weights/'manifest.json').write_text(json.dumps(manifest))
        (self.repo/'distill/LIVE.md').write_text('Intro\nCoarse **2 étapes** old\nLes bases distillés rejetés end\n')
        (self.repo/'distill/TASKS.md').write_text('# Tasks\n## Priorité utilisateur : terminer coarse1 et coarse2 avant le base\n')
        self.git('add', '.'); self.git('commit', '-qm', 'Initial partial weights')
        for n in (1, 2):
            run = self.data/f'ckpt/coarse-full/coarse{n}'
            (run/'export').mkdir(parents=True)
            source, payload = b'checkpoint', f'complete{n}'.encode()
            (run/'best.pt').write_bytes(source)
            (run/'export/coarse.pt').write_bytes(payload)
            export = dict(stage='coarse', inference_only=True,
                source=dict(step=195000, sha256=hashlib.sha256(source).hexdigest(),
                    config=dict(stage='coarse', solver_steps=n)),
                export=dict(file='coarse.pt', bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
            for filename, value in (
                ('status.json', dict(status='complete', step=200000)),
                ('export/manifest.json', export),
                ('delivery.json', dict(status='complete', training_steps=200000,
                    selected_step=195000, validation_count=384, finite=True,
                    export_max_abs=0., smoke_indices=[0, 192, 383], export=export))):
                (run/filename).write_text(json.dumps(value))

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.repo, text=True).strip()

    def run_delivery(self):
        with patch.object(delivery, 'REPO', self.repo), patch.object(delivery, 'DATA', self.data), \
                patch.object(delivery, 'state', return_value=dict(live=False, status='complete', returncode=0)), \
                patch.object(sys, 'argv', ['delivery', '--push']), contextlib.redirect_stdout(io.StringIO()):
            delivery.main()

    def test_delivers_only_completed_models_and_verifies_remote_commit(self):
        self.run_delivery()
        status = json.loads((self.data/'coarse-delivery-next.json').read_text())
        self.assertEqual(status['status'], 'complete')
        self.assertTrue(status['pushed'])
        self.assertEqual(status['commit'], self.git('rev-parse', 'HEAD'))
        self.assertEqual(status['commit'], self.git('ls-remote', 'origin', 'refs/heads/distill').split()[0])
        self.assertEqual(set(self.git('diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').splitlines()),
                         set(delivery.FILES))
        manifest = json.loads((self.repo/'distill/weights/manifest.json').read_text())
        self.assertEqual(manifest['exports']['decoder']['sha256'], 'preserved')
        self.assertEqual(manifest['exports']['coarse1']['training_status'], 'complete')
        self.assertIn('200 000', (self.repo/'distill/LIVE.md').read_text())

    def test_refuses_other_branch_without_replacing_any_weights(self):
        self.git('switch', '-qc', 'main')
        before = self.git('rev-parse', 'HEAD')
        with self.assertRaisesRegex(RuntimeError, 'Branch changed'):
            self.run_delivery()
        self.assertEqual(self.git('rev-parse', 'HEAD'), before)
        self.assertEqual((self.repo/'distill/weights/coarse1.pt').read_bytes(), b'partial')

    def test_partial_second_run_leaves_both_models_and_git_unchanged(self):
        path = self.data/'ckpt/coarse-full/coarse2/status.json'
        path.write_text(json.dumps(dict(status='interrupted', step=199999)))
        before = self.git('rev-parse', 'HEAD')
        with self.assertRaisesRegex(ValueError, 'Coarse2'):
            self.run_delivery()
        self.assertEqual(self.git('rev-parse', 'HEAD'), before)
        self.assertEqual(self.git('status', '--porcelain'), '')


if __name__ == '__main__':
    unittest.main()
