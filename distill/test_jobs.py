import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import uuid

from distill.jobs import state


class ExactJobSessionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('tmux'), 'tmux is required')
    def test_validation_session_cannot_keep_a_completed_training_job_live(self):
        name = 'probe-'+uuid.uuid4().hex
        session = 'distill-'+name+'-validation'
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)/name
            directory.mkdir()
            (directory/'status.json').write_text(json.dumps(dict(status='complete', returncode=0)))
            subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'sleep 30'], check=True)
            try:
                result = state(directory)
                self.assertFalse(result['live'])
                self.assertFalse(result['tmux_live'])
                self.assertEqual(result['status'], 'complete')
                self.assertTrue(state(Path(folder)/(name+'-validation'))['tmux_live'])
            finally:
                subprocess.run(['tmux', 'kill-session', '-t', '='+session], check=True)


if __name__ == '__main__':
    unittest.main()
