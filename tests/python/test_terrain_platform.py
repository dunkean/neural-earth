"""Storage and launcher portability without loading checkpoints or CUDA."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools._bootstrap import activate
activate()

import json
import os
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import terrain_paths as paths
import launch_terrain as launcher


class StorageTests(unittest.TestCase):
    def test_windows_default_and_linux_xdg(self):
        self.assertEqual(paths.runtime_root({}, 'nt').as_posix(), 'E:/TerrainDiffusionRuntime')
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(paths.runtime_root({'XDG_CACHE_HOME': directory}, 'posix'),
                             Path(directory) / 'neural-earth')
            self.assertEqual(paths.runtime_root({'TERRAIN_RUNTIME_ROOT': directory}, 'nt'),
                             Path(directory).resolve())
        self.assertEqual(paths.runtime_root({}, 'posix'), Path.home() / '.cache' / 'neural-earth')

    def test_overrides_and_snapshot_are_stable_after_chdir(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, TERRAIN_RUNTIME_ROOT=directory,
                       HF_HOME=str(Path(directory) / 'hf'),
                       HF_HUB_CACHE=str(Path(directory) / 'hub'),
                       TERRAIN_GENERATION_ROOT='settings', TERRAIN_BOOTSTRAP_CACHE='atlas',
                       TERRAIN_OUTPUT_ROOT='results')
            code = """
import json, os
import terrain_paths as p
import terrain_generation as g
import terrain_bootstrap as b
os.chdir('terrain-diffusion')
print(json.dumps([str(p.RUNTIME_ROOT), str(p.HF_HOME), str(p.model_snapshot('revision')),
                  str(g.REGISTRY_ROOT), str(b.CACHE_ROOT), str(p.OUTPUT_ROOT)]))
"""
            result = subprocess.check_output([sys.executable, '-c', code], cwd=ROOT, env=env, text=True)
            self.assertEqual(json.loads(result), [directory, str(Path(directory) / 'hf'),
                str(Path(directory) / 'hub/models--xandergos--terrain-diffusion-30m/snapshots/revision'),
                str(ROOT / 'settings'), str(ROOT / 'atlas'), str(ROOT / 'results')])


class LauncherTests(unittest.TestCase):
    def test_platform_specific_detachment(self):
        for platform in ('nt', 'posix'):
            with tempfile.TemporaryDirectory() as directory, patch.object(launcher, 'root', Path(directory)), \
                    patch.object(launcher, 'os', SimpleNamespace(name=platform)), \
                    patch.object(launcher.subprocess, 'CREATE_NO_WINDOW', 0x08000000, create=True), \
                    patch.object(launcher.subprocess, 'Popen') as spawn:
                launcher.start_server()
                options = spawn.call_args.kwargs
                self.assertEqual(options['stdin'], subprocess.DEVNULL)
                if platform == 'nt':
                    self.assertEqual(options['creationflags'], 0x08000000)
                    self.assertNotIn('start_new_session', options)
                else:
                    self.assertTrue(options['start_new_session'])
                    self.assertNotIn('creationflags', options)

    def test_existing_server_is_reused_without_opening_browser(self):
        with patch.object(launcher, 'alive', return_value=True), \
                patch.object(launcher, 'start_server') as spawn, \
                patch.object(launcher.webbrowser, 'open') as browser:
            launcher.main(['--no-open'])
            spawn.assert_not_called()
            browser.assert_not_called()

    def test_child_failure_is_reported_immediately(self):
        with patch.object(launcher, 'alive', return_value=False), \
                patch.object(launcher, 'start_server', return_value=Mock(poll=lambda: 1)), \
                patch.object(launcher.time, 'sleep') as sleep:
            with self.assertRaisesRegex(SystemExit, 'server-error.log'):
                launcher.main(['--no-open'])
            sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
