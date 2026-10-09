"""Bootstrap identity and exporter admission; no model execution."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import terrain_manifest as manifest
from terrain_reference import neural_manifest, export_neural_intermediates


class TerrestrialIdentityTests(unittest.TestCase):
    def test_rejected_initializers_and_incomplete_persistence(self):
        for old in ('A1', 'A2', 'A3', 'A4', 'earth', 'macro-a4'):
            with self.assertRaises(ValueError):
                manifest.build_manifest(0, old, file_hashes=False)
        with self.assertRaises(ValueError):
            manifest.world_identity(manifest.build_manifest(0, file_hashes=False))
        with self.assertRaises(ValueError):
            manifest.build_manifest(-1, file_hashes=False)

    def test_layout_seed_and_height_are_identity_but_timing_is_not(self):
        receipt = {name: None for name in ('native_config', 'attempts', 'raster', 'hypsometry')}
        receipt.update(requested_seed_u64='18446744073709551615', selected_seed_u64='9',
                       selected_attempt=1, style='earthlike', raw_height_sha256='raw',
                       height_sha256='height', sign_preserved=True, generation_seconds=1.)
        with patch.object(manifest, 'bootstrap_metadata', return_value=receipt):
            a = manifest.build_manifest(2**64-1, 'terrestrial-earthlike', file_hashes=False)
            receipt['generation_seconds'] = 200.
            b = manifest.build_manifest(2**64-1, 'terrestrial-earthlike', file_hashes=False)
            self.assertEqual(a['world_hash'], b['world_hash'])
            self.assertEqual(a['seed_u64'], '18446744073709551615')
            self.assertEqual(a['bootstrap']['selected_seed_u64'], '9')
            receipt['height_sha256'] = 'changed'
            c = manifest.build_manifest(2**64-1, 'terrestrial-earthlike', file_hashes=False)
            self.assertNotEqual(a['world_hash'], c['world_hash'])
            self.assertFalse(a['geography']['periodic_longitude'])
            self.assertTrue(a['geography']['bootstrap_periodic_longitude'])

    @patch.object(manifest, 'verify_implementation_identity', return_value=True)
    def test_manifest_tamper_and_changed_native_bytes_are_rejected(self, native_guard):
        files = {'bootstrap_native': {'binary_sha256': 'a'}, 'implementation': {'x': 'b'}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'config.json').write_text('{}', encoding='utf-8')
            with patch.object(manifest, '_files', return_value=deepcopy(files)):
                a = manifest.build_manifest(0, checkpoint_source=root)
                self.assertEqual(manifest.verify_manifest_files(a, root), a['world_hash'])
                native_guard.assert_called_with(a['files']['bootstrap_native'])
                bad = deepcopy(a)
                bad['files']['bootstrap_native']['binary_sha256'] = 'changed'
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    manifest.world_identity(bad)
            files['bootstrap_native']['binary_sha256'] = 'changed'
            with patch.object(manifest, '_files', return_value=files):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    manifest.verify_manifest_files(a, root)

    def test_natural_manifest_never_initializes_native_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'config.json').write_text('{}', encoding='utf-8')
            for stage in ('coarse_model','base_model','decoder_model'):
                (root/stage).mkdir()
                (root/stage/'config.json').write_text('{}', encoding='utf-8')
                (root/stage/'diffusion_pytorch_model.safetensors').write_bytes(b'fixture')
            with patch.object(manifest,'implementation_identity',side_effect=AssertionError('Natural needs no Rust')):
                a=manifest.build_manifest(0,checkpoint_source=root)
                self.assertIsNone(a['files']['bootstrap_native'])
                self.assertNotIn('terrain_bootstrap.py',a['files']['implementation'])
                self.assertEqual(manifest.verify_manifest_files(a,root),a['world_hash'])

    def test_exporter_rejects_wrong_world_or_seed_before_neural_reads(self):
        world = SimpleNamespace(seed=0, _terrain_world_profile='terrestrial-gondwana')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            neural_manifest(world, 'terrestrial-earthlike', file_hashes=False)
        with self.assertRaisesRegex(ValueError, 'seed differs'):
            export_neural_intermediates(world, {'seed': '1'}, Path('unused'))


if __name__ == '__main__':
    unittest.main()
