import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'backend'))
from terrain_illustration import illustration_root


class IllustrationIntegrationTests(unittest.TestCase):
    def test_locates_bundled_client_without_copying_shared_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'earth'
            client = root / 'native' / 'magna-urbis' / 'illustration' / 'web' / 'index.mjs'
            client.parent.mkdir(parents=True); client.write_text('export {}', encoding='utf-8')
            self.assertEqual(illustration_root(root), root / 'native' / 'magna-urbis')


if __name__ == '__main__': unittest.main()
