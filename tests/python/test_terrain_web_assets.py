"""Static web assets: every page script has a server route, and the imported
Render3D brush atlas matches its manifest."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import ast
import hashlib
import json
import re
import unittest
import zlib

from terrain_paths import WEB_ROOT, source_path


def server_routes():
    routes = set()
    # terrain_models registers its own page script on the server app.
    for name in ('terrain_server.py', 'terrain_models.py'):
        routes |= decorated_routes(ast.parse(source_path(name).read_text(encoding='utf-8')))
    return routes


def decorated_routes(tree):
    routes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                        and decorator.func.attr == 'get' and decorator.args
                        and isinstance(decorator.args[0], ast.Constant)):
                    routes.add(decorator.args[0].value)
    return routes


class WebAssetTests(unittest.TestCase):
    def test_page_scripts_and_render3d_assets_are_served(self):
        html = (WEB_ROOT / 'index.html').read_text(encoding='utf-8')
        scripts = {'/' + src.lstrip('/') for src in re.findall(r'<script src="([^"]+)"', html)}
        self.assertIn('/terrain_render3d.js', scripts)
        routes = server_routes()
        self.assertEqual(sorted(scripts - routes), [])
        self.assertIn('/forest-brushes.bin', routes)

    def test_forest_brush_atlas_matches_its_manifest(self):
        source = (WEB_ROOT / 'terrain_forest_brushes.js').read_text(encoding='utf-8')
        manifest = json.loads(source[source.index('=') + 1:source.rindex(';')])
        pixels = zlib.decompress((WEB_ROOT / manifest['url']).read_bytes())
        self.assertEqual(len(pixels), manifest['bytes'])
        self.assertEqual(manifest['bytes'], manifest['size'] ** 2 * 2 * manifest['layers'])
        self.assertEqual(manifest['layers'], len(manifest['families']) * manifest['variants'])
        self.assertEqual(hashlib.sha256(pixels).hexdigest(), manifest['sha256'])


if __name__ == '__main__':
    unittest.main()
