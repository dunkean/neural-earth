"""Serve the finite40km browser workflow on the normal application origin."""
import json
from pathlib import Path
import subprocess
import sys
import threading

from flask import Response,jsonify,redirect,request,send_file

from backend.terrain_paths import REPO_ROOT
from distill.tile40_prepare import bundle_root,sha


class Tile40Assets:
    def __init__(self,root=None):
        self.root=Path(root or bundle_root())
        self.lock=threading.Lock()
        self.process=None

    def state(self):
        catalog=self.root/'catalog.json'
        if catalog.is_file():return json.loads(catalog.read_text())
        status=self.root.with_suffix('.status.json')
        if self.process is not None and self.process.poll() is None:
            return dict(status='preparing')
        if status.is_file():
            saved=json.loads(status.read_text())
            if saved['status']=='error':return saved
            from filelock import FileLock,Timeout
            try:
                with FileLock(str(self.root)+'.lock',timeout=0):pass
            except Timeout:return dict(status='preparing')
            return dict(status='error',error='Browser preparation was interrupted;retry the preparation.')
        return dict(status='missing')

    def start(self):
        with self.lock:
            state=self.state()
            if state['status'] in ('ready','preparing'):return state
            self.root.parent.mkdir(parents=True,exist_ok=True)
            with self.root.with_suffix('.log').open('ab') as log:
                self.process=subprocess.Popen([sys.executable,'-u','-m','distill.tile40_prepare','--output',str(self.root)],
                    cwd=REPO_ROOT,stdout=log,stderr=subprocess.STDOUT)
            return dict(status='preparing')

    def model_file(self,decoder,name):
        if decoder not in ('decoder16','decoder64') or Path(name).name!=name:return None
        directory=self.root/decoder
        if self.state()['status']!='ready':return None
        manifest_path=directory/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        if name=='manifest.json':return manifest_path
        entry=next((v for v in manifest['models'].values() if v['file']==name),None)
        if entry is None:return None
        path=directory/name
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
            raise ValueError('A browser model changed;prepare the selected models again.')
        return path


def register_tile40_routes(app,assets=None,python_runtime=None):
    assets=assets or Tile40Assets()
    from backend.terrain_tile40_python import Tile40Python
    python_runtime=python_runtime or Tile40Python()
    web=REPO_ROOT/'webgpu'

    @app.after_request
    def browser_isolation(response):
        if request.path.startswith('/tile40/'):
            response.headers.update({'Cross-Origin-Opener-Policy':'same-origin',
                'Cross-Origin-Embedder-Policy':'require-corp','Cross-Origin-Resource-Policy':'same-origin'})
        return response

    @app.get('/tile40')
    def tile40_redirect():return redirect('/tile40/')

    @app.get('/tile40/')
    def tile40_index():return send_file(web/'tile40_demo.html')

    @app.get('/tile40/catalog.json')
    def tile40_catalog():return jsonify(assets.state())

    @app.post('/tile40/prepare')
    def tile40_prepare():return jsonify(assets.start())

    @app.get('/tile40/python/catalog.json')
    def tile40_python_catalog():
        try:return jsonify(python_runtime.catalog(request.args.get('decoder','decoder16')))
        except ValueError as exc:return jsonify(error=str(exc)),400

    @app.post('/tile40/python/generate')
    def tile40_python_generate():
        if request.content_length is not None and request.content_length>65536:
            return jsonify(error='Croquis trop volumineux.'),413
        try:
            body,metadata=python_runtime.generate(request.get_json(silent=True))
            return Response(body,mimetype='application/octet-stream',headers={
                'X-Terrain-Metadata':json.dumps(metadata,separators=(',',':')),'Cache-Control':'no-store'})
        except ValueError as exc:return jsonify(error=str(exc)),400
        except BlockingIOError as exc:return jsonify(error=str(exc)),409
        except Exception as exc:
            app.logger.exception('Python finite-tile generation failed')
            return jsonify(error=str(exc)),503

    @app.get('/tile40/model/<decoder>/<name>')
    def tile40_model(decoder,name):
        try:path=assets.model_file(decoder,name)
        except (ValueError,OSError) as exc:return jsonify(error=str(exc)),503
        if path is None:return jsonify(error='Unknown model or preparation incomplete.'),404
        return send_file(path,conditional=True)

    @app.get('/tile40/<path:name>')
    def tile40_asset(name):
        fixed={'demo.mjs':web/'tile40_demo.mjs','demo_helpers.mjs':web/'tile40_demo_helpers.mjs',
            'worker.js':web/'tile40_demo_worker.js','runtime/tile40_runtime.mjs':web/'tile40_runtime.mjs',
            'runtime/noise.mjs':web/'noise.mjs'}
        if name.startswith('ort/') and Path(name).name==name[4:]:
            vendor=web/'vendor'/name[4:]
            if vendor.is_file():return send_file(vendor)
        path=fixed.get(name)
        if path is None:return jsonify(error='Unknown browser asset.'),404
        return send_file(path,mimetype='text/javascript')

    return assets
