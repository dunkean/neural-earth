"""Serve the sketch-to40km browser demo and an external, verified model bundle."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import shutil
from urllib.parse import urlsplit

from distill.common import REPO, external_path


def routes(models):
    manifest = json.loads((models/'manifest.json').read_text())
    web = REPO/'webgpu'
    runtime=models/'runtime-snapshot' if (models/'runtime-snapshot').is_dir() else web
    for name in ('tile40_runtime.mjs', 'noise.mjs'):
        key = 'runtime_sha256' if name.startswith('tile40_') else 'noise_sha256'
        if hashlib.sha256((runtime/name).read_bytes()).hexdigest() != manifest['provenance'][key]:
            raise ValueError('Bundle runtime differs;version the bundle before serving it.')
    files = {'/model/manifest.json':models/'manifest.json'}
    for model in manifest['models'].values():
        file = models/model['file']
        if Path(model['file']).name != model['file'] or not file.is_file():
            raise ValueError('Invalid model path.')
        if file.stat().st_size != model['bytes'] or model['bytes'] >= 100000000:
            raise ValueError('Model size differs or exceeds100MB.')
        if hashlib.sha256(file.read_bytes()).hexdigest() != model['sha256']:
            raise ValueError('Model identity changed.')
        files['/model/'+model['file']] = file
    files.update({'/':web/'tile40_demo.html', '/demo.mjs':web/'tile40_demo.mjs',
        '/demo_helpers.mjs':web/'tile40_demo_helpers.mjs', '/worker.js':web/'tile40_demo_worker.js'})
    files.update({'/runtime/'+name:runtime/name for name in ('tile40_runtime.mjs', 'noise.mjs')})
    files.update({'/ort/'+file.name:file for file in (web/'vendor').iterdir() if file.is_file()})
    return files


def handler(files):
    class DemoHandler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

        def respond(self, body):
            file = files.get(urlsplit(self.path).path)
            if file is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
            self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            mime = 'text/javascript' if file.suffix in ('.mjs', '.js') else mimetypes.guess_type(file)[0]
            self.send_header('Content-Type', mime or 'application/octet-stream')
            self.send_header('Content-Length', str(file.stat().st_size))
            self.end_headers()
            if body:
                with file.open('rb') as source:
                    shutil.copyfileobj(source, self.wfile)
    return DemoHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, help='Explicit historical research bundle;default serves only the curated models.')
    parser.add_argument('--port', type=int, default=8780)
    args = parser.parse_args()
    if args.models is None:
        from flask import Flask,redirect
        from backend.terrain_tile40 import register_tile40_routes
        app=Flask(__name__)
        register_tile40_routes(app)
        app.add_url_rule('/',view_func=lambda:redirect('/tile40/'))
        app.run(host='127.0.0.1',port=args.port,threaded=True,debug=False)
        return
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(routes(external_path(args.models))))
    print(f'Browser demo http://127.0.0.1:{args.port}/ | inference stays in the browser.', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
