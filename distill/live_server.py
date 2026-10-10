"""Supervised local viewer with independent NN choices and post-Orogen reset.

source distill/env.sh
CUDA_VISIBLE_DEVICES=1 python -m distill.live_server --port 8765
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from distill.common import DATA, REPO, atomic_json


def serve(port, config):
    os.environ.update(TERRAIN_DISTILL_LIVE='1', DISTILL_LIVE_CONFIG=str(config),
                      TERRAIN_PREWARM='0', TERRAIN_GPU_MODE='single')
    import terrain_server as server
    from flask import Response, jsonify, request, send_file
    from distill.live_runtime import current
    models = current()
    switching = threading.Event()

    @server.app.after_request
    def review_ui(response):
        if request.path == '/' and response.status_code == 200:
            html = (REPO/'web/index.html').read_text()
            # Reference selection happens per NN in this server. An old viewer
            # link must not silently redirect the chosen models to /reference.
            html = html.replace('<script', '<script>(()=>{const url=new URL(location.href);'
                'url.searchParams.set("nn_engine","exact");history.replaceState(null,"",url);'
                '})()</script><script', 1)
            html = html.replace('</html>', '<script src="/distill/live-controls.js"></script></html>')
            return Response(html, mimetype='text/html', headers={'Cache-Control': 'no-store'})
        if request.path.startswith('/api/distill/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @server.app.get('/distill/live-controls.js')
    def controls():
        return send_file(REPO/'distill/live_controls.js', max_age=0)

    @server.app.get('/api/distill/models')
    def choices():
        counts = {stage: 0 for stage in ('coarse', 'base', 'decoder')}
        with server.runtimes_lock:
            runtimes = list(server.runtimes.values())
        for runtime in runtimes:
            with runtime.lock:
                for world in {id(world): world for world in runtime.live_worlds()}.values():
                    for stage, value in getattr(world, '_distill_counts', {}).items():
                        counts[stage] += value['calls']
        return jsonify(models.public() | dict(switching=switching.is_set(),
                        cache_profile=server.PROFILE, student_calls=counts))

    def restart():
        time.sleep(.5)
        try:
            server.generation_coordinator.suspend(True)
            server.coarse_background.close()
            with server.all_gpu_locks() as runtimes:
                server.synchronize_gpus()
                for runtime in runtimes:
                    for world in runtime.live_worlds():
                        preparation = getattr(world, '_terrain_coarse_preparation', None)
                        if preparation is not None:
                            preparation.flush(timeout=30)
            server.physical_delivery.flush()
        finally:
            os._exit(77)

    @server.app.post('/api/distill/models')
    def change():
        with server.generation_coordinator.lock:
            if switching.is_set():
                return jsonify(error='A model change is already in progress.'), 409
            try:
                config = models.save_request(request.get_json(silent=True))
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
            switching.set()
            server.generation_coordinator.suspend(True)
            threading.Thread(target=restart, name='distill-model-change', daemon=True).start()
            return jsonify(restarting=True, revision=config['revision']), 202

    # Settings are applied by the supervisor, never by the production restart
    # route (which would lose the live-review entry point).
    server.RESTART_COMMAND = None
    server.app.run(host='127.0.0.1', port=port, threaded=True, debug=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--config', type=Path, default=DATA/'live/config.json')
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    config = args.config.expanduser().resolve()
    if args.child:
        serve(args.port, config)
        return
    if not config.exists():
        atomic_json(config, dict(selection={stage: 'teacher' for stage in ('coarse', 'base', 'decoder')},
                                 revision=uuid.uuid4().hex, reset='initial'))
    stopped = False
    process = None

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if process is not None and process.poll() is None:
            process.terminate()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    while not stopped:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'distill.live_server', '--child',
                                    '--port', str(args.port), '--config', str(config)], cwd=REPO)
        code = process.wait()
        if stopped:
            break
        if code != 77:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
