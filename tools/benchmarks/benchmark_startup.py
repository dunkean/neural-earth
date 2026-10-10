"""Cold-start timeline: server spawn -> HTTP up -> /api/world -> Render overview.

Replays a browser's first requests (Orogen world, Render mode) against a fresh
server on a separate port, so a running server is left alone. Without --seed a
new random world is generated; with a seed already seen, the cached world is
reopened. Stop other servers first: they share the GPU memory.
"""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
RENDER_SETTINGS = {"season": .5, "variation": 1, "forest": 1, "moisture": 0, "rock_slope": 40, "snow": 1,
                   "vegetation_tint": [1, 1, 1], "rock_tint": [1, 1, 1], "snow_color": [.94, .96, .97]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8775)
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=True,
                        help='Run every optional Orogen stage on CUDA (default).')
    parser.add_argument('--cpu', dest='gpu', action='store_false', help='Run Orogen on the CPU.')
    parser.add_argument('--seed', type=int, help='World seed (default: a new random world).')
    parser.add_argument('--log', type=Path, help='Server output (default: discarded).')
    args = parser.parse_args(argv)
    base = f'http://127.0.0.1:{args.port}'
    started = time.perf_counter()

    def mark(name, **info):
        print(f'{time.perf_counter()-started:7.2f}s  {name} {json.dumps(info) if info else ""}', flush=True)

    def get(path, timeout=600):
        with urllib.request.urlopen(base+path, timeout=timeout) as response:
            return response.read()

    def post(path, data):
        request = urllib.request.Request(base+path, json.dumps(data).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    log = open(args.log, 'w', encoding='utf-8') if args.log else subprocess.DEVNULL
    options = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt'
               else {'start_new_session': True})
    server = subprocess.Popen([sys.executable, '-u', str(ROOT/'backend'/'terrain_server.py'), '--port', str(args.port)],
                              cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, **options)
    mark('spawn')
    try:
        while True:
            if server.poll() is not None:
                raise SystemExit('The server exited during startup' + (f'; see {args.log}' if args.log else ''))
            try:
                get('/api/status', 1)
                break
            except OSError:
                time.sleep(.05)
        mark('http up')
        stop = threading.Event()

        def watch_preload():
            last = None
            while not stop.is_set():
                try:
                    preload = json.loads(get('/api/status', 5)).get('model_preload', {})
                    current = preload.get('state'), preload.get('warming_model')
                    if current != last:
                        mark('model preload', state=current[0], warming=current[1])
                        last = current
                except OSError:
                    pass
                time.sleep(.1)
        threading.Thread(target=watch_preload, daemon=True).start()

        schema = json.loads(get('/api/generation/schema'))
        settings = dict(schema['defaults_by_profile']['orogen'], climate_source='orogen')
        settings.update({key: args.gpu for key in settings if key.startswith('orogen_gpu_')})
        seed = args.seed if args.seed is not None else random.randrange(2**63)
        session = uuid.uuid4().hex
        query = urllib.parse.urlencode(dict(seed=seed, world_profile='orogen', session=session, generation_epoch=1,
                                            generation=json.dumps(settings)))
        world = json.loads(get('/api/world?'+query))
        mark('world', seed=str(seed), orogen='gpu' if args.gpu else 'cpu')
        # The overview job only runs while a view declares interest in it.
        post('/api/view', dict(session=session, epoch=1, seed=str(seed), world_profile=world['generation_profile'],
                               mode='render', tiles=[], overview=True, overview_render_settings=RENDER_SETTINGS))
        url = world['overview']+'&'+urllib.parse.urlencode(dict(
            session=session, epoch=1, mode='render', render_settings=json.dumps(RENDER_SETTINGS),
            world_identity=world['world_identity']))
        while True:
            try:
                image = get(url)
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 409:
                    raise
                time.sleep(.25)
        mark('render overview', bytes=len(image))
        stop.set()
    finally:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(server.pid)], capture_output=True)
        else:
            os.killpg(server.pid, signal.SIGTERM)
        server.wait()
        if args.log:
            log.close()


if __name__ == '__main__':
    main()
