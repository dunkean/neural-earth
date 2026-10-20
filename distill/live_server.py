"""Supervised local viewer with independent NN choices and post-Orogen reset.

source distill/env.sh
CUDA_VISIBLE_DEVICES=1 python -m distill.live_server --port 8765
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from distill.common import DATA, REPO


def serve(port, config):
    os.environ.update(TERRAIN_DISTILL_LIVE='1', DISTILL_LIVE_CONFIG=str(config),
                      TERRAIN_PREWARM='0', TERRAIN_GPU_MODE='single')
    import terrain_server as server
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

    server.app.extensions['terrain_model_restart'] = lambda: threading.Thread(
        target=restart,name='terrain-model-change',daemon=True).start()

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
    from distill.live_runtime import catalog, prepare_config
    prepare_config(config, catalog())
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
