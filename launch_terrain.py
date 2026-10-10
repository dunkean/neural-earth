"""Start the local server when needed, wait until ready, then open the browser."""
import subprocess
import argparse
import json
import os
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

root = Path(__file__).resolve().parent
url = 'http://127.0.0.1:8765'


def alive():
    try:
        with urllib.request.urlopen(url + '/api/status', timeout=2) as response:
            data = json.load(response)
            return response.status == 200 and str(data.get('version', '')).startswith('natural-')
    except (OSError, ValueError):
        return False


def start_server():
    options = ({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt'
               else {'start_new_session': True})
    with open(root / 'server.log', 'a', encoding='utf-8') as out, open(root / 'server-error.log', 'a', encoding='utf-8') as err:
        return subprocess.Popen([sys.executable, '-u', str(root / 'backend' / 'terrain_server.py')], cwd=root,
                                stdin=subprocess.DEVNULL, stdout=out, stderr=err, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-open', action='store_true', help='Start without opening a browser (verification).')
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=True,
                        help="Enable GPU acceleration on a browser's first launch (default).")
    parser.add_argument('--cpu', dest='gpu', action='store_false',
                        help="Use the CPU options on a browser's first launch.")
    args = parser.parse_args(argv)
    if not alive():
        process = start_server()
        for _ in range(120):
            if alive():
                break
            if process.poll() is not None:
                raise SystemExit('Server failed to start. See server-error.log.')
            time.sleep(.5)
        else:
            raise SystemExit('Server failed to start. See server-error.log.')
    if not args.no_open:
        # Replaces the first-launch GPU question; a browser's saved choice wins.
        webbrowser.open(url + '/?gpu=' + ('1' if args.gpu else '0'))
    print('Neural Earth available: ' + url)


if __name__ == '__main__':
    main()
