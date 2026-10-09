"""Start the local server when needed, wait until ready, then open the browser."""
import subprocess
import argparse
import json
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

root = Path(__file__).resolve().parent
url = 'http://127.0.0.1:8765'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--no-open', action='store_true', help='Start without opening a browser (verification).')
args = parser.parse_args()


def alive():
    try:
        with urllib.request.urlopen(url + '/api/status', timeout=2) as response:
            data = json.load(response)
            return response.status == 200 and str(data.get('version', '')).startswith('natural-')
    except OSError:
        return False


if not alive():
    with open(root / 'server.log', 'a', encoding='utf-8') as out, open(root / 'server-error.log', 'a', encoding='utf-8') as err:
        subprocess.Popen([sys.executable, '-u', str(root / 'backend' / 'terrain_server.py')], cwd=root,
                         stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(120):
        if alive():
            break
        time.sleep(.5)
    else:
        raise SystemExit('Server failed to start. See server-error.log.')
if not args.no_open:
    webbrowser.open(url)
print('Neural Earth available: ' + url)
