"""Run the isolated browser probe on nativeWindows hardware fromWSL."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from distill.common import REPO, atomic_json, external_path


def winpath(path):
    return subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('capabilities', 'wasm', 'webgpu'), default='webgpu')
    parser.add_argument('--models', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--full-tile', action='store_true')
    parser.add_argument('--demo', action='store_true', help='Verify sketch editing and full CPU exports in the local browser demo.')
    parser.add_argument('--demo-matrix', action='store_true', help='Measure all six demo sketches across five climates onCPU.')
    parser.add_argument('--url', default='http://127.0.0.1:8780/')
    parser.add_argument('--decoder',choices=('decoder16','decoder64'),default='decoder16')
    args = parser.parse_args()
    if (args.demo or args.demo_matrix) and args.full_tile:
        raise ValueError('Choose the UI probe or the numerical fixture probe.')
    if args.demo_matrix and args.provider!='wasm':
        raise ValueError('The demo matrix currently verifies CPU;select --provider wasm.')
    output = external_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    edge = Path('/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
    node = Path('/mnt/c/Program Files/nodejs/node.exe')
    if not edge.exists() or not node.exists():
        raise FileNotFoundError('Native Windows Edge and Node required for hardware proof.')
    powershell = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    check = '(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:9226/json/version).Content'
    response = subprocess.run([powershell, '-NoProfile', '-Command', check], capture_output=True, text=True)
    started_edge = False
    if response.returncode:
        started_edge = True
        with (output.parent/'windows-edge-start.log').open('ab') as log:
            child = subprocess.Popen([str(edge), '--headless=new', '--no-first-run', '--disable-extensions',
                '--enable-unsafe-webgpu', '--remote-debugging-port=9226',
                '--user-data-dir=D:\\workspace\\tile40-browser-profile', 'about:blank'],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(output.parent/'windows-edge-launch.json', dict(bridge_pid=child.pid, port=9226,
                    profile='D:/workspace/tile40-browser-profile'))
        for _ in range(20):
            response = subprocess.run([powershell, '-NoProfile', '-Command', check], capture_output=True, text=True)
            if not response.returncode:
                break
            time.sleep(.5)
        else:
            raise RuntimeError('The isolated Windows browser did not expose its own debugging endpoint.')
    endpoint = json.loads(response.stdout)
    if not endpoint['Browser'].startswith('Edg/'):
        raise ValueError('Do not reuse a foreign browser debugging endpoint.')
    script = 'tile40_demo_matrix.cjs' if args.demo_matrix else 'tile40_demo_probe.cjs' if args.demo else 'tile40_full_browser_probe.cjs' if args.full_tile else 'tile40_browser_probe.cjs'
    command = [str(node), winpath(REPO/'webgpu'/script), '--cdp', 'http://127.0.0.1:9226',
               '--provider', args.provider, '--threads', str(args.threads), '--output', winpath(output)]
    if args.models:
        command.extend(['--models', winpath(args.models)])
    if args.demo or args.demo_matrix:
        command.extend(['--url', args.url])
    if args.demo:
        command.extend(['--decoder',args.decoder])
    result = subprocess.run(command)
    if result.returncode:
        raise RuntimeError(f'Native browser probe exited{result.returncode};inspect its report before retry.')
    report_file = output/'report.json' if args.demo or args.demo_matrix else output
    report = json.loads(report_file.read_text())
    report['execution_host'] = 'native-windows-node+edge;artifact files viaWSL UNC'
    report['started_isolated_browser'] = started_edge
    atomic_json(report_file, report)


if __name__ == '__main__':
    main()
