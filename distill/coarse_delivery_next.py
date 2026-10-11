"""Wait for both full coarse jobs, bundle verified weights, and optionally push.

No base training or server startup. Terminal failed/interrupted jobs and changed
delivery files stop the queue for inspection instead of restarting training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from distill.bundle_coarse import prepare
from distill.common import DATA, REPO, atomic_json, atomic_write
from distill.jobs import state

FILES = ('distill/weights/coarse1.pt', 'distill/weights/coarse2.pt',
         'distill/weights/manifest.json', 'distill/LIVE.md', 'distill/TASKS.md')


def git(*args, **kwargs):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True, **kwargs).strip()


def hashes():
    return {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--token-file', type=Path, default=Path('/mnt/d/workspace/dunk_token'))
    args = parser.parse_args()
    progress_path = DATA/'coarse-delivery-next.json'
    contract_path = DATA/'coarse-delivery-next-contract.json'
    contract = dict(branch='distill', original_hashes=hashes(), push=args.push)
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        if contract['push'] != args.push:
            raise ValueError('Delivery push setting changed; inspect the saved contract.')
    else:
        atomic_json(contract_path, contract)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)

    def progress(phase, status='running', **values):
        value = dict(status=status, phase=phase, updated_at=time.time(), **values)
        atomic_json(progress_path, value)
        print(json.dumps(value), flush=True)

    progress('waiting-for-full-coarse')
    try:
        last_observation = 0.
        while not stopped:
            jobs = {name: state(DATA/'jobs'/name) for name in ('coarse-full1', 'coarse-full2')}
            for name, job in jobs.items():
                if not job['live'] and (job.get('status') != 'complete' or job.get('returncode') != 0):
                    raise RuntimeError(f'{name} is terminal without success; inspect it before delivery.')
            if all(not job['live'] for job in jobs.values()):
                break
            if time.monotonic()-last_observation >= 60:
                progress('waiting-for-full-coarse', jobs={name: dict(live=job['live'],
                    child_pid=job.get('child_pid')) for name, job in jobs.items()})
                last_observation = time.monotonic()
            # Each pass observes real tmux/process identities, not just run files.
            time.sleep(10)
        if stopped:
            progress('waiting-for-full-coarse', status='interrupted')
            return
        if git('branch', '--show-current') != contract['branch']:
            raise RuntimeError('Branch changed while waiting; do not commit on another branch.')
        if hashes() != contract['original_hashes']:
            raise RuntimeError('Delivery files changed while waiting; inspect before replacing them.')
        if git('diff', '--cached', '--name-only'):
            raise RuntimeError('Other files are staged; inspect before the delivery commit.')
        root = DATA/'ckpt/coarse-full'
        manifest_path = REPO/'distill/weights/manifest.json'
        payloads, manifest = prepare(root, json.loads(manifest_path.read_text()))
        progress('verified-exports', selected_steps={name: manifest['exports'][name]['training_step']
                                                   for name in ('coarse1', 'coarse2')})
        live_path = REPO/'distill/LIVE.md'
        live = live_path.read_text()
        start = live.index('Coarse **2 étapes**')
        end = live.index('Les bases distillés rejetés', start)
        selections = manifest['exports']
        replacement = (
            'Coarse **2 étapes** et **1 étape** sont inclus sous `distill/weights/coarse2.pt`\n'
            'et `coarse1.pt`. Les deux runs ont terminé leurs **200 000 étapes**.\n'
            f"Les EMA retenues sur384 crops sont aux pas{selections['coarse2']['training_step']} "
            f"et{selections['coarse1']['training_step']} respectivement.\n"
            'Les états latest/best et les jalons restent archivés hors Git ; seuls les\n'
            'poids d’inférence sont distribués. Sorties finies et équivalence export/\n'
            'CUDA Graph sont vérifiées ; la qualité visuelle reste à juger dans l’app.\n'
            'Le modèle par défaut reste coarse4.\n')
        tasks_path = REPO/'distill/TASKS.md'
        tasks = tasks_path.read_text()
        marker = '## Priorité utilisateur : terminer coarse1 et coarse2 avant le base'
        note = ('## Livraison coarse1/coarse2 complets\n\n'
            'Les deux entraînements ont terminé200 000 étapes. Poids best EMA sur384\n'
            f"crops : coarse1 pas{selections['coarse1']['training_step']}, "
            f"coarse2 pas{selections['coarse2']['training_step']}.\n"
            'Les trois contrôles export/CUDA Graph par modèle donnent une différence\n'
            'maximale0 ; provenance/SHA et statut complet dans `weights/manifest.json`.\n'
            'Best/latest et jalons conservés sous `~/data/distill/ckpt/coarse-full/`.\n'
            'Aucun poids base livré. Reprendre maintenant la stratégie base décrite\n'
            'ci-dessous : supervision LOD3/LOD0, cas rares, coarse original/4/2.\n\n')
        for name, payload in payloads.items():
            atomic_write(REPO/'distill/weights'/name, lambda handle, payload=payload: handle.write(payload))
        atomic_json(manifest_path, manifest)
        atomic_write(live_path, lambda handle: handle.write((live[:start]+replacement+live[end:]).encode()))
        atomic_write(tasks_path, lambda handle: handle.write(tasks.replace(marker, note+marker, 1).encode()))
        git('diff', '--check')
        progress('bundled')
        git('add', *FILES)
        git('commit', '-m', 'Deliver fully trained one and two step coarse models')
        head = git('rev-parse', 'HEAD')
        if args.push:
            script = ('#!/usr/bin/env python3\nimport pathlib,re,sys\n'
                'if "Username" in sys.argv[1]: print("x-access-token")\nelse:\n'
                ' token=re.search(r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]+",'
                'pathlib.Path(__import__("os").environ["COARSE_DELIVERY_TOKEN_FILE"]).read_text())\n'
                ' if token is None: raise SystemExit("GitHub token format not found")\n'
                ' print(token.group())\n')
            with tempfile.TemporaryDirectory(prefix='coarse-delivery-auth-') as directory:
                askpass = Path(directory)/'askpass'
                askpass.write_text(script); askpass.chmod(0o700)
                environment = dict(os.environ, GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT='0',
                                   COARSE_DELIVERY_TOKEN_FILE=str(args.token_file))
                git('-c', 'credential.helper=', 'push', 'origin', 'distill', env=environment)
                remote = git('-c', 'credential.helper=', 'ls-remote', 'origin', 'refs/heads/distill', env=environment).split()[0]
                if remote != head:
                    raise RuntimeError('Remote branch differs from the delivery commit.')
        progress('delivered', status='complete', commit=head, pushed=args.push,
                 selected_steps={name: selections[name]['training_step'] for name in ('coarse1', 'coarse2')})
    except Exception as exc:
        progress('inspect-before-resume', status='needs-review', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
