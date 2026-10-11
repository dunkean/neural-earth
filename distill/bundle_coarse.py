"""Bundle both completed coarse runs after their full-solver export checks.

This updates inference weights and provenance only. Git commit/push remains a
separate operation, and an incomplete run never replaces the bundled models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def prepare(root, manifest, target_steps=200000):
    result = json.loads(json.dumps(manifest))
    if result['teacher_revision'] != REVISION:
        raise ValueError('Bundled teacher revision differs from the pinned reference.')
    payloads = {}
    for number in (1, 2):
        run = Path(root)/f'coarse{number}'
        status = json.loads((run/'status.json').read_text())
        if (status.get('status') != 'complete' or status.get('step') != target_steps or
                not (run/'delivery.json').is_file()):
            raise ValueError(f'Coarse{number} has not completed training and export verification.')
        proof = json.loads((run/'delivery.json').read_text())
        if (status.get('status') != 'complete' or status.get('step') != target_steps or
                proof.get('status') != 'complete' or proof.get('training_steps') != target_steps or
                proof.get('validation_count') != 384 or proof.get('finite') is not True or
                proof.get('export_max_abs') != 0. or
                proof.get('smoke_indices') != [0, 192, 383]):
            raise ValueError(f'Coarse{number} has not completed training and export verification.')
        export = json.loads((run/'export/manifest.json').read_text())
        if proof['export'] != export:
            raise ValueError(f'Coarse{number} delivery no longer matches its immutable export.')
        source, artifact = export['source'], export['export']
        if (export['stage'] != 'coarse' or not export['inference_only'] or
                source['config']['solver_steps'] != number or
                source['config']['stage'] != 'coarse' or
                source['step'] != proof['selected_step'] or
                not 5000 < source['step'] <= target_steps or artifact['file'] != 'coarse.pt'):
            raise ValueError(f'Coarse{number} source/solver selection is inconsistent.')
        payload = (run/'export/coarse.pt').read_bytes()
        if digest(payload) != artifact['sha256'] or len(payload) != artifact['bytes']:
            raise ValueError(f'Coarse{number} inference weights changed after verification.')
        if digest((run/'best.pt').read_bytes()) != source['sha256']:
            raise ValueError(f'Coarse{number} selected checkpoint changed after export.')
        name = f'coarse{number}'
        payloads[name+'.pt'] = payload
        result['exports'][name] = dict(file=name+'.pt', bytes=len(payload),
            sha256=digest(payload), stage='coarse', source_checkpoint_sha256=source['sha256'],
            training_step=source['step'], config=source['config'], selected_weights='ema',
            training_status='complete', last_training_step=target_steps,
            target_training_steps=target_steps, validation_count=384,
            export_max_abs=0., smoke_indices=proof['smoke_indices'], physical_acceptance=False)
    result['note'] = ('User-authorized Git distribution of coarse8/coarse4/coarse2/coarse1 and decoder. '
        'Coarse1/2 completed200k training runs; inference uses each full-validation best EMA. '
        'Visual quality remains subject to user review. No distilled base weights.')
    return payloads, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DATA/'ckpt/coarse-full')
    parser.add_argument('--bundled', type=Path, default=REPO/'distill/weights')
    parser.add_argument('--write', action='store_true', help='Replace the bundled coarse1/2 and their manifest.')
    args = parser.parse_args()
    manifest_path = args.bundled/'manifest.json'
    try:
        payloads, manifest = prepare(args.root, json.loads(manifest_path.read_text()))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.write:
        for name, payload in payloads.items():
            atomic_write(args.bundled/name, lambda handle, payload=payload: handle.write(payload))
        atomic_json(manifest_path, manifest)
    print(json.dumps(dict(written=args.write, exports={name: manifest['exports'][name]
                                                      for name in ('coarse1', 'coarse2')}), indent=2))


if __name__ == '__main__':
    main()
