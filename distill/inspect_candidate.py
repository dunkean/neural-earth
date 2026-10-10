"""Freeze and physically inspect a completed training pass on both case banks.

Successful execution means measurements were produced; acceptance still requires
quality, seam and speed evidence. Candidate files and SHA stay stable across sites.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from distill.common import DATA, atomic_json, atomic_write, external_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['coarse', 'base', 'decoder'], required=True)
    parser.add_argument('--minimum-step', type=int, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, help='Separate architecture trial directory.')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--rare-manifest', type=Path, default=DATA/'eval/rare-sites-coherent.json')
    args = parser.parse_args()
    directory = external_path(args.checkpoint_dir or DATA/'ckpt'/args.stage)
    status = json.loads((directory/'status.json').read_text())
    if status['status'] != 'complete' or status['step'] < args.minimum_step:
        raise ValueError('The requested training pass has not completed; inspect its live handle first.')
    payload = (directory/'best.pt').read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if saved['config']['stage'] != args.stage or saved['step'] > status['step']:
        raise ValueError('Candidate is inconsistent with the completed stage.')
    candidate = DATA/'ckpt/candidates'/f'{args.stage}-step{saved["step"]}-{digest[:12]}.pt'
    if candidate.exists() and hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
        raise ValueError('Frozen candidate bytes changed.')
    if not candidate.exists():
        atomic_write(candidate, lambda handle: handle.write(payload))
    del saved, payload
    from tools.verification.compare_base_variants import run, sheet
    from distill.evaluate import audit
    from distill.rare_cases import audit as rare_audit
    variant = dict(coarse='student_coarse', base='student', decoder='student_decoder')[args.stage]
    paths = dict(coarse='coarse_student', base='student', decoder='decoder_student')
    parameters = {paths[args.stage]: candidate}
    output = external_path(DATA/'eval'/args.tag)
    rare_output = external_path(DATA/'eval'/(args.tag+'-rare'))
    run(output, ['reference', 'fp32base', variant], **parameters)
    sheet(output)
    result = audit(json.loads((output/'report.json').read_text()), variant)
    atomic_json(output/'physical-audit.json', result | dict(accepted=False))
    run(rare_output, ['reference', 'fp32base', variant], site_manifest=args.rare_manifest, **parameters)
    sheet(rare_output)
    rare = rare_audit(args.rare_manifest, rare_output, variant, rare_output/'physical-audit.json')
    atomic_json(output/'inspection.json', dict(candidate=str(candidate), sha256=digest,
        completed_training_step=status['step'], original_passed=result['physical_passed'],
        rare_passed=rare['physical_passed'], accepted=False,
        note='Physical inspection only; combined models, seam and speed gates still required.'))
    print(json.dumps(dict(candidate=str(candidate), original_passed=result['physical_passed'],
                          rare_passed=rare['physical_passed'], accepted=False)), flush=True)


if __name__ == '__main__':
    main()
