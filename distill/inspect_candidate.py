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


def validate_candidate(saved, status, stage, minimum_step, interim=False, selection='best'):
    if (saved['config']['stage'] != stage or saved['step'] > saved['arguments']['steps'] or
            (not interim and saved['step'] > status['step'])):
        raise ValueError('Candidate is inconsistent with the training stage.')
    if (interim or selection == 'latest') and saved['step'] < minimum_step:
        raise ValueError('Selected checkpoint predates the requested training budget.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['coarse', 'base', 'decoder'], required=True)
    parser.add_argument('--minimum-step', type=int, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, help='Separate architecture trial directory.')
    parser.add_argument('--checkpoint', choices=['best', 'latest'], default='best',
                        help='Use latest for a comparison at the same completed training budget.')
    parser.add_argument('--interim', action='store_true', help='Inspect a frozen best checkpoint during training; no completion claim.')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--rare-manifest', type=Path, default=DATA/'eval/rare-sites-complete.json')
    args = parser.parse_args()
    directory = external_path(args.checkpoint_dir or DATA/'ckpt'/args.stage)
    status = json.loads((directory/'status.json').read_text())
    if (status['status'] != 'complete' and not (args.interim and status['status'] == 'running')) or status['step'] < args.minimum_step:
        raise ValueError('The requested training pass has not completed; inspect its live handle first.')
    payload = (directory/(args.checkpoint+'.pt')).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    validate_candidate(saved, status, args.stage, args.minimum_step, args.interim, args.checkpoint)
    candidate_step = saved['step']
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
        training_status=status['status'], training_step=status['step'], interim=args.interim,
        checkpoint_selection=args.checkpoint, candidate_step=candidate_step,
        original_passed=result['physical_passed'],
        rare_passed=rare['physical_passed'], accepted=False,
        note='Physical inspection only; combined models, seam and speed gates still required.'))
    print(json.dumps(dict(candidate=str(candidate), original_passed=result['physical_passed'],
                          rare_passed=rare['physical_passed'], accepted=False)), flush=True)


if __name__ == '__main__':
    main()
