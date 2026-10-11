"""Enter the new base trainer after successful matched capture and GPU smoke."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from distill.common import DATA,REPO,atomic_json
from distill.run_training import wait_job


def verify_smoke(proof):
    if proof.get('status') != 'complete' or proof.get('original_decoder_frozen') is not True:
        raise ValueError('The original-decoder GPU smoke did not complete.')
    rows = proof.get('rows',[])
    if len(rows)!=3 or {r['coarse_variant'] for r in rows} != {'original','coarse4','coarse2'}:
        raise ValueError('Smoke must cover each frozen coarse branch.')
    for row in rows:
        gradients = row['window_gradients']
        if [len(g) for g in gradients] != [25,9] or any(
            not math.isfinite(v) or v<=0 for g in gradients for v in g):
            raise ValueError('Smoke gradients must reach all34 windows.')
        if not math.isfinite(row['loss']) or not math.isfinite(row['grad_norm']):
            raise ValueError('Smoke optimizer step is not finite.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--output',type=Path,default=DATA/'ckpt/native128x2-multilod-mixed-pilot')
    parser.add_argument('--source',type=Path,default=DATA/'ckpt/native128x2-pilot/best.pt')
    parser.add_argument('--smoke-job',default='base-native-multilod-smoke')
    parser.add_argument('--smoke-root',type=Path,default=DATA/'eval/native-multilod-mixed-smoke')
    parser.add_argument('--progress',type=Path,default=DATA/'native-multilod-next.json')
    parser.add_argument('--preserve-valleys',action='store_true')
    args=parser.parse_args()
    if args.steps<1:
        parser.error('Positive optimizer-step target required.')
    progress=args.progress.expanduser().resolve()
    atomic_json(progress,dict(status='running',phase='waiting-for-capture-and-smoke'))
    wait_job(args.smoke_job)
    wait_job('base-native-mixed-capture')
    smoke_root=args.smoke_root.expanduser().resolve()
    proof=json.loads((smoke_root/'gpu-smoke.json').read_text())
    verify_smoke(proof)
    replays=json.loads((smoke_root/'teacher-control.json').read_text())
    if set(replays)!={'original','coarse4','coarse2'} or any(not r['passed'] for r in replays.values()):
        raise ValueError('Original teacher replay failed a matched branch.')
    contract=proof['contract']
    if bool(contract.get('preserve_valleys',False)) != args.preserve_valleys:
        raise ValueError('Smoke and trainer must use the same valley-preservation policy.')
    sources=contract['producer_sha256'] | {
        'dataset':contract['dataset_manifest_sha256'],
        'statistics':contract['statistics_sha256'],
        'source':contract['source_sha256']}
    paths={name:REPO/name for name in contract['producer_sha256']}
    paths.update(dataset=DATA/'crops/native-mixed-scenes-v2/manifest.json',
        statistics=DATA/'eval/native-decoded-probe/channel-statistics.json',
        source=args.source.expanduser().resolve())
    for name,sha in sources.items():
        if hashlib.sha256(paths[name].read_bytes()).hexdigest()!=sha:
            raise ValueError('Training source or producer differs from the successful smoke.')
    atomic_json(progress,dict(status='running',phase='training',steps=args.steps,output=str(args.output),
        smoke_manifest=contract['dataset_manifest_sha256'],
        note='Process becomes the trainer; terminal/quality status is recorded by trainer and durable job.'))
    # Keep the durable PID and signal handling: no orphan trainer subprocess.
    os.execv(sys.executable,[sys.executable,'-u','-m','distill.native_multilod_train',
        '--steps',str(args.steps),'--output',str(args.output),
        '--source',str(args.source.expanduser().resolve())]+(['--preserve-valleys'] if args.preserve_valleys else []))


if __name__=='__main__':
    main()
