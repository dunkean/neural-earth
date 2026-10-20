"""Package only the five selected inference models, without optimizer/EMA copies.

Run with --write to refresh the curated distribution from external research
checkpoints. This does not commit or publish anything.
"""
import argparse
import io
import json
from pathlib import Path

import torch

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write
from distill.native_weight_quant import file_sha

SOURCES = {
    'coarse8': ('coarse8.pt', 'ckpt/coarse-solver8/latest.pt', 25000, 'ema'),
    'coarse2': ('coarse2.pt', 'ckpt/coarse-full/coarse2/best.pt', 200000, 'ema'),
    'base_residual8k': ('base-residual8k.pt', 'tile40-20261014/residual-altitude48-pilot-v1/step-008000.pt', 8000, 'model'),
    'decoder64': ('decoder.pt', 'ckpt/decoder/best.pt', 200000, 'ema'),
    'decoder16': ('decoder16-10k.pt', 'tile40-20261014/decoder16-original-pilot-v1/step-010000.pt', 10000, 'ema'),
}


def prepare(root=DATA):
    payloads, exports = {}, {}
    for key, (filename, source, step, selected) in SOURCES.items():
        path = Path(root)/source
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['step'] != step:
            raise ValueError(f'{key}: expected the selected step{step}, got{saved["step"]}.')
        state = {name:value.detach().cpu().clone() for name,value in saved[selected].items()}
        package = dict(config=saved['config'], model=state, step=step,
            inference_only=True, selected_weights='raw' if selected=='model' else selected,
            teacher_revision=REVISION)
        for name in ('contract', 'flow_architecture'):
            if name in saved:
                package[name] = saved[name]
        if key=='base_residual8k':
            if not saved['contract'].get('residual_altitude'):
                raise ValueError('The finite-tile residual representation is required.')
            # The artifact pins its deployable coarse dependency, not a user's
            # unrelated local training checkpoint path.
            package['contract'] = {k:v for k,v in saved['contract'].items()
                                   if k not in ('arguments','initial','warm_start')}
        buffer = io.BytesIO()
        torch.save(package, buffer)
        payload = buffer.getvalue()
        if key=='coarse2':
            # This already-audited inference export is pinned by the immutable
            # training/VAL banks. Keep its bytes as well as its exact weights.
            canonical=Path(root)/'ckpt/coarse-full/coarse2/export/coarse.pt'
            retained=torch.load(canonical,map_location='cpu',weights_only=True)
            if (not retained.get('inference_only') or retained['step']!=step or
                    retained['model'].keys()!=state.keys() or
                    any(not torch.equal(state[name],retained['model'][name]) for name in state)):
                raise ValueError('Canonical coarse2 export no longer matches the selected EMA.')
            payload=canonical.read_bytes()
        if len(payload)>=100000000:
            raise ValueError(f'{key} exceeds100MB.')
        import hashlib
        payloads[filename] = payload
        exports[key] = dict(file=filename, bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
            stage=saved['config']['stage'],config=saved['config'],training_step=step,
            selected_weights=package['selected_weights'],source_checkpoint_sha256=file_sha(path),
            scope='tile40' if key=='base_residual8k' else 'world-and-tile40')
    # Never silently rebind the residual model to a different coarse artifact.
    base = torch.load(io.BytesIO(payloads['base-residual8k.pt']),map_location='cpu',weights_only=True)
    if base['contract']['coarse_sha256']!=exports['coarse2']['sha256']:
        raise ValueError('Residual8k requires the exact coarse2 artifact used during training.')
    return payloads, dict(schema=2,teacher_model='xandergos/terrain-diffusion-30m',teacher_revision=REVISION,
        inference_only=True,exports=exports,
        defaults=dict(world=dict(coarse='solver8',base='teacher',decoder='distilled200k'),
                      tile40=dict(coarse='coarse2',base='base_residual8k',decoder='decoder16')),
        originals=dict(download_command='python tools/download_models.py',
            stages=['coarse_model','base_model','decoder_model']),
        note='Only the user-selected five inference models are distributed. Original models use the pinned upstream snapshot. Residual8k is for a finite40km tile; quality remains experimental.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=REPO/'distill/weights')
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    payloads,manifest=prepare(args.root)
    if args.write:
        args.output.mkdir(parents=True,exist_ok=True)
        allowed=set(payloads)
        unexpected={p.name for p in args.output.glob('*.pt')}-allowed
        if unexpected:
            raise ValueError('Archive retired weights outside Git before packaging: '+', '.join(sorted(unexpected)))
        for filename,payload in payloads.items():
            atomic_write(args.output/filename,lambda handle,payload=payload:handle.write(payload))
        atomic_json(args.output/'manifest.json',manifest)
    print(json.dumps(dict(written=args.write,exports=manifest['exports']),indent=2))


if __name__=='__main__':
    main()
