"""Exact A/B-zero continuation proof, followed by finite capped B updates."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from distill.common import REPO,atomic_json,atomic_write,external_path
from distill.native_auxiliary_train import Engine,configure
from distill.native_weight_quant import file_sha


def cpu_state(engine):
    def copy(value):
        if torch.is_tensor(value):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:copy(v) for k,v in value.items()}
        if isinstance(value,list):return [copy(v) for v in value]
        if isinstance(value,tuple):return tuple(copy(v) for v in value)
        return value
    return dict(model=copy(engine.model.state_dict()),ema=copy(engine.ema.state_dict()),
        optimizer=copy(engine.optimizer.state_dict()),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--powers',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=external_path(args.output)
    if output.exists():raise ValueError('Preserve previous smoke.')
    configure();source=torch.load(args.source,map_location='cpu',weights_only=True,mmap=True)
    powers=json.loads(args.powers.read_text());calibration=json.loads(args.calibration.read_text())
    if calibration['source_sha256']!=file_sha(args.source):raise ValueError('Calibration source changed.')
    rows={};reference=None;proof={}
    for mode in ('a','zero','b'):
        engine=Engine(source,powers,calibration,'a' if mode=='a' else 'b',zero=mode=='zero')
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        rows[mode]=[engine.update() for _ in range(3)]
        proof[mode]=dict(seconds=time.monotonic()-started,peak_reserved=torch.cuda.max_memory_reserved(),
            teacher_frozen=all(p.grad is None for p in engine.teacher.parameters()))
        if mode=='a':reference=cpu_state(engine)
        if mode=='zero':
            if not equal(reference,cpu_state(engine)):raise ValueError('A vs B-zero raw/EMA/Adam/RNG differ.')
            if not all(x['loss']==y['loss'] and x['grad_norm']==y['grad_norm'] for x,y in zip(rows['a'],rows['zero'])):
                raise ValueError('A vs B-zero losses/gradients differ.')
        del engine;gc.collect();torch.cuda.empty_cache()
    files=('native_auxiliary_training.py','native_auxiliary_train.py','native_field_auxiliary.py')
    atomic_json(output,dict(status='complete',base_sha256=file_sha(args.source),calibration_sha256=file_sha(args.calibration),
        code_sha256={n:file_sha(REPO/'distill'/n) for n in files},rows=rows,proof=proof,
        zero_auxiliary_full_path_exact=True,accepted=False))


if __name__=='__main__':main()
