"""Continue a full native weight-QAT checkpoint, retaining Adam, raw, EMA and RNG."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, append_json, external_path
from distill.native_base import teacher_source
from distill.native_basis_training import inventory
from distill.native_fp8_training import QATBank
from distill.native_weight_quant import load_weight_quant, file_sha
from distill.native_weight_quant_train import step_loss
from terrain_diffusion.models.edm_unet import EDMUnet2D


def restore_training(source, device='cuda'):
    required={'model','ema','optimizer','rng','cuda_rng','step','feature_weight','contract'}
    if not required.issubset(source):raise ValueError('A full training checkpoint is required, not an EMA snapshot.')
    model=load_weight_quant(source|{'ema':source['model']},device).requires_grad_(True)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    ema.load_state_dict(source['ema'])
    optimizer=torch.optim.Adam(model.parameters(),betas=(.9,.99),fused=device=='cuda')
    optimizer.load_state_dict(source['optimizer'])
    for clone in (model,ema):
        if any(clone.net.get_submodule(n)._weight_quant_owner is not clone for n in clone.quantized_layers):
            raise ValueError('Wrong quantization owner after restoring model/EMA.')
    return model,ema,optimizer


def restore_rng(source):
    # Do this after model construction: construction consumes random numbers.
    torch.set_rng_state(source['rng'])
    torch.cuda.set_rng_state_all(source['cuda_rng'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=100000,help='Total global updates, including the parent.')
    parser.add_argument('--snapshot-every',type=int,default=5000)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-root',type=Path)
    args=parser.parse_args();output=external_path(args.output)
    if (output/'latest.pt').exists():raise ValueError('Existing continuation requires an explicit new output.')
    source=torch.load(args.source,map_location='cpu',weights_only=True,mmap=True)
    start=int(source['step'])
    if args.steps<=start or args.snapshot_every<=0:raise ValueError('Target must exceed the parent step.')
    records,root,manifest=inventory();stats_path=DATA/'eval/native-decoded-probe/channel-statistics.json'
    parent=source['contract']
    if parent['manifest_sha256']!=file_sha(root/'manifest.json') or parent['statistics_sha256']!=file_sha(stats_path):
        raise ValueError('Training data or variance changed.')
    # Do not silently change the parent quantizer/loss/sampler during continuation.
    for name in ('native_weight_quant.py','native_weight_quant_train.py','native_fp8_training.py',
                 'native_basis_training.py','native_decoded.py','native_fusion.py',
                 'native_phase_loss.py','native_feature_kd.py'):
        if parent['source_code_sha256'][name]!=file_sha(REPO/'distill'/name):raise ValueError('Parent training code changed: '+name)
    learning_rates=[g['lr'] for g in source['optimizer']['param_groups']]
    contract=dict(parent=parent,source=str(args.source),source_sha256=file_sha(args.source),
        source_code_sha256=file_sha(Path(__file__)),start_step=start,target_steps=args.steps,
        scheduler='constant inherited terminal LR; no reset or LR jump',learning_rates=learning_rates,
        snapshot_every=args.snapshot_every,feature_weight=source['feature_weight'],
        preserved=['FP32 raw weights','FP32 EMA','Adam moments and counters','CPU and CUDA RNG','global sampler step'],
        accepted=False)
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text())!=contract:
        raise ValueError('Changed continuation contract.')
    if not args.smoke:
        if args.smoke_root is None:raise ValueError('Resume GPU smoke required.')
        proof=json.loads((args.smoke_root/'gpu-smoke.json').read_text())
        if proof['contract']!=contract or proof['status']!='complete':raise ValueError('Resume smoke contract mismatch.')
        if shutil.disk_usage(DATA).free<12*2**30:raise ValueError('Insufficient space for atomic full checkpoints.')
    atomic_json(output/'contract.json',contract)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,ema,optimizer=restore_training(source)
    # Exact checkpoint restoration before the first update, including all Adam slots.
    for key,value in model.state_dict().items():
        if not torch.equal(value,source['model'][key].to(value.device)):raise ValueError('Raw restoration differs: '+key)
    for key,value in ema.state_dict().items():
        if not torch.equal(value,source['ema'][key].to(value.device)):raise ValueError('EMA restoration differs: '+key)
    actual_state=optimizer.state_dict()
    for key,slots in source['optimizer']['state'].items():
        for name,value in slots.items():
            actual=actual_state['state'][key][name]
            if torch.is_tensor(value):
                if not torch.equal(actual,value.to(actual.device)):raise ValueError('Adam restoration differs.')
            elif actual!=value:raise ValueError('Adam restoration differs.')
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    from types import SimpleNamespace
    from terrain_inference import prepare_models
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    variance=torch.tensor(json.loads(stats_path.read_text())['variance'],device='cuda')
    bank=QATBank();restore_rng(source)
    if not torch.equal(torch.get_rng_state(),source['rng']):raise ValueError('CPU RNG differs.')
    if not all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),source['cuda_rng'])):raise ValueError('CUDA RNG differs.')
    source_meta={k:source[k] for k in ('teacher_revision','teacher_weights_sha256')}
    # Release mmap references after restoring, retaining provenance and immutable parent file.
    del source,actual_state
    def update(step):
        generator=np.random.default_rng(np.random.SeedSequence([parent['seed'],step]))
        with torch.no_grad(),model.cached_weights():
            batch,selected=bank.batch(records,generator,step,model,on_policy=step%2==0)
        optimizer.zero_grad(set_to_none=True)
        loss,terms,blocks=step_loss(model,teacher,batch,variance,contract['feature_weight'])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss.')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),parent['clip'],error_if_nonfinite=True)
        if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Teacher received gradients.')
        if not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()):raise ValueError('No gradients.')
        optimizer.step()
        with torch.no_grad():
            for a,p in zip(ema.parameters(),model.parameters()):a.lerp_(p,1-math.exp(math.log(.5)/parent['ema_half_life']))
        return dict(loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
            grad_norm=float(norm),lr=optimizer.param_groups[0]['lr'],blocks=len(blocks),
            on_policy=batch['on_policy'],worlds=[r['metadata']['seed'] for _,r in selected],
            coarses=[r['coarse_variant'] for _,r in selected],peak_bytes=torch.cuda.max_memory_allocated())
    if args.smoke:
        rows=[update(step) for step in range(start,start+2)]
        atomic_json(output/'gpu-smoke.json',dict(status='complete',contract=contract,rows=rows,
            raw_ema_adam_exact=True,rng_exact=True,teacher_frozen=True,
            gpu=torch.cuda.get_device_name(),accepted=False))
        return
    step=start;started=time.monotonic();stop_requested=False
    def save(name,full=False,raw=False):
        state=dict(config=model.config,architecture='native-weight-quant',scheme=model.scheme,block=model.block,
            quantized_layers=model.quantized_layers,step=step,contract=parent,continuation_contract=contract,
            feature_weight=contract['feature_weight'],**source_meta,evaluation_only=False,accepted=False)
        if full:
            state.update(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                         rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        else:
            state['ema']={n:v.detach().bfloat16().cpu() for n,v in (model if raw else ema).state_dict().items()}
            state['selected_weights']='raw' if raw else 'ema'
        atomic_write(output/name,lambda f:torch.save(state,f))
    def stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    save(f'step{step:06d}.pt');save('latest.pt',full=True)
    atomic_json(output/'startup-proof.json',dict(raw_ema_adam_exact=True,rng_exact=True,start_step=start,
        parent_sha256=contract['source_sha256'],gpu=torch.cuda.get_device_name(),learning_rates=learning_rates))
    atomic_json(output/'status.json',dict(status='running',step=step,target_steps=args.steps))
    try:
        while step<args.steps and not stop_requested:
            row=update(step);step+=1
            if step==start+1 or step%100==0:
                row.update(step=step,elapsed_seconds=time.monotonic()-started)
                append_json(output/'train.jsonl',row);print(json.dumps(row),flush=True)
                atomic_json(output/'status.json',dict(status='running',step=step,target_steps=args.steps))
            # Archive candidates regularly, plus an early visual comparison after 1000 new updates.
            if step==start+1000 or step%args.snapshot_every==0 or step==args.steps:
                save(f'step{step:06d}.pt');save('latest.pt',full=True)
            if step in (20000,50000,args.steps):save(f'raw/step{step:06d}.pt',raw=True)
        save('latest.pt',full=True)
        if not (output/f'step{step:06d}.pt').exists():save(f'step{step:06d}.pt')
        atomic_json(output/'status.json',dict(status='interrupted' if stop_requested else 'complete',
            step=step,target_steps=args.steps,elapsed_seconds=time.monotonic()-started,accepted=False))
    except Exception as exc:
        save('latest.pt',full=True)
        atomic_json(output/'status.json',dict(status='failed',step=step,error=str(exc),accepted=False));raise


if __name__=='__main__':main()
