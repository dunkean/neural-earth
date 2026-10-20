"""Recover the changed final-kernel precision from frozen6000step dense masters."""
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

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write, append_json, external_path
from distill.native_base import teacher_source
from distill.native_weight_quant import file_sha
from distill.tile40_native3 import Native3Base, initial_scale, export_packed, load_packed
from distill.tile40_native3_bank import ClimateBank, paired_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--precision-packed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=800)
    parser.add_argument('--interval',type=int,default=400)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--scale-lr',type=float,default=1e-4)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-root',type=Path)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if min(args.steps,args.interval)<1 or min(args.lr,args.scale_lr)<=0:
        raise ValueError('Positive training budgets and learning rates required.')
    root = external_path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    source = teacher_source()
    precision=json.loads((args.precision_packed/'manifest.json').read_text())
    parent_path=args.source_root/'latest.pt'
    parent_sha=file_sha(parent_path)
    code = ('tile40_native3.py','tile40_native3_bank.py','tile40_native3_output4_recover.py')
    contract = dict(architecture='original-native-mp-gaussian3-4',teacher_revision=REVISION,
        teacher_weights_sha256=file_sha(source/'diffusion_pytorch_model.safetensors'),
        dataset=str(args.dataset.resolve()),dataset_manifest_sha256=file_sha(args.dataset/'manifest.json'),
        code_sha256={name:file_sha(REPO/'distill'/name) for name in code},steps=args.steps,
        interval=args.interval,master_lr=args.lr,scale_lr=args.scale_lr,block=256,
        parent_full_checkpoint_sha256=parent_sha,parent_step=6000,parent_weights='raw',
        precision_manifest_sha256=file_sha(args.precision_packed/'manifest.json'),
        promoted_to4bit=precision['contract']['promoted_to4bit'],
        quantization='3bit convolutions;4bit embedding projections;BF16 climate/noise input projections and sensitive endpoints.',
        sampler='Two same-geometry climate pairs per batch;one per native phase;70% mountains.',
        loss='Normalized paired teacher velocity+.05gradient+.1band-position+.1climate-difference;fixed TRAIN denominators.',
        optimizer='AdamW betas(.9,.99),weight_decay0;100step warmup,cosine to.1;gradclip1.',
        ema_half_life=100,size_limit_bytes=100000000,size_scope='Base only;original decoder retained.',
        quality_accepted=False,seed=2026102051)
    path = root/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract:
        raise ValueError('Training contract changed;preserve this run.')
    if not args.smoke:
        if args.smoke_root is None:
            raise ValueError('A successful full-size packed GPU smoke is required.')
        proof = json.loads((args.smoke_root/'gpu-smoke.json').read_text())
        if proof['status']!='complete' or proof['contract']!=contract:
            raise ValueError('GPU smoke does not match this training contract.')
    if shutil.disk_usage(root).free<20*2**30:
        raise ValueError('Preserve20GiB for full optimizer checkpoints and atomic saves.')
    atomic_json(path,contract)
    torch.set_num_threads(6)
    torch.manual_seed(contract['seed']);torch.cuda.manual_seed_all(contract['seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bank = ClimateBank(args.dataset,partial=args.smoke)
    stats = bank.statistics()
    atomic_json(root/'train-statistics.json',stats)
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    net = EDMUnet2D.from_pretrained(str(source),torch_dtype=torch.bfloat16).cuda().float().eval()
    model = Native3Base(net).cuda().eval()
    parent=torch.load(parent_path,map_location='cpu',weights_only=True,mmap=True)
    if parent['step']!=6000:raise ValueError('Completed6000step masters required.')
    model.load_state_dict(parent['model']);del parent
    with torch.no_grad():
        for name in precision['contract']['promoted_to4bit']:
            spec=model.quant_specs[name];spec['bits']=4
            getattr(model,spec['scale_name']).copy_(initial_scale(model.net.get_parameter(name),4,model.block).float())
    scale_params = [p for name,p in model.named_parameters() if name.startswith('scale_')]
    optimizer = torch.optim.AdamW([dict(params=model.net.parameters(),lr=args.lr),
        dict(params=scale_params,lr=args.scale_lr)],betas=(.9,.99),weight_decay=0.,fused=True)
    stopped = False
    def stop(signum,frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    step = 0
    ema = None if args.smoke else copy.deepcopy(model).requires_grad_(False)
    latest = root/'latest.pt'
    if latest.exists():
        if not args.resume or args.smoke:
            raise ValueError('Existing optimizer state requires an explicit --resume.')
        saved = torch.load(latest,map_location='cpu',weights_only=True,mmap=True)
        if saved['contract']!=contract:
            raise ValueError('Resume contract changed.')
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer']);step=saved['step']
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    def save_full():
        state = dict(contract=contract,step=step,model=model.state_dict(),ema=ema.state_dict(),
            optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        atomic_write(latest,lambda handle:torch.save(state,handle))
    def snapshot(selection,owner):
        return export_packed(owner,root/'packed'/f'step-{step:06d}-{selection}',step=step,
            selection=selection,contract=dict(teacher_weights_sha256=contract['teacher_weights_sha256'],
                dataset_manifest_sha256=contract['dataset_manifest_sha256'],training_contract_sha256=file_sha(path)),header_reserve=65536)
    def validation():
        if not bank.groups['val']:
            return None
        losses=[]
        with torch.no_grad(),model.cached_weights():
            for index in range(12):
                batch=bank.batch(np.random.default_rng(530019+index),'cuda',split='val')
                prediction=model(batch['inputs'],batch['labels'],[batch['condition']])
                value,terms=paired_loss(prediction,batch,stats)
                losses.append({k:float(v) for k,v in terms.items()}|dict(loss=float(value)))
        return {k:float(np.mean([r[k] for r in losses])) for k in losses[0]}
    began=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    if args.smoke:
        initial=snapshot('raw',model)
        if initial['sha256']!=precision['sha256']:raise ValueError('Recovery start differs from measured precision variant.')
        packed,_=load_packed(root/'packed/step-000000-raw','cuda')
        batch=bank.batch(np.random.default_rng(6),'cuda')
        with torch.no_grad(),model.cached_weights():
            expected=model(batch['inputs'],batch['labels'],[batch['condition']])
            actual=packed(batch['inputs'].bfloat16(),noise_labels=batch['labels'].bfloat16(),
                conditional_inputs=[batch['condition'].bfloat16()]).float()
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        del packed
        before=model.net.out_conv.weight.detach().clone()
    else:
        if step==0 and not (root/'packed/step-000000-raw').exists():snapshot('raw',model)
        atomic_json(root/'status.json',dict(status='running',step=step,target_steps=args.steps))
    rows=[]
    target=3 if args.smoke else args.steps
    try:
        while step<target and not stopped:
            factor=min(1.,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-100)/max(1,args.steps-100))))
            for group,lr in zip(optimizer.param_groups,(args.lr,args.scale_lr)):group['lr']=lr*factor
            batch=bank.batch(np.random.default_rng(np.random.SeedSequence([contract['seed'],step])),'cuda')
            optimizer.zero_grad(set_to_none=True)
            prediction=model(batch['inputs'],batch['labels'],[batch['condition']])
            loss,terms=paired_loss(prediction,batch,stats)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native climate QAT loss.')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            if ema is not None:
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-math.exp(math.log(.5)/100))
            row=dict(step=step,loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
                grad_norm=float(norm),lr=optimizer.param_groups[0]['lr'],
                peak_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-began)
            if args.smoke:rows.append(row)
            if step==1 or step%10==0 or args.smoke:
                print(json.dumps(row),flush=True);append_json(root/'train.jsonl',row)
                atomic_json(root/'status.json',dict(status='running',step=step,target_steps=target,elapsed_seconds=row['elapsed_seconds']))
            if not args.smoke and (step%args.interval==0 or step==args.steps):
                proxy=validation();atomic_json(root/f'operator-val-{step:06d}.json',dict(step=step,metrics=proxy,scope='Proxy only;full native decoded VAL selects quality.'))
                snapshot('raw',model);snapshot('ema',ema);save_full()
        if args.smoke:
            if stopped:raise RuntimeError('Interrupted GPU smoke.')
            if torch.equal(before,model.net.out_conv.weight):raise ValueError('No real parameter update.')
            atomic_json(root/'gpu-smoke.json',dict(status='complete',contract=contract,rows=rows,
                packed_forward_exact=True,packed_artifact_bytes=initial['total_artifact_bytes'],
                parameter_update=True,peak_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(),
                scope='Mechanics only on complete available TRAIN climate groups;not quality evidence.'))
        else:
            save_full()
        atomic_json(root/'status.json',dict(status='interrupted' if stopped else 'complete',step=step,
            target_steps=target,elapsed_seconds=time.monotonic()-began,quality_accepted=False))
    except Exception as exc:
        if not args.smoke:save_full()
        atomic_json(root/'status.json',dict(status='failed',step=step,error=str(exc)));raise


if __name__=='__main__':main()
