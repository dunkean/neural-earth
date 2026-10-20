"""Bounded conditioning-residual pilot;immutable packed base,climate lossx5."""
import argparse
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from distill.common import REPO,atomic_json,atomic_write,append_json,external_path
from distill.native_weight_quant import file_sha
from distill.tile40_native3_bank import ClimateBank,paired_loss
from distill.tile40_native3_lora import from_base,export_residual,load_residual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--statistics',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--selection-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-root',type=Path)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--interval',type=int,default=1000)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    root=external_path(args.output);root.mkdir(parents=True,exist_ok=True)
    if min(args.steps,args.interval)<1:raise ValueError('Positive bounded training budget required.')
    proof=None
    if args.smoke:
        source=args.source
        if source is None:
            from distill.tile40_native3_watch import physical_summary
            candidates=[]
            for report_path in sorted(args.selection_root.glob('val-*/report.json')):
                report=json.loads(report_path.read_text())
                if report['status']!='complete':continue
                candidates.append(dict(packed=report['packed_base'],report=str(report_path),
                    summary=physical_summary(report,report_path.parent)))
            if not candidates:raise ValueError('No complete physical candidate to initialize residuals.')
            selected=min(candidates,key=lambda row:row['summary']['quality_gap'])
            atomic_json(root/'source-selection.json',dict(selected=selected,candidates=candidates,
                policy='Minimum predeclared full-VAL quality gap among all complete physical evaluations.'))
            source=Path(selected['packed'])
    else:
        if args.smoke_root is None:raise ValueError('A matching packed GPU smoke is required.')
        proof=json.loads((args.smoke_root/'gpu-smoke.json').read_text())
        source=Path(proof['contract']['source'])
    contract=dict(source=str(source.resolve()),source_manifest_sha256=file_sha(source/'manifest.json'),
        source_weights_sha256=file_sha(source/'weights.bin'),dataset_manifest_sha256=file_sha(args.dataset/'manifest.json'),
        statistics_sha256=file_sha(args.statistics),steps=args.steps,interval=args.interval,rank=8,
        lr=.003,ema_half_life=100,seed=2026102059,base_frozen=True,limit_bytes=100000000,
        loss='Operator+.05gradient+.1band-position+.5climate-difference;TRAIN-only denominators.',
        code_sha256={n:file_sha(REPO/'distill'/n) for n in ('tile40_native3_lora.py','tile40_native3_lora_train.py','tile40_native3_bank.py','tile40_native3_watch.py')},
        selection='Freeze best full-VAL quality-gap source at smoke start;then pin exact source bytes.',quality_accepted=False)
    if proof is not None and (proof['status']!='complete' or proof['contract']!=contract):
        raise ValueError('Smoke contract changed.')
    path=root/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('Residual contract changed.')
    atomic_json(path,contract)
    torch.set_num_threads(6);torch.manual_seed(contract['seed']);torch.cuda.manual_seed_all(contract['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bank=ClimateBank(args.dataset)
    stats=json.loads(args.statistics.read_text())
    model,_=from_base(source,'cuda',rank=8)
    model.eval()
    optimizer=torch.optim.AdamW(model.adapters.parameters(),lr=.003,betas=(.9,.99),weight_decay=0.,fused=True)
    averages={k:v.detach().clone() for k,v in model.adapters.items()}
    latest=root/'latest.pt';step=0
    if latest.exists():
        if not args.resume or args.smoke:raise ValueError('Explicit --resume required.')
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        if saved['contract']!=contract:raise ValueError('Resume source changed.')
        model.adapters.load_state_dict(saved['adapters']);optimizer.load_state_dict(saved['optimizer'])
        averages={k:v.cuda() for k,v in saved['ema'].items()};step=saved['step']
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def save():
        atomic_write(latest,lambda handle:torch.save(dict(step=step,contract=contract,
            adapters=model.adapters.state_dict(),ema=averages,optimizer=optimizer.state_dict()),handle))
    def export(selection,adapters=None):
        return export_residual(model,source,root/'packed'/f'step-{step:06d}-{selection}',
            step=step,selection=selection,adapters=adapters,contract=contract)
    def proxy():
        rows=[]
        with torch.no_grad():
            for i in range(12):
                batch=bank.batch(np.random.default_rng(530019+i),'cuda',split='val')
                prediction=model(batch['inputs'],batch['labels'],[batch['condition']])
                loss,terms=paired_loss(prediction,batch,stats)
                rows.append({k:float(v) for k,v in terms.items()}|dict(loss=float(loss+.4*terms['climate_pair'])))
        return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
    began=time.monotonic();torch.cuda.reset_peak_memory_stats();rows=[]
    target=3 if args.smoke else args.steps
    try:
        while step<target and not stopped:
            factor=min(1.,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-100)/max(1,args.steps-100))))
            optimizer.param_groups[0]['lr']=.003*factor
            batch=bank.batch(np.random.default_rng(np.random.SeedSequence([contract['seed'],step])),'cuda')
            optimizer.zero_grad(set_to_none=True)
            prediction=model(batch['inputs'],batch['labels'],[batch['condition']])
            loss,terms=paired_loss(prediction,batch,stats);loss=loss+.4*terms['climate_pair']
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite climate residual loss.')
            loss.backward()
            if any(p.grad is not None for p in model.net.parameters()):raise ValueError('Frozen base received gradients.')
            norm=torch.nn.utils.clip_grad_norm_(model.adapters.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            with torch.no_grad():
                for k,p in model.adapters.items():averages[k].lerp_(p,1-math.exp(math.log(.5)/100))
            row=dict(step=step,loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
                grad_norm=float(norm),elapsed_seconds=time.monotonic()-began,peak_bytes=torch.cuda.max_memory_allocated())
            if args.smoke:rows.append(row)
            if args.smoke or step==1 or step%10==0:
                print(json.dumps(row),flush=True);append_json(root/'train.jsonl',row)
                atomic_json(root/'status.json',dict(status='running',step=step,target_steps=target,elapsed_seconds=row['elapsed_seconds']))
            if not args.smoke and (step%args.interval==0 or step==args.steps):
                atomic_json(root/f'operator-val-{step:06d}.json',dict(step=step,metrics=proxy()))
                export('raw');export('ema',averages);save()
        if args.smoke:
            if stopped:raise RuntimeError('Interrupted smoke.')
            if not any(float(p.abs().sum())>0 for k,p in model.adapters.items() if k.endswith('_b')):
                raise ValueError('No residual update.')
            packed=export('raw');actual_net,_=load_residual(root/'packed'/f'step-{step:06d}-raw','cuda')
            with torch.no_grad():
                expected=model(batch['inputs'],batch['labels'],[batch['condition']])
                actual=actual_net(batch['inputs'].bfloat16(),noise_labels=batch['labels'].bfloat16(),
                    conditional_inputs=[batch['condition'].bfloat16()]).float()
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            atomic_json(root/'gpu-smoke.json',dict(status='complete',contract=contract,rows=rows,
                packed_forward_exact=True,packed_bytes=packed['total_artifact_bytes'],base_frozen=True,
                residuals_updated=True,quality_accepted=False))
        else:save()
        atomic_json(root/'status.json',dict(status='interrupted' if stopped else 'complete',step=step,
            target_steps=target,elapsed_seconds=time.monotonic()-began,quality_accepted=False))
    except Exception as exc:
        if not args.smoke:save()
        atomic_json(root/'status.json',dict(status='failed',step=step,error=str(exc)));raise


if __name__=='__main__':main()
