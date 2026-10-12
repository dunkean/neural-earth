"""Bounded native precision distillation, starting from the original teacher."""
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from distill.common import DATA,REPO,atomic_json,atomic_write,append_json,external_path
from distill.native_fp8 import load_fp8
from distill.native_fp8_training import QATBank,kd_step,operator_loss
from distill.native_basis_training import inventory,capture_blocks
from distill.native_feature_kd import representation_loss
from distill.native_base import teacher_source
from distill.wet_mountains import ROOT,digest
from terrain_diffusion.models.edm_unet import EDMUnet2D

STOP=False


def main():
    global STOP
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DATA/'eval/native-fp8-calibration-v2/initial.pt')
    parser.add_argument('--output',type=Path,default=DATA/'ckpt/native-fp8-pilot')
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-root',type=Path)
    args=parser.parse_args();output=external_path(args.output)
    if (output/'latest.pt').exists():raise ValueError('No implicit QAT restart.')
    records,root,manifest=inventory();source=torch.load(args.source,map_location='cpu',weights_only=True)
    stats_path=DATA/'eval/native-decoded-probe/channel-statistics.json';bands_path=DATA/'eval/native-phase-train-band-statistics.json'
    stats,bands=json.loads(stats_path.read_text()),json.loads(bands_path.read_text())
    if bands['split']!='train' or bands['dataset_manifest_sha256']!=digest(root/'manifest.json'):
        raise ValueError('Frozen TRAIN calibration required.')
    sources=('native_fp8.py','native_fp8_train.py','native_fp8_training.py','native_fp8_evaluate.py','native_fp8_watch.py',
             'native_basis_training.py','native_phase_loss.py','native_feature_kd.py','native_decoded.py','native_fusion.py')
    contract=dict(source_sha256=digest(args.source),activation_calibration=source['contract'],
        teacher_revision=source['teacher_revision'],teacher_weights_sha256=source['teacher_weights_sha256'],
        sources_sha256={n:digest(REPO/'distill'/n) for n in sources},
        manifest_sha256=digest(root/'manifest.json'),statistics_sha256=digest(stats_path),band_statistics_sha256=digest(bands_path),
        mountain_manifests={str(p):digest(p/'manifest.json') for p in (ROOT,DATA/'crops/native-wet-mountain-train-extension-v1')},
        seed=2026101215,steps=args.steps,batch=8,phases=[4,4],lr=1e-5,warmup=100,clip=1,ema_half_life=500,
        quantization='effective globally-normalized MP weights E4M3 per-output-channel; static exact TRAIN p99.99 activations; BF16 convolution compute',
        topology='original192x3,64/32,T2,scalar58conditioning; entry/exit/embeddings/FiLM/logits/softmax/skips/decoder unchanged',
        sampler='8independentTRAIN worlds,25%wet/15%oldrare/60%general,3coarses per phase',
        on_policy='exactly50% batches; all4second-phase examples use detached pupil first nine-window native blends; resident teacher evaluated on SAME inputs',
        loss='native normalized all5operator with firstch0x2,gradient/x0/TRAINband terms; KD all37blocks calibrated once to.5weightedoperator',
        feature_calibration_batches=16,accepted=False,accelerated_fp8_inference=False)
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text())!=contract:
        raise ValueError('Use a fresh output root.')
    if not args.smoke:
        if args.smoke_root is None:raise ValueError('Exact QAT GPU backward smoke required.')
        proof=json.loads((args.smoke_root/'gpu-smoke.json').read_text())
        if proof['status']!='complete' or proof['contract']!=contract:raise ValueError('QAT smoke recipe changed.')
    atomic_json(output/'contract.json',contract)
    torch.set_num_threads(8);torch.manual_seed(contract['seed']);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_fp8(source,'cuda').requires_grad_(True);model.eval()
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    from types import SimpleNamespace
    from terrain_inference import prepare_models
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    variance=torch.tensor(stats['variance'],device='cuda')
    calibration={k:torch.tensor(v,device='cuda') for k,v in dict(first_power=bands['phases']['first']['band_power'],
        second_power=bands['phases']['second']['band_power'],second_variance=bands['phases']['second']['spatial_variance']).items()}
    bank=QATBank()
    if args.smoke:
        ratios=[]
        with torch.no_grad():
            for step in range(16):
                batch,_=bank.batch(records,np.random.default_rng(180210+step),step)
                with capture_blocks(teacher) as reference:
                    target=teacher(batch['inputs'].bfloat16(),noise_labels=batch['labels'].bfloat16(),conditional_inputs=[batch['condition'].bfloat16()]).float()
                with capture_blocks(model.net) as actual:prediction=model(batch['inputs'],batch['labels'],[batch['condition']])
                feature,_=representation_loss(actual,reference);_,terms=operator_loss(prediction,batch,target,variance,calibration)
                ratios.append(float(terms['weighted_operator']/feature.clamp_min(1e-8)))
        feature_weight=.5*float(np.median(ratios));del actual,reference,target,prediction;torch.cuda.empty_cache()
    else:feature_weight=proof['feature_weight'];ratios=proof['calibration_ratios']
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,betas=(.9,.99),fused=True)
    frozen_scales={n:model.net.get_submodule(n)._fp8_scale.detach().clone() for n in model.quantized_layers}
    def training_step(step):
        for group in optimizer.param_groups:group['lr']=1e-5*min(1.,(step+1)/100)
        generator=np.random.default_rng(np.random.SeedSequence([contract['seed'],step]))
        batch,selected=bank.batch(records,generator,step,model,on_policy=step%2==0)
        optimizer.zero_grad(set_to_none=True)
        loss,terms,blocks=kd_step(model,teacher,batch,variance,calibration,feature_weight)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native QAT loss.')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Teacher received gradients.')
        if not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()):raise ValueError('No QAT gradients.')
        optimizer.step()
        if any(not torch.equal(model.net.get_submodule(n)._fp8_scale,frozen_scales[n]) for n in frozen_scales):
            raise ValueError('Static TRAIN activation scales changed.')
        return dict(loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
            grad_norm=float(norm),blocks=len(blocks),on_policy=batch['on_policy'],states=batch['state_rows'],
            worlds=[r['metadata']['seed'] for _,r in selected],coarses=[r['coarse_variant'] for _,r in selected],
            peak_bytes=torch.cuda.max_memory_allocated())
    if args.smoke:
        torch.cuda.reset_peak_memory_stats();rows=[training_step(step) for step in range(3)]
        atomic_json(output/'gpu-smoke.json',dict(status='complete',contract=contract,rows=rows,
            feature_weight=feature_weight,calibration_ratios=ratios,scales_frozen=True,teacher_frozen=True,
            gpu=torch.cuda.get_device_name(),accepted=False))
        return
    ema=copy.deepcopy(model).eval().requires_grad_(False);step=0;started=time.monotonic();guarded=False
    def save(name,full=False):
        state=dict(config=model.config,ema=ema.state_dict(),architecture='native-fp8',step=step,
            quantized_layers=model.quantized_layers,contract=contract,feature_weight=feature_weight,
            teacher_revision=source['teacher_revision'],teacher_weights_sha256=source['teacher_weights_sha256'],
            evaluation_only=False,accepted=False,computation='simulated E4M3, BF16 convolutions')
        if full:state.update(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        atomic_write(output/name,lambda f:torch.save(state,f))
    del source['ema'];save('step000000.pt')
    def stop(signum,frame):
        global STOP
        STOP=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    atomic_json(output/'status.json',dict(status='running',step=0,target_steps=args.steps))
    try:
        while step<args.steps and not STOP:
            evaluation=output/'evaluation-status.json'
            if evaluation.exists() and json.loads(evaluation.read_text()).get('hard_stop'):guarded=True;break
            row=training_step(step);step+=1
            with torch.no_grad():
                for a,p in zip(ema.parameters(),model.parameters()):a.lerp_(p,1-math.exp(math.log(.5)/500))
            if step==1 or step%25==0:
                row.update(step=step,elapsed_seconds=time.monotonic()-started)
                append_json(output/'train.jsonl',row);print(json.dumps(row),flush=True)
                atomic_json(output/'status.json',dict(status='running',step=step,target_steps=args.steps))
            if step%500==0 or step==args.steps:save(f'step{step:06d}.pt');save('latest.pt',full=True)
        save('latest.pt',full=True)
        atomic_json(output/'status.json',dict(status='guarded' if guarded else 'interrupted' if STOP else 'complete',
            step=step,target_steps=args.steps,elapsed_seconds=time.monotonic()-started,accepted=False))
    except Exception as exc:
        save('latest.pt',full=True);atomic_json(output/'status.json',dict(status='failed',step=step,error=str(exc),accepted=False));raise


if __name__=='__main__':main()
