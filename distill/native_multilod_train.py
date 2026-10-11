"""Train native base on matched coarse branches, two LODs and rare cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from distill.common import DATA, REPO, REVISION, HOLDOUT_SEEDS, append_json, atomic_json, atomic_write, external_path
from distill.native_base import NativeBase, teacher_source
from distill.native_decoded import DecodedRegion, OriginalDecoder, decoded_region_forward
from distill.native_fusion_train import check_teacher
from distill.native_mixed_bank import KINDS, VARIANTS
from distill.native_multilod import PHYSICAL_WEIGHTS, REGULARIZATION_WEIGHTS, multilod_loss, physical_terms, physical_score

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_record(records, generator, rare_fraction=.25):
    variant = VARIANTS[int(generator.integers(len(VARIANTS)))]
    kind = KINDS[int(generator.integers(len(KINDS)))] if generator.random() < rare_fraction else None
    choices = [row for row in records if row['coarse_variant']==variant and row['case_kind']==kind]
    if not choices:
        raise ValueError('A training coarse/case stratum is missing.')
    return choices[int(generator.integers(len(choices)))]


@torch.no_grad()
def validate(model, records, root, weight, decoder, variance, preserve_valleys=False):
    rows = []
    for record in records:
        region = DecodedRegion(root/record['file'], 'cuda')
        result = decoded_region_forward(model,region,weight)
        terms = physical_terms(result,region,decoder,variance)
        values = {key:float(value) for key,value in terms.items()}
        rows.append(dict(file=record['file'], seed=region.metadata['seed'], profile=region.metadata['profile'],
            coarse_variant=record['coarse_variant'], case_kind=record['case_kind'],
            physical_score=float(physical_score(terms)), **values))
        if preserve_valleys:
            from distill.native_amplitude import morphology_summary
            rows[-1]['morphology']=morphology_summary(result,region,decoder)
    groups = {}
    for variant in VARIANTS:
        profiles = sorted({row['profile'] for row in rows if row['coarse_variant']==variant})
        for category,values in [('profile',profiles),('rare',KINDS)]:
            for value in values:
                selected = [r for r in rows if r['coarse_variant']==variant and
                            (r['profile']==value if category=='profile' else r['case_kind']==value)]
                if not selected:
                    raise ValueError('Independent validation is missing a coarse/case stratum.')
                groups[f'{variant}:{category}:{value}'] = dict(count=len(selected),
                    mean_physical_score=float(np.mean([r['physical_score'] for r in selected])),
                    median_height_mae_m=float(np.median([r['height_mae_m'] for r in selected])),
                    median_low_height_mae_m=float(np.median([r['low_height_mae_m'] for r in selected])))
                if preserve_valleys:
                    summaries={}
                    for scale in ('fine','native'):
                        supported=[row['morphology'][scale] for row in selected if row['morphology'][scale] is not None]
                        summaries[scale]=({key:float(np.mean([item[key] for item in supported])) for key in supported[0]}
                                          if supported else None)
                    groups[f'{variant}:{category}:{value}']['morphology']=summaries
    details=dict(rows=rows, groups=groups,
        selection_score=max(group['mean_physical_score'] for group in groups.values()),
        feature_relative_mse=float(np.mean([row['feature_relative_mse'] for row in rows])),
        feature_by_coarse={v:float(np.mean([r['feature_relative_mse'] for r in rows if r['coarse_variant']==v])) for v in VARIANTS},
        accepted=False, note='Worst profile/rare category across three coarse branches; both physical LODs, original-decoder interior proxy.')
    if preserve_valleys:
        scores=[np.mean([value['valley_iou'] for value in group['morphology'].values() if value is not None])
                for group in groups.values() if any(value is not None for value in group['morphology'].values())]
        if not scores:
            raise ValueError('No teacher-supported morphology groups for checkpoint selection.')
        details.update(physical_selection_score=details['selection_score'],selection_score=1.-float(min(scores)),
            selection='One minus worst group mean native/fine valley IoU; fixed recall/correlation and latent guards.')
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DATA/'crops/native-mixed-scenes-v2')
    parser.add_argument('--source',type=Path,default=DATA/'ckpt/native128x2-pilot/best.pt')
    parser.add_argument('--output',type=Path,default=DATA/'ckpt/native128x2-multilod-mixed-pilot')
    parser.add_argument('--statistics',type=Path,default=DATA/'eval/native-decoded-probe/channel-statistics.json')
    parser.add_argument('--controls',type=Path,default=DATA/'eval/native-decoded-controls')
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--eval-every',type=int,default=500)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--rare-fraction',type=float,default=.25)
    parser.add_argument('--fresh-noise-fraction',type=float,default=.25)
    parser.add_argument('--seed',type=int,default=20261013)
    parser.add_argument('--wait-job')
    parser.add_argument('--smoke',action='store_true',help='Three gradient steps, one captured train scene per coarse; partial bank allowed.')
    parser.add_argument('--preserve-valleys',action='store_true',help='Add amplitude/depth loss and fixed morphology guards; requires a new run.')
    args = parser.parse_args()
    if min(args.steps,args.eval_every) < 1 or not np.isfinite(args.lr) or args.lr <= 0 or any(
            not np.isfinite(v) or not 0 <= v <= 1 for v in (args.rare_fraction,args.fresh_noise_fraction)):
        parser.error('Positive finite training settings and fractions in[0,1] required.')
    if args.wait_job:
        from distill.run_training import wait_job
        wait_job(args.wait_job)
    root,output = external_path(args.dataset),external_path(args.output)
    manifest = json.loads((root/'manifest.json').read_text())
    if not args.smoke and json.loads((root/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Finish matched teacher capture before training.')
    train = [r for r in manifest['scenes'] if r['metadata']['split']=='train']
    val = [r for r in manifest['scenes'] if r['metadata']['split']=='val']
    train_seeds,val_seeds = ({r['metadata']['seed'] for r in rows} for rows in (train,val))
    if train_seeds & val_seeds or (train_seeds | val_seeds) & HOLDOUT_SEEDS:
        raise ValueError('Mixed variants must preserve independent world splits.')
    if not args.smoke and any(not (root/r['file']).exists() for r in train+val):
        raise ValueError('A planned matched teacher scene is missing.')
    stats = json.loads(args.statistics.read_text())
    if stats['split'] != 'train' or len(stats['scenes']) < 150 or stats['dataset_manifest_sha256'] != manifest['original_manifest_sha256']:
        raise ValueError('Use the fixed original-bank TRAIN variance, shared by all coarse variants.')
    control_paths = sorted(args.controls.glob('*/decoder-control.json'))
    controls = [json.loads(p.read_text()) for p in control_paths]
    profiles = {r['metadata']['profile'] for r in train+val}
    if ({r['file']['profile'] for r in controls} != profiles or any(r['status'] != 'complete' or
        r['decoder_node_max_abs'] > 1e-5 or not r['proxy_within_floor'] or
        r.get('control_basis') != 'same runtime latents, original decoder, full pipeline' for r in controls)):
        raise ValueError('Original-decoder same-input controls required for every profile.')
    source = torch.load(args.source,map_location='cpu',weights_only=True)
    source_path = teacher_source()
    if digest(source_path/'diffusion_pytorch_model.safetensors') != manifest['teacher_weights_sha256']:
        raise ValueError('The base teacher differs from the matched capture teacher.')
    contract = dict(config=source['config'], source_sha256=digest(args.source), source_step=source['step'],
        dataset_manifest_sha256=digest(root/'manifest.json'), statistics_sha256=digest(args.statistics),
        variance_basis='Fixed576 original-coarse TRAIN scenes; no validation moments',
        teacher_weights_sha256=digest(source_path/'diffusion_pytorch_model.safetensors'),
        decoder_weights_sha256=digest(source_path.parent/'decoder_model/diffusion_pytorch_model.safetensors'),
        controls={p.parent.name:digest(p) for p in control_paths},
        physical_weights=PHYSICAL_WEIGHTS, regularization_weights=REGULARIZATION_WEIGHTS,
        lr=args.lr,seed=args.seed,rare_fraction=args.rare_fraction,fresh_noise_fraction=args.fresh_noise_fraction,
        eval_every=args.eval_every,clip_norm=1.,selection='worst coarse/profile and coarse/rare physical two-LOD proxy',
        producer_sha256={name:digest(REPO/name) for name in ('distill/native_multilod_train.py',
            'distill/native_multilod.py','distill/native_decoded.py','distill/native_fusion.py',
            'distill/native_base.py','distill/native_fusion_train.py','distill/decoded_loss.py')})
    if args.preserve_valleys:
        from distill.native_amplitude import AMPLITUDE_WEIGHTS, amplitude_terms, morphology_degraded
        contract.update(preserve_valleys=True,amplitude_weights=AMPLITUDE_WEIGHTS,
            morphology_guard=dict(retention_absolute_margin=.01,correlation_absolute_margin=.02,scales=[2.,8.]),
            selection='worst-group native/fine valley IoU')
        contract['producer_sha256'].update({name:digest(REPO/name) for name in (
            'distill/native_amplitude.py','distill/terrain_shape.py')})
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text()) != contract:
        raise ValueError('Multi-LOD run contract changed; use a separate run.')
    atomic_json(output/'contract.json',contract)
    torch.set_num_threads(8);torch.manual_seed(args.seed);torch.backends.cudnn.benchmark = True
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    from terrain_inference import prepare_models
    from types import SimpleNamespace
    teacher = EDMUnet2D.from_pretrained(str(source_path)).to('cuda',torch.bfloat16).eval().requires_grad_(False)
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    replay = {}
    smoke_records = []
    for variant in VARIANTS:
        record = next(r for r in train if r['coarse_variant']==variant and (root/r['file']).exists())
        smoke_records.append(record)
        replay[variant] = check_teacher(teacher,root/record['file'])
        if not replay[variant]['passed']:
            raise RuntimeError('Matched original-base replay failed its measured precision floor.')
    atomic_json(output/'teacher-control.json',replay)
    decoder = OriginalDecoder('cuda');weight = linear_weight_window(64,'cuda',torch.float32)
    variance = torch.tensor(stats['variance'],device='cuda')
    saved = torch.load(output/'latest.pt',map_location='cpu',weights_only=True) if (output/'latest.pt').exists() else None
    model = NativeBase(source['config']).cuda().eval()
    model.load_state_dict(saved['model'] if saved else source['ema'])
    ema = NativeBase(source['config']).cuda().eval().requires_grad_(False)
    ema.load_state_dict(saved['ema'] if saved else model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.99),weight_decay=1e-4,fused=True)
    step,best = (saved['step'],saved['best_score']) if saved else (0,float('inf'))
    if step > args.steps:
        raise ValueError('Cannot resume to an earlier optimizer step.')
    if saved:
        optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    del saved,source

    def training_step(record,refresh=False):
        region = DecodedRegion(root/record['file'],'cuda')
        if refresh:
            region.refresh_noise(teacher,weight,int(generator.integers(1_000_000,2**31-1)))
        optimizer.zero_grad(set_to_none=True)
        result = decoded_region_forward(model,region,weight,checkpointed=True)
        if args.smoke:
            result['velocity0'].retain_grad();result['velocity1'].retain_grad()
        loss,terms = multilod_loss(result,region,decoder,teacher,variance)
        if args.preserve_valleys:
            extra=amplitude_terms(result,region,decoder)
            terms.update(extra)
            loss=loss+sum(AMPLITUDE_WEIGHTS[name]*value for name,value in extra.items())
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite multi-LOD physical loss.')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        gradients = [v.grad.abs().flatten(1).sum(1).detach().cpu().tolist()
                     for v in (result['velocity0'],result['velocity1'])] if args.smoke else None
        if gradients is not None and any(not np.isfinite(g).all() or min(g)<=0 for g in gradients):
            raise RuntimeError('Multi-LOD loss did not reach all34 windows.')
        optimizer.step()
        return dict(loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},grad_norm=float(norm),
                    window_gradients=gradients,coarse_variant=record['coarse_variant'],case_kind=record['case_kind'],
                    file=record['file'],noise_seed=getattr(region,'noise_seed',region.metadata['seed']),
                    peak_bytes=torch.cuda.max_memory_allocated())

    if args.smoke:
        rows = [training_step(r) for r in smoke_records]
        if any(p.grad is not None for p in decoder.parameters()):
            raise RuntimeError('The original decoder was not frozen.')
        atomic_json(output/'gpu-smoke.json',dict(status='complete',gpu=torch.cuda.get_device_name(),rows=rows,
            original_decoder_frozen=True,contract=contract,accepted=False))
        print(json.dumps(dict(smoke='complete',variants=[r['coarse_variant'] for r in rows])),flush=True)
        return
    started,last_save = time.monotonic(),time.monotonic()
    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum,stop)

    def checkpoint(name):
        atomic_write(output/name,lambda f:torch.save(dict(config=model.config,model=model.state_dict(),
            ema=ema.state_dict(),optimizer=optimizer.state_dict(),step=step,best_score=best,contract=contract,
            teacher_revision=REVISION,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),f))

    atomic_json(output/'run.json',dict(contract=contract,target_steps=args.steps,start_step=step,
        train_scenes=len(train),validation_scenes=len(val),accepted=False))
    if not (output/'baseline.json').exists():
        baseline = validate(ema,val,root,weight,decoder,variance,args.preserve_valleys)
        atomic_json(output/'baseline.json',baseline);best=baseline['selection_score']
        checkpoint('initial.pt');checkpoint('best.pt')
        append_json(output/'validation.jsonl',dict(step=0,**baseline))
        print(json.dumps(dict(step=0,selection_score=best,feature_relative_mse=baseline['feature_relative_mse'])),flush=True)
    baseline = json.loads((output/'baseline.json').read_text())
    guarded = False
    try:
        while step < args.steps and not STOP and not guarded:
            generator = np.random.default_rng(np.random.SeedSequence([args.seed,step]))
            record = choose_record(train,generator,args.rare_fraction)
            row = training_step(record,generator.random()<args.fresh_noise_fraction)
            step += 1
            with torch.no_grad():
                decay = min(.999,(1+step)/(10+step))
                for average,parameter in zip(ema.parameters(),model.parameters()):
                    average.lerp_(parameter,1-decay)
            if step%25==0 or step==1:
                row.update(step=step,elapsed_seconds=time.monotonic()-started)
                append_json(output/'train.jsonl',row);print(json.dumps(row),flush=True)
            if step%args.eval_every==0 or step==args.steps:
                details = validate(ema,val,root,weight,decoder,variance,args.preserve_valleys)
                guarded = step>=500 and (details['feature_relative_mse']>1.1*baseline['feature_relative_mse'] or any(
                    details['feature_by_coarse'][v]>1.15*baseline['feature_by_coarse'][v] for v in VARIANTS))
                if args.preserve_valleys and step>=500:
                    guarded=guarded or morphology_degraded(details['groups'],baseline['groups'])
                append_json(output/'validation.jsonl',dict(step=step,quality_guard=guarded,**details))
                if not guarded and details['selection_score']<best:
                    best=details['selection_score'];checkpoint('best.pt')
                checkpoint(f'step-{step:06d}.pt')
                print(json.dumps(dict(step=step,selection_score=details['selection_score'],quality_guard=guarded)),flush=True)
            if time.monotonic()-last_save>=120:
                checkpoint('latest.pt');last_save=time.monotonic()
                atomic_json(output/'status.json',dict(status='running',step=step,target_steps=args.steps,best_score=best))
        checkpoint('latest.pt')
        atomic_json(output/'status.json',dict(status='quality-guard' if guarded else 'interrupted' if STOP else 'complete',
            step=step,best_score=best,elapsed_seconds=time.monotonic()-started,accepted=False))
    except Exception as exc:
        checkpoint('latest.pt')
        atomic_json(output/'status.json',dict(status='failed',step=step,error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
